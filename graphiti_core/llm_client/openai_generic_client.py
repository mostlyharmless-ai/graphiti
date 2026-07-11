"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import re
import typing
from typing import Any, Literal

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient, get_extraction_language_instruction
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import EmptyResponseError, RateLimitError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4.1-mini'

StructuredOutputMode = Literal['json_schema', 'json_object']


def _example_value(schema: dict[str, Any], defs: dict[str, Any], depth: int = 0) -> Any:
    """Minimal plausible instance value for a JSON-Schema fragment."""
    if depth > 6:
        return None
    if '$ref' in schema:
        return _example_value(defs.get(schema['$ref'].split('/')[-1], {}), defs, depth + 1)
    if schema.get('enum'):
        return schema['enum'][0]
    for combinator in ('anyOf', 'oneOf', 'allOf'):
        options = schema.get(combinator)
        if options:
            non_null = [o for o in options if o.get('type') != 'null']
            return _example_value(non_null[0] if non_null else options[0], defs, depth + 1)
    schema_type = schema.get('type')
    if schema_type == 'array':
        return []
    if schema_type == 'string':
        return ''
    if schema_type == 'integer':
        return 0
    if schema_type == 'number':
        return 0
    if schema_type == 'boolean':
        return False
    if schema_type == 'object' or 'properties' in schema:
        return {
            name: _example_value(sub, defs, depth + 1)
            for name, sub in schema.get('properties', {}).items()
        }
    return None


def example_instruction_for_model(response_model: type[BaseModel]) -> str:
    """Prompt tail describing the expected reply as an example instance, not a schema.

    Appending ``model_json_schema()`` verbatim leads weaker ``json_object``
    providers (e.g. DeepSeek) to occasionally echo the schema wrapper itself
    back — responses whose top-level keys are ``properties``/``required`` —
    which then fail response-model validation and abort the whole episode.
    An example-shaped skeleton plus a plain-text field list gives the model
    nothing schema-shaped to parrot. Measured on deepseek-v4-flash with the
    ``dedupe_edges.resolve_edge`` prompt: ~10% of calls failed with the schema
    paste vs ~0-2% with the example form (N=20 per arm, temperature 1).
    """
    schema = response_model.model_json_schema()
    defs = schema.get('$defs', {})
    properties: dict[str, Any] = schema.get('properties', {})
    required = set(schema.get('required', properties.keys()))
    example = {name: _example_value(sub, defs) for name, sub in properties.items()}
    field_lines = []
    for name, sub in properties.items():
        resolved = sub
        if '$ref' in resolved:
            resolved = defs.get(resolved['$ref'].split('/')[-1], {})
        field_type = sub.get('type', resolved.get('type', 'object'))
        line = f'- {name} ({field_type}, {"required" if name in required else "optional"})'
        description = sub.get('description', resolved.get('description', ''))
        if description:
            line += f': {description}'
        field_lines.append(line)
    return (
        '\n\nRespond ONLY with a JSON object of exactly this shape — fill in the values; '
        'it is the answer instance, not a schema (do not output "properties" or '
        '"required" keys):\n\n'
        f'{json.dumps(example)}\n\nFields:\n' + '\n'.join(field_lines)
    )


class OpenAIGenericClient(LLMClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the LLMClient and provides methods to initialize the client,
    get an embedder, and generate responses from the language model.

    This client targets any OpenAI-compatible ``/chat/completions`` endpoint (OpenAI,
    vLLM, llama.cpp, Ollama, DeepSeek, Together, etc.). It defaults to native
    ``json_schema`` structured output (constrained decoding) and can fall back to
    ``json_object`` for the minority of providers that do not support ``json_schema``.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.
        structured_output_mode (StructuredOutputMode): How structured output is requested.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = 16384,
        structured_output_mode: StructuredOutputMode = 'json_schema',
    ):
        """
        Initialize the OpenAIGenericClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 16384 (16K) for better compatibility with local models.
            structured_output_mode (StructuredOutputMode): Whether to request structured
                output via native ``json_schema`` (the default, uses constrained decoding)
                or to fall back to ``json_object``. Set to ``'json_object'`` for providers
                that do not support the ``json_schema`` response format (e.g. DeepSeek); in
                that mode the schema is injected into the prompt instead of being enforced
                by the API.

        """
        # removed caching to simplify the `generate_response` override
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        # Override max_tokens to support higher limits for local models
        self.max_tokens = max_tokens
        self.structured_output_mode: StructuredOutputMode = structured_output_mode

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

    def _build_response_format(self, response_model: type[BaseModel] | None) -> dict[str, Any]:
        """Build the ``response_format`` payload for the chat completion request.

        Uses native ``json_schema`` when a response model is provided and the client is in
        ``json_schema`` mode; otherwise falls back to ``json_object``. In ``json_object``
        mode the schema is not enforced by the API — ``generate_response`` injects it into
        the prompt instead.
        """
        if response_model is None or self.structured_output_mode == 'json_object':
            return {'type': 'json_object'}

        # Native json_schema. We intentionally omit "strict": true — strict mode requires
        # the schema to meet OpenAI's strict subset (additionalProperties: false, every
        # field required), which raw model_json_schema() routinely violates (that's why the
        # dedicated OpenAIClient uses responses.parse() instead). So adherence is best-effort
        # on OpenAI-proper; constrained-decoding servers (vLLM, llama.cpp) still enforce it.
        return {
            'type': 'json_schema',
            'json_schema': {
                'name': getattr(response_model, '__name__', 'structured_response'),
                'schema': response_model.model_json_schema(),
            },
        }

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Strip a wrapping markdown code fence from a JSON payload.

        OpenAI-compatible models served via Ollama/llama.cpp etc. frequently wrap their
        output in a ```json … ``` fence even when a json_schema/json_object response_format
        is requested, which breaks a bare ``json.loads``. No-op when there is no fence.
        """
        stripped = text.strip()
        if stripped.startswith('```'):
            stripped = re.sub(r'^```[a-zA-Z0-9_-]*[ \t]*\r?\n?', '', stripped)
            stripped = re.sub(r'\r?\n?```[ \t]*$', '', stripped)
        return stripped.strip()

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})
        try:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                response_format=self._build_response_format(response_model),  # type: ignore[arg-type]
            )
            result = response.choices[0].message.content or ''
            # An empty body (refusal, length finish_reason, or a flaky endpoint) would make
            # json.loads raise a cryptic JSONDecodeError; surface a clear error instead.
            if not result:
                raise EmptyResponseError('LLM returned an empty response')
            # Many OpenAI-compatible/local models wrap JSON in a ```json fence even under a
            # structured response_format; strip it before parsing.
            return json.loads(self._strip_code_fences(result))
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
    ) -> dict[str, typing.Any]:
        self._apply_attribute_extraction_preamble(messages, attribute_extraction)
        if max_tokens is None:
            max_tokens = self.max_tokens

        # In json_object fallback mode the API does not enforce the schema, so guide the
        # model with an example-shaped instruction. Deliberately NOT the raw JSON Schema:
        # weaker providers (e.g. DeepSeek) sometimes echo a pasted schema back verbatim
        # ('properties'/'required' as top-level keys), failing validation. In json_schema
        # mode the schema is enforced via response_format, so no prompt injection is needed.
        if response_model is not None and self.structured_output_mode == 'json_object':
            messages[-1].content += example_instruction_for_model(response_model)

        # Add multilingual extraction instructions
        messages[0].content += get_extraction_language_instruction(group_id)

        # Wrap entire operation in tracing span
        with self.tracer.start_span('llm.generate') as span:
            attributes = {
                'llm.provider': 'openai',
                'model.size': model_size.value,
                'max_tokens': max_tokens,
            }
            if prompt_name:
                attributes['prompt.name'] = prompt_name
            span.add_attributes(attributes)

            try:
                # Delegate to the base tenacity wrapper so transient JSONDecodeError /
                # RateLimitError get backoff-retried (4 attempts) — most relevant in the
                # json_object fallback path for less-reliable providers. This is the clean
                # retry mechanism (same pattern as Gliner2Client); the old hand-rolled
                # re-prompt loop is intentionally not reinstated.
                return await self._generate_response_with_retry(
                    messages, response_model, max_tokens=max_tokens, model_size=model_size
                )
            except Exception as e:
                span.set_status('error', str(e))
                span.record_exception(e)
                raise
