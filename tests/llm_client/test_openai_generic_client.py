import json
from types import SimpleNamespace

import openai
import pytest
from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import EmptyResponseError, RateLimitError
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message


class DummyChatCompletions:
    def __init__(self, content: str = '{}', error: Exception | None = None):
        self.create_calls: list[dict] = []
        self._content = content
        self._error = error

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        if self._error is not None:
            raise self._error
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class DummyChat:
    def __init__(self, completions: DummyChatCompletions):
        self.completions = completions


class DummyClient:
    def __init__(self, completions: DummyChatCompletions):
        self.chat = DummyChat(completions)


class ResponseModel(BaseModel):
    foo: str


def _messages() -> list[Message]:
    return [
        Message(role='system', content='system message'),
        Message(role='user', content='user message'),
    ]


def _make_client(content: str = '{"foo": "bar"}', error: Exception | None = None, **kwargs):
    completions = DummyChatCompletions(content=content, error=error)
    client = OpenAIGenericClient(
        config=LLMConfig(api_key='test', model='test-model'),
        client=DummyClient(completions),
        **kwargs,
    )
    return client, completions


@pytest.mark.asyncio
async def test_defaults_to_json_schema_response_format():
    client, completions = _make_client()

    await client.generate_response(_messages(), response_model=ResponseModel)

    response_format = completions.create_calls[0]['response_format']
    assert response_format['type'] == 'json_schema'
    assert response_format['json_schema']['name'] == 'ResponseModel'
    assert response_format['json_schema']['schema'] == ResponseModel.model_json_schema()


@pytest.mark.asyncio
async def test_json_schema_mode_does_not_inject_schema_into_prompt():
    client, completions = _make_client()
    messages = _messages()

    await client.generate_response(messages, response_model=ResponseModel)

    sent_user_content = completions.create_calls[0]['messages'][-1]['content']
    assert 'Respond ONLY with a JSON object of exactly this shape' not in sent_user_content


@pytest.mark.asyncio
async def test_json_object_mode_injects_example_instance_not_schema():
    client, completions = _make_client(structured_output_mode='json_object')

    await client.generate_response(_messages(), response_model=ResponseModel)

    call = completions.create_calls[0]
    assert call['response_format'] == {'type': 'json_object'}
    # The API does not enforce the schema in json_object mode, so the prompt must carry
    # guidance — but as an example instance, NOT the raw schema (weak providers echo a
    # pasted schema back verbatim: 'properties'/'required' as top-level keys).
    sent_user_content = call['messages'][-1]['content']
    assert 'Respond ONLY with a JSON object of exactly this shape' in sent_user_content
    assert json.dumps({'foo': ''}) in sent_user_content
    assert json.dumps(ResponseModel.model_json_schema()) not in sent_user_content


@pytest.mark.asyncio
async def test_no_response_model_uses_json_object_without_injection():
    client, completions = _make_client(content='{"any": "thing"}')

    result = await client.generate_response(_messages())

    call = completions.create_calls[0]
    assert call['response_format'] == {'type': 'json_object'}
    assert (
        'Respond ONLY with a JSON object of exactly this shape'
        not in call['messages'][-1]['content']
    )
    assert result == {'any': 'thing'}


@pytest.mark.asyncio
async def test_rate_limit_error_is_translated():
    rate_limit = openai.RateLimitError(
        message='slow down',
        response=SimpleNamespace(status_code=429, headers={}, request=None),
        body=None,
    )
    client, _ = _make_client(error=rate_limit)

    # Assert translation at the _generate_response level. Going through generate_response
    # would invoke the inherited tenacity retry wrapper (RateLimitError is retryable), which
    # adds real backoff sleeps and would make this unit test slow.
    with pytest.raises(RateLimitError):
        await client._generate_response(_messages(), response_model=ResponseModel)


@pytest.mark.asyncio
async def test_empty_content_raises_empty_response_error():
    # Empty body (flaky endpoint / refusal / length cutoff) must raise a clear error,
    # not a cryptic JSONDecodeError from json.loads(''). Asserted at _generate_response
    # level: EmptyResponseError is retryable, so generate_response would invoke the real
    # backoff retry and slow this unit test.
    client, _ = _make_client(content='')

    with pytest.raises(EmptyResponseError):
        await client._generate_response(_messages(), response_model=ResponseModel)


def test_empty_response_error_is_retryable():
    # An empty body is treated as a transient provider hiccup (common on local/compatible
    # endpoints), so the base retry wrapper retries it rather than failing on first try.
    from graphiti_core.llm_client.client import is_server_or_retry_error

    assert is_server_or_retry_error(EmptyResponseError('empty')) is True


@pytest.mark.asyncio
async def test_strips_markdown_code_fence_before_parsing():
    # Local/compatible models (e.g. Ollama/gemma) often wrap JSON in a ```json fence even
    # under a structured response_format; the client must strip it before json.loads.
    fenced = '```json\n{"foo": "bar"}\n```'
    client, _ = _make_client(content=fenced)

    result = await client.generate_response(_messages(), response_model=ResponseModel)

    assert result == {'foo': 'bar'}


def _extract_example_json(instruction: str) -> dict:
    """The example instance is the only line in the instruction starting with '{'."""
    for line in instruction.splitlines():
        if line.startswith('{'):
            return json.loads(line)
    raise AssertionError('no example JSON line found in instruction')


def test_example_instruction_is_instance_shaped_for_edge_duplicate():
    from graphiti_core.llm_client.openai_generic_client import example_instruction_for_model
    from graphiti_core.prompts.dedupe_edges import EdgeDuplicate

    instruction = example_instruction_for_model(EdgeDuplicate)
    example = _extract_example_json(instruction)
    assert set(example.keys()) == set(EdgeDuplicate.model_fields.keys())
    # The example itself must validate against the model.
    EdgeDuplicate.model_validate(example)
    # Field descriptions carried over as plain text for semantic guidance.
    assert 'duplicate_facts' in instruction
    assert 'EXISTING FACTS' in instruction


def test_example_instruction_recurses_nested_models_and_optionals():
    from pydantic import Field

    from graphiti_core.llm_client.openai_generic_client import example_instruction_for_model

    class Nested(BaseModel):
        label: str = Field(..., description='a label')
        score: float = Field(..., description='a score')

    class Outer(BaseModel):
        items: list[int] = Field(..., description='list of idx values')
        nested: Nested = Field(..., description='nested model')
        maybe: str | None = Field(None, description='optional field')

    example = _extract_example_json(example_instruction_for_model(Outer))
    assert example['items'] == []
    assert example['nested'] == {'label': '', 'score': 0}
    Outer.model_validate(example)


def test_example_value_handles_enum_anyof_and_scalars():
    from graphiti_core.llm_client.openai_generic_client import _example_value

    assert _example_value({'enum': ['a', 'b']}, {}) == 'a'
    assert _example_value({'anyOf': [{'type': 'null'}, {'type': 'integer'}]}, {}) == 0
    assert _example_value({'type': 'boolean'}, {}) is False


def test_array_of_objects_gets_exemplar_item_and_field_docs():
    # The main extraction models are arrays of objects; the model needs the item
    # shape (the old schema paste carried it via $defs). Scalar arrays stay [].
    from graphiti_core.llm_client.openai_generic_client import example_instruction_for_model
    from graphiti_core.prompts.extract_nodes import ExtractedEntities

    instruction = example_instruction_for_model(ExtractedEntities)
    example = _extract_example_json(instruction)
    assert len(example['extracted_entities']) == 1
    item = example['extracted_entities'][0]
    assert set(item.keys()) == {'name', 'entity_type_id', 'episode_indices'}
    ExtractedEntities.model_validate(example)
    # Item fields documented one level deep.
    assert 'each item has:' in instruction
    assert 'entity_type_id' in instruction
    # Scalar-array fields keep an empty-list example (no suggested answer bias).
    from graphiti_core.prompts.dedupe_edges import EdgeDuplicate

    dedupe_example = _extract_example_json(example_instruction_for_model(EdgeDuplicate))
    assert dedupe_example == {'duplicate_facts': [], 'contradicted_facts': []}


@pytest.mark.asyncio
async def test_non_retryable_error_is_not_retried():
    # The old hand-rolled re-prompt loop is gone. Retry is now delegated to the base
    # tenacity wrapper, which only retries transient errors (RateLimitError /
    # JSONDecodeError). A non-retryable error (e.g. ValueError) propagates after a
    # single create call.
    client, completions = _make_client(error=ValueError('bad response'))

    with pytest.raises(ValueError):
        await client.generate_response(_messages(), response_model=ResponseModel)

    assert len(completions.create_calls) == 1
