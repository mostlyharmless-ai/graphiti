"""Shared FalkorDB fulltext-query construction."""

import re

from graphiti_core.driver.falkordb import STOPWORDS
from graphiti_core.helpers import validate_group_ids

MAX_QUERY_LENGTH = 128

# watercooler fork mod (39d3eee / dea618b): FalkorDB's fulltext index can time out
# on queries with many OR terms because it must scan and union results per term;
# 8-10+ terms were observed to time out during episode indexing on large graphs.
# Keep the first N distinct terms — the most important keywords.
MAX_FULLTEXT_TERMS = 6

# FalkorDB separator characters that break text into tokens.
_SEPARATOR_MAP = str.maketrans(
    {
        ',': ' ',
        '.': ' ',
        '<': ' ',
        '>': ' ',
        '{': ' ',
        '}': ' ',
        '[': ' ',
        ']': ' ',
        '"': ' ',
        "'": ' ',
        ':': ' ',
        ';': ' ',
        '!': ' ',
        '@': ' ',
        '#': ' ',
        '$': ' ',
        '%': ' ',
        '^': ' ',
        '&': ' ',
        '*': ' ',
        '(': ' ',
        ')': ' ',
        '-': ' ',
        '+': ' ',
        '=': ' ',
        '~': ' ',
        '?': ' ',
        '|': ' ',
        '/': ' ',
        '\\': ' ',
        '`': ' ',
    }
)


def sanitize_falkor_fulltext_query(query: str) -> str:
    """Replace FalkorDB special characters with whitespace."""
    return ' '.join(query.translate(_SEPARATOR_MAP).split())


def _escape_fulltext_group_id(group_id: str) -> str:
    """Escape a validated group ID for RediSearch fulltext syntax."""
    return re.sub(r'([^a-zA-Z0-9])', r'\\\1', group_id)


def build_falkor_fulltext_query(
    query: str,
    group_ids: list[str] | None = None,
    max_query_length: int = MAX_QUERY_LENGTH,
) -> str:
    """Build a FalkorDB RedisSearch fulltext query."""
    validate_group_ids(group_ids)

    group_filter = ''
    if group_ids:
        escaped_group_ids = [f'"{_escape_fulltext_group_id(group_id)}"' for group_id in group_ids]
        group_filter = f'(@group_id:{"|".join(escaped_group_ids)})'

    filtered_words = [
        word
        for word in sanitize_falkor_fulltext_query(query).split()
        if word.lower() not in STOPWORDS
    ]
    if not filtered_words:
        return ''

    # watercooler fork mod: deduplicate (first occurrence wins) then cap term count.
    seen: set[str] = set()
    unique_words: list[str] = []
    for word in filtered_words:
        key = word.lower()
        if key not in seen:
            seen.add(key)
            unique_words.append(word)
    filtered_words = unique_words[:MAX_FULLTEXT_TERMS]

    sanitized_query = ' | '.join(filtered_words)
    if len(sanitized_query.split(' ')) + len(group_ids or []) >= max_query_length:
        return ''

    return f'{group_filter} ({sanitized_query})'
