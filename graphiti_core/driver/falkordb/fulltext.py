"""Shared FalkorDB fulltext-query construction."""

import os
import re

from graphiti_core.driver.falkordb import STOPWORDS
from graphiti_core.helpers import validate_group_ids

MAX_QUERY_LENGTH = 128

# watercooler fork mod (39d3eee / dea618b): FalkorDB's fulltext index can time out
# on queries with many OR terms because it must scan and union results per term;
# 8-10+ terms were observed to time out during episode indexing on large graphs.
# Keep the first N distinct terms — the most important keywords.
# Default unchanged (6); override with GRAPHITI_MAX_FULLTEXT_TERMS. With 6, an
# entry-length query is reduced to roughly its title.
MAX_FULLTEXT_TERMS = int(os.environ.get('GRAPHITI_MAX_FULLTEXT_TERMS', '6'))

# Which distinct terms survive the cap: 'first' (default, unchanged) keeps the first N
# in query order; 'rare' keeps the N most specific. Document frequency is not available
# here (the builder has no driver), so 'rare' uses a proxy: longest terms first, ties by
# position. Override with GRAPHITI_FULLTEXT_TERM_SELECTION.
FULLTEXT_TERM_SELECTIONS = ('first', 'rare')
FULLTEXT_TERM_SELECTION = os.environ.get('GRAPHITI_FULLTEXT_TERM_SELECTION', 'first')
if FULLTEXT_TERM_SELECTION not in FULLTEXT_TERM_SELECTIONS:
    raise ValueError(
        f'GRAPHITI_FULLTEXT_TERM_SELECTION must be one of {FULLTEXT_TERM_SELECTIONS}, '
        f'got {FULLTEXT_TERM_SELECTION!r}'
    )

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


def select_fulltext_terms(
    words: list[str],
    limit: int | None = None,
    selection: str | None = None,
) -> list[str]:
    """Keep `limit` of the distinct `words` by `selection`, in their query order."""
    limit = MAX_FULLTEXT_TERMS if limit is None else limit
    selection = FULLTEXT_TERM_SELECTION if selection is None else selection
    if selection == 'first' or len(words) <= limit:
        return words[:limit]
    keep = sorted(range(len(words)), key=lambda i: (-len(words[i]), i))[:limit]
    return [words[i] for i in sorted(keep)]


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
    filtered_words = select_fulltext_terms(unique_words)

    sanitized_query = ' | '.join(filtered_words)
    if len(sanitized_query.split(' ')) + len(group_ids or []) >= max_query_length:
        return ''

    return f'{group_filter} ({sanitized_query})'
