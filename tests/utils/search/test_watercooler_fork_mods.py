"""Regression coverage for the watercooler fork's FalkorDB search mods.

These behaviours live in code that upstream keeps refactoring; they must
survive every upstream merge (see v0.29.3-wc1 sync).
"""

from types import SimpleNamespace

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb.fulltext import (
    MAX_FULLTEXT_TERMS,
    build_falkor_fulltext_query,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    HNSW_OVERSAMPLE_FACTOR,
    edge_similarity_search,
    node_similarity_search,
)


# --- fulltext term dedup + cap (fork mods 39d3eee / dea618b) -------------------


def test_fulltext_query_dedups_terms_case_insensitively():
    q = build_falkor_fulltext_query('Sync sync SYNC queue Queue')
    assert q == ' (Sync | queue)'


def test_fulltext_query_caps_term_count():
    words = [f'term{i}' for i in range(MAX_FULLTEXT_TERMS + 4)]
    q = build_falkor_fulltext_query(' '.join(words))
    kept = q.strip(' ()').split(' | ')
    assert kept == words[:MAX_FULLTEXT_TERMS]


def test_fulltext_query_group_filter_survives_cap():
    words = [f'term{i}' for i in range(MAX_FULLTEXT_TERMS + 1)]
    q = build_falkor_fulltext_query(' '.join(words), ['grp_a'])
    assert q.startswith('(@group_id:"grp\\_a")')
    assert q.count(' | ') == MAX_FULLTEXT_TERMS - 1


# --- HNSW paths ---------------------------------------------------------------


class _CapturingDriver:
    """FalkorDB-shaped driver that records execute_query kwargs."""

    provider = GraphProvider.FALKORDB
    search_interface = None

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def execute_query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [], None, None


@pytest.mark.asyncio
async def test_edge_similarity_empty_edge_uuids_short_circuits():
    """edge_uuids=[] means 'these candidate edges: none' — never a group-wide search."""
    driver = _CapturingDriver()
    result = await edge_similarity_search(
        driver, [0.1, 0.2], None, None, SearchFilters(edge_uuids=[]), group_ids=['g'], limit=10
    )
    assert result == []
    assert driver.calls == []


@pytest.mark.asyncio
async def test_edge_similarity_oversamples_hnsw_k_when_filtered():
    driver = _CapturingDriver()
    await edge_similarity_search(
        driver, [0.1, 0.2], None, None, SearchFilters(), group_ids=['g'], limit=10
    )
    query, kwargs = driver.calls[0]
    assert '$hnsw_k' in query and 'queryRelationships' in query
    assert kwargs['hnsw_k'] == 10 * HNSW_OVERSAMPLE_FACTOR
    assert kwargs['limit'] == 10


@pytest.mark.asyncio
async def test_edge_similarity_uses_plain_k_when_unfiltered():
    driver = _CapturingDriver()
    await edge_similarity_search(driver, [0.1, 0.2], None, None, SearchFilters(), limit=10)
    _, kwargs = driver.calls[0]
    assert kwargs['hnsw_k'] == 10


@pytest.mark.asyncio
async def test_node_similarity_oversamples_hnsw_k_when_filtered():
    driver = _CapturingDriver()
    await node_similarity_search(driver, [0.1, 0.2], SearchFilters(), group_ids=['g'], limit=7)
    query, kwargs = driver.calls[0]
    assert '$hnsw_k' in query and 'queryNodes' in query
    assert kwargs['hnsw_k'] == 7 * HNSW_OVERSAMPLE_FACTOR
    assert kwargs['limit'] == 7


# --- HNSW distance → similarity conversion ------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('node_labels', [None, ['Entity']])
async def test_edge_hnsw_converts_distance_to_similarity(node_labels):
    """queryRelationships YIELDs cosine distance; scores must be (2 - d) / 2."""
    driver = _CapturingDriver()
    await edge_similarity_search(
        driver, [0.1], None, None, SearchFilters(node_labels=node_labels), group_ids=['g']
    )
    query, _ = driver.calls[0]
    assert '(2 - score) / 2 AS score' in query
    assert query.index('(2 - score) / 2') < query.index('score > $min_score')


@pytest.mark.asyncio
async def test_node_hnsw_converts_distance_to_similarity():
    driver = _CapturingDriver()
    await node_similarity_search(driver, [0.1], SearchFilters(), group_ids=['g'])
    query, _ = driver.calls[0]
    assert '(2 - score) / 2 AS score' in query
    assert query.index('(2 - score) / 2') < query.index('score > $min_score')


# --- adaptive widening + candidate-scoped exact path --------------------------

from graphiti_core.search.search_utils import HNSW_MAX_K, HNSW_WIDEN_FACTOR  # noqa: E402


class _ShortDriver(_CapturingDriver):
    """Returns ``rows_per_call`` rows each call so widening can be observed."""

    def __init__(self, rows_per_call: int):
        super().__init__()
        self.rows_per_call = rows_per_call

    async def execute_query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [None] * self.rows_per_call, None, None


@pytest.mark.asyncio
async def test_filtered_hnsw_widens_k_until_cap_when_short(monkeypatch):
    import graphiti_core.search.search_utils as su

    monkeypatch.setattr(su, 'get_entity_edge_from_record', lambda r, p: r)
    driver = _ShortDriver(rows_per_call=0)
    await edge_similarity_search(driver, [0.1], None, None, SearchFilters(), group_ids=['g'], limit=10)
    ks = [kw['hnsw_k'] for _, kw in driver.calls]
    expected, k = [], 10 * HNSW_OVERSAMPLE_FACTOR
    while True:
        expected.append(k)
        if k >= HNSW_MAX_K:
            break
        k = min(k * HNSW_WIDEN_FACTOR, HNSW_MAX_K)
    assert ks == expected  # 50, 200, 800, 3200, 4096 — bounded exhaustion


@pytest.mark.asyncio
async def test_filtered_hnsw_stops_once_limit_is_met(monkeypatch):
    import graphiti_core.search.search_utils as su

    monkeypatch.setattr(su, 'get_entity_node_from_record', lambda r, p: r)
    driver = _ShortDriver(rows_per_call=10)
    await node_similarity_search(driver, [0.1], SearchFilters(), group_ids=['g'], limit=10)
    assert [kw['hnsw_k'] for _, kw in driver.calls] == [10 * HNSW_OVERSAMPLE_FACTOR]


@pytest.mark.asyncio
async def test_unfiltered_hnsw_runs_once_with_plain_k(monkeypatch):
    import graphiti_core.search.search_utils as su

    monkeypatch.setattr(su, 'get_entity_edge_from_record', lambda r, p: r)
    driver = _ShortDriver(rows_per_call=0)
    await edge_similarity_search(driver, [0.1], None, None, SearchFilters(), limit=10)
    assert [kw['hnsw_k'] for _, kw in driver.calls] == [10]


@pytest.mark.asyncio
async def test_small_explicit_edge_uuids_use_exact_candidate_scoped_path():
    driver = _CapturingDriver()
    await edge_similarity_search(
        driver, [0.1], None, None, SearchFilters(edge_uuids=['a', 'b']), group_ids=['g'], limit=10
    )
    query, kwargs = driver.calls[0]
    assert 'queryRelationships' not in query
    assert 'vec.cosineDistance' in query and 'e.uuid in $edge_uuids' in query
    assert kwargs['edge_uuids'] == ['a', 'b']
    assert len(driver.calls) == 1
