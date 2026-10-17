"""Search-tuning knobs (watercooler fork): each is an env override whose default keeps
the previous behaviour. Offline: no driver, no services."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb import fulltext
from graphiti_core.search import search, search_utils
from graphiti_core.search.search_config import EdgeReranker, EdgeSearchConfig, EdgeSearchMethod
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import rrf

REPO = Path(__file__).resolve().parents[3]


def _module_value(module: str, name: str, env: dict) -> str:
    """`module.name` as imported in a fresh interpreter with `env` added (import-time knobs)."""
    clean = {k: v for k, v in os.environ.items() if not k.startswith('GRAPHITI_')}
    clean.update(env, PYTHONPATH=str(REPO))
    out = subprocess.run(
        [sys.executable, '-c', f'import {module} as m; print(repr(m.{name}))'],
        env=clean,
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    )
    return out.stdout.strip()


# --- RRF rank constant -----------------------------------------------------------

# X is first in leg A and absent from leg B; Y is second in A and third in B.
LEGS = [['X', 'Y', 'Z'], ['W', 'V', 'Y']]


def test_rrf_rank_const_1_lets_a_single_leg_top_hit_win():
    uuids, scores = rrf(LEGS, rank_const=1)
    assert uuids[0] == 'X'
    assert scores[0] == 1.0


def test_rrf_rank_const_60_prefers_agreement_across_legs():
    uuids, scores = rrf(LEGS, rank_const=60)
    assert uuids[:2] == ['Y', 'X']
    assert scores[0] == 1 / 61 + 1 / 62


def test_rrf_without_rank_const_uses_module_default(monkeypatch):
    monkeypatch.setattr(search_utils, 'RRF_RANK_CONST', 60)
    assert rrf(LEGS)[0][0] == 'Y'
    monkeypatch.setattr(search_utils, 'RRF_RANK_CONST', 1)
    assert rrf(LEGS)[0][0] == 'X'


def test_rrf_rank_const_default_and_env_override():
    mod = 'graphiti_core.search.search_utils'
    assert _module_value(mod, 'RRF_RANK_CONST', {}) == '1'
    assert _module_value(mod, 'RRF_RANK_CONST', {'GRAPHITI_RRF_RANK_CONST': '60'}) == '60'


# --- Fulltext term cap -----------------------------------------------------------


def test_max_fulltext_terms_default_and_env_override():
    mod = 'graphiti_core.driver.falkordb.fulltext'
    assert _module_value(mod, 'MAX_FULLTEXT_TERMS', {}) == '6'
    env = {'GRAPHITI_MAX_FULLTEXT_TERMS': '20'}
    assert _module_value(mod, 'MAX_FULLTEXT_TERMS', env) == '20'


# --- Fulltext term selection -----------------------------------------------------


def test_select_first_keeps_query_order_prefix():
    words = ['Sync', 'the', 'queue', 'reconciliation', 'daemon', 'backoff']
    assert fulltext.select_fulltext_terms(words, 3, 'first') == ['Sync', 'the', 'queue']


def test_select_rare_keeps_longest_terms_in_query_order():
    words = ['Sync', 'queue', 'reconciliation', 'daemon', 'backoff', 'retry']
    # longest: reconciliation(14), backoff(7), daemon(6); returned in query order
    got = fulltext.select_fulltext_terms(words, 3, 'rare')
    assert got == ['reconciliation', 'daemon', 'backoff']


def test_select_rare_breaks_length_ties_by_position():
    words = ['alpha', 'bravo', 'delta', 'kilo']
    assert fulltext.select_fulltext_terms(words, 2, 'rare') == ['alpha', 'bravo']


def test_select_rare_under_the_cap_is_unchanged():
    words = ['b', 'aaa', 'cc']
    assert fulltext.select_fulltext_terms(words, 5, 'rare') == words


def test_builder_uses_module_selection(monkeypatch):
    monkeypatch.setattr(fulltext, 'MAX_FULLTEXT_TERMS', 2)
    monkeypatch.setattr(fulltext, 'FULLTEXT_TERM_SELECTION', 'rare')
    q = fulltext.build_falkor_fulltext_query('Sync queue reconciliation daemon')
    assert q == ' (reconciliation | daemon)'
    monkeypatch.setattr(fulltext, 'FULLTEXT_TERM_SELECTION', 'first')
    assert fulltext.build_falkor_fulltext_query('Sync queue reconciliation daemon') == (
        ' (Sync | queue)'
    )


def test_term_selection_default_env_override_and_rejection():
    mod = 'graphiti_core.driver.falkordb.fulltext'
    assert _module_value(mod, 'FULLTEXT_TERM_SELECTION', {}) == "'first'"
    env = {'GRAPHITI_FULLTEXT_TERM_SELECTION': 'rare'}
    assert _module_value(mod, 'FULLTEXT_TERM_SELECTION', env) == "'rare'"
    try:
        _module_value(mod, 'FULLTEXT_TERM_SELECTION', {'GRAPHITI_FULLTEXT_TERM_SELECTION': 'x'})
    except subprocess.CalledProcessError as error:
        assert 'GRAPHITI_FULLTEXT_TERM_SELECTION' in error.stderr
    else:
        raise AssertionError('an unknown selection must fail at import')


# --- Candidate multiplier --------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('multiplier', [2, 4])
async def test_edge_search_fetches_multiplier_times_limit_per_leg(monkeypatch, multiplier):
    seen = {}

    async def fulltext_leg(driver, query, search_filter, group_ids, limit):
        seen['bm25'] = limit
        return []

    async def cosine_leg(driver, vector, src, tgt, search_filter, group_ids, limit, min_score):
        seen['cosine'] = limit
        return []

    monkeypatch.setattr(search, 'edge_fulltext_search', fulltext_leg)
    monkeypatch.setattr(search, 'edge_similarity_search', cosine_leg)
    monkeypatch.setattr(search, 'SEARCH_CANDIDATE_MULTIPLIER', multiplier)
    config = EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.rrf,
    )
    await search.edge_search(None, None, 'q', [0.0], None, config, SearchFilters(), limit=50)
    assert seen == {'bm25': 50 * multiplier, 'cosine': 50 * multiplier}


def test_candidate_multiplier_default_and_env_override():
    mod = 'graphiti_core.search.search'
    assert _module_value(mod, 'SEARCH_CANDIDATE_MULTIPLIER', {}) == '2'
    env = {'GRAPHITI_SEARCH_CANDIDATE_MULTIPLIER': '4'}
    assert _module_value(mod, 'SEARCH_CANDIDATE_MULTIPLIER', env) == '4'


# --- FalkorDB edge-embedding load ---------------------------------------------------


class _EmbeddingDriver:
    graph_operations_interface = None
    search_interface = None

    def __init__(self, provider):
        self.provider = provider
        self.calls = []

    async def execute_query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return (
            [{'uuid': 'e1', 'fact_embedding': [1.0, 0.0]}, {'uuid': 'e2', 'fact_embedding': None}],
            None,
            None,
        )


def _edge(uuid):
    return SimpleNamespace(uuid=uuid)


@pytest.mark.asyncio
async def test_falkordb_edge_embeddings_use_a_directed_label_free_match():
    driver = _EmbeddingDriver(GraphProvider.FALKORDB)
    got = await search_utils.get_embeddings_for_edges(driver, [_edge('e1'), _edge('e2')])
    query, kwargs = driver.calls[0]
    flat = ' '.join(query.split())
    assert 'MATCH ()-[e:RELATES_TO]->()' in flat
    assert ':Entity' not in flat
    assert kwargs['edge_uuids'] == ['e1', 'e2']
    assert got == {'e1': [1.0, 0.0]}


@pytest.mark.asyncio
async def test_neo4j_edge_embeddings_keep_the_generic_match():
    driver = _EmbeddingDriver(GraphProvider.NEO4J)
    await search_utils.get_embeddings_for_edges(driver, [_edge('e1')])
    assert '(n:Entity)-[e:RELATES_TO]-(m:Entity)' in driver.calls[0][0]
