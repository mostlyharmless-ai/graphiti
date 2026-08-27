"""Query-shape regression tests for edge_bfs_search (no live database).

Completes #1500 on the generic path in ``graphiti_core/search/search_utils.py``,
which is what executes today for FalkorDB and Neo4j (``driver.search_interface``
is never assigned, so the driver-level operations modules patched by #1500 are
not reached at runtime).

The BFS query must consume the relationships already produced by
``UNWIND relationships(path)`` directly (``startNode``/``endNode``) instead of
re-matching every hit by uuid against the whole graph
(``MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]-(m:Entity)``), which caused
an O(matches x graph) scan per row. Because the BFS path traverses
``RELATES_TO|MENTIONS``, the explicit ``type(e) = 'RELATES_TO'`` guard must be
kept: the old re-MATCH filtered MENTIONS edges out implicitly.

Mirrors the RecordingExecutor approach of tests/test_falkordb_search_ops.py
(#1500): a recording driver captures the emitted Cypher, so no database
connection is required.
"""

from typing import Any

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neptune.operations.search_ops import NeptuneSearchOperations
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    edge_bfs_search,
    edge_fulltext_search,
    node_bfs_search,
)


class RecordingDriver:
    """Captures the Cypher and params a search function emits, returning no rows."""

    provider = GraphProvider.NEO4J
    search_interface = None
    fulltext_syntax = ''

    def __init__(self):
        self.cypher_query = ''
        self.params: dict[str, Any] = {}

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.cypher_query = cypher_query_
        self.params = kwargs
        return [], None, None


class RecordingNeptuneDriver(RecordingDriver):
    provider = GraphProvider.NEPTUNE

    def run_aoss_query(self, index_name: str, query: str):
        return {
            'hits': {
                'total': {'value': 1},
                'hits': [{'_source': {'uuid': 'edge-uuid'}, '_score': 1}],
            }
        }


@pytest.mark.asyncio
async def test_edge_bfs_search_consumes_path_relationships_directly():
    driver = RecordingDriver()

    await edge_bfs_search(
        driver,  # type: ignore[arg-type]
        ['origin-uuid'],
        2,
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert 'WITH rel AS e, startNode(rel) AS n, endNode(rel) AS m' in driver.cypher_query
    # The path traverses RELATES_TO|MENTIONS; the old re-MATCH filtered MENTIONS
    # implicitly, so the rewrite must keep an explicit type guard.
    assert "WHERE type(e) = 'RELATES_TO'" in driver.cypher_query
    # The expensive per-row re-MATCH must be gone.
    assert 'uuid: rel.uuid' not in driver.cypher_query
    # Downstream filters still see the relationship. The return helper derives
    # direction from it so undirected matching cannot swap edge endpoints.
    assert 'e.group_id IN $group_ids' in driver.cypher_query
    assert 'startNode(e).uuid AS source_node_uuid' in driver.cypher_query
    assert 'endNode(e).uuid AS target_node_uuid' in driver.cypher_query


@pytest.mark.asyncio
async def test_edge_fulltext_search_not_rewritten_by_bfs_fix():
    """The fulltext variant of this rewrite belongs to #1500; this change is
    scoped to the BFS query and must leave the fulltext query alone."""
    driver = RecordingDriver()

    await edge_fulltext_search(
        driver,  # type: ignore[arg-type]
        'api test system',
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert 'WITH rel AS e, startNode(rel) AS n, endNode(rel) AS m' not in driver.cypher_query
    assert "type(e) = 'RELATES_TO'" not in driver.cypher_query


@pytest.mark.asyncio
async def test_neptune_generic_edge_searches_return_reference_time():
    driver = RecordingNeptuneDriver()

    await edge_bfs_search(
        driver,  # type: ignore[arg-type]
        ['origin-uuid'],
        2,
        SearchFilters(),
        group_ids=['group-a'],
    )
    assert 'e.reference_time AS reference_time' in driver.cypher_query

    await edge_fulltext_search(
        driver,  # type: ignore[arg-type]
        'api test system',
        SearchFilters(),
        group_ids=['group-a'],
    )
    assert 'e.reference_time AS reference_time' in driver.cypher_query


@pytest.mark.asyncio
async def test_neptune_operations_bfs_search_returns_reference_time():
    executor = RecordingNeptuneDriver()

    await NeptuneSearchOperations().edge_bfs_search(
        executor,  # type: ignore[arg-type]
        ['origin-uuid'],
        2,
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert 'e.reference_time AS reference_time' in executor.cypher_query


class RecordingKuzuDriver(RecordingDriver):
    provider = GraphProvider.KUZU


class InterfaceDriver(RecordingDriver):
    """A driver that delegates to a search_interface, as non-generic drivers do."""

    search_interface = object()


@pytest.mark.asyncio
async def test_edge_bfs_search_is_outgoing_only_by_default():
    """watercooler fork mod: the default must stay outgoing-only so existing
    callers keep upstream semantics."""
    driver = RecordingDriver()

    await edge_bfs_search(
        driver,  # type: ignore[arg-type]
        ['origin-uuid'],
        2,
        SearchFilters(),
        group_ids=['group-a'],
    )

    assert ']->(:Entity)' in driver.cypher_query


@pytest.mark.asyncio
async def test_edge_bfs_search_undirected_expands_both_directions():
    """watercooler fork mod: bfs_undirected traverses inbound edges too, while the
    returned edge keeps its own direction (startNode/endNode, not traversal order).

    Rev. 4 of the T2 multi-hop brainstorm corrected "bidirectional seeding" to
    dual-origin *outgoing* traversal; a real both-direction arm needs this.
    """
    driver = RecordingDriver()

    await edge_bfs_search(
        driver,  # type: ignore[arg-type]
        ['origin-uuid'],
        2,
        SearchFilters(),
        group_ids=['group-a'],
        bfs_undirected=True,
    )

    assert ']-(:Entity)' in driver.cypher_query
    assert ']->(:Entity)' not in driver.cypher_query
    # Direction of each returned edge is still its own, not the traversal's.
    assert 'WITH rel AS e, startNode(rel) AS n, endNode(rel) AS m' in driver.cypher_query
    assert 'startNode(e).uuid AS source_node_uuid' in driver.cypher_query
    assert 'endNode(e).uuid AS target_node_uuid' in driver.cypher_query


@pytest.mark.asyncio
async def test_node_bfs_search_undirected_expands_both_directions():
    driver = RecordingDriver()

    await node_bfs_search(
        driver,  # type: ignore[arg-type]
        ['origin-uuid'],
        SearchFilters(),
        2,
        group_ids=['group-a'],
        bfs_undirected=True,
    )

    assert ']-(n:Entity)' in driver.cypher_query
    assert ']->(n:Entity)' not in driver.cypher_query


@pytest.mark.asyncio
async def test_undirected_bfs_refuses_kuzu_rather_than_silently_traversing_outgoing():
    """Kuzu splits an entity edge across RelatesToNode_, so an undirected
    variable-length pattern would change depth semantics. Fail closed."""
    for call in (
        lambda d: edge_bfs_search(
            d, ['origin-uuid'], 2, SearchFilters(), group_ids=['g'], bfs_undirected=True
        ),
        lambda d: node_bfs_search(
            d, ['origin-uuid'], SearchFilters(), 2, group_ids=['g'], bfs_undirected=True
        ),
    ):
        with pytest.raises(NotImplementedError):
            await call(RecordingKuzuDriver())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_undirected_bfs_refuses_search_interface_drivers():
    """A search_interface cannot express the flag, so silently returning
    outgoing-only results would be the exact failure this flag removes."""
    with pytest.raises(NotImplementedError):
        await edge_bfs_search(
            InterfaceDriver(),  # type: ignore[arg-type]
            ['origin-uuid'],
            2,
            SearchFilters(),
            group_ids=['group-a'],
            bfs_undirected=True,
        )
