"""Unit tests for whitelisted episode_metadata provenance properties.

Watercooler fork feature: EPISODIC_PROVENANCE_KEYS from episode_metadata
are persisted as first-class Episodic node properties on FalkorDB/Neo4j
(whose save queries write a full property map — a post-hoc stamp would be
wiped on the next re-save). Driver-free: asserts the params passed to
execute_query and the query text per provider.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.models.nodes.node_db_queries import (
    get_episode_node_save_bulk_query,
    get_episode_node_save_query,
)
from graphiti_core.nodes import EPISODIC_PROVENANCE_KEYS, EpisodeType, EpisodicNode

METADATA = {
    'entry_id': '01TESTAAAAAAAAAAAAAAAAAAAA',
    'thread_id': 'some-topic',
    'chunk_index': 2,
    'total_chunks': 3,
}


def _node(metadata=None) -> EpisodicNode:
    return EpisodicNode(
        name='ep',
        group_id='g',
        source=EpisodeType.message,
        source_description='desc',
        content='body',
        valid_at=datetime.now(timezone.utc),
        episode_metadata=metadata,
    )


def _driver(provider: GraphProvider):
    driver = MagicMock()
    driver.provider = provider
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock(return_value=None)
    return driver


def test_falkordb_save_params_carry_provenance():
    driver = _driver(GraphProvider.FALKORDB)
    asyncio.run(_node(METADATA).save(driver))

    _, kwargs = driver.execute_query.call_args
    for key in EPISODIC_PROVENANCE_KEYS:
        assert kwargs[key] == METADATA[key]


def test_falkordb_save_params_default_to_none_without_metadata():
    driver = _driver(GraphProvider.FALKORDB)
    asyncio.run(_node(None).save(driver))

    _, kwargs = driver.execute_query.call_args
    for key in EPISODIC_PROVENANCE_KEYS:
        assert key in kwargs and kwargs[key] is None


def test_kuzu_save_params_do_not_carry_provenance():
    """Kuzu has a typed schema; unknown params must not be sent."""
    driver = _driver(GraphProvider.KUZU)
    asyncio.run(_node(METADATA).save(driver))

    _, kwargs = driver.execute_query.call_args
    for key in EPISODIC_PROVENANCE_KEYS:
        assert key not in kwargs


def test_save_queries_reference_provenance_fields():
    for provider in (GraphProvider.FALKORDB, GraphProvider.NEO4J):
        q = get_episode_node_save_query(provider)
        bq = get_episode_node_save_bulk_query(provider)
        for key in EPISODIC_PROVENANCE_KEYS:
            assert f'{key}: ${key}' in q
            assert f'{key}: episode.{key}' in bq


def test_kuzu_and_neptune_queries_unchanged():
    for provider in (GraphProvider.KUZU, GraphProvider.NEPTUNE):
        q = get_episode_node_save_query(provider)
        for key in EPISODIC_PROVENANCE_KEYS:
            assert key not in q


# ---------------------------------------------------------------------------
# Operations-path coverage (PR #2 review): the per-driver
# EpisodeNodeOperations implementations build their own params/episode
# dicts against the same changed queries — they must carry the whitelisted
# provenance too, or fail at runtime with missing parameters.
# ---------------------------------------------------------------------------


def _ops_executor():
    executor = MagicMock()
    executor.execute_query = AsyncMock(return_value=None)
    return executor


def test_falkordb_operations_save_carries_provenance():
    from graphiti_core.driver.falkordb.operations.episode_node_ops import (
        FalkorEpisodeNodeOperations,
    )

    executor = _ops_executor()
    asyncio.run(FalkorEpisodeNodeOperations().save(executor, _node(METADATA)))

    _, kwargs = executor.execute_query.call_args
    for key in EPISODIC_PROVENANCE_KEYS:
        assert kwargs[key] == METADATA[key]


def test_falkordb_operations_save_defaults_to_none_without_metadata():
    from graphiti_core.driver.falkordb.operations.episode_node_ops import (
        FalkorEpisodeNodeOperations,
    )

    executor = _ops_executor()
    asyncio.run(FalkorEpisodeNodeOperations().save(executor, _node(None)))

    _, kwargs = executor.execute_query.call_args
    for key in EPISODIC_PROVENANCE_KEYS:
        assert key in kwargs and kwargs[key] is None


def test_falkordb_operations_save_bulk_flattens_provenance():
    from graphiti_core.driver.falkordb.operations.episode_node_ops import (
        FalkorEpisodeNodeOperations,
    )

    executor = _ops_executor()
    asyncio.run(
        FalkorEpisodeNodeOperations().save_bulk(
            executor, [_node(METADATA), _node(None)]
        )
    )

    _, kwargs = executor.execute_query.call_args
    episodes = kwargs['episodes']
    for key in EPISODIC_PROVENANCE_KEYS:
        assert episodes[0][key] == METADATA[key]
        assert episodes[1][key] is None


def test_neo4j_operations_save_carries_provenance():
    from graphiti_core.driver.neo4j.operations.episode_node_ops import (
        Neo4jEpisodeNodeOperations,
    )

    executor = _ops_executor()
    asyncio.run(Neo4jEpisodeNodeOperations().save(executor, _node(METADATA)))

    _, kwargs = executor.execute_query.call_args
    for key in EPISODIC_PROVENANCE_KEYS:
        assert kwargs[key] == METADATA[key]


def test_neo4j_operations_save_bulk_flattens_provenance():
    from graphiti_core.driver.neo4j.operations.episode_node_ops import (
        Neo4jEpisodeNodeOperations,
    )

    executor = _ops_executor()
    asyncio.run(
        Neo4jEpisodeNodeOperations().save_bulk(executor, [_node(METADATA)])
    )

    _, kwargs = executor.execute_query.call_args
    for key in EPISODIC_PROVENANCE_KEYS:
        assert kwargs['episodes'][0][key] == METADATA[key]
