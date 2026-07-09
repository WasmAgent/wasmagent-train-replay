"""Tests for PROV-DM causal graph."""

from __future__ import annotations

from train_replay.graph.prov_graph import (
    ProvActivity,
    ProvAgent,
    ProvEntity,
    ProvGraph,
)


def _make_graph() -> ProvGraph:
    g = ProvGraph()
    g.add_agent(ProvAgent(id="rank:0:pg:default", rank=0, process_group="default"))
    g.add_activity(ProvActivity(id="act:0:all_reduce:1", label="all_reduce",
                                rank=0, process_group="default",
                                timestamp_ns=1000, collective_type="all_reduce"))
    g.add_entity(ProvEntity(id="tensor:0:1:in", digest=None, rank=0, step=1))
    g.add_entity(ProvEntity(id="tensor:0:1:out", digest=None, rank=0, step=1))
    g.used("act:0:all_reduce:1", "tensor:0:1:in")
    g.was_generated_by("tensor:0:1:out", "act:0:all_reduce:1")
    return g


def _make_multi_hop_graph() -> ProvGraph:
    """Build a three-hop causal chain:

    entity_2 --wasGeneratedBy--> act_2 --used--> entity_1
                                         --wasGeneratedBy--> act_1 --used--> entity_0

    So entity_2's causal ancestors (activities) are [act_2, act_1].
    """
    g = ProvGraph()
    # Entities
    g.add_entity(ProvEntity(id="entity_0", digest=None, rank=0, step=0))
    g.add_entity(ProvEntity(id="entity_1", digest=None, rank=0, step=1))
    g.add_entity(ProvEntity(id="entity_2", digest=None, rank=0, step=2))
    # Activities
    g.add_activity(ProvActivity(id="act_1", label="compute",
                                rank=0, process_group="default",
                                timestamp_ns=100, collective_type="all_reduce"))
    g.add_activity(ProvActivity(id="act_2", label="compute",
                                rank=0, process_group="default",
                                timestamp_ns=200, collective_type="all_reduce"))
    # Edges — effect -> cause direction
    g.used("act_1", "entity_0")
    g.was_generated_by("entity_1", "act_1")
    g.used("act_2", "entity_1")
    g.was_generated_by("entity_2", "act_2")
    return g


def test_ancestors_found() -> None:
    g = _make_graph()
    ancestors = g.ancestors_of("tensor:0:1:out")
    assert "act:0:all_reduce:1" in ancestors


def test_leaf_entity_no_ancestors() -> None:
    g = _make_graph()
    assert g.ancestors_of("tensor:0:1:in") == []


def test_causal_subgraph() -> None:
    g = _make_graph()
    sub = g.causal_subgraph("tensor:0:1:out")
    node_ids = [n for n, _ in sub.nodes()]
    assert "act:0:all_reduce:1" in node_ids


def test_ancestors_traversal_direction() -> None:
    """Verify ancestors_of follows causality (effect -> cause).

    In the chain entity_2 -> act_2 -> entity_1 -> act_1 -> entity_0,
    ancestors_of(entity_2) must return [act_2, act_1]: both activities
    causally contributed to entity_2.
    """
    g = _make_multi_hop_graph()

    ancestors = g.ancestors_of("entity_2")

    # act_2 directly generated entity_2 via wasGeneratedBy
    assert "act_2" in ancestors, (
        "act_2 directly generated entity_2 (wasGeneratedBy), "
        "so it must be an ancestor"
    )
    # act_1 generated entity_1 which was consumed by act_2
    assert "act_1" in ancestors, (
        "act_1 generated entity_1 which was used by act_2, "
        "so act_1 is a transitive causal ancestor of entity_2"
    )
    # entity_0 is not an activity, should never appear in ancestors
    assert "entity_0" not in ancestors
