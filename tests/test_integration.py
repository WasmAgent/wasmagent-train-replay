"""Integration tests for cross-rank causal graph construction."""

from train_replay.collector.flight_recorder import CollectiveEvent
from train_replay.graph.builder import build_from_events


def _make_events() -> list[CollectiveEvent]:
    """Create mock collective events for two ranks."""
    return [
        CollectiveEvent(
            rank=0,
            process_group="default",
            collective_type="all_reduce",
            src_rank=None,
            dst_rank=None,
            tensor_size=4096,
            enqueue_time_ns=1000,
            start_time_ns=1100,
            end_time_ns=1200,
            sequence_id=1,
        ),
        CollectiveEvent(
            rank=1,
            process_group="default",
            collective_type="all_reduce",
            src_rank=None,
            dst_rank=None,
            tensor_size=4096,
            enqueue_time_ns=1000,
            start_time_ns=1100,
            end_time_ns=1200,
            sequence_id=1,
        ),
        CollectiveEvent(
            rank=0,
            process_group="default",
            collective_type="all_reduce",
            src_rank=None,
            dst_rank=None,
            tensor_size=8192,
            enqueue_time_ns=2000,
            start_time_ns=2100,
            end_time_ns=2200,
            sequence_id=2,
        ),
        CollectiveEvent(
            rank=1,
            process_group="default",
            collective_type="all_reduce",
            src_rank=None,
            dst_rank=None,
            tensor_size=8192,
            enqueue_time_ns=2000,
            start_time_ns=2100,
            end_time_ns=2200,
            sequence_id=2,
        ),
    ]


def test_build_from_events_multi_rank() -> None:
    """build_from_events should incorporate events from all ranks into one graph."""
    events = _make_events()
    graph = build_from_events(events)

    # Count nodes by kind
    activities = []
    entities = []
    agents = []
    for nid, data in graph.nodes():
        kind = data.get("kind")
        if kind == "activity":
            activities.append(nid)
        elif kind == "entity":
            entities.append(nid)
        elif kind == "agent":
            agents.append(nid)

    # Both ranks have agents
    assert "rank:0:pg:default" in agents
    assert "rank:1:pg:default" in agents

    # Both ranks have activities
    assert "act:0:all_reduce:1" in activities
    assert "act:1:all_reduce:1" in activities
    assert "act:0:all_reduce:2" in activities
    assert "act:1:all_reduce:2" in activities

    # Both ranks have entities (input and output per event)
    assert "tensor:0:1:in" in entities
    assert "tensor:0:1:out" in entities
    assert "tensor:1:1:in" in entities
    assert "tensor:1:1:out" in entities
    assert "tensor:0:2:in" in entities
    assert "tensor:0:2:out" in entities
    assert "tensor:1:2:in" in entities
    assert "tensor:1:2:out" in entities


def test_causal_ancestor_tracing_within_rank() -> None:
    """ancestors_of should trace causal ancestors within the same rank."""
    events = _make_events()
    graph = build_from_events(events)

    # Tensor out on rank 0 seq 1 should trace back to its generating activity
    ancestors = graph.ancestors_of("tensor:0:1:out")
    assert "act:0:all_reduce:1" in ancestors

    # Tensor out on rank 1 seq 1 should trace back to its generating activity
    ancestors = graph.ancestors_of("tensor:1:1:out")
    assert "act:1:all_reduce:1" in ancestors

    # Tensor out on rank 1 seq 2 should trace back to its generating activity
    ancestors = graph.ancestors_of("tensor:1:2:out")
    assert "act:1:all_reduce:2" in ancestors


def test_causal_subgraph_multi_rank() -> None:
    """causal_subgraph should include only the subgraph for a given entity."""
    events = _make_events()
    graph = build_from_events(events)

    sub = graph.causal_subgraph("tensor:0:2:out")
    node_ids = [n for n, _ in sub.nodes()]
    # Should include the generating activity
    assert "act:0:all_reduce:2" in node_ids
    # Should include the entity itself
    assert "tensor:0:2:out" in node_ids
    # Should NOT include unrelated entities from other ranks
    assert "tensor:1:1:out" not in node_ids
