"""Tests for backend-agnostic graph abstractions (graph/base.py)."""

from __future__ import annotations

from dataclasses import dataclass

from train_replay.collector.flight_recorder import CollectiveEvent
from train_replay.graph.base import BackendEvent, GraphBuilder
from train_replay.graph.builder import NCCLGraphBuilder, build_from_events
from train_replay.graph.prov_graph import ProvGraph

# ---------------------------------------------------------------------------
# CollectiveEvent satisfies BackendEvent
# ---------------------------------------------------------------------------


def test_collective_event_is_backend_event() -> None:
    """CollectiveEvent must satisfy the BackendEvent protocol."""
    evt = CollectiveEvent(
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
    )
    assert isinstance(evt, BackendEvent)
    assert evt.rank == 0
    assert evt.operation_type == "all_reduce"
    assert evt.sequence_id == 1
    assert evt.timestamp_ns == 1100
    assert evt.group_id == "default"


def test_collective_event_property_aliases() -> None:
    """BackendEvent properties on CollectiveEvent must match the source fields."""
    evt = CollectiveEvent(
        rank=2,
        process_group="pg1",
        collective_type="broadcast",
        src_rank=None,
        dst_rank=None,
        tensor_size=1024,
        enqueue_time_ns=5000,
        start_time_ns=5100,
        end_time_ns=5200,
        sequence_id=7,
    )
    assert evt.operation_type == evt.collective_type
    assert evt.timestamp_ns == evt.start_time_ns
    assert evt.group_id == evt.process_group


# ---------------------------------------------------------------------------
# Custom backend event implementing BackendEvent
# ---------------------------------------------------------------------------


@dataclass
class GlooEvent:
    """Hypothetical Gloo backend event satisfying BackendEvent."""

    rank: int
    operation_type: str
    sequence_id: int
    timestamp_ns: int
    group_id: str
    tensor_bytes: int = 0


def test_custom_backend_event_is_backend_event() -> None:
    """Any dataclass with the right properties satisfies BackendEvent."""
    evt = GlooEvent(
        rank=1,
        operation_type="allgather",
        sequence_id=3,
        timestamp_ns=9000,
        group_id="gloo-default",
    )
    assert isinstance(evt, BackendEvent)


# ---------------------------------------------------------------------------
# NCCLGraphBuilder implements GraphBuilder
# ---------------------------------------------------------------------------


def test_nccl_builder_is_graph_builder() -> None:
    """NCCLGraphBuilder must satisfy the GraphBuilder protocol."""
    builder = NCCLGraphBuilder()
    assert isinstance(builder, GraphBuilder)


def test_nccl_builder_produces_valid_graph() -> None:
    """NCCLGraphBuilder.build() must produce a graph with correct nodes."""
    builder = NCCLGraphBuilder()
    events: list[CollectiveEvent] = [
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
    ]
    graph = builder.build(events)  # type: ignore[arg-type]

    node_ids = [n for n, _ in graph.nodes()]
    assert "rank:0:pg:default" in node_ids
    assert "rank:1:pg:default" in node_ids
    assert "act:0:all_reduce:1" in node_ids
    assert "act:1:all_reduce:1" in node_ids
    assert "tensor:0:1:in" in node_ids
    assert "tensor:0:1:out" in node_ids


def test_nccl_builder_causal_ancestors() -> None:
    """NCCLGraphBuilder-built graph must support causal ancestor traversal."""
    builder = NCCLGraphBuilder()
    events: list[CollectiveEvent] = [
        CollectiveEvent(
            rank=0,
            process_group="default",
            collective_type="all_reduce",
            src_rank=None,
            dst_rank=None,
            tensor_size=2048,
            enqueue_time_ns=1000,
            start_time_ns=1100,
            end_time_ns=1200,
            sequence_id=5,
        ),
    ]
    graph = builder.build(events)  # type: ignore[arg-type]
    ancestors = graph.ancestors_of("tensor:0:5:out")
    assert "act:0:all_reduce:5" in ancestors


# ---------------------------------------------------------------------------
# build_from_events backward compatibility
# ---------------------------------------------------------------------------


def test_build_from_events_matches_builder() -> None:
    """Legacy build_from_events() must produce identical output to NCCLGraphBuilder."""
    events: list[CollectiveEvent] = [
        CollectiveEvent(
            rank=0,
            process_group="pg",
            collective_type="barrier",
            src_rank=None,
            dst_rank=None,
            tensor_size=0,
            enqueue_time_ns=3000,
            start_time_ns=3100,
            end_time_ns=3200,
            sequence_id=2,
        ),
    ]

    graph_legacy = build_from_events(events)
    graph_new = NCCLGraphBuilder().build(events)  # type: ignore[arg-type]

    legacy_ids = {n for n, _ in graph_legacy.nodes()}
    new_ids = {n for n, _ in graph_new.nodes()}
    assert legacy_ids == new_ids


# ---------------------------------------------------------------------------
# Custom GraphBuilder implementation
# ---------------------------------------------------------------------------


class GlooGraphBuilder:
    """Example GraphBuilder for a hypothetical Gloo backend."""

    def build(self, events: list[BackendEvent]) -> ProvGraph:
        graph = ProvGraph()
        agents: dict[tuple[int, str], str] = {}

        for evt in events:
            agent_key = (evt.rank, evt.group_id)
            if agent_key not in agents:
                agent_id = f"gloo:{evt.rank}:grp:{evt.group_id}"
                from train_replay.graph.prov_graph import ProvAgent

                graph.add_agent(ProvAgent(id=agent_id, rank=evt.rank, process_group=evt.group_id))
                agents[agent_key] = agent_id

            from train_replay.graph.prov_graph import ProvActivity

            act_id = f"gloo-act:{evt.rank}:{evt.operation_type}:{evt.sequence_id}"
            graph.add_activity(ProvActivity(
                id=act_id,
                label=evt.operation_type,
                rank=evt.rank,
                process_group=evt.group_id,
                timestamp_ns=evt.timestamp_ns,
                collective_type=evt.operation_type,
            ))
            graph.was_associated_with(act_id, agents[agent_key])

        return graph


def test_custom_builder_is_graph_builder() -> None:
    """Any class with a build() method satisfies GraphBuilder."""
    builder = GlooGraphBuilder()
    assert isinstance(builder, GraphBuilder)


def test_custom_builder_produces_graph() -> None:
    """GlooGraphBuilder must produce a valid ProvGraph."""
    builder = GlooGraphBuilder()
    events: list[BackendEvent] = [
        GlooEvent(
            rank=0,
            operation_type="allgather",
            sequence_id=1,
            timestamp_ns=1000,
            group_id="gloo-default",
        ),
        GlooEvent(
            rank=1,
            operation_type="allgather",
            sequence_id=1,
            timestamp_ns=1000,
            group_id="gloo-default",
        ),
    ]
    graph = builder.build(events)
    node_ids = {n for n, _ in graph.nodes()}
    assert "gloo:0:grp:gloo-default" in node_ids
    assert "gloo:1:grp:gloo-default" in node_ids
    assert "gloo-act:0:allgather:1" in node_ids
