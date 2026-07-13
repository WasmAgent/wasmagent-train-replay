"""Tests for backend-neutral graph builder (builder.py)."""

from train_replay.collector.flight_recorder import CollectiveEvent
from train_replay.graph.builder import build_from_events, build_from_specs
from train_replay.graph.ops import Backend, CollectiveOp, OpSpec
from train_replay.graph.prov_graph import ProvGraph


class TestBuildFromSpecs:
    def test_single_rank(self) -> None:
        specs = [
            OpSpec(
                op=CollectiveOp.ALL_REDUCE,
                backend=Backend.NCCL,
                rank=0,
                process_group="default",
                sequence_id=1,
                start_time_ns=1000,
                end_time_ns=2000,
            ),
        ]
        g = build_from_specs(specs)
        node_ids = [n for n, _ in g.nodes()]
        assert "rank:0:pg:default" in node_ids
        assert "act:0:all_reduce:1" in node_ids

    def test_multi_rank(self) -> None:
        specs = [
            OpSpec(
                op=CollectiveOp.ALL_REDUCE,
                backend=Backend.GLOO,
                rank=0,
                process_group="default",
                sequence_id=1,
            ),
            OpSpec(
                op=CollectiveOp.ALL_REDUCE,
                backend=Backend.GLOO,
                rank=1,
                process_group="default",
                sequence_id=1,
            ),
        ]
        g = build_from_specs(specs)
        node_ids = [n for n, _ in g.nodes()]
        assert "rank:0:pg:default" in node_ids
        assert "rank:1:pg:default" in node_ids

    def test_gloo_backend(self) -> None:
        specs = [
            OpSpec(
                op=CollectiveOp.BARRIER,
                backend=Backend.GLOO,
                rank=0,
                process_group="tp",
                sequence_id=1,
            ),
        ]
        g = build_from_specs(specs)
        # Should produce a valid graph regardless of backend
        assert isinstance(g, ProvGraph)
        node_ids = [n for n, _ in g.nodes()]
        assert "act:0:barrier:1" in node_ids

    def test_mtia_backend(self) -> None:
        specs = [
            OpSpec(
                op=CollectiveOp.ALL_GATHER,
                backend=Backend.MTIA,
                rank=0,
                process_group="default",
                sequence_id=1,
            ),
        ]
        g = build_from_specs(specs)
        node_ids = [n for n, _ in g.nodes()]
        assert "act:0:all_gather:1" in node_ids

    def test_collective_type_raw_preserved(self) -> None:
        specs = [
            OpSpec(
                op=CollectiveOp.ALL_REDUCE,
                backend=Backend.NCCL,
                rank=0,
                process_group="default",
                sequence_id=1,
                collective_type_raw="all_reduce:fp16",
            ),
        ]
        g = build_from_specs(specs)
        node_ids = [n for n, _ in g.nodes()]
        # The activity ID should use the collective_type property
        assert "act:0:all_reduce:fp16:1" in node_ids

    def test_p2p_ops(self) -> None:
        specs = [
            OpSpec(
                op=CollectiveOp.SEND,
                backend=Backend.GLOO,
                rank=0,
                process_group="default",
                sequence_id=1,
                src_rank=0,
                dst_rank=1,
            ),
            OpSpec(
                op=CollectiveOp.RECV,
                backend=Backend.GLOO,
                rank=1,
                process_group="default",
                sequence_id=1,
                src_rank=0,
                dst_rank=1,
            ),
        ]
        g = build_from_specs(specs)
        node_ids = [n for n, _ in g.nodes()]
        assert "act:0:send:1" in node_ids
        assert "act:1:recv:1" in node_ids

    def test_empty_specs_empty_graph(self) -> None:
        g = build_from_specs([])
        node_ids = [n for n, _ in g.nodes()]
        assert node_ids == []

    def test_causal_ancestors_traversable(self) -> None:
        specs = [
            OpSpec(
                op=CollectiveOp.ALL_REDUCE,
                backend=Backend.NCCL,
                rank=0,
                process_group="default",
                sequence_id=1,
            ),
        ]
        g = build_from_specs(specs)
        ancestors = g.ancestors_of("tensor:0:1:out")
        assert "act:0:all_reduce:1" in ancestors


class TestBuildFromEvents:
    def test_nccl_events_mapped_to_graph(self) -> None:
        events = [
            CollectiveEvent(
                rank=0,
                process_group="default",
                collective_type="all_reduce",
                src_rank=None,
                dst_rank=None,
                tensor_size=1024,
                enqueue_time_ns=500,
                start_time_ns=1000,
                end_time_ns=2000,
                sequence_id=1,
            ),
        ]
        g = build_from_events(events)
        node_ids = [n for n, _ in g.nodes()]
        assert "rank:0:pg:default" in node_ids
        assert "act:0:all_reduce:1" in node_ids

    def test_unknown_collective_type_preserved(self) -> None:
        events = [
            CollectiveEvent(
                rank=0,
                process_group="default",
                collective_type="custom_op_xyz",
                src_rank=None,
                dst_rank=None,
                tensor_size=0,
                enqueue_time_ns=0,
                start_time_ns=0,
                end_time_ns=0,
                sequence_id=1,
            ),
        ]
        g = build_from_events(events)
        node_ids = [n for n, _ in g.nodes()]
        # Raw string should be preserved even if not in CollectiveOp enum
        assert "act:0:custom_op_xyz:1" in node_ids

    def test_build_from_events_uses_build_from_specs(self) -> None:
        """Both entry points produce the same graph for equivalent data."""
        events = [
            CollectiveEvent(
                rank=0,
                process_group="default",
                collective_type="all_reduce",
                src_rank=None,
                dst_rank=None,
                tensor_size=1024,
                enqueue_time_ns=500,
                start_time_ns=1000,
                end_time_ns=2000,
                sequence_id=1,
            ),
        ]
        g_from_events = build_from_events(events)

        specs = [
            OpSpec(
                op=CollectiveOp.ALL_REDUCE,
                backend=Backend.NCCL,
                rank=0,
                process_group="default",
                sequence_id=1,
                src_rank=None,
                dst_rank=None,
                tensor_size=1024,
                start_time_ns=1000,
                end_time_ns=2000,
            ),
        ]
        g_from_specs = build_from_specs(specs)

        # Both produce graphs with the same structure
        assert g_from_events.digest() == g_from_specs.digest()
