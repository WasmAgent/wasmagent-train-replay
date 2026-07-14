"""Tests for backend-neutral collective operation abstractions (ops.py)."""

from train_replay.graph.ops import Backend, CollectiveOp, OpSpec


class TestCollectiveOp:
    def test_all_reduce_value(self) -> None:
        assert CollectiveOp.ALL_REDUCE.value == "all_reduce"

    def test_all_values_unique(self) -> None:
        values = {op.value for op in CollectiveOp}
        assert len(values) == len(CollectiveOp)

    def test_string_enum_comparison(self) -> None:
        assert CollectiveOp.BROADCAST == "broadcast"
        assert CollectiveOp.ALL_TO_ALL == "all_to_all"


class TestBackend:
    def test_nccl_value(self) -> None:
        assert Backend.NCCL.value == "NCCL"

    def test_gloo_value(self) -> None:
        assert Backend.GLOO.value == "GLOO"

    def test_mtia_value(self) -> None:
        assert Backend.MTIA.value == "MTIA"

    def test_custom_value(self) -> None:
        assert Backend.CUSTOM.value == "CUSTOM"


class TestOpSpec:
    def test_collective_type_from_enum(self) -> None:
        spec = OpSpec(
            op=CollectiveOp.ALL_REDUCE,
            backend=Backend.NCCL,
            rank=0,
            process_group="default",
            sequence_id=1,
        )
        assert spec.collective_type == "all_reduce"

    def test_collective_type_raw_overrides_enum(self) -> None:
        spec = OpSpec(
            op=CollectiveOp.ALL_REDUCE,
            backend=Backend.NCCL,
            rank=0,
            process_group="default",
            sequence_id=1,
            collective_type_raw="all_reduce:fp16",
        )
        assert spec.collective_type == "all_reduce:fp16"

    def test_frozen_immutability(self) -> None:
        spec = OpSpec(
            op=CollectiveOp.BARRIER,
            backend=Backend.GLOO,
            rank=2,
            process_group="tp",
            sequence_id=5,
        )
        try:
            spec.rank = 99  # type: ignore[misc]
            assert False, "Should not be able to set attribute on frozen dataclass"
        except AttributeError:
            pass

    def test_p2p_ops_allow_src_dst(self) -> None:
        spec = OpSpec(
            op=CollectiveOp.SEND,
            backend=Backend.GLOO,
            rank=0,
            process_group="default",
            sequence_id=3,
            src_rank=0,
            dst_rank=1,
        )
        assert spec.src_rank == 0
        assert spec.dst_rank == 1

    def test_default_none_for_src_dst(self) -> None:
        spec = OpSpec(
            op=CollectiveOp.ALL_REDUCE,
            backend=Backend.NCCL,
            rank=0,
            process_group="default",
            sequence_id=1,
        )
        assert spec.src_rank is None
        assert spec.dst_rank is None

    def test_backend_from_string(self) -> None:
        backend = Backend("GLOO")
        assert backend is Backend.GLOO
