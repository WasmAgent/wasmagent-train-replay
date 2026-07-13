"""Backend-neutral collective operation definitions.

These types decouple the graph layer from any specific communication backend
(NCCL, Gloo, MTIA, etc.).  The collector layer maps backend-specific trace
formats into these abstractions; everything downstream operates on them
without knowing which backend produced the event.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Backend(str, Enum):
    """Communication backend that produced a collective event."""

    NCCL = "NCCL"
    GLOO = "GLOO"
    MTIA = "MTIA"
    CUSTOM = "CUSTOM"


class CollectiveOp(str, Enum):
    """Collective operation types common across NCCL, Gloo, and MTIA.

    Each backend may support a subset of these.  The graph layer treats
    all ops uniformly; backend-specific capabilities are handled in the
    collector adapters.
    """

    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    BROADCAST = "broadcast"
    REDUCE_SCATTER = "reduce_scatter"
    SEND = "send"
    RECV = "recv"
    BARRIER = "barrier"
    ALL_TO_ALL = "all_to_all"
    REDUCE = "reduce"
    GATHER = "gather"
    SCATTER = "scatter"


@dataclass(frozen=True)
class OpSpec:
    """Backend-neutral description of a single collective operation.

    This is the graph layer's input contract — the collector translates
    backend-specific trace events into ``OpSpec`` instances, and the
    graph/replay layers consume ``OpSpec`` without inspecting any
    backend-specific fields.

    Attributes:
        op: The collective operation type.
        backend: Which communication backend produced this event.
        rank: Rank that executed the operation.
        process_group: Process group name.
        sequence_id: Monotonic sequence number within the rank.
        src_rank: Source rank for point-to-point ops (``None`` for collectives).
        dst_rank: Destination rank for point-to-point ops (``None`` for collectives).
        tensor_size: Number of elements in the tensor.
        start_time_ns: Operation start time in nanoseconds.
        end_time_ns: Operation end time in nanoseconds.
        collective_type_raw: Original string from the backend trace,
            preserved for collectors that need round-trip fidelity.
    """

    op: CollectiveOp
    backend: Backend
    rank: int
    process_group: str
    sequence_id: int
    src_rank: int | None = None
    dst_rank: int | None = None
    tensor_size: int = 0
    start_time_ns: int = 0
    end_time_ns: int = 0
    collective_type_raw: str | None = None

    @property
    def collective_type(self) -> str:
        """String label for the operation — ``collective_type_raw`` if set,
        otherwise the enum value."""
        if self.collective_type_raw is not None:
            return self.collective_type_raw
        return self.op.value
