"""Parse PyTorch Flight Recorder pickle dumps into CollectiveEvent records."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CollectiveEvent:
    rank: int
    process_group: str
    collective_type: str
    src_rank: int | None
    dst_rank: int | None
    tensor_size: int
    enqueue_time_ns: int
    start_time_ns: int
    end_time_ns: int
    call_stack: list[str] = field(default_factory=list)
    sequence_id: int = 0

    # -- BackendEvent protocol (graph/base.py) ----------------------------
    # These properties map NCCL-specific fields to the backend-agnostic
    # interface so CollectiveEvent satisfies the BackendEvent protocol.

    @property
    def operation_type(self) -> str:
        """Backend-agnostic alias for ``collective_type``."""
        return self.collective_type

    @property
    def timestamp_ns(self) -> int:
        """Backend-agnostic alias for ``start_time_ns``."""
        return self.start_time_ns

    @property
    def group_id(self) -> str:
        """Backend-agnostic alias for ``process_group``."""
        return self.process_group


def load_flight_recorder(path: Path) -> list[CollectiveEvent]:
    """Load a Flight Recorder pickle dump produced by
    ``torch._C._distributed_c10d._dump_nccl_trace()``.
    """
    with open(path, "rb") as f:
        raw: dict[str, Any] = pickle.load(f)

    events: list[CollectiveEvent] = []
    for entry in raw.get("entries", []):
        events.append(CollectiveEvent(
            rank=entry.get("rank", 0),
            process_group=entry.get("pg_name", "default"),
            collective_type=entry.get("collective_seq", "unknown"),
            src_rank=entry.get("p2p_src", None),
            dst_rank=entry.get("p2p_dst", None),
            tensor_size=entry.get("input_sizes", [[0]])[0][0] if entry.get("input_sizes") else 0,
            enqueue_time_ns=entry.get("time_created_ns", 0),
            start_time_ns=entry.get("time_started_ns", 0),
            end_time_ns=entry.get("time_finished_ns", 0),
            call_stack=entry.get("frames", []),
            sequence_id=entry.get("seq_id", 0),
        ))
    return events
