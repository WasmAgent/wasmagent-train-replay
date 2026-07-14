"""Parse PyTorch Flight Recorder pickle dumps into CollectiveEvent records.

Security
--------
Flight Recorder dumps are ``pickle`` streams produced by
``torch._C._distributed_c10d._dump_nccl_trace()``. Because pickle can carry
arbitrary Python objects, loading an untrusted dump with the bare
:func:`pickle.load` would execute arbitrary code. We therefore deserialize
through :class:`_RestrictedUnpickler`, which refuses every ``GLOBAL``/``REDUCE``
opcode: a genuine NCCL trace dump is composed entirely of built-in containers
and scalars (``dict``/``list``/``int``/``str``/``None``) and never triggers
``find_class``, whereas a crafted file that tries to import or call an
arbitrary callable is rejected with :class:`UnsafeFlightRecorderDumpError`
before any such callable runs.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class UnsafeFlightRecorderDumpError(ValueError):
    """Raised when a flight recorder dump would execute arbitrary code.

    Pickle's ``GLOBAL``/``REDUCE`` opcodes can import and call arbitrary
    callables; :class:`_RestrictedUnpickler` refuses them so that an attacker
    who controls the trace file cannot achieve code execution at load time.
    """


class _RestrictedUnpickler(pickle.Unpickler):
    """A :class:`pickle.Unpickler` that forbids importing or calling code.

    Only the built-in atomic/container types emitted by the Flight Recorder
    can be reconstructed; any ``find_class`` resolution (an attempt to import a
    class or function) raises :class:`UnsafeFlightRecorderDumpError`.
    """

    def find_class(self, module: str, name: str) -> Any:
        raise UnsafeFlightRecorderDumpError(
            f"Refusing to unpickle {module!r}.{name!r}: a flight recorder dump "
            "must contain only built-in containers and scalars. Importing or "
            "calling arbitrary code during deserialization is refused."
        )


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


def load_flight_recorder(path: Path) -> list[CollectiveEvent]:
    """Load a Flight Recorder pickle dump produced by
    ``torch._C._distributed_c10d._dump_nccl_trace()``.

    The dump is deserialized through :class:`_RestrictedUnpickler` so that an
    untrusted or crafted file cannot execute arbitrary code; only plain
    built-in containers and scalars are accepted.
    """
    with open(path, "rb") as f:
        raw: Any = _RestrictedUnpickler(f).load()

    if not isinstance(raw, dict):
        raise UnsafeFlightRecorderDumpError(
            "Flight recorder dump must deserialize to a dict with an "
            "'entries' list of built-in containers and scalars."
        )
    if "entries" not in raw or not isinstance(raw["entries"], list):
        raise UnsafeFlightRecorderDumpError(
            "Flight recorder dump must deserialize to a dict with an "
            "'entries' list of built-in containers and scalars."
        )

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
