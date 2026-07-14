"""Security tests for the Flight Recorder loader.

The loader consumes ``pickle`` files, which can carry arbitrary code. These
tests pin the restricted-unpickler guarantee: a plain built-in-only dump still
loads, while a crafted pickle that tries to invoke a callable (the classic
``__reduce__`` RCE) is rejected before any code runs.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pytest

from train_replay.collector.flight_recorder import (
    UnsafeFlightRecorderDumpError,
    load_flight_recorder,
)


def _write_plain_dump(path: Path) -> None:
    """Write a minimal built-in-only dump (no GLOBAL/REDUCE opcodes)."""
    data = {
        "entries": [
            {
                "rank": 0,
                "pg_name": "default",
                "collective_seq": "all_reduce",
                "p2p_src": None,
                "p2p_dst": None,
                "input_sizes": [[4096]],
                "time_created_ns": 1000,
                "time_started_ns": 1100,
                "time_finished_ns": 1200,
                "frames": [],
                "seq_id": 1,
            }
        ]
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _malicious_pickle_bytes() -> bytes:
    """Pickle bytes whose ``__reduce__`` resolves to ``os.system``.

    Under the bare :func:`pickle.load` this would call ``os.system`` and run
    arbitrary shell commands. The restricted unpickler must reject it at
    ``find_class`` ("posix.system") *before* the callable is ever invoked, so
    the command below is never executed.
    """

    class _Bomb:
        def __reduce__(self) -> object:
            # Harmless command on purpose; the loader must reject before run.
            return (os.system, ("echo should-never-run",))

    return pickle.dumps(_Bomb())


def test_plain_dump_loads(tmp_path: Path) -> None:
    """A built-in-only dump deserializes through the restricted unpickler."""
    trace = tmp_path / "trace.pkl"
    _write_plain_dump(trace)
    events = load_flight_recorder(trace)
    assert len(events) == 1
    assert events[0].rank == 0
    assert events[0].collective_type == "all_reduce"


def test_malicious_pickle_is_rejected(tmp_path: Path) -> None:
    """A pickle that tries to execute code is refused, not run."""
    trace = tmp_path / "trace.pkl"
    with open(trace, "wb") as f:
        f.write(_malicious_pickle_bytes())
    with pytest.raises(UnsafeFlightRecorderDumpError):
        load_flight_recorder(trace)


def test_non_dict_payload_is_rejected(tmp_path: Path) -> None:
    """A pickle that deserializes to a non-dict (e.g. a list) is rejected."""
    trace = tmp_path / "trace.pkl"
    with open(trace, "wb") as f:
        pickle.dump([1, 2, 3], f)
    with pytest.raises(UnsafeFlightRecorderDumpError):
        load_flight_recorder(trace)


def test_dict_without_entries_key_is_rejected(tmp_path: Path) -> None:
    """A pickle that deserializes to a dict without 'entries' is rejected."""
    trace = tmp_path / "trace.pkl"
    with open(trace, "wb") as f:
        pickle.dump({"other_key": [1, 2, 3]}, f)
    with pytest.raises(UnsafeFlightRecorderDumpError):
        load_flight_recorder(trace)


def test_dict_with_non_list_entries_is_rejected(tmp_path: Path) -> None:
    """A pickle that deserializes to a dict with 'entries' as a non-list is rejected."""
    trace = tmp_path / "trace.pkl"
    with open(trace, "wb") as f:
        pickle.dump({"entries": "not-a-list"}, f)
    with pytest.raises(UnsafeFlightRecorderDumpError):
        load_flight_recorder(trace)
