"""Tests for the Gloo trace collector.

Gloo collectives are observed as JSON logs (Gloo ships no binary dumper). These
tests pin the contract that :mod:`train_replay.collector.gloo` parses such traces
into the shared, backend-agnostic :class:`CollectiveEvent` schema — the same type
the flight-recorder loader emits — so Gloo events feed the same PROV-DM graph
builder. No real PyTorch/Gloo runtime is used: every trace is a fixture dict.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from train_replay.collector import gloo
from train_replay.collector.flight_recorder import CollectiveEvent
from train_replay.collector.gloo import (
    MalformedGlooTraceError,
    load_gloo_trace,
    parse_gloo_trace,
)
from train_replay.graph.builder import build_from_events


def _full_entry(**overrides: object) -> dict[str, object]:
    """A canonical Gloo trace entry using the schema's primary field names."""
    entry: dict[str, object] = {
        "rank": 1,
        "process_group": "default_pg",
        "op": "allreduce",
        "src_rank": None,
        "dst_rank": None,
        "input_size": 8192,
        "time_created_ns": 1000,
        "time_started_ns": 1100,
        "time_finished_ns": 1300,
        "stack": ["train.py:42", "model.py:88"],
        "sequence_id": 7,
    }
    entry.update(overrides)
    return entry


def test_parses_dict_with_entries_into_collective_events() -> None:
    """A top-level object with an 'entries' list maps field-by-field."""
    events = parse_gloo_trace({"entries": [_full_entry()]})

    assert len(events) == 1
    evt = events[0]
    assert isinstance(evt, CollectiveEvent)
    assert evt.rank == 1
    assert evt.process_group == "default_pg"
    assert evt.collective_type == "allreduce"
    assert evt.src_rank is None
    assert evt.dst_rank is None
    assert evt.tensor_size == 8192
    assert evt.enqueue_time_ns == 1000
    assert evt.start_time_ns == 1100
    assert evt.end_time_ns == 1300
    assert evt.call_stack == ["train.py:42", "model.py:88"]
    assert evt.sequence_id == 7


def test_parses_bare_json_array_of_entries() -> None:
    """A bare JSON array of entry objects is accepted, not just {entries: [...]}."""
    events = parse_gloo_trace([_full_entry(rank=0), _full_entry(rank=2)])

    assert [e.rank for e in events] == [0, 2]


def test_string_encoded_numbers_are_coerced() -> None:
    """JSON logs sometimes carry numeric fields as strings; coerce to int."""
    entry = _full_entry(
        rank="3",
        input_size="4096",
        time_created_ns="10",
        time_started_ns="20",
        time_finished_ns="30",
        sequence_id="9",
    )
    events = parse_gloo_trace([entry])

    assert events[0].rank == 3
    assert events[0].tensor_size == 4096
    assert events[0].enqueue_time_ns == 10
    assert events[0].start_time_ns == 20
    assert events[0].end_time_ns == 30
    assert events[0].sequence_id == 9


def test_alias_field_names_are_accepted() -> None:
    """Ad-hoc debug logs use aliases; each maps to the same CollectiveEvent field."""
    entry = {
        "rank": 0,
        "pg_name": "world",
        "collective_type": "broadcast",
        "p2p_src": 4,
        "p2p_dst": 5,
        "tensor_size": 64,
        "enqueue_time_ns": 7,
        "start_time_ns": 8,
        "end_time_ns": 9,
        "frames": ["a", "b"],
        "seq_id": 2,
    }
    events = parse_gloo_trace([entry])

    evt = events[0]
    assert evt.process_group == "world"
    assert evt.collective_type == "broadcast"
    assert evt.src_rank == 4
    assert evt.dst_rank == 5
    assert evt.tensor_size == 64
    assert evt.enqueue_time_ns == 7
    assert evt.start_time_ns == 8
    assert evt.end_time_ns == 9
    assert evt.call_stack == ["a", "b"]
    assert evt.sequence_id == 2


def test_missing_fields_use_sensible_defaults() -> None:
    """An entry with only an op still yields a valid event with defaults."""
    events = parse_gloo_trace([{"op": "all_gather"}])

    evt = events[0]
    assert evt.rank == 0
    assert evt.process_group == "default"
    assert evt.collective_type == "all_gather"
    assert evt.src_rank is None
    assert evt.dst_rank is None
    assert evt.tensor_size == 0
    assert evt.enqueue_time_ns == 0
    assert evt.start_time_ns == 0
    assert evt.end_time_ns == 0
    assert evt.call_stack == []
    assert evt.sequence_id == 0


def test_empty_trace_yields_empty_list() -> None:
    """An empty entries list parses to an empty event list."""
    assert parse_gloo_trace({"entries": []}) == []
    assert parse_gloo_trace([]) == []


def test_multiple_entries_preserve_order() -> None:
    """Events are returned in trace order."""
    entries = [_full_entry(op=f"op{n}", sequence_id=n) for n in range(3)]
    events = parse_gloo_trace(entries)

    assert [e.sequence_id for e in events] == [0, 1, 2]
    assert [e.collective_type for e in events] == ["op0", "op1", "op2"]


@pytest.mark.parametrize(
    "payload",
    [
        {"entries": "not-a-list"},
        {"no_entries_key": [1, 2, 3]},
        "a-bare-string",
        42,
        None,
        [{"op": "allreduce"}, "not-a-dict"],
    ],
    ids=[
        "entries-not-list",
        "missing-entries-key",
        "bare-string",
        "bare-number",
        "none",
        "non-dict-entry",
    ],
)
def test_malformed_traces_are_rejected(payload: object) -> None:
    """Structurally invalid traces raise MalformedGlooTraceError."""
    with pytest.raises(MalformedGlooTraceError):
        parse_gloo_trace(payload)


def test_load_gloo_trace_reads_json_file(tmp_path: Path) -> None:
    """load_gloo_trace reads and parses a JSON trace from disk."""
    trace = tmp_path / "gloo_trace.json"
    trace.write_text(
        json.dumps({"entries": [_full_entry(rank=3, op="reduce_scatter")]}),
        encoding="utf-8",
    )

    events = load_gloo_trace(trace)

    assert len(events) == 1
    assert events[0].rank == 3
    assert events[0].collective_type == "reduce_scatter"


def test_load_gloo_trace_invalid_json_raises(tmp_path: Path) -> None:
    """A file that is not valid JSON surfaces json.JSONDecodeError."""
    trace = tmp_path / "bad.json"
    trace.write_text("{ not json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_gloo_trace(trace)


def test_gloo_events_feed_the_causal_graph_builder() -> None:
    """Gloo events are CollectiveEvents that build_from_events accepts directly.

    This is the framework-agnostic payoff: Gloo and flight-recorder traces
    produce the same record type, so a single graph builder consumes both.
    """
    entries = [_full_entry(rank=r, op="allreduce", sequence_id=s)
               for r in range(2) for s in range(2)]
    events = parse_gloo_trace(entries)

    graph = build_from_events(events)

    # Bucket nodes by PROV-DM kind (same query style as test_integration.py).
    agents: list[str] = []
    activities: list[str] = []
    for nid, data in graph.nodes():
        kind = data.get("kind")
        if kind == "agent":
            agents.append(nid)
        elif kind == "activity":
            activities.append(nid)

    # One agent per (rank, process_group); both ranks present.
    assert sorted(agents) == ["rank:0:pg:default_pg", "rank:1:pg:default_pg"]
    # One activity per collective (2 ranks x 2 sequences = 4).
    assert len(activities) == 4


def test_no_nccl_specific_imports_in_gloo_collector() -> None:
    """The Gloo collector must not depend on backend-specific decoders."""
    source = inspect.getsource(gloo)
    assert "import nccl" not in source
    assert "from nccl" not in source
    assert "_dump_nccl_trace" not in source
    # CollectiveEvent is the shared schema, not a backend decoder — importing it
    # is explicitly allowed.
    assert "from .flight_recorder import CollectiveEvent" in source
