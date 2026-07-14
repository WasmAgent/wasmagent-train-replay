"""Tests for MTIA collector adapter and graph builder integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from train_replay.collector.mtia import (
    MtiaEvent,
    MtiaTraceParseError,
    parse_mtia_trace,
)
from train_replay.graph.builder import build_from_mtia_events

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_mtia_trace(tmp_path: Path) -> Path:
    """Write a valid MTIA profiler JSON trace and return its path."""
    data = {
        "events": [
            {
                "rank": 0,
                "process_group": "default",
                "op_type": "all_reduce",
                "src_rank": None,
                "dst_rank": None,
                "tensor_size": 4096,
                "start_time_ns": 100_000_000,
                "end_time_ns": 100_001_000,
                "sequence_id": 1,
                "call_stack": ["train.py:42", "model.py:17"],
            },
            {
                "rank": 1,
                "process_group": "default",
                "op_type": "all_reduce",
                "src_rank": None,
                "dst_rank": None,
                "tensor_size": 4096,
                "start_time_ns": 100_000_500,
                "end_time_ns": 100_001_500,
                "sequence_id": 1,
                "call_stack": [],
            },
            {
                "rank": 0,
                "process_group": "tp",
                "op_type": "all_gather",
                "src_rank": 0,
                "dst_rank": 1,
                "tensor_size": 2048,
                "start_time_ns": 200_000_000,
                "end_time_ns": 200_002_000,
                "sequence_id": 2,
                "call_stack": ["train.py:55"],
            },
        ]
    }
    p = tmp_path / "mtia_trace.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def multi_rank_events() -> list[MtiaEvent]:
    """Pre-built list of MtiaEvents spanning two ranks."""
    return [
        MtiaEvent(
            rank=0, process_group="default", op_type="all_reduce",
            tensor_size=8192, start_time_ns=500, end_time_ns=600,
            sequence_id=10,
        ),
        MtiaEvent(
            rank=1, process_group="default", op_type="all_reduce",
            tensor_size=8192, start_time_ns=510, end_time_ns=620,
            sequence_id=10,
        ),
        MtiaEvent(
            rank=0, process_group="default", op_type="broadcast",
            src_rank=0, dst_rank=1, tensor_size=4096,
            start_time_ns=700, end_time_ns=750, sequence_id=11,
        ),
    ]


# ---------------------------------------------------------------------------
# MtiaEvent dataclass
# ---------------------------------------------------------------------------

class TestMtiaEvent:
    def test_defaults(self) -> None:
        evt = MtiaEvent(rank=0, process_group="pg", op_type="all_reduce")
        assert evt.src_rank is None
        assert evt.dst_rank is None
        assert evt.tensor_size == 0
        assert evt.start_time_ns == 0
        assert evt.end_time_ns == 0
        assert evt.sequence_id == 0
        assert evt.call_stack == []

    def test_full_construction(self) -> None:
        evt = MtiaEvent(
            rank=2, process_group="pg0", op_type="reduce_scatter",
            src_rank=0, dst_rank=3, tensor_size=1024,
            start_time_ns=1000, end_time_ns=2000, sequence_id=5,
            call_stack=["a.py:1"],
        )
        assert evt.rank == 2
        assert evt.op_type == "reduce_scatter"
        assert evt.tensor_size == 1024
        assert evt.call_stack == ["a.py:1"]


# ---------------------------------------------------------------------------
# parse_mtia_trace
# ---------------------------------------------------------------------------

class TestParseMtiaTrace:
    def test_valid_trace(self, sample_mtia_trace: Path) -> None:
        events = parse_mtia_trace(sample_mtia_trace)
        assert len(events) == 3

    def test_event_fields(self, sample_mtia_trace: Path) -> None:
        events = parse_mtia_trace(sample_mtia_trace)
        first = events[0]
        assert first.rank == 0
        assert first.process_group == "default"
        assert first.op_type == "all_reduce"
        assert first.tensor_size == 4096
        assert first.start_time_ns == 100_000_000
        assert first.end_time_ns == 100_001_000
        assert first.sequence_id == 1
        assert first.call_stack == ["train.py:42", "model.py:17"]

    def test_empty_events_list(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text('{"events": []}', encoding="utf-8")
        events = parse_mtia_trace(p)
        assert events == []

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(MtiaTraceParseError, match="Cannot read"):
            parse_mtia_trace(tmp_path / "nonexistent.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{broken", encoding="utf-8")
        with pytest.raises(MtiaTraceParseError, match="Invalid JSON"):
            parse_mtia_trace(p)

    def test_not_a_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "list.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(MtiaTraceParseError, match="JSON object"):
            parse_mtia_trace(p)

    def test_missing_events_key(self, tmp_path: Path) -> None:
        p = tmp_path / "no_events.json"
        p.write_text('{"trace": []}', encoding="utf-8")
        with pytest.raises(MtiaTraceParseError, match="events"):
            parse_mtia_trace(p)

    def test_events_not_a_list(self, tmp_path: Path) -> None:
        p = tmp_path / "bad_events.json"
        p.write_text('{"events": "oops"}', encoding="utf-8")
        with pytest.raises(MtiaTraceParseError, match="events"):
            parse_mtia_trace(p)

    def test_event_not_a_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "evt_str.json"
        p.write_text('{"events": [42]}', encoding="utf-8")
        with pytest.raises(MtiaTraceParseError, match="Event at index 0"):
            parse_mtia_trace(p)

    def test_minimal_event_uses_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "minimal.json"
        p.write_text('{"events": [{"rank": 0}]}', encoding="utf-8")
        events = parse_mtia_trace(p)
        assert len(events) == 1
        assert events[0].rank == 0
        assert events[0].op_type == "unknown"
        assert events[0].process_group == "default"


# ---------------------------------------------------------------------------
# build_from_mtia_events
# ---------------------------------------------------------------------------

class TestBuildFromMtiaEvents:
    def test_import_succeeds(self) -> None:
        """Acceptance: import must succeed (tested by pytest invocation)."""
        from train_replay.graph.builder import build_from_mtia_events as fn  # noqa: F401

    def test_single_event_graph(self, multi_rank_events: list[MtiaEvent]) -> None:
        single = multi_rank_events[:1]
        graph = build_from_mtia_events(single)
        node_ids = [n for n, _ in graph.nodes()]
        # Should have: 1 agent, 1 activity, 2 entities (in + out)
        assert "rank:0:pg:default" in node_ids
        assert "act:0:all_reduce:10" in node_ids

    def test_multi_rank_creates_two_agents(self, multi_rank_events: list[MtiaEvent]) -> None:
        graph = build_from_mtia_events(multi_rank_events)
        node_ids = [n for n, _ in graph.nodes()]
        assert "rank:0:pg:default" in node_ids
        assert "rank:1:pg:default" in node_ids

    def test_causal_ancestor_traversal(self, multi_rank_events: list[MtiaEvent]) -> None:
        graph = build_from_mtia_events(multi_rank_events)
        # Output tensor of the first event should have its activity as ancestor
        out_id = "tensor:0:10:out"
        ancestors = graph.ancestors_of(out_id)
        assert "act:0:all_reduce:10" in ancestors

    def test_leaf_input_no_ancestors(self, multi_rank_events: list[MtiaEvent]) -> None:
        graph = build_from_mtia_events(multi_rank_events)
        in_id = "tensor:0:10:in"
        assert graph.ancestors_of(in_id) == []

    def test_op_type_preserved_in_activity(self, multi_rank_events: list[MtiaEvent]) -> None:
        graph = build_from_mtia_events(multi_rank_events)
        # The broadcast event (sequence 11) should appear with op_type
        node_ids = [n for n, _ in graph.nodes()]
        assert "act:0:broadcast:11" in node_ids

    def test_no_nccl_imports_in_mtia_module(self) -> None:
        """Acceptance: mtia.py must not import NCCL-specific symbols."""
        import ast

        import train_replay.collector.mtia as mod

        source = mod.__file__
        assert source is not None
        with open(source, encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "nccl" not in alias.name.lower()
            elif isinstance(node, ast.ImportFrom):
                assert "nccl" not in node.module.lower() if node.module else True
