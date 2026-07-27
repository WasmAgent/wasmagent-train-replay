"""Tests for :meth:`EpochReplayer.replay_diff` (issue #364 demo dependency).

Scope: the ``replay_diff(baseline_dump, candidate_dump)`` method that the
``examples/divergence_demo.py`` script invokes — loading two Flight Recorder
dumps, comparing their rank-aligned action streams, and emitting a per-rank
:class:`DivergenceReport`. The full DivergenceReplayer / ``diff`` CLI flow is
a separate Milestone 6 bullet and is not exercised here.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from train_replay.collector.flight_recorder import CollectiveEvent, load_flight_recorder
from train_replay.graph.builder import build_from_events
from train_replay.replay.replayer import EpochReplayer


def _event(rank: int, seq: int, ctype: str = "all_reduce") -> CollectiveEvent:
    return CollectiveEvent(
        rank=rank,
        process_group="default",
        collective_type=ctype,
        src_rank=None,
        dst_rank=None,
        tensor_size=1024,
        enqueue_time_ns=seq * 1_000_000,
        start_time_ns=seq * 1_000_000 + 100_000,
        end_time_ns=seq * 1_000_000 + 500_000,
        sequence_id=seq,
    )


def _dump(events: list[CollectiveEvent], path: Path) -> Path:
    """Write events as a Flight Recorder pickle dump (plain built-in data)."""
    entries = [
        {
            "rank": e.rank,
            "pg_name": e.process_group,
            "collective_seq": e.collective_type,
            "p2p_src": e.src_rank,
            "p2p_dst": e.dst_rank,
            "input_sizes": [[e.tensor_size]],
            "time_created_ns": e.enqueue_time_ns,
            "time_started_ns": e.start_time_ns,
            "time_finished_ns": e.end_time_ns,
            "frames": e.call_stack,
            "seq_id": e.sequence_id,
        }
        for e in events
    ]
    with open(path, "wb") as f:
        pickle.dump({"entries": entries}, f)
    return path


def _aligned_events(
    ranks: int = 4,
    steps: int = 3,
    divergent_ctype: str | None = None,
    divergent_rank: int = 1,
    divergent_step: int = 1,
) -> list[CollectiveEvent]:
    events: list[CollectiveEvent] = []
    for rank in range(ranks):
        for seq in range(steps):
            ctype = "all_reduce"
            if (
                divergent_ctype is not None
                and rank == divergent_rank
                and seq == divergent_step
            ):
                ctype = divergent_ctype
            events.append(_event(rank, seq, ctype))
    return events


def _replayer(events: list[CollectiveEvent]) -> EpochReplayer:
    return EpochReplayer(build_from_events(events))


def test_replay_diff_detects_first_divergence(tmp_path: Path) -> None:
    baseline_events = _aligned_events()
    candidate_events = _aligned_events(divergent_ctype="all_gather")

    baseline_dump = _dump(baseline_events, tmp_path / "baseline.pkl")
    candidate_dump = _dump(candidate_events, tmp_path / "candidate.pkl")

    replayer = _replayer(baseline_events)
    reports = replayer.replay_diff(str(baseline_dump), str(candidate_dump))

    # Only the divergent rank is reported; aligned ranks are absent.
    assert set(reports) == {1}
    report = reports[1]
    assert report.rank == 1
    assert report.first_divergence_step == 1
    assert report.baseline_action.collective_type == "all_reduce"
    assert report.candidate_action.collective_type == "all_gather"
    # Context window brackets the divergence with the surrounding baseline steps.
    assert [a.step for a in report.context_window] == [0, 2]
    # Collision/escalation correlation is a separate Milestone 6 bullet.
    assert report.correlated_collision is False
    assert report.correlated_escalation is None


def test_replay_diff_identical_dumps_yield_empty_report(tmp_path: Path) -> None:
    events = _aligned_events()
    baseline_dump = _dump(events, tmp_path / "a.pkl")
    candidate_dump = _dump(events, tmp_path / "b.pkl")

    replayer = _replayer(events)
    reports = replayer.replay_diff(str(baseline_dump), str(candidate_dump))

    assert reports == {}


def test_replay_diff_reads_back_through_restricted_unpickler(tmp_path: Path) -> None:
    """Dumps written by the demo round-trip through load_flight_recorder."""
    events = _aligned_events()
    dump = _dump(events, tmp_path / "dump.pkl")
    loaded = load_flight_recorder(dump)
    assert [e.collective_type for e in loaded] == [e.collective_type for e in events]
