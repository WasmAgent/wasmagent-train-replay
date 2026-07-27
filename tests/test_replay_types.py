"""Tests for :class:`train_replay.replay.types.DivergenceReport` (issue #357).

Scope: only the dataclass itself (the line-53 "Differential Divergence
Replay" contract) — required-field round-trip, defaults, and mutable-default
isolation. The diff/replay_diff flow that *produces* reports is a separate
Milestone 6 bullet and is not exercised here.
"""

from __future__ import annotations

from train_replay.recording.evidence import AEPRecord
from train_replay.recording.modes import RecordingMode
from train_replay.replay.types import DivergenceReport


def _record(action_id: str, rank: int, step: int) -> AEPRecord:
    return AEPRecord(
        action_id=action_id,
        rank=rank,
        step=step,
        collective_type="all_reduce",
        recording_mode=RecordingMode.FULL,
    )


def test_divergence_report_round_trips_required_fields_and_defaults() -> None:
    report = DivergenceReport(
        rank=1,
        first_divergence_step=3,
        baseline_action=_record("act-b-3", rank=1, step=3),
        candidate_action=_record("act-c-3", rank=1, step=3),
    )

    assert report.rank == 1
    assert report.first_divergence_step == 3
    assert report.baseline_action.action_id == "act-b-3"
    assert report.candidate_action.action_id == "act-c-3"
    # Optional fields default to "no correlation / empty window".
    assert report.context_window == []
    assert report.correlated_collision is False
    assert report.correlated_escalation is None


def test_divergence_report_records_context_window_and_correlation_flags() -> None:
    window = [_record("act-b-2", rank=1, step=2), _record("act-b-4", rank=1, step=4)]

    report = DivergenceReport(
        rank=1,
        first_divergence_step=3,
        baseline_action=_record("act-b-3", rank=1, step=3),
        candidate_action=_record("act-c-3", rank=1, step=3),
        context_window=window,
        correlated_collision=True,
        correlated_escalation="nccl_anomaly_score=0.95",
    )

    assert report.context_window == window
    assert report.correlated_collision is True
    assert report.correlated_escalation == "nccl_anomaly_score=0.95"


def test_divergence_report_context_window_default_is_per_instance() -> None:
    """``field(default_factory=list)`` must not share state across instances."""
    first = DivergenceReport(
        rank=0,
        first_divergence_step=1,
        baseline_action=_record("act-b-1", rank=0, step=1),
        candidate_action=_record("act-c-1", rank=0, step=1),
    )
    second = DivergenceReport(
        rank=0,
        first_divergence_step=1,
        baseline_action=_record("act-b-1", rank=0, step=1),
        candidate_action=_record("act-c-1", rank=0, step=1),
    )

    first.context_window.append(_record("act-x", rank=0, step=0))

    assert first.context_window != second.context_window
    assert second.context_window == []
