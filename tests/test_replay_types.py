"""Tests for replay-stage types (issues #357, #345).

Covers :class:`train_replay.replay.types.Divergence` (per-rank detail)
and :class:`train_replay.replay.types.DivergenceReport` (cross-run aggregate
with JSON/CBOR round-trip). The diff/replay_diff flow that *produces*
these objects is a separate Milestone 6 bullet.
"""

from __future__ import annotations

import json

from train_replay.recording.evidence import AEPRecord
from train_replay.recording.modes import RecordingMode
from train_replay.replay.types import Divergence, DivergenceReport


def _record(action_id: str, rank: int, step: int) -> AEPRecord:
    return AEPRecord(
        action_id=action_id,
        rank=rank,
        step=step,
        collective_type="all_reduce",
        recording_mode=RecordingMode.FULL,
    )


def _divergence(rank: int = 0, step: int = 1) -> Divergence:
    return Divergence(
        rank=rank,
        first_divergence_step=step,
        baseline_action=_record(f"act-b-{step}", rank=rank, step=step),
        candidate_action=_record(f"act-c-{step}", rank=rank, step=step),
    )


# -- Divergence (per-rank detail) -------------------------------------------


def test_divergence_round_trips_required_fields_and_defaults() -> None:
    div = Divergence(
        rank=1,
        first_divergence_step=3,
        baseline_action=_record("act-b-3", rank=1, step=3),
        candidate_action=_record("act-c-3", rank=1, step=3),
    )

    assert div.rank == 1
    assert div.first_divergence_step == 3
    assert div.baseline_action.action_id == "act-b-3"
    assert div.candidate_action.action_id == "act-c-3"
    # Optional fields default to "no correlation / empty window".
    assert div.context_window == []
    assert div.correlated_collision is False
    assert div.correlated_escalation is None


def test_divergence_records_context_window_and_correlation_flags() -> None:
    window = [_record("act-b-2", rank=1, step=2), _record("act-b-4", rank=1, step=4)]

    div = Divergence(
        rank=1,
        first_divergence_step=3,
        baseline_action=_record("act-b-3", rank=1, step=3),
        candidate_action=_record("act-c-3", rank=1, step=3),
        context_window=window,
        correlated_collision=True,
        correlated_escalation="nccl_anomaly_score=0.95",
    )

    assert div.context_window == window
    assert div.correlated_collision is True
    assert div.correlated_escalation == "nccl_anomaly_score=0.95"


def test_divergence_context_window_default_is_per_instance() -> None:
    """``field(default_factory=list)`` must not share state across instances."""
    first = Divergence(
        rank=0,
        first_divergence_step=1,
        baseline_action=_record("act-b-1", rank=0, step=1),
        candidate_action=_record("act-c-1", rank=0, step=1),
    )
    second = Divergence(
        rank=0,
        first_divergence_step=1,
        baseline_action=_record("act-b-1", rank=0, step=1),
        candidate_action=_record("act-c-1", rank=0, step=1),
    )

    first.context_window.append(_record("act-x", rank=0, step=0))

    assert first.context_window != second.context_window
    assert second.context_window == []


# -- DivergenceReport (cross-run aggregate) ----------------------------------


def test_divergence_report_default_fields() -> None:
    report = DivergenceReport()

    # None — "no divergence observed" — until a diff populates it (#351).
    assert report.first_divergence_step is None
    assert report.divergences == []
    assert report.per_rank_similarity == {}
    assert report.summary == ""


def test_divergence_report_to_json_round_trip() -> None:
    report = DivergenceReport(
        first_divergence_step=5,
        divergences=[_divergence(rank=0, step=5), _divergence(rank=1, step=5)],
        per_rank_similarity={0: 0.98, 1: 0.95},
        summary="Ranks 0-1 diverged at step 5",
    )

    restored = DivergenceReport.from_json(report.to_json())

    assert restored.first_divergence_step == 5
    assert len(restored.divergences) == 2
    assert restored.divergences[0].rank == 0
    assert restored.divergences[0].baseline_action.action_id == "act-b-5"
    assert restored.divergences[1].rank == 1
    assert restored.per_rank_similarity == {0: 0.98, 1: 0.95}
    assert restored.summary == "Ranks 0-1 diverged at step 5"


def test_divergence_report_to_cbor_round_trip() -> None:
    report = DivergenceReport(
        first_divergence_step=3,
        divergences=[_divergence(rank=0, step=3)],
        per_rank_similarity={0: 1.0},
        summary="Single rank divergence",
    )

    restored = DivergenceReport.from_cbor(report.to_cbor())

    assert restored.first_divergence_step == 3
    assert len(restored.divergences) == 1
    assert restored.divergences[0].rank == 0
    assert restored.summary == "Single rank divergence"


def test_divergence_report_json_is_valid_json() -> None:
    report = DivergenceReport(
        first_divergence_step=1,
        divergences=[_divergence()],
        per_rank_similarity={0: 0.9},
        summary="test",
    )

    raw = report.to_json()
    parsed = json.loads(raw)  # must not raise
    assert parsed["first_divergence_step"] == 1
    assert parsed["summary"] == "test"


def test_divergence_report_empty_round_trips() -> None:
    report = DivergenceReport()

    assert DivergenceReport.from_json(report.to_json()) == report
    assert DivergenceReport.from_cbor(report.to_cbor()) == report


def test_divergence_report_divergences_default_is_per_instance() -> None:
    """``field(default_factory=list)`` must not share state across instances."""
    first = DivergenceReport()
    second = DivergenceReport()
    first.divergences.append(_divergence())

    assert len(first.divergences) == 1
    assert second.divergences == []
