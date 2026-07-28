"""Tests for DivergenceReplayer (issue #356)."""

from __future__ import annotations

from train_replay.recording.escalation import EscalationSignal
from train_replay.recording.evidence import AEPRecord
from train_replay.recording.modes import RecordingMode
from train_replay.replay.diff import DiffConfig, DivergenceReplayer
from train_replay.replay.replayer import ReplayResult


def _record(action_id: str, rank: int, step: int,
            collective_type: str = "all_reduce",
            mode: RecordingMode = RecordingMode.FULL) -> AEPRecord:
    return AEPRecord(
        action_id=action_id,
        rank=rank,
        step=step,
        collective_type=collective_type,
        recording_mode=mode,
    )


def _result(rank: int, actions: list[AEPRecord]) -> ReplayResult:
    # Build a minimal result directly without going through EpochReplayer
    return ReplayResult(
        epoch=0,
        rank=rank,
        causal_ancestors=[],
        suspicious_actions=actions,
    )


# -- no divergence -----------------------------------------------------------

def test_no_divergence_when_streams_identical() -> None:
    actions = [_record("a1", rank=0, step=1), _record("a2", rank=0, step=2)]
    baseline = _result(0, actions)
    candidate = _result(0, list(actions))  # identical copy

    report = DivergenceReplayer().diff(baseline, candidate)

    assert report.divergences == []
    assert report.per_rank_similarity[0] == 1.0
    assert "no divergence" in report.summary


# -- divergence detected -----------------------------------------------------

def test_diff_finds_first_divergent_step() -> None:
    b_actions = [
        _record("b1", rank=0, step=1),
        _record("b2", rank=0, step=2, collective_type="all_gather"),
    ]
    c_actions = [
        _record("c1", rank=0, step=1),
        _record("c2", rank=0, step=2, collective_type="all_reduce"),  # differs
    ]
    report = DivergenceReplayer().diff(_result(0, b_actions), _result(0, c_actions))

    assert len(report.divergences) == 1
    assert report.first_divergence_step == 2
    assert report.divergences[0].baseline_action.collective_type == "all_gather"
    assert report.divergences[0].candidate_action.collective_type == "all_reduce"


def test_diff_missing_action_in_candidate_counts_as_divergence() -> None:
    b_actions = [_record("b1", rank=0, step=1), _record("b2", rank=0, step=2)]
    c_actions = [_record("c1", rank=0, step=1)]  # step 2 missing

    report = DivergenceReplayer().diff(_result(0, b_actions), _result(0, c_actions))

    assert len(report.divergences) == 1
    assert report.first_divergence_step == 2


def test_diff_empty_streams_yield_no_divergence() -> None:
    report = DivergenceReplayer().diff(_result(0, []), _result(0, []))

    assert report.divergences == []
    assert report.per_rank_similarity == {0: 1.0}


# -- context window ----------------------------------------------------------

def test_context_window_respects_config_size() -> None:
    steps = list(range(1, 11))  # steps 1-10
    b_actions = [_record(f"b{s}", rank=0, step=s) for s in steps]
    c_actions = [
        _record(f"c{s}", rank=0, step=s,
                collective_type="all_gather" if s == 5 else "all_reduce")
        for s in steps
    ]

    cfg = DiffConfig(context_window_size=2)
    report = DivergenceReplayer(cfg).diff(_result(0, b_actions), _result(0, c_actions))

    assert len(report.divergences) == 1
    window = report.divergences[0].context_window
    # 2 steps before + 2 steps after (excluding pivot), all from baseline
    assert len(window) <= 4
    window_steps = {r.step for r in window}
    assert 5 not in window_steps  # pivot excluded


# -- escalation correlation (#361) ------------------------------------------

def test_escalation_signal_populates_correlated_escalation() -> None:
    b_actions = [_record("b1", rank=0, step=1, collective_type="all_gather")]
    c_actions = [_record("c1", rank=0, step=1, collective_type="all_reduce")]
    signal = EscalationSignal(source="prom", severity=0.92, metric_name="nccl_anomaly_score")

    report = DivergenceReplayer().diff(
        _result(0, b_actions), _result(0, c_actions),
        escalation_signals=[signal],
    )

    assert len(report.divergences) == 1
    assert report.divergences[0].correlated_escalation == "nccl_anomaly_score=0.92"


def test_no_escalation_signal_leaves_correlated_escalation_none() -> None:
    b_actions = [_record("b1", rank=0, step=1, collective_type="all_gather")]
    c_actions = [_record("c1", rank=0, step=1, collective_type="all_reduce")]

    report = DivergenceReplayer().diff(_result(0, b_actions), _result(0, c_actions))

    assert report.divergences[0].correlated_escalation is None


# -- similarity score --------------------------------------------------------

def test_similarity_score_reflects_agreement_fraction() -> None:
    b_actions = [
        _record("b1", rank=0, step=1),
        _record("b2", rank=0, step=2, collective_type="all_gather"),
        _record("b3", rank=0, step=3),
    ]
    c_actions = [
        _record("c1", rank=0, step=1),
        _record("c2", rank=0, step=2, collective_type="all_reduce"),  # differs
        _record("c3", rank=0, step=3),
    ]
    report = DivergenceReplayer().diff(_result(0, b_actions), _result(0, c_actions))

    # 2 out of 3 steps agree before we stop at first divergence
    # similarity = (3-1)/3 ≈ 0.6667 (we only stop at first divergence)
    assert 0.0 < report.per_rank_similarity[0] <= 1.0
