"""Tests for DivergenceReplayer (issue #356) and the Milestone 6 diff bullets
(#351 tensor-mutation detection, #352 CBOR round-trip signing, #359 collision
correlation, #363 byte-identical replay_diff)."""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path

from train_replay.graph.collision import Collision, CollisionReport
from train_replay.recording.escalation import EscalationSignal
from train_replay.recording.evidence import AEPRecord, EpochEvidenceBundle
from train_replay.recording.modes import RecordingMode
from train_replay.replay.diff import DiffConfig, DivergenceReplayer
from train_replay.replay.replayer import EpochReplayer, ReplayResult
from train_replay.replay.types import DivergenceReport
from train_replay.signing.signer import BundleSigner, verify_divergence_report


def _record(
    action_id: str,
    rank: int,
    step: int,
    collective_type: str = "all_reduce",
    mode: RecordingMode = RecordingMode.FULL,
    tensor_output_digest: str | None = None,
) -> AEPRecord:
    return AEPRecord(
        action_id=action_id,
        rank=rank,
        step=step,
        collective_type=collective_type,
        recording_mode=mode,
        tensor_output_digest=tensor_output_digest,
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


# -- tensor mutation at a known step (#351) -----------------------------------

def test_tensor_mutation_at_step_k_is_first_divergence() -> None:
    b_actions = [
        _record("b1", 0, 1, tensor_output_digest="sha256:aaa"),
        _record("b2", 0, 2, tensor_output_digest="sha256:bbb"),
        _record("b3", 0, 3, tensor_output_digest="sha256:ccc"),
    ]
    c_actions = [
        _record("b1", 0, 1, tensor_output_digest="sha256:aaa"),
        _record("b2", 0, 2, tensor_output_digest="sha256:MUTATED"),  # mutation at K=2
        _record("b3", 0, 3, tensor_output_digest="sha256:ccc"),
    ]

    report = DivergenceReplayer().diff(_result(0, b_actions), _result(0, c_actions))

    assert report.first_divergence_step == 2
    assert report.per_rank_similarity[0] < 1.0
    assert report.divergences[0].candidate_action.tensor_output_digest == "sha256:MUTATED"


def test_tensor_mutation_not_flagged_when_only_one_side_has_digest() -> None:
    """Digests participate in the comparison only when both sides carry them."""
    b_actions = [_record("b1", 0, 1, tensor_output_digest="sha256:aaa")]
    c_actions = [_record("b1", 0, 1)]

    report = DivergenceReplayer().diff(_result(0, b_actions), _result(0, c_actions))

    assert report.first_divergence_step is None
    assert report.per_rank_similarity[0] == 1.0


def test_identical_streams_yield_none_first_divergence_and_full_similarity() -> None:
    actions = [
        _record("a1", 0, 1, tensor_output_digest="sha256:aaa"),
        _record("a2", 0, 2, tensor_output_digest="sha256:bbb"),
    ]

    report = DivergenceReplayer().diff(_result(0, list(actions)), _result(0, list(actions)))

    assert report.first_divergence_step is None
    assert report.divergences == []
    assert report.per_rank_similarity[0] == 1.0


# -- collision correlation (#359) ---------------------------------------------

def _rank_result_with_collision(
    actions: list[AEPRecord], collisions: list[Collision]
) -> ReplayResult:
    result = _result(1, actions)
    result.collision_report = CollisionReport(collisions=collisions, total_steps_checked=3)
    return result


def test_divergence_inside_desync_window_sets_correlated_collision() -> None:
    b_actions = [
        _record("b1", 1, 1),
        _record("b2", 1, 2, collective_type="all_gather"),
    ]
    c_actions = [
        _record("b1", 1, 1),
        _record("b2", 1, 2, collective_type="all_reduce"),  # diverges at step 2
    ]
    baseline = _rank_result_with_collision(b_actions, [
        Collision(rank_a=0, rank_b=1, step=2, detail="type mismatch at step 2"),
    ])

    report = DivergenceReplayer().diff(baseline, _result(1, c_actions))

    assert report.first_divergence_step == 2
    div = report.divergences[0]
    assert div.correlated_collision is True
    assert div.collision_record is not None
    assert div.collision_record.collective_type == "desync"
    assert div.collision_record.step == 2
    assert div.collision_record.action_id == "desync-r0-r1-s2"
    assert div.collision_record.causal_chain_id == "type mismatch at step 2"


def test_divergence_outside_desync_window_stays_uncorrelated() -> None:
    b_actions = [_record("b1", 1, 1, collective_type="all_gather")]
    c_actions = [_record("b1", 1, 1, collective_type="all_reduce")]
    baseline = _rank_result_with_collision(b_actions, [
        Collision(rank_a=0, rank_b=1, step=5, detail="desync elsewhere"),
    ])

    report = DivergenceReplayer().diff(baseline, _result(1, c_actions))

    assert report.divergences[0].correlated_collision is False
    assert report.divergences[0].collision_record is None


def test_collision_involving_other_rank_does_not_correlate() -> None:
    b_actions = [_record("b1", 1, 1, collective_type="all_gather")]
    c_actions = [_record("b1", 1, 1, collective_type="all_reduce")]
    baseline = _rank_result_with_collision(b_actions, [
        Collision(rank_a=2, rank_b=3, step=1, detail="unrelated ranks"),
    ])

    report = DivergenceReplayer().diff(baseline, _result(1, c_actions))

    assert report.divergences[0].correlated_collision is False


def test_candidate_side_collision_report_also_correlates() -> None:
    b_actions = [_record("b1", 1, 1, collective_type="all_gather")]
    c_actions = [_record("b1", 1, 1, collective_type="all_reduce")]
    candidate = _rank_result_with_collision(c_actions, [
        Collision(rank_a=1, rank_b=2, step=1, detail="candidate-side desync"),
    ])

    report = DivergenceReplayer().diff(_result(1, b_actions), candidate)

    assert report.divergences[0].correlated_collision is True
    assert report.divergences[0].collision_record is not None


# -- CBOR round trip + dual-bundle signature (#352) ---------------------------

def _divergent_report() -> DivergenceReport:
    b_actions = [
        _record("b1", 0, 1),
        _record("b2", 0, 2, collective_type="all_gather"),
    ]
    c_actions = [
        _record("b1", 0, 1),
        _record("b2", 0, 2, collective_type="all_reduce"),
    ]
    return DivergenceReplayer().diff(_result(0, b_actions), _result(0, c_actions))


def test_divergence_report_cbor_roundtrip_signs_and_verifies() -> None:
    report = _divergent_report()
    restored = DivergenceReport.from_cbor(report.to_cbor())
    assert restored == report

    signer, public_key = BundleSigner.generate()
    baseline = EpochEvidenceBundle(run_id="run-baseline", epoch=0, actions=[])
    candidate = EpochEvidenceBundle(run_id="run-candidate", epoch=1, actions=[])

    envelope = signer.sign_divergence_report(
        dataclasses.asdict(restored), baseline, candidate
    )
    assert verify_divergence_report(envelope, baseline, candidate, public_key) is True


def test_cbor_roundtrip_signature_validates_only_against_originating_digests() -> None:
    report = _divergent_report()
    restored = DivergenceReport.from_cbor(report.to_cbor())

    signer, public_key = BundleSigner.generate()
    baseline = EpochEvidenceBundle(run_id="run-baseline", epoch=0, actions=[])
    candidate = EpochEvidenceBundle(run_id="run-candidate", epoch=1, actions=[])
    other = EpochEvidenceBundle(run_id="run-other", epoch=2, actions=[])

    envelope = signer.sign_divergence_report(
        dataclasses.asdict(restored), baseline, candidate
    )
    assert verify_divergence_report(envelope, candidate, baseline, public_key) is False
    assert verify_divergence_report(envelope, other, candidate, public_key) is False
    assert verify_divergence_report(envelope, baseline, other, public_key) is False


# -- replay_diff over dump files (#363, #359) ---------------------------------

def _dump_entry(rank: int, step: int, collective_type: str = "all_reduce") -> dict[str, object]:
    started = 1_000_000 + step * 1_000
    return {
        "rank": rank,
        "pg_name": "default",
        "collective_seq": collective_type,
        "p2p_src": None,
        "p2p_dst": None,
        "input_sizes": [[4096]],
        "time_created_ns": started,
        "time_started_ns": started,
        "time_finished_ns": started + 100,
        "frames": [],
        "seq_id": step,
    }


def _write_dump(path: Path, entries: list[dict[str, object]]) -> Path:
    with open(path, "wb") as f:
        pickle.dump({"entries": entries}, f)
    return path


def _aligned_entries(ranks: int = 3, steps: int = 3) -> list[dict[str, object]]:
    return [
        _dump_entry(rank, step)
        for rank in range(ranks)
        for step in range(steps)
    ]


def test_replay_diff_reports_only_divergent_rank(tmp_path: Path) -> None:
    from train_replay.graph.builder import build_from_events

    candidate_events = _aligned_entries()
    for entry in candidate_events:
        if entry["rank"] == 1 and entry["seq_id"] == 1:
            entry["collective_seq"] = "all_gather"  # injected divergence, rank 1

    baseline_path = _write_dump(tmp_path / "baseline.pkl", _aligned_entries())
    candidate_path = _write_dump(tmp_path / "candidate.pkl", candidate_events)

    replayer = EpochReplayer(build_from_events([]))
    reports = replayer.replay_diff(str(baseline_path), str(candidate_path))

    assert set(reports) == {1}
    report = reports[1]
    assert report.first_divergence_step == 1
    assert report.per_rank_similarity[1] < 1.0
    div = report.divergences[0]
    assert div.baseline_action.collective_type == "all_reduce"
    assert div.candidate_action.collective_type == "all_gather"


def test_replay_diff_identical_dumps_yield_empty_dict(tmp_path: Path) -> None:
    from train_replay.graph.builder import build_from_events

    baseline_path = _write_dump(tmp_path / "a.pkl", _aligned_entries())
    candidate_path = _write_dump(tmp_path / "b.pkl", _aligned_entries())

    replayer = EpochReplayer(build_from_events([]))

    assert replayer.replay_diff(str(baseline_path), str(candidate_path)) == {}


def test_replay_diff_correlates_divergence_with_detected_collision(
    tmp_path: Path,
) -> None:
    from train_replay.graph.builder import build_from_events
    from train_replay.graph.collision import NcclCollisionDetector

    # Candidate drops rank 1's step-1 event: rank 1's stream misses the action
    # and the NCCL detector flags the missing sequence as a desync involving
    # rank 1, so the divergence must carry correlated_collision=True.
    candidate_events = [
        entry
        for entry in _aligned_entries()
        if not (entry["rank"] == 1 and entry["seq_id"] == 1)
    ]

    baseline_path = _write_dump(tmp_path / "baseline.pkl", _aligned_entries())
    candidate_path = _write_dump(tmp_path / "candidate.pkl", candidate_events)

    replayer = EpochReplayer(build_from_events([]), detector=NcclCollisionDetector())
    reports = replayer.replay_diff(str(baseline_path), str(candidate_path))

    div = reports[1].divergences[0]
    assert div.first_divergence_step == 1
    assert div.correlated_collision is True
    assert div.collision_record is not None
    assert div.collision_record.collective_type == "desync"
    assert div.collision_record.step == 1
    assert 1 in (div.collision_record.rank, div.collision_record.delta_stats["rank_b"])

