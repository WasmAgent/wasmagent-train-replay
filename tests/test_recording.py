"""Tests for recording mode logic."""

from train_replay.collector.flight_recorder import CollectiveEvent
from train_replay.recording.escalation import EscalationSignal
from train_replay.recording.modes import (
    RecordingMode,
    RiskContext,
    SideEffectClass,
    compile_recording_policy,
)
from train_replay.recording.recorder import EpochRecorder


def test_read_yields_validation():
    ctx = RiskContext(side_effect_class=SideEffectClass.READ)
    assert compile_recording_policy(ctx).mode == RecordingMode.VALIDATION


def test_mutate_local_yields_delta():
    ctx = RiskContext(side_effect_class=SideEffectClass.MUTATE_LOCAL)
    assert compile_recording_policy(ctx).mode == RecordingMode.DELTA


def test_network_egress_yields_full():
    ctx = RiskContext(side_effect_class=SideEffectClass.NETWORK_EGRESS)
    assert compile_recording_policy(ctx).mode == RecordingMode.FULL


def test_vetted_always_full():
    ctx = RiskContext(was_vetted=True, side_effect_class=SideEffectClass.READ)
    assert compile_recording_policy(ctx).mode == RecordingMode.FULL


def test_taint_chain_on_mutate_yields_full():
    ctx = RiskContext(taint_chain_length=2, side_effect_class=SideEffectClass.MUTATE_LOCAL)
    assert compile_recording_policy(ctx).mode == RecordingMode.FULL


def test_unknown_class_yields_full():
    ctx = RiskContext(side_effect_class=SideEffectClass.UNKNOWN)
    assert compile_recording_policy(ctx).mode == RecordingMode.FULL


def test_escalation_signal_yields_full_with_external_reason():
    ctx = RiskContext(side_effect_class=SideEffectClass.READ)
    policy = compile_recording_policy(ctx, escalation={"source": "nccl-inspector"})

    assert policy.mode == RecordingMode.FULL
    assert policy.reason == "external escalation signal"


def test_epoch_recorder_record_with_escalation_passes_signal_to_policy():
    recorder = EpochRecorder(run_id="run-1", epoch=3)
    event = CollectiveEvent(
        rank=2,
        process_group="default",
        collective_type="barrier",
        src_rank=None,
        dst_rank=None,
        tensor_size=0,
        enqueue_time_ns=100,
        start_time_ns=200,
        end_time_ns=300,
        sequence_id=7,
    )

    recorder.record_with_escalation(event, {"source": "nccl-inspector"})

    [record] = recorder.bundle().actions
    assert record.action_id == "r2:seq7"
    assert record.rank == 2
    assert record.step == 7
    assert record.collective_type == "barrier"
    assert record.recording_mode == RecordingMode.FULL
    assert record.timestamp_ns == 200


def test_epoch_recorder_record_with_escalation_accepts_typed_signal():
    """The escalation parameter's declared contract is the EscalationSignal dataclass,
    not just a duck-typed dict. A real EscalationSignal (as produced by
    PrometheusAnomalySource.poll()) must flow through record_with_escalation into
    compile_recording_policy and yield FULL mode."""
    recorder = EpochRecorder(run_id="run-1", epoch=3)
    event = CollectiveEvent(
        rank=4,
        process_group="default",
        collective_type="all_reduce",
        src_rank=None,
        dst_rank=None,
        tensor_size=1024,
        enqueue_time_ns=1000,
        start_time_ns=2000,
        end_time_ns=3000,
        sequence_id=9,
    )
    signal = EscalationSignal(
        source="nccl-inspector", severity=0.95, metric_name="nccl_anomaly_score"
    )

    recorder.record_with_escalation(event, signal)

    [record] = recorder.bundle().actions
    assert record.action_id == "r4:seq9"
    assert record.rank == 4
    assert record.step == 9
    assert record.collective_type == "all_reduce"
    assert record.recording_mode == RecordingMode.FULL
    assert record.timestamp_ns == 2000


def test_record_with_escalation_threads_signal_past_baseline_read():
    """The bullet's contract is that record_with_escalation *passes the signal
    through to compile_recording_policy()*. A 'recv' collective is a READ
    side-effect, so record_collective records it as VALIDATION; the same event
    passed through record_with_escalation must come back as FULL. This contrast
    proves the escalation signal (not the side-effect class) drove the policy,
    and that it actually reached compile_recording_policy."""
    event = CollectiveEvent(
        rank=7,
        process_group="default",
        collective_type="recv",
        src_rank=None,
        dst_rank=None,
        tensor_size=512,
        enqueue_time_ns=10_000,
        start_time_ns=20_000,
        end_time_ns=30_000,
        sequence_id=11,
    )

    # Baseline: without escalation, a recv is a READ -> VALIDATION.
    baseline = EpochRecorder(run_id="run-1", epoch=3)
    baseline.record_collective(event)
    [baseline_record] = baseline.bundle().actions
    assert baseline_record.recording_mode == RecordingMode.VALIDATION

    # With the signal threaded through, the same READ event becomes FULL.
    recorder = EpochRecorder(run_id="run-1", epoch=3)
    signal = EscalationSignal(
        source="nccl-inspector", severity=0.9, metric_name="nccl_anomaly_score"
    )
    recorder.record_with_escalation(event, signal)

    [record] = recorder.bundle().actions
    assert record.action_id == "r7:seq11"
    assert record.rank == 7
    assert record.step == 11
    assert record.collective_type == "recv"
    assert record.recording_mode == RecordingMode.FULL
    assert record.timestamp_ns == 20_000


def test_anomaly_signal_above_threshold_yields_full():
    from train_replay.recording.modes import ANOMALY_SCORE_THRESHOLD, AnomalySignal

    ctx = RiskContext(side_effect_class=SideEffectClass.READ)
    signal = AnomalySignal(score=ANOMALY_SCORE_THRESHOLD + 0.1)
    policy = compile_recording_policy(ctx, anomaly_signal=signal)

    assert policy.mode == RecordingMode.FULL
    assert policy.reason == "statistical anomaly detected"


def test_anomaly_signal_below_threshold_does_not_escalate():
    from train_replay.recording.modes import ANOMALY_SCORE_THRESHOLD, AnomalySignal

    ctx = RiskContext(side_effect_class=SideEffectClass.READ)
    signal = AnomalySignal(score=ANOMALY_SCORE_THRESHOLD - 0.1)
    policy = compile_recording_policy(ctx, anomaly_signal=signal)

    assert policy.mode == RecordingMode.VALIDATION


def test_anomaly_signal_none_is_noop():
    """Passing anomaly_signal=None behaves the same as omitting it."""
    ctx = RiskContext(side_effect_class=SideEffectClass.READ)
    policy_with_none = compile_recording_policy(ctx, anomaly_signal=None)
    policy_without = compile_recording_policy(ctx)

    assert policy_with_none.mode == policy_without.mode
    assert policy_with_none.reason == policy_without.reason


def test_anomaly_signal_at_exact_threshold_does_not_escalate():
    from train_replay.recording.modes import ANOMALY_SCORE_THRESHOLD, AnomalySignal

    ctx = RiskContext(side_effect_class=SideEffectClass.READ)
    signal = AnomalySignal(score=ANOMALY_SCORE_THRESHOLD)
    policy = compile_recording_policy(ctx, anomaly_signal=signal)

    # Threshold check is strictly greater-than, so exactly at threshold is no-op.
    assert policy.mode == RecordingMode.VALIDATION


def test_escalation_takes_priority_over_anomaly_signal():
    """External escalation signal wins over anomaly signal."""
    from train_replay.recording.modes import ANOMALY_SCORE_THRESHOLD, AnomalySignal

    ctx = RiskContext(side_effect_class=SideEffectClass.READ)
    anomaly = AnomalySignal(score=ANOMALY_SCORE_THRESHOLD + 0.5)
    esc = EscalationSignal(source="prom", severity=1.0, metric_name="x")
    policy = compile_recording_policy(ctx, escalation=esc, anomaly_signal=anomaly)
    assert policy.mode == RecordingMode.FULL
    assert policy.reason == "external escalation signal"


def test_record_determinism_anchor_appends_to_bundle():
    from train_replay.recording.evidence import DeterminismAnchor

    recorder = EpochRecorder(run_id='run-1', epoch=5)
    recorder.record_determinism_anchor(
        step=10,
        rank=2,
        rng_seed_snapshot=42,
        collective_order_hash='abc123',
        reduce_scatter_checksum='def456',
    )

    b = recorder.bundle()
    assert len(b.determinism_anchors) == 1
    anchor = b.determinism_anchors[0]
    assert isinstance(anchor, DeterminismAnchor)
    assert anchor.step == 10
    assert anchor.rank == 2
    assert anchor.rng_seed_snapshot == 42
    assert anchor.collective_order_hash == 'abc123'
    assert anchor.reduce_scatter_checksum == 'def456'


def test_multiple_determinism_anchors_per_rank():
    recorder = EpochRecorder(run_id='run-2', epoch=1)
    for step in range(3):
        for rank in range(2):
            recorder.record_determinism_anchor(
                step=step,
                rank=rank,
                rng_seed_snapshot=step * 100 + rank,
                collective_order_hash=f'hash-{step}-{rank}',
                reduce_scatter_checksum=f'cksum-{step}-{rank}',
            )

    anchors = recorder.bundle().determinism_anchors
    assert len(anchors) == 6
    assert anchors[0].step == 0
    assert anchors[0].rank == 0
    assert anchors[-1].step == 2
    assert anchors[-1].rank == 1


def test_determinism_anchor_roundtrip_through_json():

    recorder = EpochRecorder(run_id='run-rt', epoch=3)
    recorder.record_determinism_anchor(
        step=1, rank=0, rng_seed_snapshot=99,
        collective_order_hash='h1', reduce_scatter_checksum='c1',
    )
    recorder.record_determinism_anchor(
        step=2, rank=1, rng_seed_snapshot=100,
        collective_order_hash='h2', reduce_scatter_checksum='c2',
    )

    b = recorder.bundle()
    restored = b.from_json(b.to_json())

    assert len(restored.determinism_anchors) == 2
    a0 = restored.determinism_anchors[0]
    assert a0.step == 1
    assert a0.rank == 0
    assert a0.rng_seed_snapshot == 99
    assert a0.collective_order_hash == 'h1'
    assert a0.reduce_scatter_checksum == 'c1'
    a1 = restored.determinism_anchors[1]
    assert a1.step == 2
    assert a1.rank == 1


def test_determinism_anchor_included_in_canonical_bytes():

    r1 = EpochRecorder(run_id='canon', epoch=0)
    r2 = EpochRecorder(run_id='canon', epoch=0)

    r1.record_determinism_anchor(1, 0, 10, 'ha', 'ca')
    r2.record_determinism_anchor(1, 0, 99, 'hb', 'cb')

    # Different anchors must produce different digests.
    assert r1.bundle().digest() != r2.bundle().digest()
