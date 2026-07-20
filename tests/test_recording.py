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
