"""Tests for recording mode logic."""

from train_replay.recording.modes import (
    RecordingMode,
    RiskContext,
    SideEffectClass,
    compile_recording_policy,
)


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
