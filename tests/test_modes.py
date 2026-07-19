"""Tests for the escalation-source integration interface in ``compile_recording_policy``.

These exercise the *integration contract* (issue #118): a mock object
implementing :class:`EscalationSource` can be accepted by
:func:`apply_escalation_source` and its polled signal drives the recording
decision — no dependency on the concrete ``PrometheusAnomalySource`` from
``escalation.py`` is required.
"""

from __future__ import annotations

from typing import Any

from train_replay.recording.modes import (
    ESCALATION_SEVERITY_HIGH,
    EscalationSource,
    RecordingMode,
    RiskContext,
    SideEffectClass,
    apply_escalation_source,
    compile_recording_policy,
)


class _MockSource:
    """Minimal EscalationSource returning a fixed signal on each ``poll``."""

    def __init__(self, signal: Any) -> None:
        self._signal = signal
        self.poll_calls = 0

    def poll(self) -> Any:
        self.poll_calls += 1
        return self._signal


def test_mock_source_satisfies_escalation_source_protocol() -> None:
    # Structural typing: any object with a poll() -> X | None method is a source.
    assert isinstance(_MockSource(None), EscalationSource)


def test_apply_escalation_source_polls_exactly_once() -> None:
    source = _MockSource(None)
    apply_escalation_source(RiskContext(side_effect_class=SideEffectClass.READ), source)
    assert source.poll_calls == 1


def test_apply_escalation_source_without_signal_compiles_from_context() -> None:
    # A nominal source (no anomaly) must not perturb the ctx-based policy.
    ctx = RiskContext(side_effect_class=SideEffectClass.READ)
    policy = apply_escalation_source(ctx, _MockSource(None))

    assert policy.mode == RecordingMode.VALIDATION
    assert policy.reason == "read-only, no anomaly"


def test_apply_escalation_source_with_signal_forces_full_regardless_of_ctx() -> None:
    # Any active signal overrides even a low-risk read-only context.
    ctx = RiskContext(side_effect_class=SideEffectClass.READ)
    signal = {
        "source": "nccl-inspector",
        "severity": 0.95,
        "metric_name": "nccl_anomaly_score",
    }
    policy = apply_escalation_source(ctx, _MockSource(signal))

    assert policy.mode == RecordingMode.FULL
    # All three signal fields surface on the recorded reason for audit.
    assert "nccl-inspector" in policy.reason
    assert "nccl_anomaly_score" in policy.reason
    assert "high" in policy.reason


def test_signal_severity_band_influences_recorded_reason() -> None:
    high = compile_recording_policy(
        RiskContext(),
        escalation={
            "source": "nccl-inspector",
            "severity": ESCALATION_SEVERITY_HIGH,
            "metric_name": "x",
        },
    )
    low = compile_recording_policy(
        RiskContext(),
        escalation={"source": "nccl-inspector", "severity": 0.1, "metric_name": "x"},
    )
    assert high.mode == RecordingMode.FULL
    assert low.mode == RecordingMode.FULL
    assert "high" in high.reason
    assert "low" in low.reason


def test_signal_without_severity_keeps_canonical_reason() -> None:
    # Backward-compatible: a bare signal (no severity) reads as the milestone
    # string, preserving the contract from the parent compile_recording_policy bullet.
    policy = compile_recording_policy(RiskContext(), escalation={"source": "nccl-inspector"})
    assert policy.mode == RecordingMode.FULL
    assert policy.reason == "external escalation signal"


def test_apply_escalation_source_accepts_attribute_backed_signal() -> None:
    # The contract must also hold for the attribute-based shape that the real
    # EscalationSignal dataclass (frozen, from escalation.py) will expose.
    class _Signal:
        source = "nccl-inspector"
        severity = 0.8
        metric_name = "nccl_anomaly_score"

    policy = apply_escalation_source(
        RiskContext(side_effect_class=SideEffectClass.READ), _MockSource(_Signal())
    )
    assert policy.mode == RecordingMode.FULL
    assert "nccl-inspector" in policy.reason
    assert "nccl_anomaly_score" in policy.reason
