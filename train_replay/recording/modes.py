"""AEP recording modes — mirrors @wasmagent/capability-compiler logic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, TypeAlias, runtime_checkable

__all__ = [
    "ESCALATION_SEVERITY_HIGH",
    "EscalationSignal",
    "EscalationSource",
    "RecordingMode",
    "RecordingPolicy",
    "RiskContext",
    "SideEffectClass",
    "apply_escalation_source",
    "compile_recording_policy",
]

# ``EscalationSignal`` is a permissive alias on purpose. The concrete frozen
# dataclass lives in ``train_replay.recording.escalation`` (shipped by the
# sibling parent issue) and is *not* imported here so this module stays a
# standalone integration contract with no hard dependency on that package.
# Any object exposing the three structural fields below satisfies the
# contract; the helpers below read them defensively so both plain dicts
# (used in tests and via the CLI) and the real dataclass work unchanged.
#
#   source:      str   — detector that raised the signal (e.g. "nccl-inspector")
#   severity:    float — anomaly score; higher == stronger escalation
#   metric_name: str   — metric that crossed the detector threshold
EscalationSignal: TypeAlias = Any

# Severity band boundary for *annotating* escalation reasons. Any non-None
# signal still forces RecordingMode.FULL regardless of value; this constant
# only controls whether the recorded reason is labelled ``high`` vs ``low``
# severity so an auditor can see how strongly the detector fired.
ESCALATION_SEVERITY_HIGH: float = 0.9


class RecordingMode(str, Enum):
    VALIDATION = "validation"
    DELTA      = "delta"
    FULL       = "full"


class SideEffectClass(str, Enum):
    READ             = "read"
    MUTATE_LOCAL     = "mutate-local"
    MUTATE_EXTERNAL  = "mutate-external"
    NETWORK_EGRESS   = "network-egress"
    UNKNOWN          = "unknown"


@dataclass
class RiskContext:
    was_vetted: bool = False
    has_consent_anomaly: bool = False
    taint_chain_length: int = 0
    side_effect_class: SideEffectClass = SideEffectClass.UNKNOWN


@dataclass
class RecordingPolicy:
    mode: RecordingMode
    reason: str


@runtime_checkable
class EscalationSource(Protocol):
    """Integration interface for an external anomaly detector.

    The canonical implementation is ``PrometheusAnomalySource`` in
    ``train_replay.recording.escalation``; any class exposing a compatible
    ``poll`` method satisfies this Protocol structurally. A source is polled
    once per call and returns an active :data:`EscalationSignal` when a metric
    crosses its threshold, or ``None`` when everything is nominal. The recorder
    bridges a source into the policy via :func:`apply_escalation_source`.

    The returned signal influences the recording decision through its three
    fields:

    * presence  — any non-``None`` signal forces ``RecordingMode.FULL``
      (recording is triggered) ahead of every ctx-based rule;
    * ``severity``    — a numeric score ``>= ESCALATION_SEVERITY_HIGH`` is
      annotated on the policy ``reason`` as *high* severity, lower scores as
      *low* severity, so the audit trail records detector confidence;
    * ``source`` / ``metric_name`` — propagated onto the policy ``reason`` so
      an auditor can see which detector and metric fired.
    """

    def poll(self) -> EscalationSignal | None:
        """Return an active escalation signal, or ``None`` when no anomaly is present."""
        ...


def _signal_field(signal: EscalationSignal, name: str) -> Any:
    """Read a field from an escalation signal whether dict- or attribute-backed."""
    if isinstance(signal, dict):
        return signal.get(name)
    return getattr(signal, name, None)


def _coerce_severity(signal: EscalationSignal) -> float | None:
    """Return the signal's numeric ``severity`` or ``None`` if it is absent/invalid."""
    raw = _signal_field(signal, "severity")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _escalation_reason(signal: EscalationSignal) -> str:
    """Build the recording reason for an active escalation signal.

    The base reason is the canonical milestone string
    ``"external escalation signal"``. When the signal carries a numeric
    ``severity`` (the field that quantifies how strongly the detector fired),
    the source, metric name and severity band are annotated onto it so the
    audit trail records which detector fired and how hard. A signal without a
    severity keeps the base string unchanged — preserving the contract from
    the parent milestone bullet for callers that pass a bare signal.
    """
    severity = _coerce_severity(signal)
    if severity is None:
        return "external escalation signal"

    band = "high" if severity >= ESCALATION_SEVERITY_HIGH else "low"
    source = _signal_field(signal, "source")
    metric = _signal_field(signal, "metric_name")

    parts: list[str] = [f"severity={severity:g} ({band})"]
    if source:
        parts.append(f"source={source}")
    if metric:
        parts.append(f"metric={metric}")
    return "external escalation signal (" + ", ".join(parts) + ")"


def compile_recording_policy(
    ctx: RiskContext, escalation: EscalationSignal | None = None
) -> RecordingPolicy:
    """Port of capability-compiler's compileToRecordingPolicy. Priority order matches TS."""
    if escalation is not None:
        return RecordingPolicy(RecordingMode.FULL, _escalation_reason(escalation))
    if ctx.was_vetted:
        return RecordingPolicy(RecordingMode.FULL, "tool flagged by vetting")
    if ctx.has_consent_anomaly:
        return RecordingPolicy(RecordingMode.FULL, "consent anomaly recorded")
    if ctx.taint_chain_length > 0 and ctx.side_effect_class != SideEffectClass.READ:
        return RecordingPolicy(RecordingMode.FULL, "tainted input reaching state-changing call")
    if ctx.side_effect_class == SideEffectClass.UNKNOWN:
        return RecordingPolicy(RecordingMode.FULL, "unknown side-effect class")
    if ctx.side_effect_class in (SideEffectClass.MUTATE_EXTERNAL, SideEffectClass.NETWORK_EGRESS):
        return RecordingPolicy(RecordingMode.FULL, "external mutation")
    if ctx.side_effect_class == SideEffectClass.MUTATE_LOCAL:
        return RecordingPolicy(RecordingMode.DELTA, "local mutation, low risk")
    return RecordingPolicy(RecordingMode.VALIDATION, "read-only, no anomaly")


def apply_escalation_source(
    ctx: RiskContext, source: EscalationSource
) -> RecordingPolicy:
    """Poll an escalation source once and compile the resulting recording policy.

    This is the integration bridge between an external anomaly detector (any
    :class:`EscalationSource`, e.g. ``PrometheusAnomalySource``) and
    :func:`compile_recording_policy`: the source is polled once, and when it
    reports an active signal the policy escalates to ``RecordingMode.FULL`` for
    the given context; otherwise the policy is compiled from ``ctx`` alone.
    Wiring a source through this function is how the NCCL Inspector bridge
    connects to recording.
    """
    signal = source.poll()
    return compile_recording_policy(ctx, escalation=signal)
