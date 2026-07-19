"""AEP recording modes — mirrors @wasmagent/capability-compiler logic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from train_replay.recording.escalation import EscalationSignal


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


def compile_recording_policy(
    ctx: RiskContext, escalation: EscalationSignal | None = None
) -> RecordingPolicy:
    """Port of capability-compiler's compileToRecordingPolicy. Priority order matches TS."""
    if escalation is not None:
        return RecordingPolicy(RecordingMode.FULL, "external escalation signal")
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
