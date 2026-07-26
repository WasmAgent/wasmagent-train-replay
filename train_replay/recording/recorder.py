"""EpochRecorder — collects evidence for one training epoch."""

from __future__ import annotations

from ..collector.flight_recorder import CollectiveEvent
from .evidence import AEPRecord, DeterminismAnchor, EpochEvidenceBundle
from .modes import (
    EscalationSignal,
    RecordingMode,
    RiskContext,
    SideEffectClass,
    compile_recording_policy,
)


def _collective_side_effect(ctype: str) -> SideEffectClass:
    reads = {"recv", "barrier"}
    return SideEffectClass.READ if ctype.lower() in reads else SideEffectClass.MUTATE_EXTERNAL


class EpochRecorder:
    """Records AEP evidence for one epoch across all ranks."""

    def __init__(
        self,
        run_id: str,
        epoch: int,
        default_mode: RecordingMode = RecordingMode.VALIDATION,
    ) -> None:
        self._bundle = EpochEvidenceBundle(run_id=run_id, epoch=epoch)
        self._default_mode = default_mode

    def record_collective(
        self, evt: CollectiveEvent, risk_override: RiskContext | None = None,
    ) -> None:
        ctx = risk_override or RiskContext(
            side_effect_class=_collective_side_effect(evt.collective_type)
        )
        policy = compile_recording_policy(ctx)
        self._bundle.actions.append(AEPRecord(
            action_id=f"r{evt.rank}:seq{evt.sequence_id}",
            rank=evt.rank,
            step=evt.sequence_id,
            collective_type=evt.collective_type,
            recording_mode=policy.mode,
            timestamp_ns=evt.start_time_ns,
        ))

    def record_with_escalation(
        self, event: CollectiveEvent, escalation: EscalationSignal
    ) -> None:
        ctx = RiskContext(
            side_effect_class=_collective_side_effect(event.collective_type)
        )
        policy = compile_recording_policy(ctx, escalation=escalation)
        self._bundle.actions.append(AEPRecord(
            action_id=f"r{event.rank}:seq{event.sequence_id}",
            rank=event.rank,
            step=event.sequence_id,
            collective_type=event.collective_type,
            recording_mode=policy.mode,
            timestamp_ns=event.start_time_ns,
        ))

    def record_determinism_anchor(
        self,
        step: int,
        rank: int,
        rng_seed_snapshot: int,
        collective_order_hash: str,
        reduce_scatter_checksum: str,
    ) -> None:
        """Capture a per-step determinism anchor into the evidence bundle.

        These lightweight fingerprints allow two replays to be compared
        without re-reading raw tensors.
        """
        self._bundle.determinism_anchors.append(DeterminismAnchor(
            step=step,
            rank=rank,
            rng_seed_snapshot=rng_seed_snapshot,
            collective_order_hash=collective_order_hash,
            reduce_scatter_checksum=reduce_scatter_checksum,
        ))

    def escalate_rank(self, rank: int) -> None:
        """Escalate recording mode to FULL for all existing actions on a rank."""
        for action in self._bundle.actions:
            if action.rank == rank:
                action.recording_mode = RecordingMode.FULL

    def bundle(self) -> EpochEvidenceBundle:
        return self._bundle
