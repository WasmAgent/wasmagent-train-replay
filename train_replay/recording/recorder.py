"""EpochRecorder — collects evidence for one training epoch."""

from __future__ import annotations

from ..collector.flight_recorder import CollectiveEvent
from .evidence import EpochEvidenceBundle, TrainActionEvidence
from .modes import RecordingMode, RiskContext, SideEffectClass, compile_recording_policy


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
        self._bundle.actions.append(TrainActionEvidence(
            action_id=f"r{evt.rank}:seq{evt.sequence_id}",
            rank=evt.rank,
            step=evt.sequence_id,
            collective_type=evt.collective_type,
            recording_mode=policy.mode,
            timestamp_ns=evt.start_time_ns,
        ))

    def escalate_rank(self, rank: int) -> None:
        """Escalate recording mode to FULL for all existing actions on a rank."""
        for action in self._bundle.actions:
            if action.rank == rank:
                action.recording_mode = RecordingMode.FULL

    def bundle(self) -> EpochEvidenceBundle:
        return self._bundle
