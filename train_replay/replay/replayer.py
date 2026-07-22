"""Deterministic replayer — reconstruct training state from evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..collector.flight_recorder import CollectiveEvent
from ..graph.prov_graph import ProvGraph
from ..recording.evidence import AEPRecord, EpochEvidenceBundle
from ..recording.modes import RecordingMode

if TYPE_CHECKING:
    from ..graph.collision import CollisionDetector, CollisionReport


@dataclass
class ReplayResult:
    epoch: int
    rank: int
    causal_ancestors: list[str]
    suspicious_actions: list[AEPRecord]
    collision_report: CollisionReport | None = None


class EpochReplayer:
    """Replay evidence bundles to identify causal chains for anomalous tensors."""

    def __init__(
        self,
        graph: ProvGraph,
        detector: CollisionDetector | None = None,
    ) -> None:
        self._graph = graph
        self._detector = detector

    def find_root_cause(self, entity_id: str) -> list[str]:
        """Return activity IDs that causally contributed to entity_id."""
        return self._graph.ancestors_of(entity_id)

    def suspicious_actions(self, bundle: EpochEvidenceBundle) -> list[AEPRecord]:
        """Return actions that were recorded in FULL mode — highest risk signals.

        If a :class:`CollisionDetector` was provided at construction time,
        desyncs detected by the backend-specific detector are also treated
        as suspicious by augmenting the returned list with synthetic records.
        """
        full_mode_actions = [
            a for a in bundle.actions if a.recording_mode == RecordingMode.FULL
        ]

        if self._detector is None:
            return full_mode_actions

        # Build per-rank timelines from the bundle and run the detector.
        timelines: dict[int, list[CollectiveEvent]] = {}
        for action in bundle.actions:
            timelines.setdefault(action.rank, []).append(
                self._record_to_collective_event(action)
            )
        report = self._detector.detect(timelines)

        # Convert each detected collision into a synthetic AEPRecord so
        # callers get a uniform list[AEPRecord] regardless of backend.
        desync_records = [
            AEPRecord(
                action_id=f"desync-r{c.rank_a}-r{c.rank_b}-s{c.step}",
                rank=c.rank_a,
                step=c.step,
                collective_type="desync",
                recording_mode=RecordingMode.FULL,
                delta_stats={"rank_b": float(c.rank_b)},
                causal_chain_id=c.detail,
            )
            for c in report.collisions
        ]
        return full_mode_actions + desync_records

    def check_collisions(
        self,
        timelines: dict[int, list[CollectiveEvent]],
    ) -> CollisionReport:
        """Run the configured backend detector over per-rank event timelines.

        Raises :exc:`RuntimeError` if no detector was configured.
        """
        if self._detector is None:
            raise RuntimeError(
                "No CollisionDetector configured — pass a detector to "
                "EpochReplayer.__init__() to enable collision detection."
            )
        # Avoid circular import at module level; collision imports CollectiveEvent
        # from the same module we TYPE_CHECK-guard above.
        from ..graph.collision import CollisionReport as _  # noqa: F401 — ensure importable
        return self._detector.detect(timelines)

    def replay_rank(
        self,
        bundle: EpochEvidenceBundle,
        rank: int,
        entity_id: str,
    ) -> ReplayResult:
        ancestors = self.find_root_cause(entity_id)
        suspicious = [
            a for a in self.suspicious_actions(bundle) if a.rank == rank
        ]
        events = [
            self._record_to_collective_event(a)
            for a in bundle.actions
            if a.rank == rank
        ]
        collision_report = (
            self.check_collisions({rank: events})
            if self._detector is not None
            else None
        )
        return ReplayResult(
            epoch=bundle.epoch,
            rank=rank,
            causal_ancestors=ancestors,
            suspicious_actions=suspicious,
            collision_report=collision_report,
        )

    @staticmethod
    def _record_to_collective_event(record: AEPRecord) -> CollectiveEvent:
        return CollectiveEvent(
            rank=record.rank,
            process_group="default",
            collective_type=record.collective_type,
            src_rank=None,
            dst_rank=None,
            tensor_size=0,
            enqueue_time_ns=record.timestamp_ns,
            start_time_ns=record.timestamp_ns,
            end_time_ns=record.timestamp_ns,
            sequence_id=record.step,
        )
