"""Deterministic replayer — reconstruct training state from evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..anomaly.detector import AnomalySignal, StatisticalAnomalyDetector
from ..collector.flight_recorder import CollectiveEvent
from ..graph.prov_graph import ProvGraph
from ..recording.evidence import AEPRecord, EpochEvidenceBundle
from ..recording.modes import RecordingMode
from .types import DivergenceReport

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
    """Replay evidence bundles to identify causal chains for anomalous tensors.

    The causal graph is only required for :meth:`find_root_cause`; the
    divergence-comparison entry points (:meth:`replay_diff`,
    :meth:`diff_events`) accept ``graph=None`` so callers comparing two dumps
    don't pay for graph construction.
    """

    def __init__(
        self,
        graph: ProvGraph | None = None,
        detector: CollisionDetector | None = None,
    ) -> None:
        self._graph = graph
        self._detector = detector

    def find_root_cause(self, entity_id: str) -> list[str]:
        """Return activity IDs that causally contributed to entity_id."""
        if self._graph is None:
            raise RuntimeError(
                "EpochReplayer was constructed without a ProvGraph — "
                "find_root_cause() requires one."
            )
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

    def replay_diff(
        self,
        baseline_dump: str,
        candidate_dump: str,
        context_window_size: int = 5,
    ) -> dict[int, DivergenceReport]:
        """Compare two Flight Recorder dump files rank-by-rank.

        Convenience wrapper that loads both dumps and delegates to
        :meth:`diff_events`.  Callers that already hold the parsed event lists
        should call :meth:`diff_events` directly to avoid re-reading files.

        Returns:
            A report per divergent rank, keyed by rank.  Byte-identical dumps
            yield an empty dict.
        """
        from ..collector.flight_recorder import load_flight_recorder

        return self.diff_events(
            load_flight_recorder(Path(baseline_dump)),
            load_flight_recorder(Path(candidate_dump)),
            context_window_size=context_window_size,
        )

    def diff_events(
        self,
        baseline_events: list[CollectiveEvent],
        candidate_events: list[CollectiveEvent],
        context_window_size: int = 5,
    ) -> dict[int, DivergenceReport]:
        """Compare two rank-aligned event lists and report divergences.

        Maps the rank-aligned action streams pairwise and delegates to
        :class:`DivergenceReplayer`.  When a :class:`CollisionDetector` was
        configured at construction time it runs over each side's full
        multi-rank timeline and the resulting desyncs are attached to the
        owning rank's comparison, so a divergence that lands on a desync step
        is reported with ``correlated_collision=True``.

        Returns:
            A report per divergent rank, keyed by rank.  Identical streams
            yield an empty dict.
        """
        from .diff import DiffConfig, DivergenceReplayer

        baseline_by_rank = _records_by_rank(baseline_events)
        candidate_by_rank = _records_by_rank(candidate_events)

        baseline_collisions: dict[int, CollisionReport] = {}
        candidate_collisions: dict[int, CollisionReport] = {}
        if self._detector is not None:
            baseline_collisions = _collisions_by_rank(
                self._detector.detect(_group_by_rank(baseline_events))
            )
            candidate_collisions = _collisions_by_rank(
                self._detector.detect(_group_by_rank(candidate_events))
            )

        differ = DivergenceReplayer(DiffConfig(context_window_size=context_window_size))
        reports: dict[int, DivergenceReport] = {}
        for rank in sorted(set(baseline_by_rank) | set(candidate_by_rank)):
            baseline_result = ReplayResult(
                epoch=0,
                rank=rank,
                causal_ancestors=[],
                suspicious_actions=baseline_by_rank.get(rank, []),
                collision_report=baseline_collisions.get(rank),
            )
            candidate_result = ReplayResult(
                epoch=0,
                rank=rank,
                causal_ancestors=[],
                suspicious_actions=candidate_by_rank.get(rank, []),
                collision_report=candidate_collisions.get(rank),
            )
            report = differ.diff(baseline_result, candidate_result)
            if report.divergences:
                reports[rank] = report
        return reports

    def anomaly_scan(
        self,
        bundle: EpochEvidenceBundle,
        z_threshold: float = 3.0,
    ) -> list[AnomalySignal]:
        """Run :class:`StatisticalAnomalyDetector` over the bundle's event
        timeline and return ranked anomalies sorted by severity descending.

        Each :class:`AEPRecord` in *bundle.actions* is passed to the
        detector, which computes Z-scores on inter-event timing intervals
        (per rank) and on ``delta_stats`` numeric values.  Results are
        ranked so the highest-confidence anomalies appear first.

        Parameters:
            bundle: The evidence bundle to scan.
            z_threshold: Absolute Z-score threshold forwarded to the
                :class:`StatisticalAnomalyDetector`.

        Returns:
            Anomaly signals sorted by severity (highest first).
        """
        detector = StatisticalAnomalyDetector(z_threshold=z_threshold)
        signals = detector.detect(bundle.actions)
        return sorted(signals, key=lambda s: s.severity, reverse=True)

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


# ---------------------------------------------------------------------------
# Module-level helpers for replay_diff
# ---------------------------------------------------------------------------

def _group_by_rank(
    events: list[CollectiveEvent],
) -> dict[int, list[CollectiveEvent]]:
    """Group Flight Recorder events into per-rank timelines."""
    grouped: dict[int, list[CollectiveEvent]] = {}
    for evt in events:
        grouped.setdefault(evt.rank, []).append(evt)
    return grouped


def _records_by_rank(
    events: list[CollectiveEvent],
) -> dict[int, list[AEPRecord]]:
    """Map Flight Recorder events to per-rank ``AEPRecord`` action streams."""
    grouped: dict[int, list[AEPRecord]] = {}
    for evt in events:
        grouped.setdefault(evt.rank, []).append(_event_to_record(evt))
    return grouped


def _event_to_record(evt: CollectiveEvent) -> AEPRecord:
    """Map one Flight Recorder event to its replay-side ``AEPRecord``."""
    return AEPRecord(
        action_id=f"evt-r{evt.rank}-s{evt.sequence_id}",
        rank=evt.rank,
        step=evt.sequence_id,
        collective_type=evt.collective_type,
        recording_mode=RecordingMode.FULL,
        timestamp_ns=evt.start_time_ns,
    )


def _collisions_by_rank(
    report: CollisionReport,
) -> dict[int, CollisionReport]:
    """Split a collision report into per-rank reports.

    Each collision is attributed to both ranks of the desync pair, so every
    rank's diff sees the desyncs it participated in.
    """
    from ..graph.collision import CollisionReport as _CollisionReport

    by_rank: dict[int, CollisionReport] = {}
    for collision in report.collisions:
        for rank in {collision.rank_a, collision.rank_b}:
            rank_report = by_rank.setdefault(
                rank, _CollisionReport(total_steps_checked=report.total_steps_checked)
            )
            rank_report.collisions.append(collision)
    return by_rank
