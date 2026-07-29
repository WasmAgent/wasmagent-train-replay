"""Deterministic replayer — reconstruct training state from evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..anomaly.detector import AnomalySignal, StatisticalAnomalyDetector
from ..collector.flight_recorder import CollectiveEvent, load_flight_recorder
from ..graph.prov_graph import ProvGraph
from ..recording.evidence import AEPRecord, EpochEvidenceBundle
from ..recording.modes import RecordingMode
from ..recording.recorder import EpochRecorder
from .types import DivergenceReport

if TYPE_CHECKING:
    from ..graph.collision import CollisionDetector, CollisionReport

# Number of action records on either side of the first divergence captured
# in a :class:`DivergenceReport`'s ``context_window``.
_DIFF_CONTEXT_WINDOW = 2


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

    def replay_diff(
        self,
        baseline_dump: str,
        candidate_dump: str,
    ) -> dict[int, DivergenceReport]:
        """Compare two Flight Recorder dumps rank-by-rank and report divergences.

        Loads both dumps, records an :class:`EpochEvidenceBundle` for each,
        then walks the rank-aligned action streams pairwise. The first step at
        which a rank's baseline and candidate actions disagree is captured in
        a :class:`DivergenceReport` alongside an N-step context window on
        either side. Byte-identical dumps yield an empty ``dict`` (no
        divergence); a report is only emitted for ranks that disagree.

        .. note::
           Collision and escalation correlation (``correlated_collision`` /
           ``correlated_escalation``) is populated by separate Milestone 6
           bullets; this method leaves those fields at their defaults.
        """
        baseline_bundle = self._load_bundle(baseline_dump)
        candidate_bundle = self._load_bundle(candidate_dump)

        baseline_by_rank = self._group_actions_by_rank(baseline_bundle)
        candidate_by_rank = self._group_actions_by_rank(candidate_bundle)

        reports: dict[int, DivergenceReport] = {}
        for rank in sorted(set(baseline_by_rank) | set(candidate_by_rank)):
            report = self._first_divergence(
                rank,
                baseline_by_rank.get(rank, []),
                candidate_by_rank.get(rank, []),
            )
            if report is not None:
                reports[rank] = report
        return reports

    @staticmethod
    def _load_bundle(dump_path: str) -> EpochEvidenceBundle:
        """Load a Flight Recorder dump and record it into an evidence bundle."""
        events = load_flight_recorder(Path(dump_path))
        recorder = EpochRecorder(run_id="replay-diff", epoch=0)
        for evt in events:
            recorder.record_collective(evt)
        return recorder.bundle()

    @staticmethod
    def _group_actions_by_rank(
        bundle: EpochEvidenceBundle,
    ) -> dict[int, list[AEPRecord]]:
        grouped: dict[int, list[AEPRecord]] = {}
        for action in bundle.actions:
            grouped.setdefault(action.rank, []).append(action)
        for actions in grouped.values():
            actions.sort(key=lambda a: a.step)
        return grouped

    @staticmethod
    def _actions_agree(a: AEPRecord, b: AEPRecord) -> bool:
        """True if two records represent the same replayed action.

        Compares replay-meaningful fields (step, collective type, recording
        mode, tensor digests, delta stats). ``action_id`` and ``timestamp_ns``
        are intentionally excluded: they identify or locate a record but do
        not change what the replayed computation did.
        """
        return (
            a.step == b.step
            and a.collective_type == b.collective_type
            and a.recording_mode == b.recording_mode
            and a.tensor_input_digest == b.tensor_input_digest
            and a.tensor_output_digest == b.tensor_output_digest
            and a.delta_stats == b.delta_stats
        )

    @staticmethod
    def _missing_record(rank: int, step: int) -> AEPRecord:
        """Sentinel marking an action present in one stream but not the other."""
        return AEPRecord(
            action_id=f"r{rank}:seq{step}:missing",
            rank=rank,
            step=step,
            collective_type="<missing>",
            recording_mode=RecordingMode.VALIDATION,
        )

    @staticmethod
    def _first_divergence(
        rank: int,
        baseline: list[AEPRecord],
        candidate: list[AEPRecord],
    ) -> DivergenceReport | None:
        """Locate the first step where the two rank-aligned streams disagree."""
        common = min(len(baseline), len(candidate))
        for i in range(common):
            if not EpochReplayer._actions_agree(baseline[i], candidate[i]):
                return EpochReplayer._build_divergence_report(rank, i, baseline, candidate)
        if len(baseline) != len(candidate):
            # Common prefix agrees but one stream has extra trailing actions.
            return EpochReplayer._build_divergence_report(rank, common, baseline, candidate)
        return None

    @staticmethod
    def _build_divergence_report(
        rank: int,
        index: int,
        baseline: list[AEPRecord],
        candidate: list[AEPRecord],
    ) -> DivergenceReport:
        baseline_action = (
            baseline[index]
            if index < len(baseline)
            else EpochReplayer._missing_record(rank, candidate[index].step)
        )
        candidate_action = (
            candidate[index]
            if index < len(candidate)
            else EpochReplayer._missing_record(rank, baseline[index].step)
        )
        # Context window: N actions on either side of the divergence, drawn
        # from the baseline stream (the reference run).
        start = max(0, index - _DIFF_CONTEXT_WINDOW)
        end = index + 1 + _DIFF_CONTEXT_WINDOW
        window = [*baseline[start:index], *baseline[index + 1 : end]]
        return DivergenceReport(
            rank=rank,
            first_divergence_step=baseline_action.step,
            baseline_action=baseline_action,
            candidate_action=candidate_action,
            context_window=window,
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
