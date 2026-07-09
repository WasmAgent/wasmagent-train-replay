"""Deterministic replayer — reconstruct training state from evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass

from ..graph.prov_graph import ProvGraph
from ..recording.evidence import EpochEvidenceBundle, TrainActionEvidence
from ..recording.modes import RecordingMode


@dataclass
class ReplayResult:
    epoch: int
    rank: int
    causal_ancestors: list[str]
    suspicious_actions: list[TrainActionEvidence]


class EpochReplayer:
    """Replay evidence bundles to identify causal chains for anomalous tensors."""

    def __init__(self, graph: ProvGraph) -> None:
        self._graph = graph

    def find_root_cause(self, entity_id: str) -> list[str]:
        """Return activity IDs that causally contributed to entity_id."""
        return self._graph.ancestors_of(entity_id)

    def suspicious_actions(self, bundle: EpochEvidenceBundle) -> list[TrainActionEvidence]:
        """Return actions that were recorded in FULL mode — highest risk signals."""
        return [a for a in bundle.actions if a.recording_mode == RecordingMode.FULL]

    def replay_rank(self, bundle: EpochEvidenceBundle, rank: int, entity_id: str) -> ReplayResult:
        ancestors = self.find_root_cause(entity_id)
        suspicious = [
            a for a in self.suspicious_actions(bundle) if a.rank == rank
        ]
        return ReplayResult(
            epoch=bundle.epoch,
            rank=rank,
            causal_ancestors=ancestors,
            suspicious_actions=suspicious,
        )
