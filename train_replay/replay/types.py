"""Replay-stage data types.

Hosts :class:`Divergence` (per-rank divergence detail) and
:class:`DivergenceReport` (cross-run aggregate with JSON/CBOR serialization).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..recording.evidence import AEPRecord


@dataclass
class Divergence:
    """Per-rank divergence between a baseline and a candidate replay.

    Produced by pairwise comparison of two rank-aligned action streams
    (the ``DivergenceReplayer`` / ``EpochReplayer.replay_diff`` pipeline).
    Captures the first step at which the streams disagree, the disagreeing
    actions on either side, an N-step context window around the divergence,
    and correlation flags tying the divergence to detected desync collisions
    and external escalation signals.
    """

    rank: int
    first_divergence_step: int
    baseline_action: AEPRecord
    candidate_action: AEPRecord
    context_window: list[AEPRecord] = field(default_factory=list)
    correlated_collision: bool = False
    correlated_escalation: str | None = None
    # Populated when ``correlated_collision`` is True: the synthetic desync
    # ``AEPRecord`` for the collision that overlaps the divergence step.
    collision_record: AEPRecord | None = None


@dataclass
class DivergenceReport:
    """Cross-run divergence report with serialization support.

    Aggregates per-rank :class:`Divergence` entries alongside similarity
    scores and a human-readable summary.  Mirrors :class:`EpochEvidenceBundle`'s
    ``to_json()`` / ``to_cbor()`` serialization pair.
    """

    first_divergence_step: int | None = None
    divergences: list[Divergence] = field(default_factory=list)
    per_rank_similarity: dict[int, float] = field(default_factory=dict)
    summary: str = ""

    # -- Serialization ---------------------------------------------------------

    def to_json(self) -> str:
        """Serialize the report to canonical JSON."""
        import dataclasses

        d = dataclasses.asdict(self)
        return json.dumps(d, sort_keys=True, default=str)

    @classmethod
    def from_json(cls: type[DivergenceReport], data: str) -> DivergenceReport:
        """Deserialize from canonical JSON produced by ``to_json``."""
        d: dict[str, Any] = json.loads(data)
        return cls._from_dict(d)

    def to_cbor(self) -> bytes:
        """Serialize the report to CBOR."""
        import dataclasses

        import cbor2

        d = dataclasses.asdict(self)
        return cbor2.dumps(d, default=str)  # type: ignore[no-any-return]

    @classmethod
    def from_cbor(cls: type[DivergenceReport], data: bytes) -> DivergenceReport:
        """Deserialize from CBOR produced by ``to_cbor``."""
        import cbor2

        d: dict[str, Any] = cbor2.loads(data)
        return cls._from_dict(d)

    # -- Internal deserialization helpers -------------------------------------

    @classmethod
    def _from_dict(cls: type[DivergenceReport], d: dict[str, Any]) -> DivergenceReport:
        """Reconstruct a report from a plain dict, restoring nested types."""
        from ..recording.modes import RecordingMode

        divergences_raw: list[dict[str, Any]] = d.get("divergences", [])
        divergences = [_divergence_from_dict(dr, RecordingMode) for dr in divergences_raw]
        first_step = d.get("first_divergence_step")
        return cls(
            first_divergence_step=int(first_step) if first_step is not None else None,
            divergences=divergences,
            per_rank_similarity={
                int(k): v for k, v in d.get("per_rank_similarity", {}).items()
            },
            summary=d.get("summary", ""),
        )


def _aep_record_from_dict(
    raw: dict[str, Any],
    mode_cls: type[Any],
) -> AEPRecord:
    """Reconstruct an :class:`AEPRecord` from a plain dict."""
    return AEPRecord(
        action_id=raw["action_id"],
        rank=raw["rank"],
        step=raw["step"],
        collective_type=raw["collective_type"],
        recording_mode=mode_cls(raw["recording_mode"]),
        tensor_input_digest=raw.get("tensor_input_digest"),
        tensor_output_digest=raw.get("tensor_output_digest"),
        delta_stats=raw.get("delta_stats"),
        timestamp_ns=raw.get("timestamp_ns", 0),
        causal_chain_id=raw.get("causal_chain_id"),
        parent_action_id=raw.get("parent_action_id"),
    )


def _divergence_from_dict(
    d: dict[str, Any],
    mode_cls: type[Any],
) -> Divergence:
    """Reconstruct a :class:`Divergence` from a plain dict."""
    return Divergence(
        rank=d["rank"],
        first_divergence_step=d["first_divergence_step"],
        baseline_action=_aep_record_from_dict(d["baseline_action"], mode_cls),
        candidate_action=_aep_record_from_dict(d["candidate_action"], mode_cls),
        context_window=[
            _aep_record_from_dict(w, mode_cls)
            for w in d.get("context_window", [])
        ],
        correlated_collision=d.get("correlated_collision", False),
        correlated_escalation=d.get("correlated_escalation"),
        collision_record=(
            _aep_record_from_dict(d["collision_record"], mode_cls)
            if d.get("collision_record") is not None
            else None
        ),
    )
