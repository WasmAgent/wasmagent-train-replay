"""Divergence analysis types for cross-run regression detection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import cbor2


class DivergenceKind(str, Enum):
    """Category of a single divergence."""

    OP_MISMATCH = "op_mismatch"
    TENSOR_MISMATCH = "tensor_mismatch"
    STEP_MISMATCH = "step_mismatch"
    RANK_MISSING = "rank_missing"
    OTHER = "other"


@dataclass
class Divergence:
    """A single detected divergence between two training runs."""

    step: int
    rank: int
    kind: DivergenceKind
    field_path: str
    expected: str
    actual: str


@dataclass
class DivergenceReport:
    """Aggregated cross-run divergence analysis, mirroring ``EpochEvidenceBundle``
    with ``to_json()`` / ``to_cbor()`` serialization.
    """

    first_divergence_step: int
    divergences: list[Divergence] = field(default_factory=list)
    per_rank_similarity: dict[int, float] = field(default_factory=dict)
    summary: str = ""

    # -- serialization (mirrors EpochEvidenceBundle) --------------------------------

    def to_json(self) -> str:
        """Serialize to deterministic JSON."""
        d = self._as_serializable_dict()
        return json.dumps(d, sort_keys=True, default=str)

    def to_cbor(self) -> bytes:
        """Serialize to CBOR."""
        d = self._as_serializable_dict()
        return cbor2.dumps(d, default=str)  # type: ignore[no-any-return]

    @classmethod
    def from_json(cls, raw: str) -> DivergenceReport:
        """Deserialize from JSON produced by ``to_json()``."""
        d = json.loads(raw)
        return cls._from_dict(d)

    @classmethod
    def from_cbor(cls, raw: bytes) -> DivergenceReport:
        """Deserialize from CBOR produced by ``to_cbor()``."""
        d = cbor2.loads(raw)
        return cls._from_dict(d)

    # -- internal -------------------------------------------------------------------

    def _as_serializable_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for JSON/CBOR serialization."""
        d: dict[str, Any] = asdict(self)
        # Convert DivergenceKind enums to their string values.
        for div in d.get("divergences", []):
            div["kind"] = div["kind"].value
        return d

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> DivergenceReport:
        """Reconstruct a ``DivergenceReport`` from a plain dict.

        Handles enum restoration for ``Divergence.kind``.
        """
        divergences = d.get("divergences", [])
        restored_divs: list[Divergence] = []
        for item in divergences:
            item = dict(item)  # ensure mutable copy
            if "kind" in item and not isinstance(item["kind"], DivergenceKind):
                item["kind"] = DivergenceKind(item["kind"])
            restored_divs.append(Divergence(**item))

        per_rank = d.get("per_rank_similarity", {})
        # Ensure int keys (CBOR may return them as ints, JSON as strings).
        per_rank_int: dict[int, float] = {
            int(k): float(v) for k, v in per_rank.items()
        }

        return cls(
            first_divergence_step=int(d["first_divergence_step"]),
            divergences=restored_divs,
            per_rank_similarity=per_rank_int,
            summary=str(d.get("summary", "")),
        )
