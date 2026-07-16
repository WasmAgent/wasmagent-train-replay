"""AEP evidence types adapted for distributed training actions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .modes import RecordingMode


@dataclass
class AEPRecord:
    """One Agent Evidence Protocol (AEP) record per recorded collective."""
    action_id: str
    rank: int
    step: int
    collective_type: str
    recording_mode: RecordingMode
    tensor_input_digest: str | None = None
    tensor_output_digest: str | None = None
    delta_stats: dict[str, float] | None = None
    timestamp_ns: int = 0
    causal_chain_id: str | None = None
    parent_action_id: str | None = None


_SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"train-aep/v0.1"})


@dataclass
class EpochEvidenceBundle:
    schema_version: str = "train-aep/v0.1"
    run_id: str = ""
    epoch: int = 0
    actions: list[AEPRecord] = field(default_factory=list)
    signature: dict[str, str] | None = None

    def canonical_bytes(self) -> bytes:
        import dataclasses
        d = dataclasses.asdict(self)
        d.pop("signature", None)
        return json.dumps(d, sort_keys=True, default=str).encode()

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    # -- Serialization ---------------------------------------------------------

    def to_json(self) -> str:
        """Serialize the full bundle (including signature) to canonical JSON."""
        import dataclasses
        d = dataclasses.asdict(self)
        return json.dumps(d, sort_keys=True, default=str)

    @classmethod
    def from_json(cls: type[EpochEvidenceBundle], data: str) -> EpochEvidenceBundle:
        """Deserialize from canonical JSON produced by ``to_json``."""
        return cls._from_dict(json.loads(data))

    def to_cbor(self) -> bytes:
        """Serialize the full bundle (including signature) to CBOR."""
        import dataclasses

        import cbor2
        d = dataclasses.asdict(self)
        return cbor2.dumps(d, default=str)  # type: ignore[no-any-return]

    @classmethod
    def from_cbor(cls: type[EpochEvidenceBundle], data: bytes) -> EpochEvidenceBundle:
        """Deserialize from CBOR produced by ``to_cbor``."""
        import cbor2
        return cls._from_dict(cbor2.loads(data))

    # -- Internal deserialization helpers -------------------------------------

    @classmethod
    def _from_dict(cls: type[EpochEvidenceBundle], d: dict[str, Any]) -> EpochEvidenceBundle:
        """Reconstruct a bundle from a plain dict, restoring enum types."""
        version = d.get("schema_version", "train-aep/v0.1")
        if version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema_version: {version}")
        actions_raw: list[dict[str, Any]] = d.get("actions", [])
        actions = [
            AEPRecord(
                action_id=a["action_id"],
                rank=a["rank"],
                step=a["step"],
                collective_type=a["collective_type"],
                recording_mode=RecordingMode(a["recording_mode"]),
                tensor_input_digest=a.get("tensor_input_digest"),
                tensor_output_digest=a.get("tensor_output_digest"),
                delta_stats=a.get("delta_stats"),
                timestamp_ns=a.get("timestamp_ns", 0),
                causal_chain_id=a.get("causal_chain_id"),
                parent_action_id=a.get("parent_action_id"),
            )
            for a in actions_raw
        ]
        return cls(
            schema_version=version,
            run_id=d.get("run_id", ""),
            epoch=d.get("epoch", 0),
            actions=actions,
            signature=d.get("signature"),
        )


TrainActionEvidence = AEPRecord
