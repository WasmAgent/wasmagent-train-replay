"""AEP evidence types adapted for distributed training actions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

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


@dataclass
class EpochEvidenceBundle:
    schema_version: str = "train-aep/v0.1"
    run_id: str = ""
    epoch: int = 0
    actions: list[AEPRecord] = field(default_factory=list)
    signature: dict[str, str] | None = None

    def canonical_bytes(self) -> bytes:
        import dataclasses
        import json
        d = dataclasses.asdict(self)
        d.pop("signature", None)
        return json.dumps(d, sort_keys=True, default=str).encode()

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


TrainActionEvidence = AEPRecord
