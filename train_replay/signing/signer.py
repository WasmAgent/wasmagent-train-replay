"""Ed25519 signing for EpochEvidenceBundle — DSSE-style envelope."""

from __future__ import annotations

import base64
import dataclasses
import json
from collections.abc import Mapping
from typing import Protocol, TypeAlias, runtime_checkable

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ..recording.evidence import EpochEvidenceBundle

# DSSE payload type for an auditor-facing divergence-report comparison output.
_DIV_REPORT_PAYLOAD_TYPE = (
    "application/vnd.wasmagent.train-replay.divergence-report.v0.1"
)


def load_private_key_hex(hex_str: str) -> Ed25519PrivateKey:
    """Load an Ed25519PrivateKey from a 64-char hex string (32 raw bytes).

    Lets the CLI accept a raw hex key without callers constructing
    cryptography objects directly.
    """
    try:
        raw = bytes.fromhex(hex_str)
    except ValueError as exc:
        raise ValueError(f"private key is not valid hex: {exc}") from exc
    if len(raw) != 32:
        raise ValueError(
            "Ed25519 private key must be 32 bytes (64 hex chars), "
            f"got {len(raw)} byte(s)"
        )
    return Ed25519PrivateKey.from_private_bytes(raw)


@runtime_checkable
class _Canonicalizable(Protocol):
    """Anything exposing ``canonical_bytes()`` (e.g. a future ``DivergenceReport``)."""

    def canonical_bytes(self) -> bytes: ...


# A divergence report may arrive as a typed object (duck-typed on
# ``canonical_bytes``), raw canonical bytes, or a JSON-serialisable mapping.
# The not-yet-shipped ``DivergenceReport`` dataclass (milestone 6) is expected
# to mirror ``EpochEvidenceBundle`` and therefore expose ``canonical_bytes()``.
_ReportLike: TypeAlias = "_Canonicalizable | bytes | bytearray | Mapping[str, object]"


def _report_canonical_bytes(report: _ReportLike) -> bytes:
    """Return canonical bytes for a divergence-report-like object."""
    if isinstance(report, (bytes, bytearray)):
        return bytes(report)
    if isinstance(report, Mapping):
        return json.dumps(report, sort_keys=True, default=str).encode()
    if isinstance(report, _Canonicalizable):
        return report.canonical_bytes()
    raise TypeError(
        "report must expose canonical_bytes(), be bytes, or be a Mapping; "
        f"got {type(report).__name__}"
    )


def _dsse_pae(
    payload_type: str,
    payload: bytes,
    baseline_digest: str,
    candidate_digest: str,
) -> bytes:
    """DSSE-style pre-auth encoding binding a divergence report to its sources.

    Standard DSSE length-prefixed framing ("DSSEv1" followed by
    ``<len> <bytes>`` per field) extended with the two source-bundle digests,
    so the signature ties the report to the exact pair of bundles it was
    computed from.
    """
    fields = (
        payload_type.encode("ascii"),
        payload,
        baseline_digest.encode("ascii"),
        candidate_digest.encode("ascii"),
    )
    out = b"DSSEv1"
    for field in fields:
        out += b" " + str(len(field)).encode("ascii") + b" " + field
    return out


@dataclasses.dataclass
class DivergenceReportEnvelope:
    """DSSE-style envelope over a divergence report and its two source bundles.

    Carries the report payload plus the SHA-256 digests of the baseline and
    candidate ``EpochEvidenceBundle`` it was derived from, and an Ed25519
    signature over the pre-auth encoding (``pae_bytes``) of all four. An
    auditor can therefore attest that a comparison output was produced from
    two specific signed bundles.
    """

    payload_type: str
    payload: str
    source_digests: dict[str, str]
    signature: dict[str, str] | None = None

    def pae_bytes(self) -> bytes:
        return _dsse_pae(
            self.payload_type,
            base64.b64decode(self.payload),
            self.source_digests["baseline"],
            self.source_digests["candidate"],
        )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, data: str) -> DivergenceReportEnvelope:
        d = json.loads(data)
        return cls(
            payload_type=d["payload_type"],
            payload=d["payload"],
            source_digests=dict(d["source_digests"]),
            signature=d.get("signature"),
        )


class BundleSigner:
    def __init__(self, private_key: Ed25519PrivateKey, key_id: str) -> None:
        self._key = private_key
        self.key_id = key_id

    def sign(self, bundle: EpochEvidenceBundle) -> EpochEvidenceBundle:
        payload = bundle.canonical_bytes()
        sig_bytes = self._key.sign(payload)
        bundle.signature = {
            "alg": "ed25519",
            "key_id": self.key_id,
            "sig": base64.b64encode(sig_bytes).decode(),
        }
        return bundle

    @classmethod
    def generate(cls, key_id: str = "dev-key") -> tuple[BundleSigner, Ed25519PublicKey]:
        key = Ed25519PrivateKey.generate()
        return cls(key, key_id), key.public_key()

    def sign_divergence_report(
        self,
        report: _ReportLike,
        baseline: EpochEvidenceBundle,
        candidate: EpochEvidenceBundle,
    ) -> DivergenceReportEnvelope:
        """Sign a divergence report into a DSSE envelope over both source bundles.

        The envelope records the SHA-256 digest of each source bundle and an
        Ed25519 signature over the pre-auth encoding of the payload type, the
        report payload, and both digests — so the auditor-facing comparison
        output is cryptographically bound to the exact pair of bundles it was
        derived from. See :func:`verify_divergence_report`.
        """
        payload = _report_canonical_bytes(report)
        envelope = DivergenceReportEnvelope(
            payload_type=_DIV_REPORT_PAYLOAD_TYPE,
            payload=base64.b64encode(payload).decode("ascii"),
            source_digests={
                "baseline": baseline.digest(),
                "candidate": candidate.digest(),
            },
        )
        sig_bytes = self._key.sign(envelope.pae_bytes())
        envelope.signature = {
            "alg": "ed25519",
            "key_id": self.key_id,
            "sig": base64.b64encode(sig_bytes).decode("ascii"),
        }
        return envelope


def verify_bundle(bundle: EpochEvidenceBundle, public_key: Ed25519PublicKey) -> bool:
    if not bundle.signature:
        return False
    try:
        sig = base64.b64decode(bundle.signature["sig"])
        public_key.verify(sig, bundle.canonical_bytes())
        return True
    except Exception:
        return False


def verify_divergence_report(
    envelope: DivergenceReportEnvelope,
    baseline: EpochEvidenceBundle,
    candidate: EpochEvidenceBundle,
    public_key: Ed25519PublicKey,
) -> bool:
    """Verify a divergence-report envelope against its two source bundles.

    Returns ``True`` only when the envelope's recorded digests match the
    supplied bundles AND the signature validates — i.e. the comparison output
    originated from exactly these two bundles. Swapping the bundles (or
    substituting a different one) fails the digest check.
    """
    if not envelope.signature:
        return False
    if envelope.source_digests.get("baseline") != baseline.digest():
        return False
    if envelope.source_digests.get("candidate") != candidate.digest():
        return False
    try:
        sig = base64.b64decode(envelope.signature["sig"])
        public_key.verify(sig, envelope.pae_bytes())
    except Exception:
        return False
    return True
