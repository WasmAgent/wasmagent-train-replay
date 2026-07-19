"""Ed25519 signing for EpochEvidenceBundle — DSSE-style envelope."""

from __future__ import annotations

import base64

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ..recording.evidence import EpochEvidenceBundle


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


def verify_bundle(bundle: EpochEvidenceBundle, public_key: Ed25519PublicKey) -> bool:
    if not bundle.signature:
        return False
    try:
        sig = base64.b64decode(bundle.signature["sig"])
        public_key.verify(sig, bundle.canonical_bytes())
        return True
    except Exception:
        return False
