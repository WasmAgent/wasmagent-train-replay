"""Ed25519 signing for EpochEvidenceBundle — DSSE-style envelope."""

from __future__ import annotations

import base64

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ..recording.evidence import EpochEvidenceBundle


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
