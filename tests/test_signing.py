"""Tests for the ``load_private_key_hex`` helper (issue #74).

Scope note: the full Ed25519 signing suite is issue #11; these tests cover
only the new ``load_private_key_hex`` factory so the CLI can accept raw hex.
"""

from __future__ import annotations

import binascii

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from train_replay.signing.signer import load_private_key_hex


def _key_hex(key: Ed25519PrivateKey) -> str:
    return binascii.hexlify(key.private_bytes_raw()).decode()


def test_load_private_key_hex_roundtrips_raw_bytes() -> None:
    key = Ed25519PrivateKey.generate()
    loaded = load_private_key_hex(_key_hex(key))
    assert loaded.private_bytes_raw() == key.private_bytes_raw()


def test_load_private_key_hex_yields_usable_signing_key() -> None:
    key = Ed25519PrivateKey.generate()
    loaded = load_private_key_hex(_key_hex(key))
    message = b"auditor-evidence"
    # verify() raises on a bad signature and returns None on success.
    assert loaded.public_key().verify(key.sign(message), message) is None


def test_load_private_key_hex_rejects_non_hex() -> None:
    with pytest.raises(ValueError):
        load_private_key_hex("not-valid-hex-zz")


def test_load_private_key_hex_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        load_private_key_hex("00")
