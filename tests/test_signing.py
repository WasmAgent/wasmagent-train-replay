"""Tests for the ``load_private_key_hex`` helper (issue #74).

Scope note: the full Ed25519 signing suite is issue #11; these tests cover
only the new ``load_private_key_hex`` factory so the CLI can accept raw hex,
plus the divergence-report DSSE envelope (issue #350).
"""

from __future__ import annotations

import binascii

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from train_replay.recording.evidence import EpochEvidenceBundle
from train_replay.signing.signer import (
    BundleSigner,
    DivergenceReportEnvelope,
    load_private_key_hex,
    verify_divergence_report,
)


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


# -- Divergence-report DSSE envelope (issue #350) ------------------------------


def _bundle(run_id: str, epoch: int) -> EpochEvidenceBundle:
    """Distinct empty bundles — differing run_id/epoch yield differing digests."""
    return EpochEvidenceBundle(run_id=run_id, epoch=epoch, actions=[])


def _report_mapping() -> dict[str, object]:
    return {"first_divergence_step": 7, "per_rank_similarity": {0: 0.98}, "summary": "x"}


def test_sign_divergence_report_roundtrips_and_verifies() -> None:
    signer, public_key = BundleSigner.generate()
    baseline = _bundle("run-baseline", 0)
    candidate = _bundle("run-candidate", 1)

    envelope = signer.sign_divergence_report(_report_mapping(), baseline, candidate)

    assert isinstance(envelope, DivergenceReportEnvelope)
    assert envelope.payload_type == (
        "application/vnd.wasmagent.train-replay.divergence-report.v0.1"
    )
    assert envelope.source_digests == {
        "baseline": baseline.digest(),
        "candidate": candidate.digest(),
    }
    assert envelope.signature is not None
    assert envelope.signature["alg"] == "ed25519"
    assert verify_divergence_report(envelope, baseline, candidate, public_key) is True


def test_verify_divergence_report_rejects_swapped_sources() -> None:
    """The signature validates only against the originating digests."""
    signer, public_key = BundleSigner.generate()
    baseline = _bundle("run-baseline", 0)
    candidate = _bundle("run-candidate", 1)
    envelope = signer.sign_divergence_report(_report_mapping(), baseline, candidate)

    # Swapping baseline/candidate must fail even though both digests are present.
    assert (
        verify_divergence_report(envelope, candidate, baseline, public_key) is False
    )


def test_verify_divergence_report_rejects_substituted_bundle() -> None:
    signer, public_key = BundleSigner.generate()
    baseline = _bundle("run-baseline", 0)
    candidate = _bundle("run-candidate", 1)
    envelope = signer.sign_divergence_report(_report_mapping(), baseline, candidate)

    other = _bundle("run-other", 2)
    assert (
        verify_divergence_report(envelope, baseline, other, public_key) is False
    )
    assert (
        verify_divergence_report(envelope, other, candidate, public_key) is False
    )


def test_verify_divergence_report_rejects_wrong_key() -> None:
    signer, _ = BundleSigner.generate()
    other_signer, other_public_key = BundleSigner.generate()
    baseline = _bundle("run-baseline", 0)
    candidate = _bundle("run-candidate", 1)
    envelope = signer.sign_divergence_report(_report_mapping(), baseline, candidate)

    assert (
        verify_divergence_report(envelope, baseline, candidate, other_public_key)
        is False
    )


def test_verify_divergence_report_rejects_unsigned_envelope() -> None:
    signer, public_key = BundleSigner.generate()
    baseline = _bundle("run-baseline", 0)
    candidate = _bundle("run-candidate", 1)
    envelope = signer.sign_divergence_report(_report_mapping(), baseline, candidate)
    envelope.signature = None

    assert (
        verify_divergence_report(envelope, baseline, candidate, public_key) is False
    )


def test_divergence_report_envelope_json_roundtrip() -> None:
    signer, public_key = BundleSigner.generate()
    baseline = _bundle("run-baseline", 0)
    candidate = _bundle("run-candidate", 1)
    envelope = signer.sign_divergence_report(_report_mapping(), baseline, candidate)

    restored = DivergenceReportEnvelope.from_json(envelope.to_json())
    assert restored == envelope
    assert (
        verify_divergence_report(restored, baseline, candidate, public_key) is True
    )


def test_sign_divergence_report_accepts_canonical_bytes_ducktype() -> None:
    """Forward-compat: a future DivergenceReport exposing canonical_bytes()."""

    class _FakeReport:
        def canonical_bytes(self) -> bytes:
            return b'{"first_divergence_step":7}'

    signer, public_key = BundleSigner.generate()
    baseline = _bundle("run-baseline", 0)
    candidate = _bundle("run-candidate", 1)
    envelope = signer.sign_divergence_report(_FakeReport(), baseline, candidate)

    assert verify_divergence_report(envelope, baseline, candidate, public_key) is True


# -- Issue #376: PAE covers source-bundle digests -----------------------------


def test_pae_bytes_contains_source_digests() -> None:
    """The pre-auth encoding must embed both source-bundle digests verbatim."""
    signer, public_key = BundleSigner.generate()
    baseline = _bundle("run-baseline", 0)
    candidate = _bundle("run-candidate", 1)
    envelope = signer.sign_divergence_report(_report_mapping(), baseline, candidate)

    pae = envelope.pae_bytes()
    assert pae.startswith(b"DSSEv1")
    # Both digests must appear in the PAE so the signature is bound to them.
    assert baseline.digest().encode() in pae
    assert candidate.digest().encode() in pae


def test_pae_digest_binding_tampered_digest_fails_verification() -> None:
    """Tampering with source_digests after signing must break verification."""
    signer, public_key = BundleSigner.generate()
    baseline = _bundle("run-baseline", 0)
    candidate = _bundle("run-candidate", 1)
    envelope = signer.sign_divergence_report(_report_mapping(), baseline, candidate)

    assert verify_divergence_report(envelope, baseline, candidate, public_key) is True

    # Corrupt the baseline digest in the envelope.
    envelope.source_digests["baseline"] = "00" * 32
    assert verify_divergence_report(envelope, baseline, candidate, public_key) is False


def test_pae_digest_binding_different_order_in_pae() -> None:
    """Swapping baseline/candidate in source_digests must invalidate the signature.

    Because the PAE encodes baseline then candidate in fixed order,
    swapping their positions produces different PAE bytes.
    """
    signer, public_key = BundleSigner.generate()
    baseline = _bundle("run-baseline", 0)
    candidate = _bundle("run-candidate", 1)
    envelope = signer.sign_divergence_report(_report_mapping(), baseline, candidate)

    assert verify_divergence_report(envelope, baseline, candidate, public_key) is True

    # Swap the digests in the envelope — PAE will no longer match the signature.
    orig_baseline = envelope.source_digests["baseline"]
    orig_candidate = envelope.source_digests["candidate"]
    envelope.source_digests["baseline"] = orig_candidate
    envelope.source_digests["candidate"] = orig_baseline
    assert verify_divergence_report(envelope, baseline, candidate, public_key) is False
