"""Tests for EpochEvidenceBundle JSON/CBOR serialization round-trips."""

from train_replay.recording.evidence import AEPRecord, EpochEvidenceBundle
from train_replay.recording.modes import RecordingMode


def _sample_bundle(
    *,
    with_signature: bool = False,
    with_optional_fields: bool = False,
) -> EpochEvidenceBundle:
    """Build a representative bundle for round-trip tests."""
    actions = [
        AEPRecord(
            action_id="r0:seq1",
            rank=0,
            step=1,
            collective_type="all_reduce",
            recording_mode=RecordingMode.FULL,
            tensor_input_digest="aabb" if with_optional_fields else None,
            tensor_output_digest="ccdd" if with_optional_fields else None,
            delta_stats={"mean": 0.5, "var": 0.1} if with_optional_fields else None,
            timestamp_ns=1_000_000,
            causal_chain_id="chain-42" if with_optional_fields else None,
            parent_action_id="r0:seq0" if with_optional_fields else None,
        ),
        AEPRecord(
            action_id="r1:seq2",
            rank=1,
            step=2,
            collective_type="barrier",
            recording_mode=RecordingMode.VALIDATION,
            timestamp_ns=2_000_000,
        ),
    ]
    sig = (
        {"alg": "ed25519", "key_id": "dev-key", "sig": "ZmFrZVNpZ25hdHVyZQ=="}
        if with_signature
        else None
    )
    return EpochEvidenceBundle(
        schema_version="train-aep/v0.1",
        run_id="test-run",
        epoch=7,
        actions=actions,
        signature=sig,
    )


# -- JSON round-trip --------------------------------------------------------


class TestJsonRoundTrip:
    def test_basic_fields_preserved(self):
        bundle = _sample_bundle()
        restored = EpochEvidenceBundle.from_json(bundle.to_json())
        assert restored == bundle

    def test_with_signature_preserved(self):
        bundle = _sample_bundle(with_signature=True)
        restored = EpochEvidenceBundle.from_json(bundle.to_json())
        assert restored == bundle

    def test_with_optional_fields_preserved(self):
        bundle = _sample_bundle(with_optional_fields=True)
        restored = EpochEvidenceBundle.from_json(bundle.to_json())
        assert restored == bundle

    def test_empty_bundle(self):
        bundle = EpochEvidenceBundle()
        restored = EpochEvidenceBundle.from_json(bundle.to_json())
        assert restored == bundle

    def test_digest_unchanged_after_round_trip(self):
        """canonical_bytes must produce the same digest before and after."""
        bundle = _sample_bundle(with_signature=True)
        original_digest = bundle.digest()
        restored = EpochEvidenceBundle.from_json(bundle.to_json())
        assert restored.digest() == original_digest

    def test_recording_mode_enum_restored(self):
        """RecordingMode must be a proper enum member, not a bare string."""
        bundle = _sample_bundle()
        restored = EpochEvidenceBundle.from_json(bundle.to_json())
        for orig, res in zip(bundle.actions, restored.actions):
            assert isinstance(res.recording_mode, RecordingMode)
            assert res.recording_mode is orig.recording_mode


# -- CBOR round-trip ---------------------------------------------------------


class TestCborRoundTrip:
    def test_basic_fields_preserved(self):
        bundle = _sample_bundle()
        restored = EpochEvidenceBundle.from_cbor(bundle.to_cbor())
        assert restored == bundle

    def test_with_signature_preserved(self):
        bundle = _sample_bundle(with_signature=True)
        restored = EpochEvidenceBundle.from_cbor(bundle.to_cbor())
        assert restored == bundle

    def test_with_optional_fields_preserved(self):
        bundle = _sample_bundle(with_optional_fields=True)
        restored = EpochEvidenceBundle.from_cbor(bundle.to_cbor())
        assert restored == bundle

    def test_empty_bundle(self):
        bundle = EpochEvidenceBundle()
        restored = EpochEvidenceBundle.from_cbor(bundle.to_cbor())
        assert restored == bundle

    def test_digest_unchanged_after_round_trip(self):
        bundle = _sample_bundle(with_signature=True)
        original_digest = bundle.digest()
        restored = EpochEvidenceBundle.from_cbor(bundle.to_cbor())
        assert restored.digest() == original_digest

    def test_recording_mode_enum_restored(self):
        bundle = _sample_bundle()
        restored = EpochEvidenceBundle.from_cbor(bundle.to_cbor())
        for orig, res in zip(bundle.actions, restored.actions):
            assert isinstance(res.recording_mode, RecordingMode)
            assert res.recording_mode is orig.recording_mode

    def test_cbor_output_is_bytes(self):
        bundle = _sample_bundle()
        data = bundle.to_cbor()
        assert isinstance(data, bytes)
        assert len(data) > 0
