"""Tests for train_replay.replay.types — DivergenceReport, Divergence, DivergenceKind."""

from __future__ import annotations

import json

from train_replay.replay.types import Divergence, DivergenceKind, DivergenceReport


class TestDivergenceKind:
    def test_values(self) -> None:
        assert DivergenceKind.OP_MISMATCH.value == "op_mismatch"
        assert DivergenceKind.TENSOR_MISMATCH.value == "tensor_mismatch"
        assert DivergenceKind.STEP_MISMATCH.value == "step_mismatch"
        assert DivergenceKind.RANK_MISSING.value == "rank_missing"
        assert DivergenceKind.OTHER.value == "other"

    def test_from_string(self) -> None:
        assert DivergenceKind("op_mismatch") is DivergenceKind.OP_MISMATCH


class TestDivergence:
    def test_fields(self) -> None:
        d = Divergence(
            step=10,
            rank=2,
            kind=DivergenceKind.TENSOR_MISMATCH,
            field_path="loss",
            expected="0.5",
            actual="0.7",
        )
        assert d.step == 10
        assert d.rank == 2
        assert d.kind is DivergenceKind.TENSOR_MISMATCH
        assert d.field_path == "loss"
        assert d.expected == "0.5"
        assert d.actual == "0.7"


class TestDivergenceReport:
    def _make_report(self) -> DivergenceReport:
        return DivergenceReport(
            first_divergence_step=42,
            divergences=[
                Divergence(
                    step=42,
                    rank=0,
                    kind=DivergenceKind.OP_MISMATCH,
                    field_path="allreduce",
                    expected="nccl",
                    actual="gloo",
                ),
            ],
            per_rank_similarity={0: 0.95, 1: 1.0},
            summary="Rank 0 diverged at step 42",
        )

    def test_fields(self) -> None:
        r = self._make_report()
        assert r.first_divergence_step == 42
        assert len(r.divergences) == 1
        assert r.per_rank_similarity == {0: 0.95, 1: 1.0}
        assert r.summary == "Rank 0 diverged at step 42"

    def test_defaults(self) -> None:
        r = DivergenceReport(first_divergence_step=0)
        assert r.divergences == []
        assert r.per_rank_similarity == {}
        assert r.summary == ""

    # -- to_json / from_json ----------------------------------------------------

    def test_to_json_round_trip(self) -> None:
        r = self._make_report()
        raw = r.to_json()
        parsed = json.loads(raw)
        # Enum serialized as string value
        assert parsed["divergences"][0]["kind"] == "op_mismatch"
        # Deterministic sort_keys
        keys = list(parsed.keys())
        assert keys == sorted(keys)
        # Round-trip
        restored = DivergenceReport.from_json(raw)
        assert restored.first_divergence_step == r.first_divergence_step
        assert restored.summary == r.summary
        assert restored.per_rank_similarity == r.per_rank_similarity
        assert len(restored.divergences) == 1
        assert restored.divergences[0].kind is DivergenceKind.OP_MISMATCH
        assert restored.divergences[0].step == 42
        assert restored.divergences[0].rank == 0

    def test_from_json_string_keys(self) -> None:
        """JSON always has string keys; deserialization must coerce to int."""
        data = {
            "first_divergence_step": 5,
            "divergences": [],
            "per_rank_similarity": {"0": 0.9, "1": 0.8},
            "summary": "test",
        }
        raw = json.dumps(data)
        r = DivergenceReport.from_json(raw)
        assert r.per_rank_similarity == {0: 0.9, 1: 0.8}

    # -- to_cbor / from_cbor ----------------------------------------------------

    def test_to_cbor_round_trip(self) -> None:
        r = self._make_report()
        raw = r.to_cbor()
        assert isinstance(raw, bytes)
        restored = DivergenceReport.from_cbor(raw)
        assert restored.first_divergence_step == r.first_divergence_step
        assert restored.summary == r.summary
        assert restored.per_rank_similarity == r.per_rank_similarity
        assert len(restored.divergences) == 1
        assert restored.divergences[0].kind is DivergenceKind.OP_MISMATCH

    def test_empty_report_round_trip(self) -> None:
        """An empty report (no divergences) serializes and deserializes cleanly."""
        r = DivergenceReport(first_divergence_step=-1, summary="no divergence")
        for raw in (r.to_json(), r.to_cbor()):
            if isinstance(raw, bytes):
                restored = DivergenceReport.from_cbor(raw)
            else:
                restored = DivergenceReport.from_json(raw)
            assert restored.first_divergence_step == -1
            assert restored.divergences == []
            assert restored.per_rank_similarity == {}
            assert restored.summary == "no divergence"
