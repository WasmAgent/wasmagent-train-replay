"""Tests for abstract collision/desync detection protocol (collision.py)."""

from train_replay.graph.collision import (
    CollisionDetector,
    CollisionEvent,
    CollisionSeverity,
)


class _StubDetector(CollisionDetector):
    """Minimal concrete implementation for testing the protocol."""

    def detect(
        self,
        timelines: dict[int, list[tuple[int, str]]],
        *,
        tolerance_ns: int = 0,
    ) -> list[CollisionEvent]:
        events: list[CollisionEvent] = []
        # Detect mismatched collective types at the same sequence_id across ranks
        all_seqs: set[int] = set()
        for ops in timelines.values():
            for seq_id, _ in ops:
                all_seqs.add(seq_id)

        for seq_id in sorted(all_seqs):
            types_per_rank: dict[str, list[int]] = {}
            for rank, ops in timelines.items():
                for s, ctype in ops:
                    if s == seq_id:
                        types_per_rank.setdefault(ctype, []).append(rank)
            if len(types_per_rank) > 1:
                ranks = []
                for rank_list in types_per_rank.values():
                    ranks.extend(rank_list)
                events.append(CollisionEvent(
                    severity=CollisionSeverity.CRITICAL,
                    rank_a=ranks[0],
                    rank_b=ranks[1] if len(ranks) > 1 else None,
                    sequence_id=seq_id,
                    description=f"Collective type mismatch at seq {seq_id}",
                    backend="stub",
                ))
        return events


def test_collision_event_fields() -> None:
    evt = CollisionEvent(
        severity=CollisionSeverity.WARNING,
        rank_a=0,
        sequence_id=5,
        description="rank 0 ahead of rank 1",
        backend="NCCL",
        rank_b=1,
        details={"drift_ns": "1000"},
    )
    assert evt.severity == CollisionSeverity.WARNING
    assert evt.rank_a == 0
    assert evt.rank_b == 1
    assert evt.details["drift_ns"] == "1000"


def test_collision_severity_values() -> None:
    assert CollisionSeverity.INFO.value == "info"
    assert CollisionSeverity.WARNING.value == "warning"
    assert CollisionSeverity.CRITICAL.value == "critical"


def test_stub_detector_no_collision() -> None:
    detector = _StubDetector()
    timelines = {
        0: [(1, "all_reduce"), (2, "all_gather")],
        1: [(1, "all_reduce"), (2, "all_gather")],
    }
    events = detector.detect(timelines)
    assert events == []


def test_stub_detector_finds_mismatch() -> None:
    detector = _StubDetector()
    timelines = {
        0: [(1, "all_reduce"), (2, "barrier")],
        1: [(1, "all_reduce"), (2, "all_gather")],  # mismatch at seq 2
    }
    events = detector.detect(timelines)
    assert len(events) == 1
    assert events[0].sequence_id == 2
    assert events[0].severity == CollisionSeverity.CRITICAL


def test_stub_detector_sorted_by_sequence_id() -> None:
    detector = _StubDetector()
    timelines = {
        0: [(1, "all_reduce"), (3, "barrier"), (2, "all_gather")],
        1: [(1, "all_reduce"), (3, "reduce"), (2, "all_gather")],
    }
    events = detector.detect(timelines)
    seq_ids = [e.sequence_id for e in events]
    assert seq_ids == sorted(seq_ids)


def test_collision_event_default_details_empty() -> None:
    evt = CollisionEvent(
        severity=CollisionSeverity.INFO,
        rank_a=0,
        sequence_id=1,
        description="test",
        backend="GLOO",
    )
    assert evt.details == {}
    assert evt.rank_b is None


def test_cannot_instantiate_abstract_detector() -> None:
    try:
        CollisionDetector()  # type: ignore[abstract]
        assert False, "Should not instantiate abstract class"
    except TypeError:
        pass
