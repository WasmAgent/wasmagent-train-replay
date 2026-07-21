"""Tests for backend-specific CollisionDetector implementations."""

import pytest

from train_replay.collector.flight_recorder import CollectiveEvent
from train_replay.graph.collision import (
    Collision,
    CollisionDetector,
    CollisionReport,
    GlooCollisionDetector,
    MtiaCollisionDetector,
    NcclCollisionDetector,
    detect_collisions,
)
from train_replay.graph.prov_graph import ProvActivity, ProvEntity, ProvGraph
from train_replay.recording.evidence import AEPRecord, EpochEvidenceBundle
from train_replay.recording.modes import RecordingMode
from train_replay.replay.replayer import EpochReplayer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evt(
    rank: int,
    collective_type: str = "all_reduce",
    seq_id: int = 0,
    start_time_ns: int = 0,
) -> CollectiveEvent:
    """Build a minimal CollectiveEvent for testing."""
    return CollectiveEvent(
        rank=rank,
        process_group="default",
        collective_type=collective_type,
        src_rank=None,
        dst_rank=None,
        tensor_size=1024,
        enqueue_time_ns=start_time_ns,
        start_time_ns=start_time_ns,
        end_time_ns=start_time_ns + 1000,
        sequence_id=seq_id,
    )


def _make_nccl_aligned() -> dict[int, list[CollectiveEvent]]:
    """Two ranks with perfectly aligned NCCL sequences."""
    return {
        0: [_evt(0, "all_reduce", seq_id=1), _evt(0, "all_gather", seq_id=2)],
        1: [_evt(1, "all_reduce", seq_id=1), _evt(1, "all_gather", seq_id=2)],
    }


def _make_nccl_type_mismatch() -> dict[int, list[CollectiveEvent]]:
    """Rank 1 has a different collective type at seq_id 2."""
    return {
        0: [_evt(0, "all_reduce", seq_id=1), _evt(0, "all_gather", seq_id=2)],
        1: [_evt(1, "all_reduce", seq_id=1), _evt(1, "barrier", seq_id=2)],
    }


def _make_nccl_missing_step() -> dict[int, list[CollectiveEvent]]:
    """Rank 1 is missing seq_id 3."""
    return {
        0: [_evt(0, "all_reduce", seq_id=1), _evt(0, "all_reduce", seq_id=2),
            _evt(0, "all_reduce", seq_id=3)],
        1: [_evt(1, "all_reduce", seq_id=1), _evt(1, "all_reduce", seq_id=2)],
    }


# ---------------------------------------------------------------------------
# CollisionReport basics
# ---------------------------------------------------------------------------


class TestCollisionReport:
    def test_empty_report_no_collisions(self) -> None:
        report = CollisionReport()
        assert not report.has_collisions

    def test_report_with_collision(self) -> None:
        report = CollisionReport(
            collisions=[Collision(rank_a=0, rank_b=1, step=5, detail="test")],
            total_steps_checked=10,
        )
        assert report.has_collisions
        assert report.total_steps_checked == 10

    def test_collision_dataclass_fields(self) -> None:
        c = Collision(rank_a=0, rank_b=1, step=42, detail="mismatch")
        assert c.rank_a == 0
        assert c.rank_b == 1
        assert c.step == 42
        assert "mismatch" in c.detail


# ---------------------------------------------------------------------------
# NcclCollisionDetector
# ---------------------------------------------------------------------------


class TestNcclCollisionDetector:
    def test_aligned_sequences_no_collision(self) -> None:
        detector = NcclCollisionDetector()
        report = detector.detect(_make_nccl_aligned())
        assert not report.has_collisions
        assert report.total_steps_checked == 2

    def test_type_mismatch_detected(self) -> None:
        detector = NcclCollisionDetector()
        report = detector.detect(_make_nccl_type_mismatch())
        assert report.has_collisions
        assert report.total_steps_checked == 2
        # The collision should mention seq_id 2.
        mismatch = [c for c in report.collisions if c.step == 2]
        assert len(mismatch) == 1
        assert "all_gather" in mismatch[0].detail
        assert "barrier" in mismatch[0].detail

    def test_missing_step_detected(self) -> None:
        detector = NcclCollisionDetector()
        report = detector.detect(_make_nccl_missing_step())
        assert report.has_collisions
        assert report.total_steps_checked == 3
        missing = [c for c in report.collisions if c.step == 3]
        assert len(missing) == 1
        assert "missing" in missing[0].detail

    def test_single_rank_returns_empty(self) -> None:
        detector = NcclCollisionDetector()
        timelines = {0: [_evt(0, "all_reduce", seq_id=1)]}
        report = detector.detect(timelines)
        assert not report.has_collisions
        assert report.total_steps_checked == 0

    def test_empty_timelines(self) -> None:
        detector = NcclCollisionDetector()
        report = detector.detect({})
        assert not report.has_collisions

    def test_three_ranks_aligned(self) -> None:
        detector = NcclCollisionDetector()
        timelines = {
            0: [_evt(0, "all_reduce", seq_id=1)],
            1: [_evt(1, "all_reduce", seq_id=1)],
            2: [_evt(2, "all_reduce", seq_id=1)],
        }
        report = detector.detect(timelines)
        assert not report.has_collisions

    def test_three_ranks_partial_mismatch(self) -> None:
        detector = NcclCollisionDetector()
        timelines = {
            0: [_evt(0, "all_reduce", seq_id=1), _evt(0, "all_gather", seq_id=2)],
            1: [_evt(1, "all_reduce", seq_id=1), _evt(1, "all_gather", seq_id=2)],
            2: [_evt(2, "all_reduce", seq_id=1), _evt(2, "barrier", seq_id=2)],
        }
        report = detector.detect(timelines)
        assert report.has_collisions
        # Rank 2 mismatches with the reference rank (rank 0) at seq 2.
        rank2_collisions = [c for c in report.collisions if c.rank_b == 2]
        assert len(rank2_collisions) == 1
        assert "barrier" in rank2_collisions[0].detail


# ---------------------------------------------------------------------------
# GlooCollisionDetector
# ---------------------------------------------------------------------------


class TestGlooCollisionDetector:
    def test_aligned_timestamps_no_collision(self) -> None:
        detector = GlooCollisionDetector(tolerance_ns=1_000_000)
        timelines: dict[int, list[CollectiveEvent]] = {
            0: [_evt(0, start_time_ns=1000), _evt(0, start_time_ns=2000)],
            1: [_evt(1, start_time_ns=1100), _evt(1, start_time_ns=2100)],
        }
        report = detector.detect(timelines)
        assert not report.has_collisions
        assert report.total_steps_checked == 2

    def test_desync_exceeds_tolerance(self) -> None:
        detector = GlooCollisionDetector(tolerance_ns=100)
        timelines = {
            0: [_evt(0, start_time_ns=1000)],
            1: [_evt(1, start_time_ns=2000)],
        }
        report = detector.detect(timelines)
        assert report.has_collisions
        assert report.total_steps_checked == 1
        assert report.collisions[0].step == 0
        assert "1000" in report.collisions[0].detail  # delta is 1000ns

    def test_default_tolerance_one_ms(self) -> None:
        detector = GlooCollisionDetector()  # default 1_000_000
        timelines = {
            0: [_evt(0, start_time_ns=0)],
            1: [_evt(1, start_time_ns=500_000)],
        }
        report = detector.detect(timelines)
        assert not report.has_collisions  # 500µs < 1ms

    def test_event_count_mismatch(self) -> None:
        detector = GlooCollisionDetector(tolerance_ns=1_000_000)
        timelines = {
            0: [_evt(0, start_time_ns=1000), _evt(0, start_time_ns=2000)],
            1: [_evt(1, start_time_ns=1100)],
        }
        report = detector.detect(timelines)
        assert report.has_collisions
        count_mismatch = [c for c in report.collisions if "Event count" in c.detail]
        assert len(count_mismatch) == 1

    def test_single_rank_returns_empty(self) -> None:
        detector = GlooCollisionDetector()
        report = detector.detect({0: [_evt(0, start_time_ns=1000)]})
        assert not report.has_collisions
        assert report.total_steps_checked == 0

    def test_empty_timelines(self) -> None:
        detector = GlooCollisionDetector()
        report = detector.detect({})
        assert not report.has_collisions

    def test_custom_tolerance(self) -> None:
        detector = GlooCollisionDetector(tolerance_ns=10)
        timelines = {
            0: [_evt(0, start_time_ns=0)],
            1: [_evt(1, start_time_ns=15)],
        }
        report = detector.detect(timelines)
        assert report.has_collisions


# ---------------------------------------------------------------------------
# MtiaCollisionDetector
# ---------------------------------------------------------------------------


class TestMtiaCollisionDetector:
    def test_raises_not_implemented(self) -> None:
        detector = MtiaCollisionDetector()
        with pytest.raises(NotImplementedError, match="MTIA collision detection"):
            detector.detect({0: [_evt(0)]})


# ---------------------------------------------------------------------------
# detect_collisions factory
# ---------------------------------------------------------------------------


class TestDetectCollisionsFactory:
    def test_nccl_backend(self) -> None:
        report = detect_collisions("nccl", _make_nccl_aligned())
        assert not report.has_collisions

    def test_gloo_backend(self) -> None:
        timelines = {
            0: [_evt(0, start_time_ns=1000)],
            1: [_evt(1, start_time_ns=1100)],
        }
        report = detect_collisions("gloo", timelines, tolerance_ns=1_000_000)
        assert not report.has_collisions

    def test_gloo_backend_type_collision(self) -> None:
        report = detect_collisions("nccl", _make_nccl_type_mismatch())
        assert report.has_collisions

    def test_mtia_backend_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            detect_collisions("mtia", {0: [_evt(0)]})

    def test_case_insensitive_backend(self) -> None:
        report = detect_collisions("NCCL", _make_nccl_aligned())
        assert not report.has_collisions

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            detect_collisions("mpi", {0: [_evt(0)]})


# ---------------------------------------------------------------------------
# Replay integration
# ---------------------------------------------------------------------------


class _RecordingDetector(CollisionDetector):
    def __init__(self) -> None:
        self.timelines: dict[int, list[CollectiveEvent]] | None = None
        self.report = CollisionReport(
            collisions=[Collision(rank_a=2, rank_b=2, step=7, detail="detected")],
            total_steps_checked=1,
        )

    def detect(
        self,
        timelines: dict[int, list[CollectiveEvent]],
    ) -> CollisionReport:
        self.timelines = timelines
        return self.report


class TestReplayRankCollisionReport:
    def test_replay_rank_populates_collision_report_from_rank_events(self) -> None:
        detector = _RecordingDetector()
        replayer = EpochReplayer(ProvGraph(), detector=detector)
        rank_event = AEPRecord(
            action_id="rank-2-step-7",
            rank=2,
            step=7,
            collective_type="all_reduce",
            recording_mode=RecordingMode.FULL,
        )
        other_rank_event = AEPRecord(
            action_id="rank-3-step-7",
            rank=3,
            step=7,
            collective_type="all_reduce",
            recording_mode=RecordingMode.FULL,
        )
        bundle = EpochEvidenceBundle(
            epoch=11,
            actions=[rank_event, other_rank_event],
        )

        result = replayer.replay_rank(bundle, rank=2, entity_id="missing-entity")

        assert result.collision_report is detector.report
        assert detector.timelines is not None
        assert list(detector.timelines) == [2]
        replayed_event = detector.timelines[2][0]
        assert isinstance(replayed_event, CollectiveEvent)
        assert replayed_event.rank == rank_event.rank
        assert replayed_event.collective_type == rank_event.collective_type
        assert replayed_event.sequence_id == rank_event.step

    def test_replay_rank_leaves_collision_report_none_without_detector(self) -> None:
        # No CollisionDetector configured: replay_rank must not raise and must
        # leave collision_report as None — the fallback the bullet contrasts
        # against ("instead of leaving it None"). The remaining ReplayResult
        # fields still populate from the bundle.
        replayer = EpochReplayer(ProvGraph())  # detector omitted
        rank_event = AEPRecord(
            action_id="rank-1-step-3",
            rank=1,
            step=3,
            collective_type="all_reduce",
            recording_mode=RecordingMode.FULL,
        )
        bundle = EpochEvidenceBundle(epoch=5, actions=[rank_event])

        result = replayer.replay_rank(bundle, rank=1, entity_id="missing-entity")

        assert result.collision_report is None
        assert result.epoch == 5
        assert result.rank == 1
        assert rank_event in result.suspicious_actions
        assert result.causal_ancestors == []

    def test_replay_rank_populates_collision_report_with_real_detector(self) -> None:
        # End-to-end: a real (non-stub) detector wired through replay_rank
        # populates collision_report with a genuine CollisionReport via the
        # check_collisions({rank: events}) call the bullet requires. Per the
        # bullet, replay_rank passes only the replayed rank's timeline; a
        # single-rank NCCL timeline has no peer to mismatch against, so
        # has_collisions is False but the report is a real object (not None).
        detector = NcclCollisionDetector()
        replayer = EpochReplayer(ProvGraph(), detector=detector)
        rank_event = AEPRecord(
            action_id="rank-0-step-1",
            rank=0,
            step=1,
            collective_type="all_reduce",
            recording_mode=RecordingMode.FULL,
        )
        bundle = EpochEvidenceBundle(epoch=3, actions=[rank_event])

        result = replayer.replay_rank(bundle, rank=0, entity_id="missing-entity")

        assert result.collision_report is not None
        assert isinstance(result.collision_report, CollisionReport)
        assert not result.collision_report.has_collisions

    def test_replay_rank_passes_all_events_for_rank_in_order(self) -> None:
        # The bullet's "{rank: events}" population is plural: a rank with
        # several collectives must contribute its whole timeline to
        # check_collisions, in bundle order, with other ranks' interleaved
        # events filtered out. The single-event case above can't distinguish
        # "convert every matching record" from "convert the first one only".
        detector = _RecordingDetector()
        replayer = EpochReplayer(ProvGraph(), detector=detector)
        rank_events = [
            AEPRecord(
                action_id=f"rank-5-step-{step}",
                rank=5,
                step=step,
                collective_type="all_reduce",
                recording_mode=RecordingMode.FULL,
                timestamp_ns=step * 1000,
            )
            for step in (1, 2, 3)
        ]
        other_rank_event = AEPRecord(
            action_id="rank-6-step-1",
            rank=6,
            step=1,
            collective_type="all_reduce",
            recording_mode=RecordingMode.FULL,
        )
        # Interleave the foreign-rank event to confirm rank filtering, not
        # position-based selection.
        bundle = EpochEvidenceBundle(
            epoch=7,
            actions=[rank_events[0], other_rank_event, rank_events[1], rank_events[2]],
        )

        result = replayer.replay_rank(bundle, rank=5, entity_id="missing-entity")

        assert result.collision_report is detector.report
        assert list(detector.timelines) == [5]
        passed = detector.timelines[5]
        assert len(passed) == 3
        for original, converted in zip(rank_events, passed):
            assert isinstance(converted, CollectiveEvent)
            assert converted.rank == 5
            assert converted.sequence_id == original.step
            assert converted.collective_type == original.collective_type
            # _record_to_collective_event maps timestamp_ns onto the three
            # CollectiveEvent time fields — a timestamp-sensitive backend
            # (Gloo) relies on this so collision_report is meaningful.
            assert converted.enqueue_time_ns == original.timestamp_ns
            assert converted.start_time_ns == original.timestamp_ns
            assert converted.end_time_ns == original.timestamp_ns
        # Bundle order preserved across the interleaved foreign-rank event.
        assert [e.sequence_id for e in passed] == [1, 2, 3]

    def test_replay_rank_populates_collision_report_even_when_rank_has_no_events(
        self,
    ) -> None:
        # The bullet populates collision_report via check_collisions whenever
        # a detector is configured — it must NOT short-circuit to None just
        # because the replayed rank happens to contribute zero events. This
        # locks in the "instead of leaving it None" contract for the empty-
        # timeline case and guards against a future ``if events:`` guard
        # silently regressing it. Backend-agnostic: the stub detector records
        # the timeline it receives and returns its preset report regardless
        # of input.
        detector = _RecordingDetector()
        replayer = EpochReplayer(ProvGraph(), detector=detector)
        # Bundle holds only a foreign-rank action; the replayed rank (2)
        # contributes no events at all.
        foreign_event = AEPRecord(
            action_id="rank-9-step-1",
            rank=9,
            step=1,
            collective_type="all_reduce",
            recording_mode=RecordingMode.FULL,
        )
        bundle = EpochEvidenceBundle(epoch=4, actions=[foreign_event])

        result = replayer.replay_rank(bundle, rank=2, entity_id="missing-entity")

        # check_collisions was still invoked on the detector path...
        assert detector.timelines is not None
        assert list(detector.timelines) == [2]
        # ...with an empty timeline for the replayed rank (not skipped)...
        assert detector.timelines[2] == []
        # ...so collision_report is the detector's report, not None.
        assert result.collision_report is detector.report

    def test_replay_rank_populates_collision_report_independently_of_causal_ancestors(
        self,
    ) -> None:
        # The bullet populates ``collision_report`` via ``check_collisions`` — a
        # computation orthogonal to ``find_root_cause``. Every other test in
        # this class drives an *empty* ``ProvGraph`` with ``entity_id`` set to a
        # missing node, so ``causal_ancestors`` is always ``[]``. None can catch
        # a regression that gates collision detection on the entity resolving
        # (e.g. a future ``if ancestors:`` / ``if not ancestors:`` guard before
        # the ``check_collisions`` call, which would silently re-break the
        # bullet's "instead of leaving it None" contract on the resolved-entity
        # path). Build a real ancestor chain, confirm BOTH populate together,
        # locking the independence.
        graph = ProvGraph()
        graph.add_activity(
            ProvActivity(
                id="act:seed",
                label="init",
                rank=0,
                process_group="default",
                timestamp_ns=0,
                collective_type="all_reduce",
            )
        )
        graph.add_activity(
            ProvActivity(
                id="act:gen",
                label="generate",
                rank=0,
                process_group="default",
                timestamp_ns=10,
                collective_type="all_reduce",
            )
        )
        graph.add_entity(ProvEntity(id="tensor:0:1:out", digest=None, rank=0, step=1))
        # was_generated_by(entity, activity) records activity -> entity edges,
        # so act:seed -> act:gen -> tensor:0:1:out is the ancestor chain.
        graph.was_generated_by("act:gen", "act:seed")
        graph.was_generated_by("tensor:0:1:out", "act:gen")

        detector = _RecordingDetector()
        replayer = EpochReplayer(graph, detector=detector)
        rank_event = AEPRecord(
            action_id="rank-0-step-1",
            rank=0,
            step=1,
            collective_type="all_reduce",
            recording_mode=RecordingMode.FULL,
        )
        bundle = EpochEvidenceBundle(epoch=2, actions=[rank_event])

        result = replayer.replay_rank(bundle, rank=0, entity_id="tensor:0:1:out")

        # The entity resolved to a non-empty ancestor chain — the path no other
        # test in this class exercises (they all pass a missing entity_id).
        assert result.causal_ancestors == ["act:gen", "act:seed"]
        # ...yet collision_report is still populated via check_collisions with
        # the replayed rank's timeline, proving the two outputs are computed
        # independently rather than the collision path being ancestor-gated.
        assert result.collision_report is detector.report
        assert detector.timelines is not None
        assert list(detector.timelines) == [0]


# ---------------------------------------------------------------------------
# Import acceptance criteria
# ---------------------------------------------------------------------------


class TestImportAcceptanceCriteria:
    def test_import_nccl(self) -> None:
        from train_replay.graph.collision import NcclCollisionDetector as NCC
        assert NCC is not None

    def test_import_gloo(self) -> None:
        from train_replay.graph.collision import GlooCollisionDetector as GCD
        assert GCD is not None
