"""Tests for EpochReplayer.anomaly_scan()."""

from __future__ import annotations

from train_replay.anomaly import AnomalySignal
from train_replay.anomaly.profile import TrainingProfile
from train_replay.collector.flight_recorder import CollectiveEvent
from train_replay.graph.prov_graph import ProvGraph
from train_replay.replay.replayer import AnomalyDetector, EpochReplayer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evt(
    rank: int,
    collective_type: str = "all_reduce",
    seq_id: int = 0,
    start_time_ns: int = 0,
    tensor_size: int = 1024,
) -> CollectiveEvent:
    """Build a minimal CollectiveEvent for testing."""
    return CollectiveEvent(
        rank=rank,
        process_group="default",
        collective_type=collective_type,
        src_rank=None,
        dst_rank=None,
        tensor_size=tensor_size,
        enqueue_time_ns=start_time_ns,
        start_time_ns=start_time_ns,
        end_time_ns=start_time_ns + 1000,
        sequence_id=seq_id,
    )


def _normal_events() -> list[CollectiveEvent]:
    """A simple normal-run timeline for profile fitting."""
    return [
        _evt(0, "all_reduce", seq_id=1, start_time_ns=0),
        _evt(0, "all_reduce", seq_id=2, start_time_ns=10_000),
        _evt(1, "all_reduce", seq_id=1, start_time_ns=0),
        _evt(1, "all_reduce", seq_id=2, start_time_ns=10_000),
    ]


class _StubAnomalyDetector:
    """Detector that returns a fixed set of signals for testing."""

    def __init__(self, signals: list[AnomalySignal]) -> None:
        self._signals = signals
        self.last_events: list[CollectiveEvent] | None = None
        self.last_profile: TrainingProfile | None = None

    def detect(
        self,
        events: list[CollectiveEvent],
        profile: TrainingProfile,
    ) -> list[AnomalySignal]:
        self.last_events = events
        self.last_profile = profile
        return list(self._signals)


# ---------------------------------------------------------------------------
# anomaly_scan tests
# ---------------------------------------------------------------------------


class TestAnomalyScan:
    def test_no_detector_returns_empty_list(self) -> None:
        """Without an anomaly detector, anomaly_scan returns []."""
        replayer = EpochReplayer(ProvGraph())
        profile = TrainingProfile.fit_on_normal_run(_normal_events())
        result = replayer.anomaly_scan([], profile)
        assert result == []

    def test_delegates_to_configured_detector(self) -> None:
        """anomaly_scan passes events and profile to the detector."""
        signal = AnomalySignal(
            score=3.5,
            confidence=0.9,
            description="delayed all_reduce",
            event_rank=0,
            event_step=2,
            collective_type="all_reduce",
        )
        stub = _StubAnomalyDetector([signal])
        replayer = EpochReplayer(ProvGraph(), anomaly_detector=stub)  # type: ignore[arg-type]
        events = _normal_events()
        profile = TrainingProfile.fit_on_normal_run(events)

        result = replayer.anomaly_scan(events, profile)

        assert stub.last_events is events
        assert stub.last_profile is profile
        assert len(result) == 1
        assert result[0].score == 3.5

    def test_results_ranked_by_score_descending(self) -> None:
        """Signals are returned sorted by score, highest first."""
        signals = [
            AnomalySignal(
                score=1.0, confidence=0.5, description="low",
                event_rank=0, event_step=1, collective_type="all_reduce",
            ),
            AnomalySignal(
                score=5.0, confidence=0.9, description="high",
                event_rank=1, event_step=2, collective_type="barrier",
            ),
            AnomalySignal(
                score=3.0, confidence=0.7, description="mid",
                event_rank=0, event_step=3, collective_type="all_gather",
            ),
        ]
        stub = _StubAnomalyDetector(signals)
        replayer = EpochReplayer(ProvGraph(), anomaly_detector=stub)  # type: ignore[arg-type]
        profile = TrainingProfile.fit_on_normal_run(_normal_events())

        result = replayer.anomaly_scan([], profile)

        scores = [s.score for s in result]
        assert scores == [5.0, 3.0, 1.0]

    def test_empty_events_returns_empty(self) -> None:
        """Scanning an empty event list yields the detector's output (sorted)."""
        signal = AnomalySignal(
            score=1.0, confidence=0.5, description="x",
            event_rank=0, event_step=0, collective_type="all_reduce",
        )
        stub = _StubAnomalyDetector([signal])
        replayer = EpochReplayer(ProvGraph(), anomaly_detector=stub)  # type: ignore[arg-type]
        profile = TrainingProfile()

        result = replayer.anomaly_scan([], profile)

        assert len(result) == 1
        assert result[0].score == 1.0

    def test_single_signal_unchanged_by_sort(self) -> None:
        """A single signal passes through sorting unchanged."""
        signal = AnomalySignal(
            score=2.5, confidence=0.8, description="solo",
            event_rank=0, event_step=1, collective_type="all_reduce",
        )
        stub = _StubAnomalyDetector([signal])
        replayer = EpochReplayer(ProvGraph(), anomaly_detector=stub)  # type: ignore[arg-type]
        profile = TrainingProfile()

        result = replayer.anomaly_scan([], profile)

        assert len(result) == 1
        assert result[0].score == signal.score
        assert result[0].description == signal.description

    def test_equal_scores_preserve_relative_order(self) -> None:
        """Signals with equal scores remain in detector-returned order (stable sort)."""
        signals = [
            AnomalySignal(
                score=1.0, confidence=0.5, description="first",
                event_rank=0, event_step=1, collective_type="all_reduce",
            ),
            AnomalySignal(
                score=1.0, confidence=0.5, description="second",
                event_rank=1, event_step=2, collective_type="barrier",
            ),
        ]
        stub = _StubAnomalyDetector(signals)
        replayer = EpochReplayer(ProvGraph(), anomaly_detector=stub)  # type: ignore[arg-type]
        profile = TrainingProfile()

        result = replayer.anomaly_scan([], profile)

        descriptions = [s.description for s in result]
        assert descriptions == ["first", "second"]

    def test_detector_satisfies_protocol(self) -> None:
        """_StubAnomalyDetector is recognised as an AnomalyDetector."""
        stub = _StubAnomalyDetector([])
        assert isinstance(stub, AnomalyDetector)
