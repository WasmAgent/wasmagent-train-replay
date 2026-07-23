"""Tests for anomaly detection layer."""

from __future__ import annotations

import pytest

from train_replay.anomaly.detector import (
    AnomalyDetector,
    AnomalySignal,
    StatisticalAnomalyDetector,
)
from train_replay.recording.evidence import AEPRecord
from train_replay.recording.modes import RecordingMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(
    rank: int = 0,
    step: int = 0,
    timestamp_ns: int = 0,
    delta_stats: dict[str, float] | None = None,
    **kwargs: object,
) -> AEPRecord:
    return AEPRecord(
        action_id=f"r{rank}:s{step}",
        rank=rank,
        step=step,
        collective_type="all_reduce",
        recording_mode=RecordingMode.VALIDATION,
        timestamp_ns=timestamp_ns,
        delta_stats=delta_stats,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# AnomalySignal dataclass
# ---------------------------------------------------------------------------


class TestAnomalySignal:
    def test_fields(self) -> None:
        sig = AnomalySignal(
            action_id="r0:s1",
            rank=0,
            step=1,
            metric_name="timing_zscore",
            score=4.5,
            severity=1.0,
            description="outlier",
        )
        assert sig.action_id == "r0:s1"
        assert sig.score == 4.5
        assert sig.severity == 1.0
        assert sig.extra == {}

    def test_extra_defaults_to_empty_dict(self) -> None:
        sig = AnomalySignal(
            action_id="r0:s1",
            rank=0,
            step=1,
            metric_name="timing_zscore",
            score=0.0,
            severity=0.0,
            description="",
        )
        assert sig.extra == {}


# ---------------------------------------------------------------------------
# AnomalyDetector ABC
# ---------------------------------------------------------------------------


class TestAnomalyDetectorABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            AnomalyDetector()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_detect(self) -> None:
        class Bad(AnomalyDetector):  # type: ignore[misc]
            pass

        with pytest.raises(TypeError):
            Bad()

    def test_minimal_subclass(self) -> None:
        class Minimal(AnomalyDetector):
            def detect(self, events: list[object]) -> list[AnomalySignal]:
                return []

        det = Minimal()
        assert det.detect([]) == []


# ---------------------------------------------------------------------------
# StatisticalAnomalyDetector — constructor
# ---------------------------------------------------------------------------


class TestStatisticalAnomalyDetectorConstructor:
    def test_default_threshold(self) -> None:
        det = StatisticalAnomalyDetector()
        assert det._z_threshold == 3.0

    def test_custom_threshold(self) -> None:
        det = StatisticalAnomalyDetector(z_threshold=2.5)
        assert det._z_threshold == 2.5

    def test_zero_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            StatisticalAnomalyDetector(z_threshold=0.0)

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            StatisticalAnomalyDetector(z_threshold=-1.0)


# ---------------------------------------------------------------------------
# StatisticalAnomalyDetector — empty / trivial inputs
# ---------------------------------------------------------------------------


class TestStatisticalAnomalyDetectorEmptyInputs:
    def test_empty_event_list(self) -> None:
        det = StatisticalAnomalyDetector()
        assert det.detect([]) == []

    def test_single_event_no_anomaly(self) -> None:
        det = StatisticalAnomalyDetector()
        events = [_record(rank=0, step=1, timestamp_ns=100)]
        assert det.detect(events) == []

    def test_events_without_timestamp_ns(self) -> None:
        """Objects missing timestamp_ns are skipped entirely."""
        det = StatisticalAnomalyDetector()
        obj = object()  # no timestamp_ns attribute
        assert det.detect([obj]) == []

    def test_two_events_same_timestamp_emits_duplicate_signal(self) -> None:
        """Simultaneous events on the same rank produce a duplicate_timestamp signal."""
        det = StatisticalAnomalyDetector()
        events = [
            _record(rank=0, step=1, timestamp_ns=100),
            _record(rank=0, step=2, timestamp_ns=100),
        ]
        signals = det.detect(events)
        dup = [s for s in signals if s.metric_name == "duplicate_timestamp"]
        assert len(dup) == 1
        assert dup[0].step == 2
        assert dup[0].severity == 0.5


# ---------------------------------------------------------------------------
# StatisticalAnomalyDetector — timing anomalies
# ---------------------------------------------------------------------------


class TestStatisticalAnomalyDetectorTiming:
    def test_uniform_intervals_no_zscore_anomaly(self) -> None:
        """Evenly-spaced timestamps should not trigger timing Z-score signals.

        A zero_variance signal is expected but no Z-score outlier.
        """
        det = StatisticalAnomalyDetector(z_threshold=3.0)
        events = [_record(rank=0, step=i, timestamp_ns=i * 1000) for i in range(20)]
        signals = det.detect(events)
        timing = [s for s in signals if s.metric_name == "timing_zscore"]
        assert timing == []

    def test_outlier_interval_detected(self) -> None:
        """A huge gap between two consecutive events on the same rank triggers."""
        det = StatisticalAnomalyDetector(z_threshold=2.0)
        # 18 events at 1000ns intervals, then one at 500000ns gap.
        events = [_record(rank=0, step=i, timestamp_ns=i * 1000) for i in range(18)]
        events.append(_record(rank=0, step=18, timestamp_ns=18 * 1000 + 500_000))
        signals = det.detect(events)
        timing = [s for s in signals if s.metric_name == "timing_zscore"]
        assert len(timing) == 1
        assert timing[0].step == 18
        assert timing[0].score > 0

    def test_multi_rank_independent(self) -> None:
        """An outlier on rank 1 must not affect rank 0 detection."""
        det = StatisticalAnomalyDetector(z_threshold=2.0)
        # Rank 0 — uniform intervals
        r0 = [_record(rank=0, step=i, timestamp_ns=i * 1000) for i in range(20)]
        # Rank 1 — uniform intervals with a spike
        r1 = [_record(rank=1, step=i, timestamp_ns=i * 1000) for i in range(18)]
        r1.append(_record(rank=1, step=18, timestamp_ns=18 * 1000 + 500_000))
        signals = det.detect(r0 + r1)
        timing = [s for s in signals if s.metric_name == "timing_zscore"]
        # Only rank 1's outlier should be detected.
        assert all(s.rank == 1 for s in timing)


# ---------------------------------------------------------------------------
# StatisticalAnomalyDetector — delta stat anomalies
# ---------------------------------------------------------------------------


class TestStatisticalAnomalyDetectorDeltaStats:
    def test_no_delta_stats_no_signal(self) -> None:
        det = StatisticalAnomalyDetector()
        events = [_record(rank=0, step=i, timestamp_ns=i * 100) for i in range(10)]
        delta_signals = [s for s in det.detect(events) if "delta_zscore" in s.metric_name]
        assert delta_signals == []

    def test_uniform_delta_stats_no_signal(self) -> None:
        det = StatisticalAnomalyDetector()
        events = [
            _record(rank=0, step=i, timestamp_ns=i * 100, delta_stats={"loss": 0.5})
            for i in range(20)
        ]
        delta_signals = [s for s in det.detect(events) if "delta_zscore" in s.metric_name]
        assert delta_signals == []

    def test_outlier_delta_stat_detected(self) -> None:
        det = StatisticalAnomalyDetector(z_threshold=2.0)
        events = [
            _record(rank=0, step=i, timestamp_ns=i * 100, delta_stats={"loss": 0.1})
            for i in range(18)
        ]
        # Spike the loss on step 18.
        events.append(
            _record(rank=0, step=18, timestamp_ns=1800, delta_stats={"loss": 50.0})
        )
        signals = det.detect(events)
        delta_signals = [s for s in signals if "delta_zscore" in s.metric_name]
        assert len(delta_signals) == 1
        assert delta_signals[0].step == 18
        assert "loss" in delta_signals[0].metric_name

    def test_multiple_delta_keys(self) -> None:
        """Outliers in different delta stat keys should each produce a signal."""
        det = StatisticalAnomalyDetector(z_threshold=2.0)
        events = [
            _record(
                rank=0,
                step=i,
                timestamp_ns=i * 100,
                delta_stats={"loss": 0.1, "grad_norm": 1.0},
            )
            for i in range(18)
        ]
        # Spike both keys on step 18.
        events.append(
            _record(
                rank=0,
                step=18,
                timestamp_ns=1800,
                delta_stats={"loss": 50.0, "grad_norm": 100.0},
            )
        )
        signals = det.detect(events)
        delta_signals = [s for s in signals if "delta_zscore" in s.metric_name]
        metric_names = {s.metric_name for s in delta_signals}
        assert "delta_zscore:loss" in metric_names
        assert "delta_zscore:grad_norm" in metric_names

    def test_non_numeric_delta_stats_ignored(self) -> None:
        det = StatisticalAnomalyDetector()
        events = [
            _record(
                rank=0,
                step=0,
                timestamp_ns=0,
                delta_stats={"label": "ok", "count": 5.0},
            )
        ]
        # Single numeric value — not enough to compute Z-score.
        signals = det.detect(events)
        delta_signals = [s for s in signals if "delta_zscore" in s.metric_name]
        assert delta_signals == []


# ---------------------------------------------------------------------------
# StatisticalAnomalyDetector — severity clamping
# ---------------------------------------------------------------------------


class TestStatisticalAnomalyDetectorSeverity:
    def test_severity_clamped_to_one(self) -> None:
        """Even an extreme Z-score must produce severity <= 1.0."""
        det = StatisticalAnomalyDetector(z_threshold=1.0)
        events = [_record(rank=0, step=i, timestamp_ns=i * 100) for i in range(10)]
        events.append(_record(rank=0, step=10, timestamp_ns=10 * 100 + 1_000_000))
        signals = det.detect(events)
        for s in signals:
            assert 0.0 <= s.severity <= 1.0

    def test_severity_zero_for_non_anomalous(self) -> None:
        det = StatisticalAnomalyDetector(z_threshold=100.0)
        events = [_record(rank=0, step=i, timestamp_ns=i * 100) for i in range(10)]
        signals = det.detect(events)
        # Uniform intervals now emit a zero_variance signal; no Z-score
        # outliers should be present with such a high threshold.
        zscore_sigs = [s for s in signals if s.metric_name == "timing_zscore"]
        assert zscore_sigs == []
        # A zero_variance signal is expected for the perfectly uniform intervals.
        zv = [s for s in signals if "zero_variance" in s.metric_name]
        assert len(zv) == 1


# ---------------------------------------------------------------------------
# StatisticalAnomalyDetector — combined timing + delta signals
# ---------------------------------------------------------------------------


class TestStatisticalAnomalyDetectorCombined:
    def test_both_timing_and_delta_signals(self) -> None:
        """An event that is an outlier in both timing and delta stats
        produces two separate signals."""
        det = StatisticalAnomalyDetector(z_threshold=2.0)
        events = [
            _record(rank=0, step=i, timestamp_ns=i * 1000, delta_stats={"loss": 0.1})
            for i in range(18)
        ]
        events.append(
            _record(
                rank=0,
                step=18,
                timestamp_ns=18 * 1000 + 500_000,
                delta_stats={"loss": 50.0},
            )
        )
        signals = det.detect(events)
        types = {s.metric_name for s in signals}
        assert "timing_zscore" in types
        assert "delta_zscore:loss" in types


# ---------------------------------------------------------------------------
# Zero-variance detection (reviewer finding #3)
# ---------------------------------------------------------------------------


class TestStatisticalAnomalyDetectorZeroVariance:
    """When all values in a sequence are identical (stdev == 0), the detector
    emits a zero_variance signal rather than silently skipping."""

    def test_zero_variance_timing_emits_signal(self) -> None:
        """Perfectly uniform intervals with >=3 samples produce a signal."""
        det = StatisticalAnomalyDetector()
        events = [_record(rank=0, step=i, timestamp_ns=i * 1000) for i in range(10)]
        signals = det.detect(events)
        zv = [s for s in signals if s.metric_name == "zero_variance:timing"]
        assert len(zv) == 1
        assert "zero timing variance" in zv[0].description
        assert zv[0].severity == 0.3

    def test_zero_variance_timing_two_samples_no_signal(self) -> None:
        """Only 2 identical intervals — below the zero-variance threshold."""
        det = StatisticalAnomalyDetector()
        events = [_record(rank=0, step=i, timestamp_ns=i * 1000) for i in range(3)]
        signals = det.detect(events)
        zv = [s for s in signals if "zero_variance" in s.metric_name]
        assert zv == []

    def test_zero_variance_delta_stats_emits_signal(self) -> None:
        """Identical delta stat values with >=3 samples produce a signal."""
        det = StatisticalAnomalyDetector()
        events = [
            _record(rank=0, step=i, timestamp_ns=i * 100, delta_stats={"loss": 0.5})
            for i in range(10)
        ]
        signals = det.detect(events)
        zv = [s for s in signals if s.metric_name == "zero_variance:delta:loss"]
        assert len(zv) == 1
        assert "stuck metric" in zv[0].description
        assert zv[0].severity == 0.3

    def test_zero_variance_delta_stats_two_samples_no_signal(self) -> None:
        """Only 2 identical delta values — below the zero-variance threshold."""
        det = StatisticalAnomalyDetector()
        events = [
            _record(rank=0, step=i, timestamp_ns=i * 100, delta_stats={"loss": 0.5})
            for i in range(2)
        ]
        signals = det.detect(events)
        zv = [s for s in signals if "zero_variance" in s.metric_name]
        assert zv == []

    def test_zero_variance_no_false_timing_outliers(self) -> None:
        """When intervals are uniform, no timing_zscore signals should fire,
        only the zero_variance signal."""
        det = StatisticalAnomalyDetector()
        events = [_record(rank=0, step=i, timestamp_ns=i * 1000) for i in range(10)]
        signals = det.detect(events)
        timing_zs = [s for s in signals if s.metric_name == "timing_zscore"]
        assert timing_zs == []


# ---------------------------------------------------------------------------
# Duplicate timestamp detection (reviewer finding #2)
# ---------------------------------------------------------------------------


class TestStatisticalAnomalyDetectorDuplicateTimestamps:
    """Zero-time intervals (duplicate timestamps on same rank) are flagged."""

    def test_single_duplicate_pair(self) -> None:
        det = StatisticalAnomalyDetector()
        events = [
            _record(rank=0, step=1, timestamp_ns=100),
            _record(rank=0, step=2, timestamp_ns=100),
            _record(rank=0, step=3, timestamp_ns=200),
        ]
        signals = det.detect(events)
        dup = [s for s in signals if s.metric_name == "duplicate_timestamp"]
        assert len(dup) == 1
        assert dup[0].step == 2

    def test_multiple_duplicate_pairs(self) -> None:
        det = StatisticalAnomalyDetector()
        events = [
            _record(rank=0, step=1, timestamp_ns=100),
            _record(rank=0, step=2, timestamp_ns=100),
            _record(rank=0, step=3, timestamp_ns=100),
        ]
        signals = det.detect(events)
        dup = [s for s in signals if s.metric_name == "duplicate_timestamp"]
        # 3 events with same timestamp → 2 consecutive pairs (1-2, 2-3)
        assert len(dup) == 2

    def test_different_ranks_same_timestamp_not_duplicate(self) -> None:
        """Same timestamp on different ranks is normal — not a duplicate."""
        det = StatisticalAnomalyDetector()
        events = [
            _record(rank=0, step=1, timestamp_ns=100),
            _record(rank=1, step=1, timestamp_ns=100),
        ]
        signals = det.detect(events)
        dup = [s for s in signals if s.metric_name == "duplicate_timestamp"]
        assert dup == []
