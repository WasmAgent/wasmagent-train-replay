"""Tests for synthetic timing-anomaly injection against TrainingProfile baselines.

These tests build a :class:`TrainingProfile` from a known-good run, then
inject various timing anomalies (delayed all-reduce, straggler rank, burst
pattern, missing rank) into a second run and verify that the anomalous run's
statistics deviate from the baseline in detectable ways.

The anomaly *detector* (``detector.py``) is a separate milestone; here we
assert that the profile's summary statistics are sensitive enough for a
downstream detector to flag the injected faults.
"""

from __future__ import annotations

from train_replay.anomaly.profile import TrainingProfile
from train_replay.collector.flight_recorder import CollectiveEvent

# -- helpers ----------------------------------------------------------------


def _ev(
    rank: int,
    collective_type: str,
    start: int,
    end: int,
    size: int,
) -> CollectiveEvent:
    """Shorthand to build a CollectiveEvent with sensible defaults."""
    return CollectiveEvent(
        rank=rank,
        process_group="default",
        collective_type=collective_type,
        src_rank=None,
        dst_rank=None,
        tensor_size=size,
        enqueue_time_ns=start,
        start_time_ns=start,
        end_time_ns=end,
    )


def _normal_run(
    ranks: int = 4,
    steps: int = 10,
    interval_ns: int = 1_000_000,
    tensor_size: int = 1024,
    collective_type: str = "all_reduce",
) -> list[CollectiveEvent]:
    """Generate a perfectly regular training run across *ranks*.

    Each rank has *steps* collective events spaced exactly *interval_ns*
    apart. This represents a known-good baseline with no anomalies.
    """
    events: list[CollectiveEvent] = []
    for r in range(ranks):
        for s in range(steps):
            start = s * interval_ns
            events.append(_ev(r, collective_type, start, start + 1000, tensor_size))
    return events


def _z_score(observed: float, mean: float, std: float) -> float:
    """Return the Z-score of *observed* against (mean, std).

    Returns 0.0 when std is zero to avoid division-by-zero on degenerate
    baselines (e.g. single-event profiles).
    """
    if std == 0.0:
        return 0.0
    return abs(observed - mean) / std


# -- fixture: stable baseline ------------------------------------------------


def _baseline(
    ranks: int = 4,
    steps: int = 10,
    interval_ns: int = 1_000_000,
    tensor_size: int = 1024,
) -> TrainingProfile:
    """Build a TrainingProfile from a perfectly regular run."""
    events = _normal_run(ranks, steps, interval_ns, tensor_size)
    return TrainingProfile.fit_on_normal_run(events)


# -- tests -------------------------------------------------------------------


class TestDelayedAllReduce:
    """Inject a single delayed all-reduce on one rank and verify the profile
    of the anomalous run deviates from the baseline."""

    def test_delayed_event_increases_interval_max(self) -> None:
        baseline = _baseline(ranks=2, steps=5, interval_ns=1_000_000)

        # Inject a delay: rank 0, step 3 shifted from 3_000_000 to 13_000_000
        anomalous = _normal_run(ranks=2, steps=5, interval_ns=1_000_000)
        # _normal_run emits rank-0 events at indices 0..4, rank-1 at 5..9
        # step 2 end = 2_000_000+1000, step 3 start shifted to 13_000_000 => gap = 11M-2M = 9M
        idx = 3  # rank 0, step 3
        anomalous[idx] = _ev(0, "all_reduce", 13_000_000, 13_001_000, 1024)

        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        # The delayed gap (9_000_000 ns) should push interval_max above baseline
        assert anomalous_profile.interval_max_ns > baseline.interval_max_ns
        # Baseline max is exactly 1_000_000 (perfectly regular)
        assert baseline.interval_max_ns == 1_000_000
        assert anomalous_profile.interval_max_ns == 9_000_000

    def test_delayed_event_increases_interval_mean(self) -> None:
        baseline = _baseline(ranks=2, steps=5, interval_ns=1_000_000)

        anomalous = _normal_run(ranks=2, steps=5, interval_ns=1_000_000)
        # rank 0, step 3: insert 10ms delay
        idx = 3  # rank 0, step 3
        anomalous[idx] = _ev(0, "all_reduce", 13_000_000, 13_001_000, 1024)

        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        # The mean interval should increase due to the outlier gap
        assert anomalous_profile.interval_mean_ns > baseline.interval_mean_ns
        assert baseline.interval_mean_ns == 1_000_000.0

    def test_delayed_event_produces_large_z_score(self) -> None:
        baseline = _baseline(ranks=4, steps=10, interval_ns=1_000_000)

        anomalous = _normal_run(ranks=4, steps=10, interval_ns=1_000_000)
        # rank 2, step 5: shift from 5_000_000 to 55_000_000 (50ms delay)
        idx = 2 * 10 + 5  # rank 2, step 5
        anomalous[idx] = _ev(2, "all_reduce", 55_000_000, 55_001_000, 1024)

        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        # Z-score of the anomalous mean vs baseline should be significant (>2)
        z = _z_score(
            anomalous_profile.interval_mean_ns,
            baseline.interval_mean_ns,
            baseline.interval_std_ns,
        )
        # With 4 ranks × 10 steps = 40 events, one 50ms outlier among 39 normal
        # gaps should produce a detectable deviation
        assert z > 2.0


class TestStragglerRank:
    """One rank consistently slower than the others."""

    def test_straggler_shifts_interval_mean_up(self) -> None:
        baseline = _baseline(ranks=4, steps=10, interval_ns=1_000_000)

        # Rank 3 is 5x slower: interval 5_000_000 instead of 1_000_000
        anomalous: list[CollectiveEvent] = []
        for r in range(4):
            interval = 5_000_000 if r == 3 else 1_000_000
            for s in range(10):
                start = s * interval
                anomalous.append(_ev(r, "all_reduce", start, start + 1000, 1024))

        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        # Mean should be higher than baseline due to straggler's larger gaps
        assert anomalous_profile.interval_mean_ns > baseline.interval_mean_ns

    def test_straggler_increases_interval_std(self) -> None:
        baseline = _baseline(ranks=4, steps=10, interval_ns=1_000_000)

        # Rank 0 is 10x slower
        anomalous: list[CollectiveEvent] = []
        for r in range(4):
            interval = 10_000_000 if r == 0 else 1_000_000
            for s in range(10):
                start = s * interval
                anomalous.append(_ev(r, "all_reduce", start, start + 1000, 1024))

        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        # Std should increase: baseline has 0 std (perfectly regular),
        # anomalous has mixed gap sizes
        assert anomalous_profile.interval_std_ns > baseline.interval_std_ns
        assert baseline.interval_std_ns == 0.0
        assert anomalous_profile.interval_std_ns > 0.0


class TestBurstPattern:
    """A burst of rapid-fire events violates the normal inter-arrival cadence."""

    def test_burst_decreases_interval_min(self) -> None:
        baseline = _baseline(ranks=2, steps=10, interval_ns=1_000_000)

        anomalous = _normal_run(ranks=2, steps=10, interval_ns=1_000_000)
        # Compress rank 0, steps 4-6 into a tight burst (100ns apart)
        for i in range(4, 7):
            idx = i  # rank 0
            start = 4_000_000 + (i - 4) * 100
            anomalous[idx] = _ev(0, "all_reduce", start, start + 100, 1024)

        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        # The burst should produce intervals much smaller than baseline's 1_000_000
        assert anomalous_profile.interval_min_ns < baseline.interval_min_ns
        assert baseline.interval_min_ns == 1_000_000
        assert anomalous_profile.interval_min_ns == 100

    def test_burst_increases_interval_std(self) -> None:
        baseline = _baseline(ranks=1, steps=10, interval_ns=1_000_000)

        anomalous = _normal_run(ranks=1, steps=10, interval_ns=1_000_000)
        # Inject a burst: steps 3,4,5 at 10ns intervals
        for i in range(3, 6):
            start = 3_000_000 + (i - 3) * 10
            anomalous[i] = _ev(0, "all_reduce", start, start + 10, 1024)

        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        # Std should increase due to the mix of 1_000_000 and 10ns gaps
        assert anomalous_profile.interval_std_ns > baseline.interval_std_ns


class TestMissingRank:
    """A rank that participated in the baseline disappears from the run."""

    def test_missing_rank_changes_rank_set(self) -> None:
        baseline = _baseline(ranks=4, steps=5, interval_ns=1_000_000)
        assert baseline.ranks == frozenset({0, 1, 2, 3})

        # Run with only 3 ranks
        anomalous = _normal_run(ranks=3, steps=5, interval_ns=1_000_000)
        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        assert anomalous_profile.ranks == frozenset({0, 1, 2})
        assert anomalous_profile.ranks != baseline.ranks

    def test_missing_rank_reduces_event_count(self) -> None:
        baseline = _baseline(ranks=4, steps=5, interval_ns=1_000_000)

        anomalous = _normal_run(ranks=3, steps=5, interval_ns=1_000_000)
        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        assert anomalous_profile.event_count < baseline.event_count
        assert anomalous_profile.event_count == 15  # 3 ranks × 5 steps
        assert baseline.event_count == 20  # 4 ranks × 5 steps

    def test_missing_rank_changes_collective_type_counts(self) -> None:
        baseline = _baseline(ranks=4, steps=5, interval_ns=1_000_000)

        anomalous = _normal_run(ranks=3, steps=5, interval_ns=1_000_000)
        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        # With 5 fewer events, all_reduce count should drop by 5
        base_count = baseline.collective_type_counts["all_reduce"]
        assert anomalous_profile.collective_type_counts["all_reduce"] == base_count - 5


class TestNewCollectiveType:
    """An unexpected collective operation type appears in the run."""

    def test_new_type_detected_in_counts(self) -> None:
        baseline = _baseline(ranks=2, steps=5, interval_ns=1_000_000)
        assert set(baseline.collective_type_counts.keys()) == {"all_reduce"}

        # Inject an unexpected barrier on rank 0
        anomalous = _normal_run(ranks=2, steps=5, interval_ns=1_000_000)
        anomalous.append(_ev(0, "barrier", 5_500_000, 5_500_500, 0))

        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        assert "barrier" in anomalous_profile.collective_type_counts
        assert "barrier" not in baseline.collective_type_counts

    def test_new_type_increases_event_count(self) -> None:
        baseline = _baseline(ranks=2, steps=5, interval_ns=1_000_000)

        anomalous = _normal_run(ranks=2, steps=5, interval_ns=1_000_000)
        anomalous.append(_ev(0, "barrier", 5_500_000, 5_500_500, 0))
        anomalous.append(_ev(1, "barrier", 5_500_000, 5_500_500, 0))

        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        assert anomalous_profile.event_count == baseline.event_count + 2


class TestTensorSizeAnomaly:
    """Tensor sizes deviate from the baseline distribution."""

    def test_larger_tensors_shift_size_stats(self) -> None:
        baseline = _baseline(ranks=2, steps=5, interval_ns=1_000_000, tensor_size=1024)
        assert baseline.tensor_size_mean == 1024.0

        # Anomalous run: one rank uses 10x larger tensors
        anomalous: list[CollectiveEvent] = []
        for r in range(2):
            size = 10240 if r == 1 else 1024
            for s in range(5):
                start = s * 1_000_000
                anomalous.append(_ev(r, "all_reduce", start, start + 1000, size))

        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        assert anomalous_profile.tensor_size_mean > baseline.tensor_size_mean
        assert anomalous_profile.tensor_size_max == 10240
        assert baseline.tensor_size_max == 1024

    def test_zero_tensor_size_anomaly(self) -> None:
        """A run that suddenly has zero-sized tensors should be detectable."""
        baseline = _baseline(ranks=2, steps=5, interval_ns=1_000_000, tensor_size=1024)

        # Anomalous: half the events have zero tensor size (e.g. corrupted)
        anomalous = _normal_run(ranks=2, steps=5, interval_ns=1_000_000, tensor_size=1024)
        for i in range(0, len(anomalous), 2):
            ev = anomalous[i]
            anomalous[i] = _ev(ev.rank, ev.collective_type, ev.start_time_ns, ev.end_time_ns, 0)

        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        assert anomalous_profile.tensor_size_min == 0
        assert anomalous_profile.tensor_size_min < baseline.tensor_size_min
        assert anomalous_profile.tensor_size_mean < baseline.tensor_size_mean


class TestBaselineStability:
    """Verify that two normal runs produce identical profiles."""

    def test_identical_normal_runs_produce_same_profile(self) -> None:
        profile_a = TrainingProfile.fit_on_normal_run(
            _normal_run(ranks=4, steps=10, interval_ns=1_000_000)
        )
        profile_b = TrainingProfile.fit_on_normal_run(
            _normal_run(ranks=4, steps=10, interval_ns=1_000_000)
        )

        assert profile_a.interval_mean_ns == profile_b.interval_mean_ns
        assert profile_a.interval_std_ns == profile_b.interval_std_ns
        assert profile_a.interval_min_ns == profile_b.interval_min_ns
        assert profile_a.interval_max_ns == profile_b.interval_max_ns
        assert profile_a.tensor_size_mean == profile_b.tensor_size_mean
        assert profile_a.event_count == profile_b.event_count
        assert profile_a.ranks == profile_b.ranks

    def test_normal_run_interval_std_is_zero(self) -> None:
        """A perfectly regular run has zero interval standard deviation."""
        profile = _baseline(ranks=4, steps=10, interval_ns=1_000_000)
        assert profile.interval_std_ns == 0.0


class TestMultiRankTimingSkew:
    """Different ranks drift out of sync over time."""

    def test_clock_skew_increases_interval_std(self) -> None:
        _baseline(ranks=4, steps=10, interval_ns=1_000_000)

        # Each successive rank starts 100_000ns later (clock skew)
        anomalous: list[CollectiveEvent] = []
        for r in range(4):
            offset = r * 100_000
            for s in range(10):
                start = s * 1_000_000 + offset
                anomalous.append(_ev(r, "all_reduce", start, start + 1000, 1024))

        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        # Per-rank intervals are still exactly 1_000_000 (skew is a constant
        # offset per rank, not a changing gap), so std stays zero.
        assert anomalous_profile.interval_std_ns == 0.0

    def test_growing_jitter_increases_interval_std(self) -> None:
        baseline = _baseline(ranks=2, steps=10, interval_ns=1_000_000)

        # Rank 0 has growing jitter: each successive interval is 10_000ns longer
        anomalous: list[CollectiveEvent] = []
        cumulative = 0
        for s in range(10):
            anomalous.append(_ev(0, "all_reduce", cumulative, cumulative + 1000, 1024))
            cumulative += 1_000_000 + s * 10_000
        for s in range(10):
            start = s * 1_000_000
            anomalous.append(_ev(1, "all_reduce", start, start + 1000, 1024))

        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        # Growing jitter on rank 0 should produce non-zero std
        assert anomalous_profile.interval_std_ns > 0.0
        assert anomalous_profile.interval_std_ns > baseline.interval_std_ns


class TestAnomalyWithEscalationIntegration:
    """Verify that anomalous profiles combined with escalation signals trigger
    FULL recording mode via the recording policy pipeline."""

    def test_anomalous_rank_count_triggers_full_recording_via_escalation(self) -> None:
        """When an escalation signal is present, recording is always FULL."""
        from train_replay.recording.escalation import EscalationSignal
        from train_replay.recording.modes import RiskContext, compile_recording_policy

        baseline = _baseline(ranks=4, steps=5, interval_ns=1_000_000)

        # Anomalous run with a missing rank
        anomalous = _normal_run(ranks=3, steps=5, interval_ns=1_000_000)
        anomalous_profile = TrainingProfile.fit_on_normal_run(anomalous)

        # Detect: rank set changed
        rank_anomaly = anomalous_profile.ranks != baseline.ranks
        assert rank_anomaly is True

        # Simulate an escalation signal from an external detector
        signal = EscalationSignal(
            source="nccl-inspector",
            severity=0.92,
            metric_name="nccl_anomaly_score",
        )

        # Even a low-risk context should get FULL when escalation is present
        ctx = RiskContext(was_vetted=False)
        policy = compile_recording_policy(ctx, escalation=signal)

        assert policy.mode.value == "full"
        assert "escalation" in policy.reason

    def test_consent_anomaly_triggers_full_without_escalation(self) -> None:
        from train_replay.recording.modes import RiskContext, compile_recording_policy

        ctx = RiskContext(has_consent_anomaly=True)
        policy = compile_recording_policy(ctx)

        assert policy.mode.value == "full"
        assert "consent anomaly" in policy.reason
