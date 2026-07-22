"""Tests for :class:`TrainingProfile.fit_on_normal_run`."""

from __future__ import annotations

from train_replay.anomaly.profile import TrainingProfile
from train_replay.collector.flight_recorder import CollectiveEvent


def _ev(rank: int, collective_type: str, start: int, end: int, size: int) -> CollectiveEvent:
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


def test_empty_run_is_all_zero() -> None:
    profile = TrainingProfile.fit_on_normal_run([])
    assert profile.event_count == 0
    assert profile.interval_count == 0
    assert profile.tensor_count == 0
    assert profile.collective_type_counts == {}
    assert profile.ranks == frozenset()


def test_collective_type_counts_and_ranks() -> None:
    events = [
        _ev(0, "all_reduce", 0, 10, 100),
        _ev(0, "all_reduce", 20, 30, 100),
        _ev(1, "broadcast", 0, 5, 200),
    ]
    profile = TrainingProfile.fit_on_normal_run(events)
    assert profile.collective_type_counts == {"all_reduce": 2, "broadcast": 1}
    assert profile.ranks == frozenset({0, 1})
    assert profile.event_count == 3


def test_event_intervals_aggregated_across_ranks() -> None:
    # rank 0: starts 0, 20, 50 -> gaps 20, 30 ; rank 1: starts 0, 100 -> gap 100
    events = [
        _ev(0, "all_reduce", 0, 10, 8),
        _ev(0, "all_reduce", 20, 30, 8),
        _ev(0, "all_reduce", 50, 60, 8),
        _ev(1, "all_reduce", 0, 5, 8),
        _ev(1, "all_reduce", 100, 110, 8),
    ]
    profile = TrainingProfile.fit_on_normal_run(events)
    # intervals: [20, 30, 100]
    assert profile.interval_count == 3
    assert profile.interval_min_ns == 20
    assert profile.interval_max_ns == 100
    assert profile.interval_mean_ns == 50.0


def test_event_intervals_isolated_per_rank() -> None:
    # Interleaved across ranks: a global timeline would mix ranks and produce
    # spurious tiny gaps (1ns). Per-rank differencing must keep gaps at 10ns.
    events = [
        _ev(0, "all_reduce", 0, 1, 4),
        _ev(1, "all_reduce", 1, 2, 4),
        _ev(0, "all_reduce", 10, 11, 4),
        _ev(1, "all_reduce", 11, 12, 4),
    ]
    profile = TrainingProfile.fit_on_normal_run(events)
    assert profile.interval_count == 2
    assert profile.interval_min_ns == 10
    assert profile.interval_max_ns == 10


def test_tensor_size_distribution() -> None:
    events = [
        _ev(0, "all_reduce", 0, 1, 100),
        _ev(0, "all_reduce", 10, 11, 200),
        _ev(0, "all_reduce", 20, 21, 300),
    ]
    profile = TrainingProfile.fit_on_normal_run(events)
    assert profile.tensor_count == 3
    assert profile.tensor_size_min == 100
    assert profile.tensor_size_max == 300
    assert profile.tensor_size_mean == 200.0
    assert profile.tensor_size_std == 0.0 or profile.tensor_size_std > 0.0


def test_single_event_has_no_intervals() -> None:
    profile = TrainingProfile.fit_on_normal_run([_ev(0, "all_reduce", 5, 6, 42)])
    assert profile.interval_count == 0
    assert profile.interval_mean_ns == 0.0
    assert profile.interval_std_ns == 0.0
    assert profile.tensor_count == 1
    assert profile.tensor_size_mean == 42.0
    assert profile.tensor_size_std == 0.0
    assert profile.collective_type_counts == {"all_reduce": 1}
    assert profile.event_count == 1


def test_out_of_order_within_rank_is_sorted() -> None:
    # Same rank, emitted out of time order: gaps must still be non-negative.
    events = [
        _ev(0, "all_reduce", 50, 60, 8),
        _ev(0, "all_reduce", 0, 10, 8),
        _ev(0, "all_reduce", 20, 30, 8),
    ]
    profile = TrainingProfile.fit_on_normal_run(events)
    assert profile.interval_count == 2
    assert profile.interval_min_ns == 20
    assert profile.interval_max_ns == 30
