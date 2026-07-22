"""Baseline training-run statistics for the anomaly-detection pipeline.

A :class:`TrainingProfile` summarises a known-good ("normal") training run so
that downstream anomaly detection (Milestone 5, see ``docs/15-milestones.md``)
can flag deviations in later runs. It is built by the classmethod
:meth:`TrainingProfile.fit_on_normal_run`, which walks a sequence of
:class:`~train_replay.collector.flight_recorder.CollectiveEvent` records and
captures the three baseline-statistic families named in the milestone bullet:

* **event intervals** — inter-arrival gaps between consecutive collectives on
  each rank (``CollectiveEvent.start_time_ns``), aggregated across ranks.
* **tensor distributions** — byte-size statistics over every recorded tensor
  (``CollectiveEvent.tensor_size``).
* **collective operation patterns** — per-``collective_type`` frequency and the
  observed rank set, so a missing, extra, or new op on a later run is visible.

Only the standard library is used (``statistics`` / ``collections``) so that
building a profile needs neither ``torch`` nor ``scikit-learn``; the heavier
Z-score / Isolation-Forest scoring lives in the detector (``detector.py``, a
separate milestone bullet).
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..collector.flight_recorder import CollectiveEvent


def _summarize_int(values: Sequence[int]) -> tuple[float, float, int, int]:
    """Return ``(mean, population-std, min, max)`` for an int sequence.

    The *population* standard deviation is used so that a single observation
    yields a std of ``0.0`` rather than raising. An empty sequence returns
    ``(0.0, 0.0, 0, 0)``.
    """
    if not values:
        return 0.0, 0.0, 0, 0
    return (
        statistics.fmean(values),
        statistics.pstdev(values),
        min(values),
        max(values),
    )


@dataclass
class TrainingProfile:
    """Baseline statistics captured from a normal training run.

    Instances are normally produced by :meth:`fit_on_normal_run` rather than
    constructed directly. For an empty run every numeric field is ``0`` and the
    pattern containers are empty, so the profile degrades gracefully.
    """

    # -- Event intervals: inter-arrival gap between consecutive events (ns) ---
    interval_mean_ns: float = 0.0
    interval_std_ns: float = 0.0
    interval_min_ns: int = 0
    interval_max_ns: int = 0
    interval_count: int = 0

    # -- Tensor-size distribution over all recorded tensors (bytes) ----------
    tensor_size_mean: float = 0.0
    tensor_size_std: float = 0.0
    tensor_size_min: int = 0
    tensor_size_max: int = 0
    tensor_count: int = 0

    # -- Collective operation patterns ----------------------------------------
    collective_type_counts: dict[str, int] = field(default_factory=dict)
    ranks: frozenset[int] = frozenset()
    event_count: int = 0

    @classmethod
    def fit_on_normal_run(cls, events: Sequence[CollectiveEvent]) -> TrainingProfile:
        """Fit baseline statistics from a known-good training run.

        Args:
            events: Collective events from a normal run, e.g. the output of
                :func:`~train_replay.collector.flight_recorder.load_flight_recorder`.
                Events are grouped per rank and sorted by ``start_time_ns``
                before interval differencing, so a dump emitted out of order
                still yields non-negative inter-arrival gaps.

        Returns:
            A populated :class:`TrainingProfile`. An empty ``events`` sequence
            yields an all-zero profile.
        """
        event_list = list(events)
        if not event_list:
            return cls()

        # Event intervals: per-rank inter-arrival of start_time_ns. Grouping by
        # rank (rather than a single global timeline) avoids spurious cross-rank
        # gaps that would otherwise dominate the baseline for interleaved dumps.
        by_rank: dict[int, list[int]] = {}
        for ev in event_list:
            by_rank.setdefault(ev.rank, []).append(ev.start_time_ns)
        intervals: list[int] = []
        for rank_times in by_rank.values():
            rank_times.sort()
            for prev, cur in zip(rank_times, rank_times[1:]):
                gap = cur - prev
                if gap >= 0:
                    intervals.append(gap)
        i_mean, i_std, i_min, i_max = _summarize_int(intervals)

        # Tensor-size distribution over every recorded tensor.
        sizes = [ev.tensor_size for ev in event_list]
        s_mean, s_std, s_min, s_max = _summarize_int(sizes)

        # Collective operation patterns.
        counts = Counter(ev.collective_type for ev in event_list)
        ranks = frozenset(ev.rank for ev in event_list)

        return cls(
            interval_mean_ns=i_mean,
            interval_std_ns=i_std,
            interval_min_ns=i_min,
            interval_max_ns=i_max,
            interval_count=len(intervals),
            tensor_size_mean=s_mean,
            tensor_size_std=s_std,
            tensor_size_min=s_min,
            tensor_size_max=s_max,
            tensor_count=len(sizes),
            collective_type_counts=dict(counts),
            ranks=ranks,
            event_count=len(event_list),
        )
