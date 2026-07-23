"""Anomaly detection for AEP training evidence.

Provides :class:`AnomalyDetector`, an abstract base class for detectors that
consume :class:`~train_replay.recording.evidence.AEPRecord` lists and emit
:class:`AnomalySignal` results, plus :class:`StatisticalAnomalyDetector`, a
Z-score-based implementation operating on event timing and tensor delta
statistics.

Implements the core detection contract specified in issue #288 (Milestone 5).
Timing analysis operates on inter-event intervals per rank; delta-stat
analysis operates on per-key numeric values from ``AEPRecord.delta_stats``.
Both analyses flag statistical outliers via absolute Z-score thresholding and
detect degenerate cases (zero-variance sequences, duplicate timestamps).

An Isolation Forest backend can be added behind the same ABC once
``scikit-learn`` is an optional dependency.
"""

from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnomalySignal:
    """A single anomaly detected in training evidence.

    Attributes:
        action_id: The AEP record that triggered the signal.
        rank: Rank on which the anomaly was observed.
        step: Training step at time of detection.
        metric_name: Which metric crossed the threshold (e.g.
            ``"timing_zscore"``, ``"delta_zscore:loss"``,
            ``"zero_variance:timing"``, ``"duplicate_timestamp"``).
        score: Raw statistical score (e.g. absolute Z-score value).
        severity: Normalised severity in [0, 1].
        description: Human-readable explanation of the anomaly.
        extra: Optional metadata.
    """

    action_id: str
    rank: int
    step: int
    metric_name: str
    score: float
    severity: float
    description: str
    extra: dict[str, Any] = field(default_factory=dict)


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors.

    Subclasses implement :meth:`detect` to analyse a list of
    :class:`~train_replay.recording.evidence.AEPRecord` instances and return
    zero or more :class:`AnomalySignal` results.
    """

    @abstractmethod
    def detect(
        self,
        events: list[Any],  # list[AEPRecord] — avoiding forward ref indirection
    ) -> list[AnomalySignal]:
        """Analyse *events* and return detected anomalies."""


class StatisticalAnomalyDetector(AnomalyDetector):
    """Z-score anomaly detector on event timing and tensor delta statistics.

    Two independent analyses run over the supplied event list:

    1. **Timing Z-score** — inter-event intervals (nanoseconds between
       consecutive ``timestamp_ns`` values, per rank) are checked for outliers.
       Zero-time intervals (duplicate timestamps on the same rank) and
       zero-variance sequences (frozen clock) are also flagged.
    2. **Delta-stat Z-score** — for each key present in ``delta_stats``
       dictionaries across all events, a per-key Z-score is computed.
       Zero-variance sequences (all values identical) emit a signal indicating
       a potentially stuck metric.

    An interval or value whose absolute Z-score exceeds *z_threshold* triggers
    an :class:`AnomalySignal`.  Severity is the absolute Z-score clamped to 1.

    Parameters:
        z_threshold: Absolute Z-score threshold (default ``3.0``).
    """

    def __init__(self, z_threshold: float = 3.0) -> None:
        if z_threshold <= 0:
            raise ValueError(f"z_threshold must be positive, got {z_threshold}")
        self._z_threshold = z_threshold

    # -- public interface -----------------------------------------------------

    def detect(
        self,
        events: list[Any],
    ) -> list[AnomalySignal]:
        """Return anomaly signals for statistically outlying events."""
        # Keep only events that carry a timestamp_ns attribute.  After this
        # filter, direct attribute access is safe for timestamp_ns — we never
        # fall back to a misleading default of 0.
        ts_events = [e for e in events if hasattr(e, "timestamp_ns")]
        signals: list[AnomalySignal] = []

        signals.extend(self._detect_timing_anomalies(ts_events))
        signals.extend(self._detect_delta_stat_anomalies(ts_events))

        return signals

    # -- timing analysis -------------------------------------------------------

    def _detect_timing_anomalies(
        self,
        events: list[Any],
    ) -> list[AnomalySignal]:
        """Z-score outlier detection on inter-event timing intervals.

        Also flags duplicate timestamps (zero intervals) and zero-variance
        sequences (frozen clock) as anomalies.
        """
        signals: list[AnomalySignal] = []

        # Group by rank, compute inter-event intervals.
        by_rank: dict[int, list[Any]] = {}
        for evt in events:
            rank = getattr(evt, "rank", 0)
            by_rank.setdefault(rank, []).append(evt)

        intervals: list[float] = []
        interval_map: dict[float, tuple[Any, Any]] = {}  # interval -> (prev, curr)
        duplicate_pairs: list[tuple[Any, Any]] = []

        for rank_events in by_rank.values():
            sorted_evts = sorted(
                rank_events,
                key=lambda e: e.timestamp_ns,
            )
            for prev, curr in zip(sorted_evts, sorted_evts[1:]):
                prev_ts = prev.timestamp_ns
                curr_ts = curr.timestamp_ns
                iv = float(curr_ts - prev_ts)
                if iv == 0:
                    # Duplicate timestamp on the same rank — potential event
                    # duplication or clock issue.  Flag rather than silently
                    # dropping.
                    duplicate_pairs.append((prev, curr))
                elif iv > 0:
                    intervals.append(iv)
                    interval_map[iv] = (prev, curr)

        # Emit signals for duplicate timestamps.
        for prev, curr in duplicate_pairs:
            rank = getattr(curr, "rank", 0)
            step = getattr(curr, "step", 0)
            action_id = getattr(curr, "action_id", "")
            signals.append(AnomalySignal(
                action_id=action_id,
                rank=rank,
                step=step,
                metric_name="duplicate_timestamp",
                score=0.0,
                severity=0.5,
                description=(
                    f"Duplicate timestamp on rank {rank} at step {step}: "
                    f"two consecutive events share the same timestamp_ns"
                ),
            ))

        if len(intervals) < 2:
            return signals

        # Check for zero-variance intervals (frozen clock).
        try:
            stdev_iv = statistics.stdev(intervals)
        except statistics.StatisticsError:
            stdev_iv = 0.0

        if stdev_iv == 0.0 and len(intervals) >= 3:
            # All intervals are identical — frozen clock, no variation.
            # Pick the last event as anchor for the signal.
            _iv, (_prev, anchor_evt) = next(
                reversed(list(interval_map.items()))
            )
            rank = getattr(anchor_evt, "rank", 0)
            step = getattr(anchor_evt, "step", 0)
            action_id = getattr(anchor_evt, "action_id", "")
            mean_iv = statistics.mean(intervals)
            signals.append(AnomalySignal(
                action_id=action_id,
                rank=rank,
                step=step,
                metric_name="zero_variance:timing",
                score=0.0,
                severity=0.3,
                description=(
                    f"All {len(intervals)} inter-event intervals on rank {rank} "
                    f"are identical ({mean_iv:.0f} ns), indicating zero timing variance"
                ),
            ))
            return signals

        zscores = _zscore_list(intervals)
        for iv, zs in zip(intervals, zscores):
            if zs is not None and abs(zs) > self._z_threshold:
                prev, curr = interval_map[iv]
                rank = getattr(curr, "rank", 0)
                step = getattr(curr, "step", 0)
                action_id = getattr(curr, "action_id", "")
                severity = min(abs(zs), 1.0)
                signals.append(AnomalySignal(
                    action_id=action_id,
                    rank=rank,
                    step=step,
                    metric_name="timing_zscore",
                    score=abs(zs),
                    severity=severity,
                    description=(
                        f"Inter-event interval {iv:.0f} ns on rank {rank} at step "
                        f"{step} has Z-score {zs:.2f} (threshold {self._z_threshold})"
                    ),
                ))

        return signals

    # -- delta-stat analysis ---------------------------------------------------

    def _detect_delta_stat_anomalies(
        self,
        events: list[Any],
    ) -> list[AnomalySignal]:
        """Z-score outlier detection on per-key delta statistics.

        When all values for a key are identical (zero variance), a signal is
        emitted indicating a potentially stuck metric, rather than silently
        skipping.
        """
        signals: list[AnomalySignal] = []

        # Collect all delta_stats dicts.
        stat_entries: list[tuple[Any, str, float]] = []  # (event, key, value)
        for evt in events:
            ds = getattr(evt, "delta_stats", None)
            if not isinstance(ds, dict):
                continue
            for key, val in ds.items():
                if isinstance(val, (int, float)):
                    stat_entries.append((evt, str(key), float(val)))

        if not stat_entries:
            return signals

        # Group by key.
        by_key: dict[str, list[tuple[Any, float]]] = {}
        for evt, key, val in stat_entries:
            by_key.setdefault(key, []).append((evt, val))

        for key, entries in by_key.items():
            values = [v for _, v in entries]
            if len(values) < 2:
                continue

            # Check for zero variance — all values identical.
            try:
                stdev_val = statistics.stdev(values)
            except statistics.StatisticsError:
                stdev_val = 0.0

            if stdev_val == 0.0 and len(values) >= 3:
                # Stuck metric: all values are the same across enough samples.
                anchor_evt, _ = entries[-1]
                rank = getattr(anchor_evt, "rank", 0)
                step = getattr(anchor_evt, "step", 0)
                action_id = getattr(anchor_evt, "action_id", "")
                mean_val = statistics.mean(values)
                signals.append(AnomalySignal(
                    action_id=action_id,
                    rank=rank,
                    step=step,
                    metric_name=f"zero_variance:delta:{key}",
                    score=0.0,
                    severity=0.3,
                    description=(
                        f"Delta stat '{key}' has zero variance across "
                        f"{len(values)} samples (all = {mean_val}), "
                        f"indicating a potentially stuck metric"
                    ),
                ))
                continue  # No point running Z-score on zero-variance data.

            zscores = _zscore_list(values)
            for (evt, _val), zs in zip(entries, zscores):
                if zs is not None and abs(zs) > self._z_threshold:
                    rank = getattr(evt, "rank", 0)
                    step = getattr(evt, "step", 0)
                    action_id = getattr(evt, "action_id", "")
                    severity = min(abs(zs), 1.0)
                    signals.append(AnomalySignal(
                        action_id=action_id,
                        rank=rank,
                        step=step,
                        metric_name=f"delta_zscore:{key}",
                        score=abs(zs),
                        severity=severity,
                        description=(
                            f"Delta stat '{key}' on rank {rank} at step {step} "
                            f"has Z-score {zs:.2f} (threshold {self._z_threshold})"
                        ),
                    ))

        return signals


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------


def _zscore_list(values: list[float]) -> list[float | None]:
    """Return Z-scores for *values*, or ``None`` where the std-dev is zero.

    Callers that need to handle the zero-variance case should inspect stdev
    themselves *before* calling this helper, or check the returned list for
    ``None`` entries.
    """
    mean = statistics.mean(values)
    if len(values) < 2:
        return [None] * len(values)
    try:
        stdev = statistics.stdev(values)
    except statistics.StatisticsError:
        stdev = 0.0

    if stdev == 0.0:
        return [None] * len(values)

    return [(v - mean) / stdev for v in values]