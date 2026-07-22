"""Anomaly detection data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnomalySignal:
    """One detected anomaly in a training run.

    Carries an anomaly score (higher = more anomalous), a confidence value
    between 0 and 1, a human-readable description, and a reference to the
    offending event."""

    score: float
    confidence: float
    description: str
    event_rank: int
    event_step: int
    collective_type: str


__all__ = ["AnomalySignal"]
