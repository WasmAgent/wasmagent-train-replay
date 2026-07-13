"""Abstract cross-rank collision / desync detection protocol.

Each communication backend may exhibit desync differently:
- NCCL: collective sequence numbers diverge between ranks.
- Gloo: operation completion timestamps drift beyond a threshold.
- MTIA: firmware-level watchdog flags.

This module defines the **interface** only.  Backend-specific implementations
plug in via the ``CollisionDetector`` protocol without modifying the graph core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class CollisionSeverity(str, Enum):
    """Severity of a detected desync or collision."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CollisionEvent:
    """One detected cross-rank alignment anomaly.

    Attributes:
        severity: How serious the anomaly is.
        rank_a: First rank involved.
        rank_b: Second rank involved (``None`` for single-rank issues).
        sequence_id: Sequence number where the collision was detected.
        description: Human-readable description of the anomaly.
        backend: Which backend reported the collision.
        details: Optional backend-specific key-value context.
    """

    severity: CollisionSeverity
    rank_a: int
    sequence_id: int
    description: str
    backend: str
    rank_b: int | None = None
    details: dict[str, str] = field(default_factory=dict)


class CollisionDetector(ABC):
    """Protocol for backend-specific desync / collision detection.

    Subclass this per backend and pass instances to the graph or replay layer.
    The detector receives a timeline of per-rank operations and returns any
    alignment anomalies it finds.
    """

    @abstractmethod
    def detect(
        self,
        timelines: dict[int, list[tuple[int, str]]],
        *,
        tolerance_ns: int = 0,
    ) -> list[CollisionEvent]:
        """Detect cross-rank collisions in the given timelines.

        Args:
            timelines: Mapping of rank → list of (sequence_id, collective_type)
                pairs, sorted by sequence_id.
            tolerance_ns: Nanosecond tolerance for timestamp-based backends
                (ignored by sequence-number backends).

        Returns:
            List of detected anomalies, sorted by sequence_id.
        """
        ...
