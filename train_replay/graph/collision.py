"""Backend-specific cross-rank desync detection (CollisionDetector ABC + implementations).

Each backend (NCCL, Gloo, MTIA) exposes a different timeline shape:

- **NCCL**: sequence-number-based — each rank emits monotonically increasing
  ``sequence_id`` values; a desync shows up as a mismatch in which sequence
  IDs appear across ranks at the same logical step.
- **Gloo**: timestamp-based — events carry ``enqueue_time_ns`` / ``start_time_ns``
  values; a desync shows up when the same logical collective on different ranks
  has a timestamp delta exceeding a configurable tolerance.
- **MTIA**: placeholder for future MTIA trace format support.

The ABC lives here alongside concrete implementations so the replay layer can
import a single module and select the correct backend at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..collector.flight_recorder import CollectiveEvent


@dataclass
class Collision:
    """One detected desync between two ranks."""

    rank_a: int
    rank_b: int
    step: int
    detail: str


@dataclass
class CollisionReport:
    """Result of a collision-detection pass over a set of timelines."""

    collisions: list[Collision] = field(default_factory=list)
    total_steps_checked: int = 0

    @property
    def has_collisions(self) -> bool:
        return len(self.collisions) > 0


class CollisionDetector(ABC):
    """Abstract backend-agnostic desync detector.

    Subclasses implement backend-specific alignment checks.  The replay layer
    holds a detector and feeds it per-rank event lists; the detector returns
    a :class:`CollisionReport` summarising any desyncs found.
    """

    @abstractmethod
    def detect(
        self,
        timelines: dict[int, list[CollectiveEvent]],
    ) -> CollisionReport:
        """Run desync detection across *timelines* keyed by rank."""


class NcclCollisionDetector(CollisionDetector):
    """Sequence-number-based alignment check for NCCL backends.

    NCCL assigns monotonically increasing ``sequence_id`` values to collectives
    on each rank.  A cross-rank desync manifests when:
    1. Ranks disagree on the sequence of collective types at the same step.
    2. One rank is missing a sequence_id that another rank has.
    """

    def detect(
        self,
        timelines: dict[int, list[CollectiveEvent]],
    ) -> CollisionReport:
        if len(timelines) < 2:
            return CollisionReport(total_steps_checked=0)

        # Build per-rank maps: sequence_id -> (collective_type, rank)
        rank_maps: dict[int, dict[int, str]] = {}
        for rank, events in timelines.items():
            rank_maps[rank] = {
                evt.sequence_id: evt.collective_type for evt in events
            }

        collisions: list[Collision] = []
        ranks = sorted(timelines.keys())
        steps_checked = 0

        # Check that all ranks share the same set of sequence IDs and that
        # collective types align at each step.
        all_seq_ids: set[int] = set()
        for m in rank_maps.values():
            all_seq_ids |= m.keys()

        for seq_id in sorted(all_seq_ids):
            steps_checked += 1
            present_ranks = [r for r in ranks if seq_id in rank_maps[r]]
            if len(present_ranks) < len(ranks):
                # Some ranks are missing this step — that's a desync.
                missing_ranks = [r for r in ranks if r not in present_ranks]
                anchor_rank = present_ranks[0]
                for mr in missing_ranks:
                    collisions.append(Collision(
                        rank_a=anchor_rank,
                        rank_b=mr,
                        step=seq_id,
                        detail=f"Rank {mr} missing sequence_id {seq_id} "
                               f"present on rank {anchor_rank}",
                    ))
                continue

            # All ranks present — check collective type agreement.
            types = {r: rank_maps[r][seq_id] for r in ranks}
            first_type = types[ranks[0]]
            for r in ranks[1:]:
                if types[r] != first_type:
                    collisions.append(Collision(
                        rank_a=ranks[0],
                        rank_b=r,
                        step=seq_id,
                        detail=f"Collective type mismatch at seq {seq_id}: "
                               f"rank {ranks[0]}={first_type}, rank {r}={types[r]}",
                    ))

        return CollisionReport(
            collisions=collisions,
            total_steps_checked=steps_checked,
        )


class GlooCollisionDetector(CollisionDetector):
    """Timestamp-based alignment check for Gloo backend with configurable tolerance.

    Gloo events lack NCCL's monotonic sequence_id guarantee, so alignment is
    checked via ``start_time_ns`` proximity: for the *n*-th event on each rank
    (ordered by start time), if the max delta between any pair exceeds
    *tolerance_ns*, a collision is recorded.
    """

    def __init__(self, tolerance_ns: int = 1_000_000) -> None:
        """*tolerance_ns* is the maximum allowed inter-rank timestamp delta (default 1 ms)."""
        self._tolerance_ns = tolerance_ns

    def detect(
        self,
        timelines: dict[int, list[CollectiveEvent]],
    ) -> CollisionReport:
        if len(timelines) < 2:
            return CollisionReport(total_steps_checked=0)

        # Sort each rank's events by start_time_ns and pair them index-by-index.
        sorted_timelines: dict[int, list[CollectiveEvent]] = {}
        for rank, events in timelines.items():
            sorted_timelines[rank] = sorted(events, key=lambda e: e.start_time_ns)

        ranks = sorted(sorted_timelines.keys())
        # How many events to check — minimum length across ranks.
        min_len = min(len(sorted_timelines[r]) for r in ranks)

        collisions: list[Collision] = []
        steps_checked = 0

        for idx in range(min_len):
            steps_checked += 1
            timestamps = {
                r: sorted_timelines[r][idx].start_time_ns for r in ranks
            }
            ts_values = list(timestamps.values())
            max_delta = max(ts_values) - min(ts_values)
            if max_delta > self._tolerance_ns:
                # Find the pair with the largest gap.
                max_r, min_r = (
                    max(timestamps, key=timestamps.get),  # type: ignore[arg-type]
                    min(timestamps, key=timestamps.get),  # type: ignore[arg-type]
                )
                collisions.append(Collision(
                    rank_a=min_r,
                    rank_b=max_r,
                    step=idx,
                    detail=f"Timestamp delta {max_delta}ns at index {idx} "
                           f"exceeds tolerance {self._tolerance_ns}ns "
                           f"(rank {min_r}={timestamps[min_r]}ns, "
                           f"rank {max_r}={timestamps[max_r]}ns)",
                ))

        # Also flag length mismatches (one rank has more events than another).
        max_len = max(len(sorted_timelines[r]) for r in ranks)
        if max_len > min_len:
            longer_ranks = [r for r in ranks if len(sorted_timelines[r]) > min_len]
            shorter_ranks = [r for r in ranks if len(sorted_timelines[r]) == min_len]
            collisions.append(Collision(
                rank_a=shorter_ranks[0],
                rank_b=longer_ranks[0],
                step=min_len,
                detail=f"Event count mismatch: rank(s) {longer_ranks} have "
                       f"{max_len} events, rank(s) {shorter_ranks} have {min_len}",
            ))

        return CollisionReport(
            collisions=collisions,
            total_steps_checked=steps_checked,
        )


class MtiaCollisionDetector(CollisionDetector):
    """Placeholder for future MTIA trace format desync detection.

    Currently raises :exc:`NotImplementedError` on any call to
    :meth:`detect`.  Will be implemented once the MTIA trace schema is
    available.
    """

    def detect(
        self,
        timelines: dict[int, list[CollectiveEvent]],  # noqa: ARG002
    ) -> CollisionReport:
        msg = "MTIA collision detection is not yet implemented"
        raise NotImplementedError(msg)


def detect_collisions(
    backend: str,
    timelines: dict[int, list[CollectiveEvent]],
    *,
    tolerance_ns: int = 1_000_000,
) -> CollisionReport:
    """Factory helper: select the correct detector for *backend* and run it.

    Supported backend names (case-insensitive): ``nccl``, ``gloo``, ``mtia``.
    """
    backend_lower = backend.lower()
    if backend_lower == "nccl":
        detector: CollisionDetector = NcclCollisionDetector()
    elif backend_lower == "gloo":
        detector = GlooCollisionDetector(tolerance_ns=tolerance_ns)
    elif backend_lower == "mtia":
        detector = MtiaCollisionDetector()
    else:
        raise ValueError(f"Unknown backend: {backend!r} (expected nccl, gloo, or mtia)")
    return detector.detect(timelines)
