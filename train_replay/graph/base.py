"""Abstract interfaces for backend-agnostic graph construction.

The graph layer must support NCCL, Gloo, MTIA, and future backends without
code duplication.  This module defines two protocols:

- ``BackendEvent`` — the minimal contract any backend event must satisfy.
- ``GraphBuilder`` — the contract for converting a list of events into a
  ``ProvGraph``.

Concrete implementations live in backend-specific modules (e.g. the existing
``NCCLGraphBuilder`` in ``builder.py``).  The ``ProvGraph`` itself is already
backend-agnostic — it models PROV-DM Activity / Entity / Agent nodes and is
queried identically regardless of which backend produced the events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .prov_graph import ProvGraph


@runtime_checkable
class BackendEvent(Protocol):
    """Minimal interface for any distributed-training backend event.

    Concrete implementations: ``CollectiveEvent`` (NCCL), future ``GlooEvent``,
    ``MTIAEvent``, etc.  Field names are intentionally generic — the NCCL
    concept of "collective type" maps to ``operation_type``, and "process group"
    maps to ``group_id``.
    """

    @property
    def rank(self) -> int:
        """Rank (worker) that emitted this event."""

    @property
    def operation_type(self) -> str:
        """Name of the distributed operation (e.g. 'all_reduce', 'recv')."""

    @property
    def sequence_id(self) -> int:
        """Monotonic sequence number within the rank's trace."""

    @property
    def timestamp_ns(self) -> int:
        """Start time of the operation in nanoseconds."""

    @property
    def group_id(self) -> str:
        """Identifier for the communication group (process group, team, etc.)."""


@runtime_checkable
class GraphBuilder(Protocol):
    """Builds a ``ProvGraph`` from a list of backend events.

    Implementations are backend-specific (NCCL, Gloo, MTIA) but produce the
    same ``ProvGraph`` type so downstream code (recording, replay, signing) is
    fully backend-agnostic.
    """

    def build(self, events: list[BackendEvent]) -> ProvGraph:
        """Construct a cross-rank PROV-DM graph from the given events."""
        ...
