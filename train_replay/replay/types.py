"""Replay-stage data types.

Hosts :class:`DivergenceReport`, the per-rank output of differential
divergence replay (Milestone 6, "Differential Divergence Replay"). A report
is emitted only when a baseline and a candidate action stream disagree, so
``baseline_action`` / ``candidate_action`` are always populated; byte-identical
replays yield an empty ``dict`` from ``replay_diff`` rather than a report with
null actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..recording.evidence import AEPRecord


@dataclass
class DivergenceReport:
    """Per-rank divergence between a baseline and a candidate replay.

    Produced by pairwise comparison of two rank-aligned action streams
    (the planned ``DivergenceReplayer`` / ``EpochReplayer.replay_diff``).
    Captures the first step at which the streams disagree, the disagreeing
    actions on either side, an N-step context window around the divergence,
    and correlation flags tying the divergence to detected desync collisions
    and external escalation signals.
    """

    rank: int
    first_divergence_step: int
    baseline_action: AEPRecord
    candidate_action: AEPRecord
    context_window: list[AEPRecord] = field(default_factory=list)
    correlated_collision: bool = False
    correlated_escalation: str | None = None
