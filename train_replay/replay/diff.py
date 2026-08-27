"""DivergenceReplayer — pairwise comparison of two replay result streams."""

from __future__ import annotations

from dataclasses import dataclass

from ..graph.collision import Collision, CollisionReport
from ..recording.escalation import EscalationSignal
from ..recording.evidence import AEPRecord
from ..recording.modes import RecordingMode
from .replayer import ReplayResult
from .types import Divergence, DivergenceReport


@dataclass
class DiffConfig:
    context_window_size: int = 5


class DivergenceReplayer:
    """Accept two ReplayResult objects and emit divergence details.

    Walks rank-aligned action streams and finds the first step where
    actions disagree, emitting a context window of N steps on either side.
    """

    def __init__(self, config: DiffConfig | None = None) -> None:
        self._cfg = config or DiffConfig()

    def diff(
        self,
        baseline: ReplayResult,
        candidate: ReplayResult,
        escalation_signals: list[EscalationSignal] | None = None,
    ) -> DivergenceReport:
        """Compare two ReplayResult objects and return a DivergenceReport.

        Args:
            baseline: The reference replay result.
            candidate: The candidate replay result to compare against baseline.
            escalation_signals: Optional external escalation signals.  When an
                EscalationSignal's timestamp is not available (EscalationSignal
                has no timestamp field), all signals are considered to overlap
                the divergence step and the first one is used for correlation.

        Returns:
            DivergenceReport with per-rank divergences and overall summary.
        """
        baseline_by_step = {a.step: a for a in baseline.suspicious_actions}
        candidate_by_step = {a.step: a for a in candidate.suspicious_actions}

        all_steps = sorted(
            set(baseline_by_step) | set(candidate_by_step)
        )

        divergences: list[Divergence] = []
        first_divergence_step: int | None = None
        collision_by_step = _collisions_by_step(
            baseline.collision_report, candidate.collision_report
        )

        for step in all_steps:
            b_action = baseline_by_step.get(step)
            c_action = candidate_by_step.get(step)
            collision_hit, collision_rec = _collision_correlation(
                collision_by_step, step, baseline.rank
            )

            if b_action is None or c_action is None:
                # One stream has an action the other doesn't — that's a divergence.
                if first_divergence_step is None:
                    first_divergence_step = step

                # Use a placeholder AEPRecord for the missing side.
                missing = _missing_record(baseline.rank, step)
                div = Divergence(
                    rank=baseline.rank,
                    first_divergence_step=step,
                    baseline_action=b_action if b_action is not None else missing,
                    candidate_action=c_action if c_action is not None else missing,
                    context_window=self._context_window(
                        all_steps, step, baseline_by_step, candidate_by_step
                    ),
                    correlated_collision=collision_hit,
                    collision_record=collision_rec,
                    correlated_escalation=_correlate_escalation(escalation_signals),
                )
                divergences.append(div)
                break

            if not _actions_agree(b_action, c_action):
                if first_divergence_step is None:
                    first_divergence_step = step

                div = Divergence(
                    rank=baseline.rank,
                    first_divergence_step=step,
                    baseline_action=b_action,
                    candidate_action=c_action,
                    context_window=self._context_window(
                        all_steps, step, baseline_by_step, candidate_by_step
                    ),
                    correlated_collision=collision_hit,
                    collision_record=collision_rec,
                    correlated_escalation=_correlate_escalation(escalation_signals),
                )
                divergences.append(div)
                break

        # Similarity: fraction of steps that agree.
        n_total = len(all_steps)
        n_agree = n_total - len(divergences)
        similarity = n_agree / n_total if n_total > 0 else 1.0

        if divergences:
            summary = (
                f"Rank {baseline.rank} diverged at step {first_divergence_step}"
            )
        else:
            summary = f"Rank {baseline.rank}: no divergence detected"

        return DivergenceReport(
            first_divergence_step=first_divergence_step,
            divergences=divergences,
            per_rank_similarity={baseline.rank: round(similarity, 4)},
            summary=summary,
        )

    def _context_window(
        self,
        all_steps: list[int],
        pivot_step: int,
        baseline_by_step: dict[int, AEPRecord],
        candidate_by_step: dict[int, AEPRecord],
    ) -> list[AEPRecord]:
        """Return up to N baseline actions around the divergence step."""
        n = self._cfg.context_window_size
        pivot_idx = all_steps.index(pivot_step)
        lo = max(0, pivot_idx - n)
        hi = min(len(all_steps), pivot_idx + n + 1)
        window: list[AEPRecord] = []
        for step in all_steps[lo:hi]:
            if step == pivot_step:
                continue
            if step in baseline_by_step:
                window.append(baseline_by_step[step])
        return window


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _actions_agree(a: AEPRecord, b: AEPRecord) -> bool:
    """Two actions agree if type, mode and tensor digests all match.

    Tensor digests participate only when both sides carry them, so streams
    recorded without digests keep comparing on collective metadata alone,
    while a mutated tensor value on one side is flagged as a divergence.
    """
    if a.collective_type != b.collective_type or a.recording_mode != b.recording_mode:
        return False
    for attr in ("tensor_input_digest", "tensor_output_digest"):
        a_digest = getattr(a, attr)
        b_digest = getattr(b, attr)
        if a_digest is not None and b_digest is not None and a_digest != b_digest:
            return False
    return True


def _missing_record(rank: int, step: int) -> AEPRecord:
    return AEPRecord(
        action_id=f"missing-rank{rank}-step{step}",
        rank=rank,
        step=step,
        collective_type="<missing>",
        recording_mode=RecordingMode.VALIDATION,
    )


def _correlate_escalation(
    signals: list[EscalationSignal] | None,
) -> str | None:
    """Return a correlation string for the first available escalation signal."""
    if not signals:
        return None
    s = signals[0]
    return f"{s.metric_name}={s.severity}"


def _collisions_by_step(
    baseline_report: CollisionReport | None,
    candidate_report: CollisionReport | None,
) -> dict[int, Collision]:
    """Index collisions from either side's report by the step they occurred at.

    When both sides report a collision at the same step the baseline's entry
    wins — the two describe the same desync.
    """
    by_step: dict[int, Collision] = {}
    for report in (baseline_report, candidate_report):
        if report is None:
            continue
        for collision in report.collisions:
            by_step.setdefault(collision.step, collision)
    return by_step


def _collision_correlation(
    collision_by_step: dict[int, Collision],
    divergence_step: int,
    rank: int,
) -> tuple[bool, AEPRecord | None]:
    """Correlate a divergence step with a reported desync collision.

    A collision correlates when it happened at the divergence step and
    involved *rank* (on either side of the desync pair).  Returns the
    ``correlated_collision`` flag and the matching synthetic desync
    ``AEPRecord`` (or ``(False, None)``).
    """
    collision = collision_by_step.get(divergence_step)
    if collision is None or rank not in (collision.rank_a, collision.rank_b):
        return False, None
    return True, _desync_record(collision)


def _desync_record(collision: Collision) -> AEPRecord:
    """Build the synthetic desync :class:`AEPRecord` for a collision.

    Mirrors the records :meth:`EpochReplayer.suspicious_actions` synthesizes
    so callers see one uniform shape.
    """
    return AEPRecord(
        action_id=f"desync-r{collision.rank_a}-r{collision.rank_b}-s{collision.step}",
        rank=collision.rank_a,
        step=collision.step,
        collective_type="desync",
        recording_mode=RecordingMode.FULL,
        delta_stats={"rank_b": float(collision.rank_b)},
        causal_chain_id=collision.detail,
    )
