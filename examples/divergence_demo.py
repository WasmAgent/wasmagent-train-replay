"""Differential divergence demo — compare two pre-recorded runs rank-by-rank.

Usage:
    python examples/divergence_demo.py

This demo (Milestone 6, "Differential Divergence Replay"):

1. Synthesizes two Flight Recorder dumps — a *baseline* run and a *candidate*
   run that diverges from the baseline on rank 1 at step 1 (collective type
   ``all_reduce`` becomes ``all_gather``).
2. Writes both dumps to a temporary directory as pickle files in the native
   Flight Recorder format consumed by :func:`load_flight_recorder`.
3. Invokes :meth:`EpochReplayer.replay_diff` over the two dump paths.
4. Prints the per-rank :class:`DivergenceReport` to stdout.

The two dumps are byte-identical except for the injected divergence, so the
report should flag exactly rank 1 with ``first_divergence_step == 1`` and leave
the aligned ranks (0, 2, 3) absent from the report.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

from train_replay.collector.flight_recorder import CollectiveEvent
from train_replay.graph.builder import build_from_events
from train_replay.replay.replayer import EpochReplayer
from train_replay.replay.types import DivergenceReport

RANKS = 4
STEPS = 3
DIVERGENT_RANK = 1
DIVERGENT_STEP = 1
DIVERGENT_CANDIDATE_TYPE = "all_gather"


def make_events(
    ranks: int = RANKS,
    steps: int = STEPS,
    divergent_ctype: str | None = None,
) -> list[CollectiveEvent]:
    """Synthesize ``ranks`` × ``steps`` aligned all-reduce events.

    When ``divergent_ctype`` is set, the event at rank ``DIVERGENT_RANK`` /
    step ``DIVERGENT_STEP`` is emitted with that collective type instead of
    ``all_reduce`` — this is the injected divergence between baseline and
    candidate runs.
    """
    events: list[CollectiveEvent] = []
    for rank in range(ranks):
        for seq in range(steps):
            ctype = "all_reduce"
            if divergent_ctype is not None and rank == DIVERGENT_RANK and seq == DIVERGENT_STEP:
                ctype = divergent_ctype
            events.append(CollectiveEvent(
                rank=rank,
                process_group="default",
                collective_type=ctype,
                src_rank=None,
                dst_rank=None,
                tensor_size=1024 * 1024,
                enqueue_time_ns=seq * 1_000_000,
                start_time_ns=seq * 1_000_000 + 100_000,
                end_time_ns=seq * 1_000_000 + 500_000,
                sequence_id=seq,
            ))
    return events


def write_dump(events: list[CollectiveEvent], path: Path) -> Path:
    """Serialize events to a Flight Recorder pickle dump at *path*.

    The dump is plain built-in data only (``dict``/``list``/``int``/``str``/
    ``None``), so :class:`load_flight_recorder`'s restricted unpickler accepts
    it without executing arbitrary code — exactly like a genuine NCCL trace.
    """
    entries = [
        {
            "rank": evt.rank,
            "pg_name": evt.process_group,
            "collective_seq": evt.collective_type,
            "p2p_src": evt.src_rank,
            "p2p_dst": evt.dst_rank,
            "input_sizes": [[evt.tensor_size]],
            "time_created_ns": evt.enqueue_time_ns,
            "time_started_ns": evt.start_time_ns,
            "time_finished_ns": evt.end_time_ns,
            "frames": evt.call_stack,
            "seq_id": evt.sequence_id,
        }
        for evt in events
    ]
    with open(path, "wb") as f:
        pickle.dump({"entries": entries}, f)
    return path


def print_report(report: DivergenceReport) -> None:
    print(f"  rank {report.rank}: first divergence at step {report.first_divergence_step}")
    b = report.baseline_action
    c = report.candidate_action
    print(
        f"    baseline:  rank={b.rank} step={b.step}"
        f" type={b.collective_type} mode={b.recording_mode}"
    )
    print(
        f"    candidate: rank={c.rank} step={c.step}"
        f" type={c.collective_type} mode={c.recording_mode}"
    )
    if report.context_window:
        window = ", ".join(
            f"step={a.step}/type={a.collective_type}" for a in report.context_window
        )
        print(f"    context_window: [{window}]")
    else:
        print("    context_window: []")
    print(f"    correlated_collision: {report.correlated_collision}")
    print(f"    correlated_escalation: {report.correlated_escalation}")


def main() -> None:
    baseline_events = make_events()
    candidate_events = make_events(divergent_ctype=DIVERGENT_CANDIDATE_TYPE)

    with tempfile.TemporaryDirectory() as tmp:
        baseline_dump = write_dump(baseline_events, Path(tmp) / "baseline.pkl")
        candidate_dump = write_dump(candidate_events, Path(tmp) / "candidate.pkl")

        # The causal graph is the standard EpochReplayer construction; replay_diff
        # compares the two dumps' action streams directly.
        graph = build_from_events(baseline_events)
        replayer = EpochReplayer(graph)

        print("Differential divergence replay")
        print(f"  baseline:  {baseline_dump.name} ({len(baseline_events)} events)")
        print(f"  candidate: {candidate_dump.name} ({len(candidate_events)} events)")
        print()

        reports = replayer.replay_diff(str(baseline_dump), str(candidate_dump))

        if not reports:
            print("No divergence detected — runs are byte-identical.")
            return

        print(f"Diverged ranks: {sorted(reports)}")
        for rank in sorted(reports):
            print_report(reports[rank])
        print()
        print(f"Summary: {len(reports)} rank(s) diverged out of {RANKS}.")


if __name__ == "__main__":
    main()
