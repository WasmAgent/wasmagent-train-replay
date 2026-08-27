"""
Fault injection demo — validate the causal graph catches injected gradient corruption.

Usage:
    python examples/fault_injection_demo.py

This demo:
1. Synthesizes Flight Recorder events for a 4-rank all-reduce training step
2. Injects a simulated corruption at rank 2, sequence 3
3. Builds the causal graph
4. Escalates recording mode for the suspect rank
5. Traces the corrupted output tensor back to its causal ancestors
6. Prints a before/after divergence report (baseline vs corrupted dump)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from divergence_demo import write_dump

from train_replay.collector.flight_recorder import CollectiveEvent
from train_replay.graph.builder import build_from_events
from train_replay.recording.modes import RiskContext, SideEffectClass
from train_replay.recording.recorder import EpochRecorder
from train_replay.replay.replayer import EpochReplayer


def make_synthetic_events(ranks: int = 4, steps: int = 5) -> list[CollectiveEvent]:
    events = []
    for rank in range(ranks):
        for seq in range(steps):
            events.append(CollectiveEvent(
                rank=rank,
                process_group="default",
                collective_type="all_reduce",
                src_rank=None,
                dst_rank=None,
                tensor_size=1024 * 1024,
                enqueue_time_ns=seq * 1_000_000,
                start_time_ns=seq * 1_000_000 + 100_000,
                end_time_ns=seq * 1_000_000 + 500_000,
                sequence_id=seq,
            ))
    return events


def main() -> None:
    events = make_synthetic_events(ranks=4, steps=5)
    graph = build_from_events(events)
    recorder = EpochRecorder(run_id="demo-run", epoch=0)
    for evt in events:
        # Inject corruption signal: rank 2, seq 3 has anomalous taint
        if evt.rank == 2 and evt.sequence_id == 3:
            risk = RiskContext(
                was_vetted=True,
                side_effect_class=SideEffectClass.MUTATE_EXTERNAL,
            )
        else:
            risk = None
        recorder.record_collective(evt, risk)

    # Escalate rank 2 to FULL recording
    recorder.escalate_rank(2)

    bundle = recorder.bundle()
    replayer = EpochReplayer(graph)

    print(f"Total actions recorded: {len(bundle.actions)}")
    suspicious = replayer.suspicious_actions(bundle)
    print(f"Suspicious (FULL mode) actions: {len(suspicious)}")
    for a in suspicious:
        print(f"  rank={a.rank} step={a.step} type={a.collective_type} mode={a.recording_mode}")

    # Trace the corrupted output entity
    entity_id = "tensor:2:3:out"
    ancestors = replayer.find_root_cause(entity_id)
    print(f"\nCausal ancestors of {entity_id}:")
    for anc in ancestors:
        print(f"  {anc}")

    print(f"\nBundle digest: {bundle.digest()}")

    # Before/after divergence report: diff the clean baseline against a
    # candidate dump whose rank-2/step-3 collective shows the fault's
    # observable signature (all_reduce became all_gather).
    candidate_events = [
        (
            _corrupted_copy(evt) if (evt.rank == 2 and evt.sequence_id == 3)
            else evt
        )
        for evt in events
    ]
    with tempfile.TemporaryDirectory() as tmp:
        baseline_dump = write_dump(events, Path(tmp) / "baseline.pkl")
        candidate_dump = write_dump(candidate_events, Path(tmp) / "candidate.pkl")
        reports = EpochReplayer(graph).replay_diff(
            str(baseline_dump), str(candidate_dump)
        )

    print("\nBefore/after divergence report:")
    for rank in sorted(reports):
        report = reports[rank]
        print(
            f"  rank {rank}: first_divergence_step={report.first_divergence_step}"
            f" similarity={report.per_rank_similarity[rank]}"
        )
        print(f"    {report.summary}")


def _corrupted_copy(evt: CollectiveEvent) -> CollectiveEvent:
    """Return a copy of *evt* whose collective type shows the injected fault."""
    return CollectiveEvent(
        rank=evt.rank,
        process_group=evt.process_group,
        collective_type="all_gather",
        src_rank=evt.src_rank,
        dst_rank=evt.dst_rank,
        tensor_size=evt.tensor_size,
        enqueue_time_ns=evt.enqueue_time_ns,
        start_time_ns=evt.start_time_ns,
        end_time_ns=evt.end_time_ns,
        sequence_id=evt.sequence_id,
    )


if __name__ == "__main__":
    main()
