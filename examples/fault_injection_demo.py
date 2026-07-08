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
"""

from __future__ import annotations
from train_replay.collector.flight_recorder import CollectiveEvent
from train_replay.graph.builder import build_from_events
from train_replay.recording.recorder import EpochRecorder
from train_replay.recording.modes import RiskContext, SideEffectClass
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


if __name__ == "__main__":
    main()
