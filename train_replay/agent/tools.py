"""Agent tool dispatch for querying training replay evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def dispatch_tool(tool: str, dump_path: Path, args: dict[str, Any]) -> dict[str, Any]:
    """Run an agent tool against a Flight Recorder dump."""
    if tool == "trace_tensor":
        return trace_tensor(dump_path, args)
    raise ValueError(f"Unknown agent tool: {tool}")


def trace_tensor(dump_path: Path, args: dict[str, Any]) -> dict[str, Any]:
    """Trace causal ancestors for a tensor entity."""
    entity_id = args.get("entity_id")
    if not isinstance(entity_id, str) or not entity_id:
        raise ValueError("trace_tensor requires args.entity_id to be a non-empty string")

    from train_replay.collector.flight_recorder import load_flight_recorder
    from train_replay.graph.builder import build_from_events
    from train_replay.replay.replayer import EpochReplayer

    events = load_flight_recorder(dump_path)
    graph = build_from_events(events)
    replayer = EpochReplayer(graph)

    return {
        "tool": "trace_tensor",
        "entity_id": entity_id,
        "causal_ancestors": replayer.find_root_cause(entity_id),
    }
