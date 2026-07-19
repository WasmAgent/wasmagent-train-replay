"""Agent tool dispatch for querying training replay evidence.

Each tool's payload contract is described by a JSON Schema constant co-located
here (``TRACE_TENSOR_INPUT_SCHEMA`` / ``TRACE_TENSOR_OUTPUT_SCHEMA``). The
TypedDict definitions in :mod:`train_replay.agent.schema` mirror these schemas
field-for-field, keeping the runtime JSON Schema description and the static
type contract of every tool in lock-step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: JSON Schema describing the ``trace_tensor`` tool's input payload.
TRACE_TENSOR_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entity_id": {"type": "string"},
    },
    "required": ["entity_id"],
    "additionalProperties": False,
}

#: JSON Schema describing the ``trace_tensor`` tool's output payload.
TRACE_TENSOR_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": ["trace_tensor"]},
        "entity_id": {"type": "string"},
        "causal_ancestors": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["tool", "entity_id", "causal_ancestors"],
    "additionalProperties": False,
}

#: Registry mapping each agent tool name to its input/output JSON Schemas.
AGENT_TOOL_SCHEMAS: dict[str, dict[str, dict[str, Any]]] = {
    "trace_tensor": {
        "input": TRACE_TENSOR_INPUT_SCHEMA,
        "output": TRACE_TENSOR_OUTPUT_SCHEMA,
    },
}


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
