"""TypedDict contracts for agent-facing tool payloads.

Each TypedDict in this module mirrors the JSON Schema declared in
:mod:`train_replay.agent.tools` (``TRACE_TENSOR_INPUT_SCHEMA`` /
``TRACE_TENSOR_OUTPUT_SCHEMA``): the TypedDict keys equal the JSON Schema
``properties`` keys and the Python types equal the JSON Schema value types.
Keeping the two in sync means the static type contract and the runtime JSON
Schema description of every tool describe the same payload.
"""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

AgentToolName: TypeAlias = Literal["trace_tensor"]


class TraceTensorInput(TypedDict):
    """Input payload for the trace_tensor tool.

    Mirrors ``TRACE_TENSOR_INPUT_SCHEMA`` in :mod:`train_replay.agent.tools`.
    """

    entity_id: str


class TraceTensorOutput(TypedDict):
    """Output payload returned by the trace_tensor tool.

    Mirrors ``TRACE_TENSOR_OUTPUT_SCHEMA`` in :mod:`train_replay.agent.tools`.
    """

    tool: Literal["trace_tensor"]
    entity_id: str
    causal_ancestors: list[str]


AgentToolInput: TypeAlias = TraceTensorInput
AgentToolOutput: TypeAlias = TraceTensorOutput


__all__ = [
    "AgentToolInput",
    "AgentToolName",
    "AgentToolOutput",
    "TraceTensorInput",
    "TraceTensorOutput",
]
