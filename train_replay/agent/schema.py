"""TypedDict contracts for agent-facing tool payloads."""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

AgentToolName: TypeAlias = Literal["trace_tensor"]


class TraceTensorInput(TypedDict):
    """Input payload for the trace_tensor tool."""

    entity_id: str


class TraceTensorOutput(TypedDict):
    """Output payload returned by the trace_tensor tool."""

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
