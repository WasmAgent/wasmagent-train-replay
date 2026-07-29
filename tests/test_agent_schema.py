"""Tests that agent TypedDicts match the JSON Schema in ``tools.py``."""

from __future__ import annotations

from typing import get_type_hints

from train_replay.agent import schema
from train_replay.agent.tools import (
    AGENT_TOOL_SCHEMAS,
    TRACE_TENSOR_INPUT_SCHEMA,
    TRACE_TENSOR_OUTPUT_SCHEMA,
)


def test_registry_exposes_trace_tensor_schemas() -> None:
    """The schema registry points at the same constant objects."""
    assert AGENT_TOOL_SCHEMAS["trace_tensor"]["input"] is TRACE_TENSOR_INPUT_SCHEMA
    assert AGENT_TOOL_SCHEMAS["trace_tensor"]["output"] is TRACE_TENSOR_OUTPUT_SCHEMA


def test_trace_tensor_input_typeddict_matches_schema() -> None:
    """TypedDict keys equal the JSON Schema property keys."""
    assert set(get_type_hints(schema.TraceTensorInput)) == set(
        TRACE_TENSOR_INPUT_SCHEMA["properties"]
    )
    assert TRACE_TENSOR_INPUT_SCHEMA["required"] == ["entity_id"]


def test_trace_tensor_output_typeddict_matches_schema() -> None:
    """TypedDict keys equal the JSON Schema property keys, all required."""
    assert set(get_type_hints(schema.TraceTensorOutput)) == set(
        TRACE_TENSOR_OUTPUT_SCHEMA["properties"]
    )
    assert set(TRACE_TENSOR_OUTPUT_SCHEMA["required"]) == set(
        get_type_hints(schema.TraceTensorOutput)
    )
    assert TRACE_TENSOR_OUTPUT_SCHEMA["properties"]["tool"]["enum"] == ["trace_tensor"]


def test_agent_tool_name_covers_registry() -> None:
    """Every tool in the registry is a member of the AgentToolName literal."""
    # Literal["trace_tensor"] surfaces its members via __args__.
    literal_args = set(schema.AgentToolName.__args__)  # type: ignore[attr-defined]
    assert set(AGENT_TOOL_SCHEMAS) <= literal_args
