"""Parse MTIA profiler output into collective event records.

The MTIA (Matrix Tensor Intelligence Accelerator) profiler emits JSON trace
files that record collective communication operations across ranks. This module
provides a safe parser that converts those traces into :class:`MtiaEvent`
records suitable for ingestion by the causal graph builder.

This module intentionally avoids any NCCL-specific imports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class MtiaTraceParseError(ValueError):
    """Raised when an MTIA profiler trace file cannot be parsed."""


@dataclass
class MtiaEvent:
    """A single collective operation parsed from MTIA profiler output.

    Field names follow the MTIA profiler convention rather than the
    NCCL Flight Recorder naming, keeping this module free of NCCL
    references while still representing the same logical data.
    """

    rank: int
    process_group: str
    op_type: str
    src_rank: int | None = None
    dst_rank: int | None = None
    tensor_size: int = 0
    start_time_ns: int = 0
    end_time_ns: int = 0
    sequence_id: int = 0
    call_stack: list[str] = field(default_factory=list)


def parse_mtia_trace(path: Path) -> list[MtiaEvent]:
    """Parse an MTIA profiler JSON trace file into :class:`MtiaEvent` records.

    The expected format is a JSON object with an ``"events"`` list, where
    each element is a dict with keys matching :class:`MtiaEvent` fields.

    Parameters
    ----------
    path:
        Path to the MTIA profiler JSON trace file.

    Returns
    -------
    list[MtiaEvent]
        Parsed events in file order.

    Raises
    ------
    MtiaTraceParseError
        If the file cannot be read, is not valid JSON, or does not match
        the expected schema.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MtiaTraceParseError(
            f"Cannot read MTIA trace file {path}: {exc}"
        ) from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MtiaTraceParseError(
            f"Invalid JSON in MTIA trace file {path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise MtiaTraceParseError(
            f"MTIA trace must be a JSON object, got {type(raw).__name__}"
        )

    if "events" not in raw or not isinstance(raw["events"], list):
        raise MtiaTraceParseError(
            "MTIA trace must contain an 'events' list"
        )

    events: list[MtiaEvent] = []
    for idx, entry in enumerate(raw["events"]):
        if not isinstance(entry, dict):
            raise MtiaTraceParseError(
                f"Event at index {idx} must be a JSON object, "
                f"got {type(entry).__name__}"
            )
        events.append(_parse_single_event(entry))

    return events


def _parse_single_event(entry: dict[str, Any]) -> MtiaEvent:
    """Convert a raw JSON dict into an :class:`MtiaEvent`."""
    return MtiaEvent(
        rank=entry.get("rank", 0),
        process_group=entry.get("process_group", "default"),
        op_type=entry.get("op_type", "unknown"),
        src_rank=entry.get("src_rank"),
        dst_rank=entry.get("dst_rank"),
        tensor_size=entry.get("tensor_size", 0),
        start_time_ns=entry.get("start_time_ns", 0),
        end_time_ns=entry.get("end_time_ns", 0),
        sequence_id=entry.get("sequence_id", 0),
        call_stack=entry.get("call_stack", []),
    )
