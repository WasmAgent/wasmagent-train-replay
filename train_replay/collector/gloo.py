"""Parse Gloo collective traces into the backend-agnostic ``CollectiveEvent`` schema.

Gloo is PyTorch's CPU collective backend. Unlike the flight recorder, Gloo ships
no binary trace dumper: collectives are observed as JSON logs emitted by a
distributed-debug hook or a small logging shim. This module parses those JSON
traces into :class:`CollectiveEvent` records — the same schema produced by
:mod:`train_replay.collector.flight_recorder` — so Gloo and flight-recorder
events unify in one PROV-DM causal graph via :func:`build_from_events`.

The :class:`CollectiveEvent` type is imported from the flight-recorder module
because it is the single shared, backend-agnostic record format the graph layer
consumes. No backend-specific decoder, constant, or binary path is referenced
here: parsing is plain :mod:`json`, which cannot execute code, so Gloo traces are
safe to ingest from untrusted sources.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .flight_recorder import CollectiveEvent


class MalformedGlooTraceError(ValueError):
    """Raised when a Gloo trace is structurally invalid and cannot be parsed."""


def _entries_from(payload: Any) -> list[dict[str, Any]]:
    """Normalize a decoded JSON payload into a list of entry dicts.

    Accepts either a top-level object with an ``"entries"`` list (the same shape
    as the flight-recorder dump, for cross-backend symmetry) or a bare JSON array
    of entry objects. Any other shape — or a non-object entry — is rejected with
    :class:`MalformedGlooTraceError`.
    """
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise MalformedGlooTraceError(
            "Gloo trace must be a JSON object with an 'entries' list, or a JSON "
            "array of entry objects."
        )
    for entry in entries:
        if not isinstance(entry, dict):
            raise MalformedGlooTraceError(
                "Each Gloo trace entry must be a JSON object."
            )
    return entries


def _opt_int(value: Any) -> int | None:
    """Coerce an optional numeric/string field to int, preserving ``None``."""
    return int(value) if value is not None else None


def _entry_to_event(entry: dict[str, Any]) -> CollectiveEvent:
    """Map one Gloo trace entry dict to a :class:`CollectiveEvent`.

    A small set of alias keys is accepted per field so the parser handles both
    the canonical schema and ad-hoc distributed-debug log lines that name the
    same data differently.
    """
    stack = entry.get("stack")
    if stack is None:
        stack = entry.get("frames", [])
    if not isinstance(stack, list):
        stack = []

    src_rank = entry.get("src_rank")
    if src_rank is None:
        src_rank = entry.get("p2p_src")
    dst_rank = entry.get("dst_rank")
    if dst_rank is None:
        dst_rank = entry.get("p2p_dst")

    return CollectiveEvent(
        rank=int(entry.get("rank", 0)),
        process_group=str(
            entry.get("process_group") or entry.get("pg_name") or "default"
        ),
        collective_type=str(
            entry.get("op")
            or entry.get("collective")
            or entry.get("collective_type")
            or "unknown"
        ),
        src_rank=_opt_int(src_rank),
        dst_rank=_opt_int(dst_rank),
        tensor_size=int(
            entry.get("input_size", entry.get("tensor_size", 0))
        ),
        enqueue_time_ns=int(
            entry.get("time_created_ns", entry.get("enqueue_time_ns", 0))
        ),
        start_time_ns=int(
            entry.get("time_started_ns", entry.get("start_time_ns", 0))
        ),
        end_time_ns=int(
            entry.get("time_finished_ns", entry.get("end_time_ns", 0))
        ),
        call_stack=[str(frame) for frame in stack],
        sequence_id=int(entry.get("sequence_id", entry.get("seq_id", 0))),
    )


def parse_gloo_trace(payload: Any) -> list[CollectiveEvent]:
    """Parse a decoded Gloo trace (object or array) into CollectiveEvent records.

    Takes an already-decoded JSON value. Use :func:`load_gloo_trace` to read and
    parse a trace file in one step.
    """
    return [_entry_to_event(entry) for entry in _entries_from(payload)]


def load_gloo_trace(path: Path) -> list[CollectiveEvent]:
    """Load a Gloo JSON trace file into CollectiveEvent records.

    The file is decoded with :func:`json.load`, which cannot execute code, so
    Gloo traces are safe to ingest from untrusted sources.
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return parse_gloo_trace(payload)
