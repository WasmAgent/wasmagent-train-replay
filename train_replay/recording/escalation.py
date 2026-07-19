"""Escalation signals that force higher-fidelity recording."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EscalationSignal:
    source: str = ""
    reason: str = ""
