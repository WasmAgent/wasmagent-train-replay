"""Escalation signals sourced from external anomaly detectors."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _default_opener(request: Request, timeout_seconds: float) -> Any:
    return urlopen(request, timeout=timeout_seconds)


@dataclass(frozen=True)
class EscalationSignal:
    source: str
    severity: float
    metric_name: str


class PrometheusAnomalySource:
    """Poll a Prometheus query endpoint for threshold-crossing anomaly metrics."""

    def __init__(
        self,
        endpoint: str,
        query: str,
        threshold: float,
        *,
        metric_name: str,
        source: str = "prometheus",
        timeout_seconds: float = 5.0,
        opener: Callable[[Request, float], Any] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.query = query
        self.threshold = threshold
        self.metric_name = metric_name
        self.source = source
        self.timeout_seconds = timeout_seconds
        self._opener = opener or _default_opener

    def poll(self) -> EscalationSignal | None:
        """Return an escalation signal when the queried metric exceeds the threshold."""
        request = Request(self._query_url(), headers={"Accept": "application/json"})
        with self._opener(request, self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        value = self._extract_value(payload)
        if value is None or value <= self.threshold:
            return None
        return EscalationSignal(
            source=self.source,
            severity=value,
            metric_name=self.metric_name,
        )

    def _query_url(self) -> str:
        separator = "&" if "?" in self.endpoint else "?"
        return f"{self.endpoint}{separator}{urlencode({'query': self.query})}"

    @staticmethod
    def _extract_value(payload: dict[str, Any]) -> float | None:
        result = payload.get("data", {}).get("result", [])
        if not isinstance(result, list) or not result:
            return None

        first = result[0]
        if not isinstance(first, dict):
            return None

        value = first.get("value")
        if not isinstance(value, list) or len(value) < 2:
            return None

        try:
            return float(value[1])
        except (TypeError, ValueError):
            return None
