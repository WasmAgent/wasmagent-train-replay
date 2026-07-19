"""Tests for Prometheus-backed escalation signals."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import Request

from train_replay.recording.escalation import EscalationSignal, PrometheusAnomalySource


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _prometheus_payload(value: float) -> dict[str, Any]:
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {
                    "metric": {"__name__": "nccl_anomaly_score"},
                    "value": [1_725_000_000.0, str(value)],
                }
            ],
        },
    }


def test_prometheus_anomaly_source_yields_none_below_threshold() -> None:
    def opener(_request: Request, _timeout: float) -> _Response:
        return _Response(_prometheus_payload(0.25))

    source = PrometheusAnomalySource(
        "https://prometheus.example/api/v1/query",
        "nccl_anomaly_score",
        threshold=0.75,
        metric_name="nccl_anomaly_score",
        opener=opener,
    )

    assert source.poll() is None


def test_prometheus_anomaly_source_yields_signal_above_threshold() -> None:
    def opener(_request: Request, _timeout: float) -> _Response:
        return _Response(_prometheus_payload(0.95))

    source = PrometheusAnomalySource(
        "https://prometheus.example/api/v1/query",
        "nccl_anomaly_score",
        threshold=0.75,
        metric_name="nccl_anomaly_score",
        source="nccl-inspector",
        opener=opener,
    )

    assert source.poll() == EscalationSignal(
        source="nccl-inspector",
        severity=0.95,
        metric_name="nccl_anomaly_score",
    )
