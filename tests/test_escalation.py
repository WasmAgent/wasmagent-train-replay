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


def test_escalation_signal_exposes_source_severity_and_metric_name() -> None:
    signal = EscalationSignal(
        source="nccl-inspector",
        severity=0.95,
        metric_name="nccl_anomaly_score",
    )

    assert signal.source == "nccl-inspector"
    assert signal.severity == 0.95
    assert signal.metric_name == "nccl_anomaly_score"


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


def test_prometheus_anomaly_source_yields_none_for_malformed_data() -> None:
    def opener(_request: Request, _timeout: float) -> _Response:
        return _Response({"status": "success", "data": None})

    source = PrometheusAnomalySource(
        "https://prometheus.example/api/v1/query",
        "nccl_anomaly_score",
        threshold=0.75,
        metric_name="nccl_anomaly_score",
        opener=opener,
    )

    assert source.poll() is None


def test_prometheus_anomaly_source_encodes_query_url() -> None:
    seen_urls: list[str] = []

    def opener(request: Request, _timeout: float) -> _Response:
        seen_urls.append(request.full_url)
        return _Response(_prometheus_payload(0.0))

    source = PrometheusAnomalySource(
        "https://prometheus.example/api/v1/query?time=1725000000",
        'max(nccl_anomaly_score{job="trainer"})',
        threshold=0.75,
        metric_name="nccl_anomaly_score",
        opener=opener,
    )

    assert source.poll() is None
    assert seen_urls == [
        "https://prometheus.example/api/v1/query?time=1725000000"
        "&query=max%28nccl_anomaly_score%7Bjob%3D%22trainer%22%7D%29"
    ]
