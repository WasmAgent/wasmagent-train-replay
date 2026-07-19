"""Tests for train_replay.config — Prometheus configuration loading."""

import os
from unittest.mock import MagicMock, patch

from train_replay.config import (
    PrometheusConfig,
    check_prometheus_readiness,
    load_prometheus_config,
    prometheus_enabled,
)


class TestLoadPrometheusConfig:
    """load_prometheus_config reads env vars with sensible defaults."""

    def test_defaults_when_no_env_set(self, monkeypatch: object) -> None:
        monkeypatch.setattr(os, "environ", {})
        cfg = load_prometheus_config()

        assert cfg == PrometheusConfig(
            query_url="",
            poll_interval_seconds=30.0,
            timeout_seconds=10.0,
            bearer_token=None,
        )

    def test_env_vars_override_defaults(self, monkeypatch: object) -> None:
        monkeypatch.setattr(os, "environ", {
            "PROMETHEUS_QUERY_URL": "http://prom:9090",
            "PROMETHEUS_POLL_INTERVAL": "60",
            "PROMETHEUS_TIMEOUT": "5",
            "PROMETHEUS_BEARER_TOKEN": "secret",
        })
        cfg = load_prometheus_config()

        assert cfg.query_url == "http://prom:9090"
        assert cfg.poll_interval_seconds == 60.0
        assert cfg.timeout_seconds == 5.0
        assert cfg.bearer_token == "secret"

    def test_kwarg_overrides_env(self, monkeypatch: object) -> None:
        monkeypatch.setattr(os, "environ", {
            "PROMETHEUS_QUERY_URL": "http://env:9090",
        })
        cfg = load_prometheus_config(PROMETHEUS_QUERY_URL="http://kwarg:9090")

        assert cfg.query_url == "http://kwarg:9090"

    def test_empty_env_var_treated_as_empty_string(self, monkeypatch: object) -> None:
        monkeypatch.setattr(os, "environ", {
            "PROMETHEUS_QUERY_URL": "",
        })
        cfg = load_prometheus_config()

        assert cfg.query_url == ""

    def test_none_override_does_not_clear_env(self, monkeypatch: object) -> None:
        monkeypatch.setattr(os, "environ", {
            "PROMETHEUS_QUERY_URL": "http://env:9090",
        })
        # Passing explicit None for an override key should fall back to env
        cfg = load_prometheus_config(PROMETHEUS_QUERY_URL=None)

        assert cfg.query_url == "http://env:9090"


class TestPrometheusEnabled:
    """prometheus_enabled gates on non-empty query_url."""

    def test_disabled_by_default(self, monkeypatch: object) -> None:
        monkeypatch.setattr(os, "environ", {})
        assert prometheus_enabled() is False

    def test_enabled_with_url(self, monkeypatch: object) -> None:
        monkeypatch.setattr(os, "environ", {
            "PROMETHEUS_QUERY_URL": "http://localhost:9090",
        })
        assert prometheus_enabled() is True

    def test_explicit_config_passed(self) -> None:
        off = PrometheusConfig(
            query_url="",
            poll_interval_seconds=30.0,
            timeout_seconds=10.0,
            bearer_token=None,
        )
        on = PrometheusConfig(
            query_url="http://x",
            poll_interval_seconds=30.0,
            timeout_seconds=10.0,
            bearer_token=None,
        )
        assert prometheus_enabled(off) is False
        assert prometheus_enabled(on) is True


class TestPrometheusConfigFrozen:
    """PrometheusConfig is a frozen dataclass."""

    def test_immutable(self) -> None:
        cfg = PrometheusConfig(
            query_url="x",
            poll_interval_seconds=1.0,
            timeout_seconds=1.0,
            bearer_token=None,
        )
        try:
            cfg.query_url = "y"  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass  # expected — frozen dataclass


class TestCheckPrometheusReadiness:
    """check_prometheus_readiness probes the Prometheus /-/ready endpoint."""

    def test_returns_false_when_disabled(self) -> None:
        cfg = PrometheusConfig(
            query_url="",
            poll_interval_seconds=30.0,
            timeout_seconds=10.0,
            bearer_token=None,
        )
        assert check_prometheus_readiness(cfg) is False

    def test_returns_true_on_http_200(self) -> None:
        cfg = PrometheusConfig(
            query_url="http://localhost:9090",
            poll_interval_seconds=30.0,
            timeout_seconds=10.0,
            bearer_token=None,
        )
        mock_resp = MagicMock()
        mock_resp.status = 200
        with patch("train_replay.config.urllib.request.urlopen", return_value=mock_resp):
            assert check_prometheus_readiness(cfg) is True

    def test_returns_false_on_non_200(self) -> None:
        cfg = PrometheusConfig(
            query_url="http://localhost:9090",
            poll_interval_seconds=30.0,
            timeout_seconds=10.0,
            bearer_token=None,
        )
        mock_resp = MagicMock()
        mock_resp.status = 503
        with patch("train_replay.config.urllib.request.urlopen", return_value=mock_resp):
            assert check_prometheus_readiness(cfg) is False

    def test_returns_false_on_connection_error(self) -> None:
        cfg = PrometheusConfig(
            query_url="http://localhost:9090",
            poll_interval_seconds=30.0,
            timeout_seconds=10.0,
            bearer_token=None,
        )
        import urllib.error

        with patch(
            "train_replay.config.urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            assert check_prometheus_readiness(cfg) is False

    def test_returns_false_on_timeout(self) -> None:
        cfg = PrometheusConfig(
            query_url="http://localhost:9090",
            poll_interval_seconds=30.0,
            timeout_seconds=1.0,
            bearer_token=None,
        )
        with patch(
            "train_replay.config.urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            assert check_prometheus_readiness(cfg) is False

    def test_sends_bearer_token_when_configured(self) -> None:
        cfg = PrometheusConfig(
            query_url="http://localhost:9090",
            poll_interval_seconds=30.0,
            timeout_seconds=10.0,
            bearer_token="my-secret-token",
        )
        mock_resp = MagicMock()
        mock_resp.status = 200

        captured_req: list[object] = []

        def _capture(req: object, **_kw: object) -> MagicMock:
            captured_req.append(req)
            return mock_resp

        with patch("train_replay.config.urllib.request.urlopen", side_effect=_capture):
            assert check_prometheus_readiness(cfg) is True

        req = captured_req[0]
        assert hasattr(req, "headers")  # type: ignore[truthy-function]
        header_dict = dict(req.headers)  # type: ignore[arg-type]
        assert header_dict.get("Authorization") == "Bearer my-secret-token"

    def test_strips_trailing_slash_from_url(self) -> None:
        cfg = PrometheusConfig(
            query_url="http://localhost:9090/",
            poll_interval_seconds=30.0,
            timeout_seconds=10.0,
            bearer_token=None,
        )
        mock_resp = MagicMock()
        mock_resp.status = 200

        captured_url: list[str] = []

        def _capture(req: object, **_kw: object) -> MagicMock:
            captured_url.append(str(getattr(req, "full_url", "")))
            return mock_resp

        with patch("train_replay.config.urllib.request.urlopen", side_effect=_capture):
            assert check_prometheus_readiness(cfg) is True

        # Should be exactly one slash before -/ready, not a double slash
        assert captured_url[0].endswith("/-/ready")
        assert "//-/-ready" not in captured_url[0]
