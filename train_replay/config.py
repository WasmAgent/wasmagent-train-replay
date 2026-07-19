"""Project-wide configuration — environment-variable-driven settings.

Prometheus integration settings are read from environment variables with
sensible defaults so that the package works out-of-the-box without a
Prometheus server present (queries are simply skipped).
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrometheusConfig:
    """Configuration for Prometheus anomaly-source connectivity.

    Attributes:
        query_url: Base URL of the Prometheus server (e.g.
            ``http://localhost:9090``).  When empty or ``None``, Prometheus
            queries are disabled.
        poll_interval_seconds: How often (in seconds) the anomaly source
            should poll Prometheus for new data.
        timeout_seconds: Per-request HTTP timeout for Prometheus API calls.
        bearer_token: Optional ``Authorization: Bearer <token>`` value.  When
            ``None`` no authentication header is sent.
    """

    query_url: str
    poll_interval_seconds: float
    timeout_seconds: float
    bearer_token: str | None


def load_prometheus_config(**overrides: str | None) -> PrometheusConfig:
    """Build a :class:`PrometheusConfig` from environment variables.

    Recognised environment variables (in priority order: ``overrides`` >
    ``os.environ`` > built-in default):

    ================================ ===================================
    Variable                          Default
    ================================ ===================================
    ``PROMETHEUS_QUERY_URL``          ``""``  (disabled)
    ``PROMETHEUS_POLL_INTERVAL``     ``30``
    ``PROMETHEUS_TIMEOUT``           ``10``
    ``PROMETHEUS_BEARER_TOKEN``      ``None``
    ================================ ===================================

    Args:
        overrides: Keyword arguments that take precedence over environment
            variables.  Useful for testing.

    Returns:
        A frozen :class:`PrometheusConfig` instance.
    """

    def _env_or(name: str, default: str | None) -> str | None:
        """Return override > env > default."""
        if name in overrides and overrides[name] is not None:
            return overrides[name]
        return os.environ.get(name, default)

    query_url = _env_or("PROMETHEUS_QUERY_URL", "") or ""
    poll_interval = float(_env_or("PROMETHEUS_POLL_INTERVAL", "30") or "30")
    timeout = float(_env_or("PROMETHEUS_TIMEOUT", "10") or "10")
    bearer_token = _env_or("PROMETHEUS_BEARER_TOKEN", None)

    return PrometheusConfig(
        query_url=query_url,
        poll_interval_seconds=poll_interval,
        timeout_seconds=timeout,
        bearer_token=bearer_token,
    )


def prometheus_enabled(cfg: PrometheusConfig | None = None) -> bool:
    """Return ``True`` when Prometheus querying should be active.

    A non-empty ``query_url`` is the sole gate — no URL means no queries.
    """
    if cfg is None:
        cfg = load_prometheus_config()
    return bool(cfg.query_url)


def check_prometheus_readiness(cfg: PrometheusConfig | None = None) -> bool:
    """Probe the Prometheus ``/-/ready`` endpoint and return ``True`` on 200.

    This is a lightweight HTTP GET used to verify basic connectivity before
    the anomaly source begins polling.  Errors (network, timeout, non-200)
    are caught and logged rather than propagated.

    Args:
        cfg: Configuration to use.  When ``None`` the config is loaded from
            environment variables.  When Prometheus is disabled (empty
            ``query_url``) the function returns ``False`` immediately.

    Returns:
        ``True`` when the ``/-/ready`` endpoint responds with HTTP 200,
        ``False`` otherwise.
    """
    if cfg is None:
        cfg = load_prometheus_config()
    if not prometheus_enabled(cfg):
        return False

    url = f"{cfg.query_url.rstrip('/')}/-/ready"
    req = urllib.request.Request(url)
    if cfg.bearer_token:
        req.add_header("Authorization", f"Bearer {cfg.bearer_token}")

    try:
        resp = urllib.request.urlopen(req, timeout=cfg.timeout_seconds)
        ok: bool = bool(resp.status == 200)
        if not ok:
            logger.warning("Prometheus /-/ready returned status %d", resp.status)
        return ok
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("Prometheus readiness check failed: %s", exc)
        return False
