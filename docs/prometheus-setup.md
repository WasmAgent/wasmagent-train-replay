# Prometheus Integration Setup

This guide covers configuring the `prometheus-client` dependency shipped with
wasmagent-train-replay and verifying connectivity to a Prometheus server.

## 1. Install the dependency

The `prometheus-client` package is included as a core dependency.  After a
standard editable install it is already available:

```bash
pip install -e ".[dev]"
python -c "import prometheus_client; print(prometheus_client.__version__)"
```

## 2. Configure environment variables

Copy the provided template and edit as needed:

```bash
cp .env.example .env
```

| Variable                  | Default | Description                                    |
|---------------------------|---------|------------------------------------------------|
| `PROMETHEUS_QUERY_URL`    | `""`    | Base URL (e.g. `http://localhost:9090`).      |
| `PROMETHEUS_POLL_INTERVAL`| `30`    | Seconds between anomaly-source polls.          |
| `PROMETHEUS_TIMEOUT`      | `10`    | HTTP timeout per Prometheus request.           |
| `PROMETHEUS_BEARER_TOKEN` | *(none)*| Optional Bearer token for authenticated access.|

When `PROMETHEUS_QUERY_URL` is empty or unset, Prometheus querying is **disabled**
and the anomaly source is skipped entirely.

## 3. Test connectivity from Python

```python
"""Quick health check against a running Prometheus server."""

import urllib.request

from train_replay.config import load_prometheus_config, prometheus_enabled

cfg = load_prometheus_config()

if not prometheus_enabled(cfg):
    print("Prometheus is disabled (PROMETHEUS_QUERY_URL is empty).")
else:
    url = f"{cfg.query_url.rstrip('/')}/-/ready"
    print(f"Checking {url} ...")
    req = urllib.request.Request(url, headers={})
    if cfg.bearer_token:
        req.add_header("Authorization", f"Bearer {cfg.bearer_token}")
    resp = urllib.request.urlopen(req, timeout=cfg.timeout_seconds)
    print(f"Prometheus ready: {resp.status} {resp.read().decode()}")
```

Run with:

```bash
PROMETHEUS_QUERY_URL=http://localhost:9090 python healthcheck.py
```

## 4. Running Prometheus locally (for development)

Docker Compose is the fastest path:

```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus:v2.54.1
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

Minimal `prometheus.yml` (scrapes itself for testing):

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
```

```bash
docker compose up -d
curl http://localhost:9090/-/ready
# → Prometheus is Ready.
```

## 5. Next steps

Once connectivity is confirmed, the `PrometheusAnomalySource` class in
`train_replay/recording/escalation.py` will use this configuration to poll
Prometheus for anomaly-related metrics and feed them into the recording-policy
escalation pipeline.
