# Anomaly Detection Guide

> How to build baseline profiles, configure detection thresholds, interpret
> anomaly signals, and integrate with alerting pipelines.

This guide covers the Milestone 5 anomaly detection pipeline: from capturing a
baseline profile during normal training, through scanning new runs for statistical
deviations, to delivering alerts when anomalies are found.

## 1. Profile creation from normal training runs

A **TrainingProfile** captures baseline statistics from a known-good training
run so that subsequent runs can be compared against it. It is built from
`CollectiveEvent` records — the same events produced by
`load_flight_recorder()`.

### What the profile captures

| Statistic family | Fields | Meaning |
|---|---|---|
| **Event intervals** | `interval_mean_ns`, `interval_std_ns`, `interval_min_ns`, `interval_max_ns`, `interval_count` | Per-rank inter-arrival gaps (ns) between consecutive collectives. |
| **Tensor sizes** | `tensor_size_mean`, `tensor_size_std`, `tensor_size_min`, `tensor_size_max`, `tensor_count` | Byte-size distribution over all recorded tensors. |
| **Operation patterns** | `collective_type_counts`, `ranks`, `event_count` | Per-`collective_type` frequency and the set of observed ranks. |

### Building a profile from a Flight Recorder dump

```bash
# Derive a self-referential baseline directly from a known-good dump.
# This is the simplest path: the dump *is* the normal run.
train-replay anomaly /data/runs/normal/epoch_0.pkl
```

When `--profile` is omitted, the anomaly command builds an on-the-fly profile
from the dump itself and uses it as the comparison baseline. This is useful
for quick spot-checks but not for production monitoring — a stored profile
covers more representative data.

### Building a persistent profile from Python

For production use, build and persist the profile explicitly so it can be
reused across many scans:

```python
"""Build a TrainingProfile from a known-good run and save it."""

import json
from pathlib import Path

from train_replay.anomaly.profile import TrainingProfile
from train_replay.collector.flight_recorder import load_flight_recorder


def build_and_save_profile(dump_path: Path, output_path: Path) -> None:
    events = load_flight_recorder(dump_path)
    profile = TrainingProfile.fit_on_normal_run(events)

    # The profile dataclass is JSON-serialisable once frozensets are converted.
    data = {
        "interval_mean_ns": profile.interval_mean_ns,
        "interval_std_ns": profile.interval_std_ns,
        "interval_min_ns": profile.interval_min_ns,
        "interval_max_ns": profile.interval_max_ns,
        "interval_count": profile.interval_count,
        "tensor_size_mean": profile.tensor_size_mean,
        "tensor_size_std": profile.tensor_size_std,
        "tensor_size_min": profile.tensor_size_min,
        "tensor_size_max": profile.tensor_size_max,
        "tensor_count": profile.tensor_count,
        "collective_type_counts": dict(profile.collective_type_counts),
        "ranks": sorted(profile.ranks),
        "event_count": profile.event_count,
    }
    output_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Saved profile ({profile.event_count} events, "
          f"{len(profile.ranks)} ranks) to {output_path}")


build_and_save_profile(
    Path("/data/runs/normal/epoch_0.pkl"),
    Path("/data/profiles/production_baseline.json"),
)
```

### Aggregating multiple normal runs

A single epoch may not represent steady-state training. To build a more robust
baseline, aggregate statistics from multiple epochs before persisting:

```python
"""Aggregate a TrainingProfile across multiple normal epochs."""

from pathlib import Path
from statistics import fmean

from train_replay.anomaly.profile import TrainingProfile
from train_replay.collector.flight_recorder import load_flight_recorder

dump_dir = Path("/data/runs/normal")
profiles = []

for pkl in sorted(dump_dir.glob("epoch_*.pkl")):
    events = load_flight_recorder(pkl)
    profiles.append(TrainingProfile.fit_on_normal_run(events))

# Combine: average the interval and tensor-size statistics, merge patterns.
total_events = sum(p.event_count for p in profiles)
combined = TrainingProfile(
    interval_mean_ns=fmean(p.interval_mean_ns for p in profiles),
    interval_std_ns=fmean(p.interval_std_ns for p in profiles),
    interval_min_ns=min(p.interval_min_ns for p in profiles),
    interval_max_ns=max(p.interval_max_ns for p in profiles),
    interval_count=sum(p.interval_count for p in profiles),
    tensor_size_mean=fmean(p.tensor_size_mean for p in profiles),
    tensor_size_std=fmean(p.tensor_size_std for p in profiles),
    tensor_size_min=min(p.tensor_size_min for p in profiles),
    tensor_size_max=max(p.tensor_size_max for p in profiles),
    tensor_count=sum(p.tensor_count for p in profiles),
    collective_type_counts={},  # populate as needed
    ranks=frozenset().union(*(p.ranks for p in profiles)),
    event_count=total_events,
)
```

### Profile refresh cadence

Baselines degrade as training evolves (learning rate schedules, layer freezing,
dataset sharding). Recreate the profile:

- **After hyperparameter changes** — new learning rates change gradient
  distributions and collective timings.
- **When adding or removing ranks** — the observed rank set and inter-event
  intervals shift.
- **Periodically** (e.g. weekly or every N epochs) — to capture gradual drift
  in tensor sizes and timing.

## 2. Detector configuration and threshold tuning

The anomaly scanner uses Z-score comparison: each metric (tensor size,
inter-event timing) is compared to the baseline mean and standard deviation.
Events whose absolute Z-score exceeds `--threshold` are flagged.

### Using the `train-replay anomaly` command

```bash
# Scan against a stored profile with default threshold (3.0).
train-replay anomaly /data/runs/latest/epoch_10.pkl \
    --profile /data/profiles/production_baseline.json

# Scan with a tighter threshold (more sensitive).
train-replay anomaly /data/runs/latest/epoch_10.pkl \
    --profile /data/profiles/production_baseline.json \
    --threshold 2.5

# Self-referential scan (no stored profile — uses the dump as baseline).
train-replay anomaly /data/runs/latest/epoch_10.pkl
```

### Choosing a Z-score threshold

| Threshold | Sensitivity | Typical use case |
|---|---|---|
| **2.0** | High — flags ~5% of events as anomalous | Debugging a suspected issue; low tolerance for deviation. |
| **3.0** | Medium — flags ~0.3% of events | Default. Good starting point for production. |
| **4.0** | Low — flags ~0.006% of events | Stable training runs where only extreme outliers matter. |
| **5.0+** | Very low | Alerting-only on near-certain anomalies; minimal false positives. |

### Practical tuning tips

1. **Start with the default (3.0) and adjust based on false-positive rate.**
   Run the scan on a few known-good epochs: if anomalies appear, the baseline
   is too tight or the threshold too low.

2. **Distinguish metric types.** Tensor-size anomalies are often more
   meaningful than timing anomalies (timing jitter is common in shared
   GPU environments). Inspect the `metric_name` column in the output table.

3. **Account for warm-up.** The first few steps of an epoch may show
   anomalous timing as GPU kernels initialise. Exclude warm-up steps from
   the baseline or accept a higher false-positive rate early in each epoch.

4. **Validate against injected faults.** The
   `examples/fault_injection_demo.py` script shows how to inject
   `was_vetted` risk signals into the recording policy. Use similar
   techniques to inject known anomalies and verify the detector catches them.

### Prometheus anomaly source

For external metrics (GPU utilisation, NCCL errors, etc.), the
`PrometheusAnomalySource` polls a Prometheus server and produces
`EscalationSignal` records when a metric exceeds a threshold:

```python
"""Configure a Prometheus anomaly source for GPU utilisation."""

from train_replay.config import load_prometheus_config
from train_replay.recording.escalation import PrometheusAnomalySource

source = PrometheusAnomalySource(
    endpoint="http://localhost:9090/api/v1/query",
    query="avg(nccl_communicator_queue_size)",
    threshold=100.0,
    metric_name="nccl_queue_size",
)

signal = source.poll()
if signal is not None:
    print(f"Escalation: severity={signal.severity} "
          f"metric={signal.metric_name}")
```

When the `EscalationSignal` is passed to `compile_recording_policy()` as the
`escalation` parameter, it forces `RecordingMode.FULL` — retroactively
capturing full evidence for the anomalous period.

See [prometheus-setup.md](prometheus-setup.md) for Prometheus connectivity
configuration.

## 3. Interpreting anomaly signals

### Output format

The `train-replay anomaly` command prints a table of detected anomalies:

```
┏━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Rank   ┃ Step  ┃ Metric          ┃ Z-score┃ Severity┃ Description          ┃
┡━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ 2      │ 47    │ tensor_size_zsc…│ +4.21  │ 1.00    │ tensor_size 8192B … │
│ 0      │ 12    │ timing_zscore   │ +3.87  │ 1.00    │ inter-event interva…│
└────────┴───────┴────────────────┴────────┴─────────┴──────────────────────┘
```

### Column guide

| Column | Meaning |
|---|---|
| **Rank** | GPU rank where the anomaly was observed. |
| **Step** | Sequence ID (step number) of the offending collective. |
| **Metric** | `tensor_size_zscore` — the event's tensor byte size deviates from baseline. `timing_zscore` — the inter-event interval on this rank deviates from baseline. |
| **Z-score** | Signed distance from the baseline mean in standard deviations. Positive = higher than normal; negative = lower. |
| **Severity** | Clipped to [0.0, 1.0]. Higher values indicate stronger deviation. Used by alerting. |
| **Description** | Human-readable summary of the deviation. |

### Common anomaly patterns

| Pattern | Likely cause | Action |
|---|---|---|
| **High positive timing Z-score on one rank** | Slow rank (straggler) — NCCL timeout risk, GPU thermal throttling, or CPU contention. | Check GPU utilisation on that rank; consider reducing batch size or enabling gradient checkpointing. |
| **High negative timing Z-score** | Faster-than-normal collective — possibly missing data or skipped operations. | Verify the training step completed correctly; check for `nan` losses that may have short-circuited a step. |
| **High positive tensor-size Z-score** | Unexpectedly large tensor — likely a gradient explosion, `nan` overflow, or incorrect tensor shape after a bug. | Inspect the step's loss and gradient norms; enable `RecordingMode.FULL` via escalation for that rank. |
| **High negative tensor-size Z-score** | Unexpectedly small tensor — possibly zeroed gradients (dead neurons) or a partial reduction. | Check for `zero` gradients in the model; verify data pipeline is feeding correctly. |
| **Anomalies on multiple ranks at the same step** | Synchronisation issue or collective-type mismatch between ranks. | Use `train-replay trace` to inspect the causal graph at that step; check NCCL collective type alignment. |

### Integration with the recording pipeline

An anomaly signal can trigger automatic evidence escalation. When an
`EscalationSignal` is present, `compile_recording_policy()` returns
`RecordingMode.FULL` regardless of other risk factors:

```python
from train_replay.recording.escalation import EscalationSignal
from train_replay.recording.modes import compile_recording_policy, RiskContext

signal = EscalationSignal(
    source="prometheus",
    severity=4.2,
    metric_name="nccl_queue_size",
)
policy = compile_recording_policy(RiskContext(), escalation=signal)
# policy.mode == RecordingMode.FULL
# policy.reason == "external escalation signal"
```

This means anomalous collectives automatically receive full tensor snapshots,
providing forensic-grade evidence for post-mortem analysis.

## 4. Integrating with alerting pipelines

### Slack notifications

The `train-replay anomaly` command has built-in Slack support via the
`--notify` flag:

```bash
train-replay anomaly /data/runs/latest/epoch_10.pkl \
    --profile /data/profiles/production_baseline.json \
    --threshold 3.0 \
    --notify "slack:https://hooks.slack.com/services/T00/B00/xxx"
```

When anomalies are found, a summary message is posted to the webhook. When no
anomalies are detected, the notification is silently skipped (no alert fatigue).

**Slack webhook setup:**

1. Create an incoming webhook in your Slack workspace
   (Settings → Integrations → Incoming Webhooks).
2. Note the webhook URL — it will look like
   `https://hooks.slack.com/services/T00/B00/xxx`.
3. Pass it to the CLI via `--notify "slack:<url>"`.

### Programmatic Slack notifications

For custom alerting logic or non-CLI workflows, use the internal notification
function directly:

```python
"""Send a custom anomaly alert to Slack."""

import json
from urllib.request import Request, urlopen

from train_replay.anomaly.profile import TrainingProfile
from train_replay.collector.flight_recorder import load_flight_recorder


def send_anomaly_summary(dump_path: str, profile: TrainingProfile,
                          webhook_url: str, threshold: float = 3.0) -> None:
    """Scan a dump and notify Slack if anomalies are found."""
    # In production, use StatisticalAnomalyDetector.detect() when available.
    # The CLI's _scan_anomalies() provides the same Z-score logic.
    events = load_flight_recorder(dump_path)
    # ... run detection logic ...
    # When anomalies exist:
    message = (
        f"🚨 train-replay anomaly: {len(anomalies)} anomalies "
        f"(|z| > {threshold}) in {dump_path}"
    )
    payload = json.dumps({"text": message}).encode("utf-8")
    request = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5.0) as response:
        response.read()
```

### Prometheus escalation pipeline

For continuous monitoring, combine the Prometheus anomaly source with the
recording policy:

```
Prometheus → PrometheusAnomalySource.poll()
                │
                ▼
         EscalationSignal (severity, metric_name)
                │
                ▼
         compile_recording_policy(ctx, escalation=signal)
                │
                ▼
         RecordingMode.FULL (automatic evidence capture)
```

This pipeline runs without operator intervention. Configure Prometheus to
expose training metrics (NCCL queue size, gradient norms, GPU utilisation)
and the anomaly source polls at the interval defined by
`PROMETHEUS_POLL_INTERVAL` (default: 30 seconds).

### Extending with custom alert backends

The alerting interface is pluggable. To add a new backend (e.g. PagerDuty):

```python
"""Custom alert notifier for PagerDuty."""

from dataclasses import dataclass


@dataclass
class PagerDutyNotifier:
    routing_key: str
    url: str = "https://events.pagerduty.com/v2/enqueue"

    def send(self, severity: float, description: str) -> None:
        import json
        from urllib.request import Request, urlopen

        payload = json.dumps({
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": f"Training anomaly: {description}",
                "severity": "critical" if severity > 4.0 else "warning",
                "source": "wasmagent-train-replay",
            },
        }).encode("utf-8")
        request = Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5.0) as response:
            response.read()
```

### End-to-end monitoring recipe

A complete anomaly monitoring setup involves three steps:

1. **Build a baseline profile** from representative normal runs
   ([§1](#1-profile-creation-from-normal-training-runs)).

2. **Schedule periodic scans** of new dumps against the profile:
   ```bash
   # In a cron job or monitoring loop:
   train-replay anomaly /data/runs/latest/epoch_$EPOCH.pkl \
       --profile /data/profiles/production_baseline.json \
       --threshold 3.0 \
       --notify "slack:https://hooks.slack.com/services/T00/B00/xxx"
   ```

3. **Enable Prometheus escalation** for real-time external metric monitoring
   ([prometheus-setup.md](prometheus-setup.md)). This feeds `EscalationSignal`
   records into the recording policy, automatically escalating evidence capture
   when GPU utilisation, NCCL queue size, or other metrics exceed thresholds.

## See also

- [architecture.md](architecture.md) — system design and the anomaly detection pipeline overview.
- [prometheus-setup.md](prometheus-setup.md) — Prometheus connectivity and configuration.
- [cli-reference.md](cli-reference.md) — full CLI reference including the `anomaly` subcommand.
- [protocol.md](protocol.md) — schemas for `CollectiveEvent`, `AEPRecord`, and `EpochEvidenceBundle`.
