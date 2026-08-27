# Cross-run regression & divergence analysis

> Worked example: comparing two training dumps, interpreting the divergence
> report fields, and attributing a regression to a specific rank and step
> with the `train-replay diff` CLI.

## When to use this

Two replays of the *same* epoch should produce identical collective streams.
When they don't — a loss spike after a code change, a suspected nondeterministic
kernel, an unexplained accuracy regression — `train-replay diff` localizes the
first point where the two runs' collective behavior diverges, per rank, and
ties it to any detected cross-rank desync.

## The data path

```
ReplayResult (baseline) ─┐
                         ├─▶ DivergenceReplayer.diff() ─▶ DivergenceReport ─▶ signed report
ReplayResult (candidate)─┘         ▲                                (BundleSigner)
                                   │
                    collision_report / EscalationSignal correlation
```

`EpochReplayer.replay_diff(baseline_dump, candidate_dump)` drives the whole
path from two Flight Recorder dumps and returns one `DivergenceReport` per
divergent rank (an empty `dict` when the runs agree).

## Worked example

Suppose epoch 5 trained cleanly yesterday (`baseline.pkl`) and today's rerun
(`candidate.pkl`) shows a loss spike. Compare the two dumps:

```bash
train-replay diff baseline.pkl candidate.pkl
```

Output (JSON, one object keyed by divergent rank — truncated for readability):

```json
{
  "1": {
    "first_divergence_step": 42,
    "divergences": [
      {
        "rank": 1,
        "first_divergence_step": 42,
        "baseline_action":  {"step": 42, "collective_type": "all_reduce"},
        "candidate_action": {"step": 42, "collective_type": "all_gather"},
        "correlated_collision": true,
        "correlated_escalation": null
      }
    ],
    "per_rank_similarity": {"1": 0.9767},
    "summary": "Rank 1 diverged at step 42"
  }
}
```

The command exits `1` because a divergence was found (exit `0` means the dumps
agree) — so the same invocation works as a CI regression gate.

### Interpreting the fields

| Field | Meaning |
|---|---|
| `first_divergence_step` | Report-level first step at which any compared stream disagreed (`null` when the runs are identical). |
| `divergences[]` | Per-rank detail: the disagreeing actions on both sides, plus a context window of surrounding baseline steps. |
| `per_rank_similarity` | Fraction of steps that agreed before the first divergence, per rank. `1.0` = identical. |
| `correlated_collision` | `true` when the divergence step falls inside a desync reported by that rank's `collision_report` — the divergence is not merely local, ranks disagree with each other. |
| `correlated_escalation` | Populated when an external `EscalationSignal` (e.g. an NCCL Inspector metric) overlaps the divergence; carries `metric_name=severity`. |

### Attributing the regression

In the example above:

1. **Rank and step are pinned**: rank 1 changed behavior at step 42 —
   an `all_reduce` became an `all_gather`.
2. **`correlated_collision: true`** says the detector saw a cross-rank desync
   at that same step: rank 1 did not just change its own plan, it broke
   alignment with its peers. That points to a scheduling/rank-skew root cause
   (e.g. a straggler kernel launch or a topology change), not a silent
   numerical drift.
3. **`per_rank_similarity` ≈ 0.98** tells you the divergence is a single
   point event, not a gradual drift across the epoch.

To narrow further, re-run with `--rank 1 --context 20` and inspect the
context window around step 42, or sign the report for the audit trail:

```python
from dataclasses import asdict

from train_replay.signing.signer import BundleSigner, verify_divergence_report

signer, public_key = BundleSigner.generate()
envelope = signer.sign_divergence_report(asdict(report), baseline_bundle, candidate_bundle)
assert verify_divergence_report(envelope, baseline_bundle, candidate_bundle, public_key)
```

The DSSE envelope binds the report to the SHA-256 digests of both source
bundles, so an auditor can verify the comparison output originated from
exactly these two dumps (see [protocol.md](protocol.md)).

## Python API

```python
from train_replay.graph.builder import build_from_events
from train_replay.replay.replayer import EpochReplayer

replayer = EpochReplayer(build_from_events(baseline_events))
reports = replayer.replay_diff("baseline.pkl", "candidate.pkl")
for rank, report in sorted(reports.items()):
    print(rank, report.first_divergence_step, report.summary)
```

See `examples/divergence_demo.py` for a runnable end-to-end demo, and
[cli-reference.md](cli-reference.md#train-replay-diff) for the full flag
reference.
