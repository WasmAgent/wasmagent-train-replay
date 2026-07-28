# Auditor Guide: Root-Causing Divergence with `train-replay diff`

This guide walks through a complete example of using `train-replay` to isolate
the root cause of a divergent distributed training run using signed evidence bundles.

## Prerequisites

```bash
pip install -e ".[dev]"
```

You need two evidence bundles (baseline and candidate) produced by the same
`run_id` / epoch from two separate training runs.

---

## Step 1: Export Signed Evidence Bundles

Use the `record` subcommand to produce a signed bundle from a PyTorch Flight
Recorder dump. Repeat for both the baseline and the candidate run.

```bash
# Generate a fresh Ed25519 key (32 raw bytes → 64 hex chars)
SIGNING_KEY=$(python3 -c "
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import binascii
k = Ed25519PrivateKey.generate()
print(binascii.hexlify(k.private_bytes_raw()).decode())
")
PUBLIC_KEY=$(python3 -c "
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import binascii, sys
k = Ed25519PrivateKey.from_private_bytes(bytes.fromhex('$SIGNING_KEY'))
print(binascii.hexlify(k.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode())
")

# Export baseline bundle
train-replay record baseline_dump.json \
  --run-id run-baseline-001 \
  --epoch 12 \
  --signing-key-hex "$SIGNING_KEY" \
  --signing-key-id auditor-key-v1 \
  > baseline_bundle.json

# Export candidate bundle
train-replay record candidate_dump.json \
  --run-id run-candidate-001 \
  --epoch 12 \
  --signing-key-hex "$SIGNING_KEY" \
  --signing-key-id auditor-key-v1 \
  > candidate_bundle.json
```

Both commands print the bundle digest and signature. Keep the public key hex
(`$PUBLIC_KEY`) — you need it for verification.

---

## Step 2: Verify Cryptographic Signatures

Before trusting any bundle content, verify its signature:

```python
from train_replay.recording.evidence import EpochEvidenceBundle
from train_replay.signing.signer import verify_bundle
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import binascii, json

PUBLIC_KEY_HEX = "<paste $PUBLIC_KEY here>"
public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(PUBLIC_KEY_HEX))

with open("baseline_bundle.json") as f:
    baseline = EpochEvidenceBundle.from_json(f.read())
with open("candidate_bundle.json") as f:
    candidate = EpochEvidenceBundle.from_json(f.read())

assert verify_bundle(baseline, public_key), "Baseline signature INVALID"
assert verify_bundle(candidate, public_key), "Candidate signature INVALID"
print("Both signatures valid.")
print(f"Baseline digest:  {baseline.digest()}")
print(f"Candidate digest: {candidate.digest()}")
```

A failed assertion means the bundle was tampered with or signed by a different key.

---

## Step 3: Run `train-replay diff`

With verified bundles, replay both and compare their action streams:

```python
from train_replay.graph.builder import build_from_events
from train_replay.collector.flight_recorder import load_flight_recorder
from train_replay.recording.recorder import EpochRecorder
from train_replay.replay.replayer import EpochReplayer
from train_replay.replay.diff import DivergenceReplayer
from pathlib import Path

def replay_bundle(dump_path: str, run_id: str, epoch: int, rank: int, entity_id: str):
    events = load_flight_recorder(Path(dump_path))
    graph = build_from_events(events)
    replayer = EpochReplayer(graph)
    recorder = EpochRecorder(run_id=run_id, epoch=epoch)
    for evt in events:
        recorder.record_collective(evt)
    bundle = recorder.bundle()
    return replayer.replay_rank(bundle, rank, entity_id)

baseline_result = replay_bundle("baseline_dump.json", "run-baseline-001", 12, rank=0, entity_id="tensor:grad:layer3")
candidate_result = replay_bundle("candidate_dump.json", "run-candidate-001", 12, rank=0, entity_id="tensor:grad:layer3")

report = DivergenceReplayer().diff(baseline_result, candidate_result)
print(report.summary)
```

---

## Step 4: Interpret the Divergence Output

The `DivergenceReport` exposes:

| Field | Meaning |
|---|---|
| `first_divergence_step` | Training step where baseline and candidate first disagree |
| `divergences` | Per-rank `Divergence` entries; usually one per diverging rank |
| `per_rank_similarity` | Fraction of steps that agreed (1.0 = identical) |
| `summary` | Human-readable sentence describing the divergence |

Each `Divergence` entry contains:

| Field | Meaning |
|---|---|
| `rank` | Which distributed rank diverged |
| `first_divergence_step` | Step index of divergence |
| `baseline_action` | The AEP record from the reference run at that step |
| `candidate_action` | The AEP record from the candidate run at that step |
| `context_window` | Up to N baseline actions around the divergence for context |
| `correlated_escalation` | If an external escalation signal overlapped this step, its `metric_name=severity` string |
| `correlated_collision` | Whether a backend-detected desync collision occurred at this step |

### Example output inspection

```python
if report.divergences:
    div = report.divergences[0]
    print(f"Divergence at step {div.first_divergence_step} on rank {div.rank}")
    print(f"  Baseline: {div.baseline_action.collective_type} (mode={div.baseline_action.recording_mode})")
    print(f"  Candidate: {div.candidate_action.collective_type} (mode={div.candidate_action.recording_mode})")
    if div.correlated_escalation:
        print(f"  Escalation signal at divergence: {div.correlated_escalation}")
    print(f"  Context window ({len(div.context_window)} steps):")
    for rec in div.context_window:
        print(f"    step={rec.step} type={rec.collective_type}")
else:
    print("No divergence found — runs are equivalent.")
```

A `correlated_escalation` value such as `nccl_anomaly_score=0.95` indicates
that an external anomaly detector flagged this region; cross-reference with
your monitoring system at the corresponding training step to confirm.

---

## Step 5: Sign the Divergence Report (Optional)

To produce an auditor-attested report cryptographically bound to both source bundles:

```python
from train_replay.signing.signer import BundleSigner, verify_divergence_report
import dataclasses, json

signer = BundleSigner(private_key, "auditor-key-v1")
envelope = signer.sign_divergence_report(
    dataclasses.asdict(report), baseline, candidate
)
print(envelope.to_json())

# Verify later:
assert verify_divergence_report(envelope, baseline, candidate, public_key)
print("Divergence report signature verified.")
```

The envelope records the SHA-256 digest of both source bundles, so any
modification to either bundle invalidates the signature.
