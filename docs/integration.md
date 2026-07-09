# Integration guide

> How to wire `wasmagent-train-replay` into a real PyTorch distributed training
> job: capture tensor-level events, dump Flight Recorder traces, build the
> cross-rank causal graph, and trace a gradient anomaly back to its origin rank.

This guide shows the live-training side. For the formats these steps produce see
[protocol.md](protocol.md); for the CLI that consumes them see
[cli-reference.md](cli-reference.md).

## Prerequisites

- Python 3.10+
- PyTorch ≥ 2.2.0 with NCCL (the profiler hook and Flight Recorder rely on
  `torch` and `torch.distributed`)
- The package and its dev extras:

  ```bash
  pip install -e ".[dev]"
  ```

  Runtime dependencies (declared in `pyproject.toml`): `torch>=2.2.0`,
  `cryptography>=42.0`, `pydantic>=2.0`, `click>=8.0`, `rich>=13.0`,
  `networkx>=3.0`.

> **Testing without GPUs.** Every step below can be exercised without a GPU by
> synthesising events the way the fault-injection demo does — see
> `examples/fault_injection_demo.py`. The test suite (`pytest tests/`) never
> touches a real GPU or torch runtime.

## 1. Attach `EvidenceProfilerHook` to a training loop

`EvidenceProfilerHook` (in `train_replay/collector/profiler_hook.py`) collects
tensor-level `TensorEvent` records. It is deliberately minimal: you drive it
explicitly from your loop so it records exactly the tensors you care about.

```python
import torch.distributed as dist
from train_replay.collector.profiler_hook import EvidenceProfilerHook

rank = dist.get_rank()
hook = EvidenceProfilerHook(rank=rank)

for step, batch in enumerate(dataloader):
    hook.on_step_begin()                       # advances the internal step counter

    out = model(batch)
    loss = criterion(out, labels)
    loss.backward()

    # Record the tensors whose provenance you want to capture.
    hook.record_tensor("grad.weight", model.layer.weight.grad)
    hook.record_tensor("activations", out)

    optimizer.step()
    optimizer.zero_grad()
```

What this produces:

- `hook.events` → `list[TensorEvent]`, one per `record_tensor()` call.
- Each `TensorEvent.tensor_id` is `r{rank}:s{step}:{op_name}` and carries an
  optional `digest` (a cheap `sha256` over the first 4096 bytes of the flattened
  tensor, 16 hex chars). See [protocol.md § TensorEvent](protocol.md#tensorevent).

> **Where to attach deeper hooks.** The roadmap calls for wiring
  `record_tensor()` behind `torch.autograd` `register_hook` callbacks so
  gradient tensors are captured automatically; today you call `record_tensor()`
  explicitly. The `TensorEvent` schema already supports this richer integration.

## 2. Collect Flight Recorder dumps

Flight Recorder dumps are the per-rank timeline of NCCL collectives. PyTorch
exposes them through `torch._C._distributed_c10d._dump_nccl_trace()`, which
returns a dict with an `entries` list. Each entry becomes one `CollectiveEvent`
(see [protocol.md § CollectiveEvent](protocol.md#collectiveevent)).

### Capture in-process

```python
import pickle
import torch

# After your training step(s) have run collectives:
trace = torch._C._distributed_c10d._dump_nccl_trace()
with open(f"nccl_trace_rank{dist.get_rank()}.pkl", "wb") as f:
    pickle.dump(trace, f)
```

### Read it back

```python
from pathlib import Path
from train_replay.collector.flight_recorder import load_flight_recorder

events = load_flight_recorder(Path("nccl_trace_rank0.pkl"))
# events: list[CollectiveEvent]
```

`load_flight_recorder()` maps the dump's raw keys to the `CollectiveEvent`
fields (e.g. `pg_name` → `process_group`, `time_started_ns` → `start_time_ns`,
`input_sizes[0][0]` → `tensor_size`, `frames` → `call_stack`,
`seq_id` → `sequence_id`).

## 3. Build the causal graph from multi-rank dumps

Concatenate the `CollectiveEvent` lists from every rank, then build one graph.
`build_from_events()` is rank-agnostic: it creates a PROV-DM **Agent** per
unique `(rank, process_group)`, an **Activity** per collective, and input/output
**Entities** per collective, wired with `used`, `wasGeneratedBy`, and
`wasAssociatedWith` edges.

```python
from train_replay.collector.flight_recorder import load_flight_recorder
from train_replay.graph.builder import build_from_events
from pathlib import Path

# One dump per rank.
events = []
for r in range(world_size):
    events += load_flight_recorder(Path(f"nccl_trace_rank{r}.pkl"))

graph = build_from_events(events)
```

Node-id conventions (see [architecture.md § PROV-DM data model](architecture.md#prov-dm-data-model)):

| Node kind | Id pattern | Example |
|---|---|---|
| Agent | `rank:{rank}:pg:{process_group}` | `rank:2:pg:default` |
| Activity | `act:{rank}:{collective_type}:{sequence_id}` | `act:2:all_reduce:3` |
| Entity (input) | `tensor:{rank}:{sequence_id}:in` | `tensor:2:3:in` |
| Entity (output) | `tensor:{rank}:{sequence_id}:out` | `tensor:2:3:out` |

`build_from_events()` incorporates events from all ranks into a single graph —
this is exactly what `tests/test_integration.py` asserts (agents, activities,
and in/out entities present for every rank).

## 4. Record AEP evidence

Wrap the same events in an `EpochRecorder` to produce the signed evidence bundle.
The recorder runs each collective through the recording policy and emits one
`AEPRecord` per collective.

```python
from train_replay.recording.recorder import EpochRecorder
from train_replay.recording.modes import RiskContext, SideEffectClass
from train_replay.signing.signer import BundleSigner

recorder = EpochRecorder(run_id="run-42", epoch=5)
for evt in events:
    recorder.record_collective(evt)

# If your monitor flags rank 2 as anomalous, escalate that rank's evidence:
recorder.escalate_rank(2)   # rewrites every action on rank 2 to FULL mode

bundle = recorder.bundle()
signer, _pubkey = BundleSigner.generate(key_id="ci-key")
signer.sign(bundle)

print(bundle.digest())          # sha256 of canonical_bytes()
print(bundle.signature["key_id"])  # "ci-key"
```

`escalate_rank()` is how a detected anomaly retroactively upgrades evidence for
the suspect rank from `validation`/`delta` to `full`.

### Verify the signature (tamper-evidence round-trip)

Signing is only useful if an auditor can check it later. `verify_bundle()` is
the other half of the round-trip (`train_replay/signing/signer.py`): it
recomputes `canonical_bytes()` and checks the stored Ed25519 signature against
the public key. Because the signature covers the canonical bytes, *any* edit to
a signed bundle invalidates it.

```python
from train_replay.signing.signer import verify_bundle

# _pubkey is the Ed25519PublicKey returned by BundleSigner.generate() above.
assert verify_bundle(bundle, _pubkey)   # True — signature matches

# Tamper-evidence: mutate a field and the signature no longer verifies.
bundle.epoch = 999
assert not verify_bundle(bundle, _pubkey)   # False — canonical_bytes() changed
```

This is the property an auditor relies on: a recorded bundle plus its public key
proves the evidence was not modified after signing.

## 5. Trace a gradient anomaly back to its origin rank

`EpochReplayer` couples the graph (for *causality*) and the bundle (for
*risk*). Given the id of an anomalous output tensor, `find_root_cause()` walks
the `wasGeneratedBy` edges back to the producing activities, and
`suspicious_actions()` filters the bundle to FULL-mode actions.

```python
from train_replay.replay.replayer import EpochReplayer

replayer = EpochReplayer(graph)

# The output tensor that looks wrong (rank 2, sequence 3):
entity_id = "tensor:2:3:out"

result = replayer.replay_rank(bundle, rank=2, entity_id)
print(result.causal_ancestors)      # activity ids that produced this tensor
print(result.suspicious_actions)    # FULL-mode actions on rank 2
```

### End-to-end example (no GPU)

This is the same shape as `examples/fault_injection_demo.py`, which synthesises
4 ranks × 5 steps of `all_reduce` events, injects a `was_vetted` signal at
rank 2 / sequence 3, escalates rank 2, and traces `tensor:2:3:out`:

```python
from train_replay.collector.flight_recorder import CollectiveEvent
from train_replay.graph.builder import build_from_events
from train_replay.recording.recorder import EpochRecorder
from train_replay.recording.modes import RiskContext, SideEffectClass
from train_replay.replay.replayer import EpochReplayer

events = [
    CollectiveEvent(
        rank=rank, process_group="default", collective_type="all_reduce",
        src_rank=None, dst_rank=None, tensor_size=1024 * 1024,
        enqueue_time_ns=seq * 1_000_000,
        start_time_ns=seq * 1_000_000 + 100_000,
        end_time_ns=seq * 1_000_000 + 500_000,
        sequence_id=seq,
    )
    for rank in range(4) for seq in range(5)
]

graph = build_from_events(events)
recorder = EpochRecorder(run_id="demo-run", epoch=0)
for evt in events:
    risk = (RiskContext(was_vetted=True, side_effect_class=SideEffectClass.MUTATE_EXTERNAL)
            if (evt.rank == 2 and evt.sequence_id == 3) else None)
    recorder.record_collective(evt, risk)

recorder.escalate_rank(2)
bundle = recorder.bundle()

replayer = EpochReplayer(graph)
ancestors = replayer.find_root_cause("tensor:2:3:out")   # → includes act:2:all_reduce:3
suspicious = replayer.suspicious_actions(bundle)          # every FULL-mode action in the bundle
# (all 20 here: all_reduce classifies as mutate-external → full, see protocol.md)
# For rank-filtered results use replay_rank():
# replayer.replay_rank(bundle, rank=2, entity_id="tensor:2:3:out")
```

Run the shipped version with `make demo` (or
`python examples/fault_injection_demo.py`).

## CLI shortcuts

For interactive use, the same pipeline is available as CLI subcommands:

```bash
# Build the causal graph from one dump (optionally filter to a rank)
train-replay ingest path/to/nccl_trace.pkl --rank 2

# Trace a tensor's causal ancestors
train-replay trace "tensor:2:3:out" path/to/nccl_trace.pkl

# Record AEP evidence for an epoch
train-replay record path/to/nccl_trace.pkl --run-id run-42 --epoch 5
```

Full flag reference: [cli-reference.md](cli-reference.md).
