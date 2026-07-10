# wasmagent-train-replay

> Causal evidence layer for distributed GPU training — cross-rank PROV-DM provenance graph and deterministic replay

Reads PyTorch Flight Recorder dumps and profiler hooks, builds a cross-rank causal graph,
records layered AEP evidence, and supports tracing any anomalous tensor back to its
origin rank and collective operation.

## Problem

PyTorch Flight Recorder gives you per-rank linear timestamp logs. When a gradient looks
wrong after an all-reduce, you still need to manually cross-reference dumps across all
ranks to find the desync. This project adds the missing layer:

- Cross-rank causal graph (PROV-DM Activity/Entity/Agent)
- `validation → delta → full` evidence recording — low cost by default, auto-escalates on risk signals
- Ed25519-signed `EpochEvidenceBundle` — tamper-evident, auditable
- CLI for ingestion, tracing, and replaying from any epoch

## Architecture

```
PyTorch Flight Recorder dump (.pkl)
        │
        ▼
┌─────────────────────────┐
│  collector              │  parse collective events per rank
│  ├── flight_recorder.py │
│  └── profiler_hook.py   │  tensor-level evidence via autograd hooks
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  graph                  │  build PROV-DM causal graph
│  ├── prov_graph.py      │  Activity / Entity / Agent + edges
│  └── builder.py         │  cross-rank graph construction
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  recording              │  AEP evidence collection
│  ├── modes.py           │  RecordingMode + compile_recording_policy
│  ├── evidence.py        │  AEPRecord + EpochEvidenceBundle
│  └── recorder.py        │  EpochRecorder (per-epoch accumulator)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐   ┌──────────────────┐
│  signing                │   │  replay           │
│  └── signer.py          │   │  └── replayer.py  │
│  Ed25519 / DSSE         │   │  causal ancestor  │
└─────────────────────────┘   │  traversal + CLI  │
                              └──────────────────┘
```

## Quick start

```bash
pip install -e ".[dev]"

# Run tests
make test

# Fault injection demo (no GPU required)
make demo

# CLI: ingest a Flight Recorder dump
train-replay ingest path/to/nccl_trace.pkl

# CLI: trace a tensor's causal ancestors
train-replay trace "tensor:2:3:out" path/to/nccl_trace.pkl

# CLI: record AEP evidence
train-replay record path/to/nccl_trace.pkl --run-id my-run --epoch 5
```

## Documentation

The [`docs/`](docs/) directory holds the full reference:

| Document | What it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System flow, component responsibilities, the PROV-DM data model, recording policy, and Ed25519 signing. |
| [docs/protocol.md](docs/protocol.md) | Field-by-field schemas for `EpochEvidenceBundle`, `AEPRecord`, `CollectiveEvent`, and `TensorEvent`. |
| [docs/integration.md](docs/integration.md) | Wiring `EvidenceProfilerHook` into a training loop, collecting Flight Recorder dumps, and an end-to-end trace example. |
| [docs/cli-reference.md](docs/cli-reference.md) | Complete reference for the `ingest`, `trace`, and `record` subcommands. |

## Recording modes

| Mode | Trigger | Stored |
|---|---|---|
| `validation` | read-only collectives, no anomaly | tensor hash + ordering metadata |
| `delta` | local mutations, low risk | statistical diff (mean/var/percentiles) |
| `full` | tainted input, external mutation, anomaly detected | full tensor snapshot (sampled) |

Mode escalation is automatic — driven by loss spikes, gradient norm anomalies,
or DCGM/GCM XID events. Mirrors the `compileToRecordingPolicy` logic from
[@wasmagent/capability-compiler](https://github.com/WasmAgent/wasmagent-js/tree/main/packages/capability-compiler).

## Relationship to wasmagent-js

Shares the AEP protocol schema and `RecordingMode` semantics with
[@wasmagent/aep](https://github.com/WasmAgent/wasmagent-js/tree/main/packages/aep).
The PROV-DM graph format is compatible — cross-environment causal chains
(gateway → agent process → training job) can be joined by shared `trace_id`.

## License

Apache-2.0
