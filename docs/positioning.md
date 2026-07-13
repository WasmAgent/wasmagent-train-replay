# Positioning

> How `wasmagent-train-replay` differs from PyTorch `fr_trace` and NVIDIA NCCL
> Inspector, and why it occupies a distinct niche in the distributed-training
> observability landscape.

## The competitive landscape

| Tool | Vendor | Primary function | Integrity guarantees |
|---|---|---|---|
| **PyTorch Flight Recorder + `fr_trace`** | Meta (PyTorch) | Per-rank NCCL collective alignment, cross-rank desync detection, heuristic root-cause classification | None — local pickle files, no signing or hashing |
| **NVIDIA NCCL Inspector** | NVIDIA | Real-time NCCL performance monitoring, Prometheus/Grafana dashboards | None — live metrics only, no persistent evidence |
| **wasmagent-train-replay** | WasmAgent | Cross-rank PROV-DM causal graph, Ed25519-signed evidence bundles, deterministic replay | **Ed25519 / DSSE envelope** per epoch bundle, SHA-256 digests |

## Where official tools stop

### 1. Tamper-proof evidence chain — the auditor gap

`fr_trace` emits local pickle files with no integrity guarantees. An operator can
open the file in a text editor, modify a timestamp, and the tooling will happily
consume the altered dump. This is fine for *debugging* but unacceptable for
*accountability*.

Scenarios where integrity matters:

- **Multi-tenant GPU clusters** — when a training job corrupts shared state,
  the infrastructure operator needs to prove to the tenant (or an external
  auditor) *which rank* and *which collective* introduced the fault.
- **Regulated ML pipelines** — financial or healthcare models may require
  tamper-evident training logs for compliance audits.
- **SLA enforcement** — cloud GPU providers need cryptographic proof that a
  training run executed as claimed before issuing SLA credits.

`wasmagent-train-replay` fills this gap:

- Every `EpochEvidenceBundle` is canonicalised (JSON with sorted keys,
  signature stripped) and SHA-256 digested before Ed25519 signing.
- The DSSE-style envelope (`BundleSigner`) attaches `alg`, `key_id`, and
  `sig` fields so verification is self-contained.
- Post-hoc modification of *any* field in a signed bundle invalidates the
  digest and the signature — detectable by `verify_bundle()`.
- The recording policy (`validation → delta → full`) provides cost-aware
  evidence capture: cheap by default, escalating only on risk signals.

This is not a feature `fr_trace` plans to add — its roadmap focuses on
performance analysis, not cryptographic auditing.

### 2. Framework-agnostic causal graph — the backend-neutral gap

`fr_trace` currently targets NCCL and plans to expand to MTIA and Gloo
(PyTorch March 2026 blog). Its architecture ties collective tracing to the
NCCL trace dump format (`_dump_nccl_trace()`).

`wasmagent-train-replay` builds on a **backend-neutral abstraction layer**:

- `BackendEvent` protocol (`train_replay/graph/base.py`) defines the minimal
  interface any backend event must satisfy — `rank`, `operation_type`,
  `sequence_id`, `timestamp_ns`, `group_id`.
- `GraphBuilder` protocol (`train_replay/graph/base.py`) defines how a builder
  converts a list of backend events into a `ProvGraph`.
- The existing `NCCLGraphBuilder` is one concrete implementation; future
  `GlooGraphBuilder` and `MTIAGraphBuilder` implementations plug into the same
  graph without code duplication.
- The PROV-DM graph (`ProvGraph`) itself is backend-agnostic — it models
  Activity / Entity / Agent with PROV-DM edges and is queried identically
  regardless of which backend produced the events.

The window to establish this abstraction is now: PyTorch has announced MTIA/Gloo
as future targets but has not shipped them. By building the abstraction layer
before official tooling covers these backends, `wasmagent-train-replay` can
offer RecSys/Gloo/MTIA causal graph support ahead of `fr_trace`.

## What this project does NOT try to replace

| Capability | Handled by |
|---|---|
| Real-time NCCL performance profiling | NCCL Inspector |
| NCCL collective alignment analysis | `fr_trace` |
| GPU hardware telemetry (SM occupancy, memory bandwidth) | DCGM / Nsight Systems |
| Training loop optimisation (gradient accumulation, mixed precision) | PyTorch native tooling |

`wasmagent-train-replay` consumes the *outputs* of these tools (Flight Recorder
dumps, profiler hook events) and adds the provenance and integrity layer they
don't provide.

## Architecture of the evidence chain

```
Backend event source                    wasmagent-train-replay
─────────────────                       ──────────────────────
NCCL trace dump ──────┐
                       ├── BackendEvent ──→ GraphBuilder ──→ ProvGraph
Gloo trace dump ──────┤    (protocol)       (protocol)      (backend-agnostic)
                       │
MTIA trace dump ──────┘
                                                            │
                                                            ▼
                                                     EpochRecorder
                                                            │
                                                            ▼
                                                   EpochEvidenceBundle
                                                            │
                                                            ▼
                                                     BundleSigner
                                                     (Ed25519 / DSSE)
```

The left column (event sources) is where backend-specific parsing happens.
Everything to the right of `BackendEvent` is backend-neutral.

## Follow-up work

Concrete abstractions still needed (open issues):

1. **Gloo backend adapter** — parse Gloo trace dumps into `BackendEvent`
   implementations and implement a `GlooGraphBuilder`.
2. **MTIA backend adapter** — same for MTIA/NVIDIA hardware traces.
3. **Collector-layer `BackendAdapter` protocol** — abstract the parsing step
   so `EpochRecorder.record_collective()` accepts any `BackendEvent` rather
   than `CollectiveEvent` directly.
4. **Evidence bundle persistence** — serialize `EpochEvidenceBundle` to JSON/CBOR
   for on-disk storage and cross-process verification.
5. **Replay CLI subcommand** — wire `EpochReplayer` into the `train-replay`
   CLI (issue #10).

## Cross-environment compatibility

The `EpochEvidenceBundle` format and `RecordingMode` semantics are shared with
[`@wasmagent/aep`](https://github.com/WasmAgent/wasmagent-js/tree/main/packages/aep).
Cross-environment causal chains (gateway → agent process → training job) can
be joined by a shared `trace_id`, so a tensor anomaly traced here correlates
with agent-layer evidence recorded elsewhere.
