# Positioning: why wasmagent-train-replay alongside official tooling

> How `wasmagent-train-replay` differentiates from PyTorch Flight Recorder /
> `fr_trace` and NVIDIA NCCL Inspector, and why the project should exist despite
> overlapping surface areas.

## Audience

Infrastructure engineers, ML platform teams, and auditors who need to
**prove** what happened in a distributed training job — not just see it.

---

## Official tooling landscape (as of 2026)

| Tool | Maintainer | Focus | Integrity guarantees |
|------|-----------|-------|---------------------|
| **PyTorch Flight Recorder** (`torch._C._distributed_c10d._dump_nccl_trace`) | PyTorch Core | Per-rank linear NCCL timeline, heuristic root-cause classification | None (local pickle files) |
| **`fr_trace`** | PyTorch OSS | NCCL collective alignment, cross-rank desync detection, heuristic classification | None (local pickle files) |
| **NCCL Inspector** | NVIDIA | Real-time NCCL performance monitoring, bandwidth/latency diagnostics | None (live metrics, no persistent audit trail) |
| **`wasmagent-train-replay`** | WasmAgent | Cross-rank **PROV-DM causal graph**, **Ed25519-signed** evidence chains, framework-agnostic graph abstraction | ✅ Ed25519 signatures over canonical JSON, hash-linked attachment chain |

## Two gaps official tools do not fill

### 1. Tamper-proof evidence chain

`fr_trace` emits local pickle files with **no integrity guarantees**. An operator
or auditor has no cryptographic proof that the dump has not been modified after
collection — or that a specific dump corresponds to the training run it claims
to represent.

`wasmagent-train-replay` addresses this with:

- **`EpochEvidenceBundle`** — a canonical JSON record of every collective's
  recording mode and metadata, serialised with `json.dumps(sort_keys=True)`.
- **Ed25519 signing** (via `BundleSigner`) — `canonical_bytes()` is signed
  into a DSSE-style envelope stored directly on the bundle.
- **`EvidenceAttachment` hash chain** — each attachment binds a graph state
  digest to a bundle digest, forming a hash-linked chain that makes removal,
  reordering, or insertion of attachments cryptographically detectable.

This is essential for:

- **Regulated environments** (finance, healthcare, defence) where training
  outcomes must be auditable by a third party.
- **Fault attribution disputes** between teams (“rank 3 produced a bad
  gradient, but the evidence bundle proves the collective ran correctly on
  rank 3’s side”).
- **Post-hoc forensic analysis** where the original dumps are suspect.

`fr_trace` and NCCL Inspector do not offer any equivalent.

### 2. Framework-agnostic causal graph abstraction

Both `fr_trace` and NCCL Inspector are **NCCL-only**. PyTorch’s March 2026
Flight Recorder blog announced MTIA and Gloo as planned future targets, but
today neither tool works outside NCCL.

`wasmagent-train-replay` decouples the causal graph from the backend:

- The **`CollectiveEvent`** schema (`rank`, `process_group`, `collective_type`,
  timestamps, `sequence_id`) is backend-neutral — it represents a generic
  collective operation.
- The **`build_from_events()`** graph builder operates on `CollectiveEvent`
  lists, regardless of whether they originated from NCCL, Gloo, MTIA, or
  synthetic test data.
- **`ProvGraph`** (wrapping `networkx.DiGraph`) and the PROV-DM data model
  (Activity / Entity / Agent) are purely topological — they have no concept of
  “NCCL” or “GPU”.
- **`EpochReplayer`** for causal ancestor traversal works on the graph alone;
  the recording policy is decoupled from the graph layer.

This means the same replay code that traces a `tensor:0:1:out` from an NCCL
`all_reduce` will also trace a `tensor:0:1:out` from a Gloo `all_gather` or
an MTIA `barrier` — **without code changes** to the graph or replay layers.

**The window to establish this framework-agnostic position is now**, before
PyTorch expands `fr_trace` into MTIA/Gloo territory with the same NCCL-centric
assumptions baked in.

## What we are not

- **Not a Flight Recorder replacement.** We consume Flight Recorder dumps
  (via `load_flight_recorder()`) as one input source among possible future
  sources. We do not compete with FR’s real-time tracing.
- **Not a performance monitor.** NCCL Inspector is the right tool for
  bandwidth, latency, and topology diagnostics in real time.
- **Not a desync detector.** `fr_trace`’s cross-rank alignment and heuristic
  root-cause classification are complementary to our replay layer. We can
  consume `fr_trace` output as a `RiskContext` signal to escalate recording
  mode (see `compile_recording_policy`).

## When to reach for which tool

| Scenario | Tool |
|----------|------|
| “Is my NCCL collective running at expected bandwidth?” | **NCCL Inspector** |
| “Rank 3 desynced from rank 0 — where in the timeline?” | **fr_trace** |
| “I need to prove to an external auditor that this gradient anomaly was not caused by tampering.” | **wasmagent-train-replay** |
| “My training uses Gloo / MTIA / a custom backend — I still want causal tracing.” | **wasmagent-train-replay** |
| “I want an end-to-end CI pipeline that signs and verifies evidence bundles for every epoch.” | **wasmagent-train-replay** |

## Summary

`wasmagent-train-replay` occupies the **evidence chain** and
**framework-agnostic causal graph** spaces that official tools explicitly leave
unfilled. It does not try to replace `fr_trace`’s timeline alignment or NCCL
Inspector’s real-time monitoring; instead it adds a durable, signed, cross-backend
provenance layer on top.

---

### See also

- [`docs/architecture.md`](architecture.md) — system design and the PROV-DM data model.
- [`docs/protocol.md`](protocol.md) — record schemas and signing format.
- [`docs/integration.md`](integration.md) — wiring the profiler into a training loop.
- [`docs/follow-up-issues.md`](follow-up-issues.md) — concrete abstractions identified by the audit that are needed for full framework-agnostic coverage.
