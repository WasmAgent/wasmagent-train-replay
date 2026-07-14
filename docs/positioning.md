# Positioning — wasmagent-train-replay vs. fr\_trace & NCCL Inspector

> How this project complements rather than competes with official PyTorch and NVIDIA
> tooling, and why a framework-agnostic, tamper-evident position matters now.

## Landscape (March 2026)

| Tool | Vendor | Scope | Integrity guarantees |
|---|---|---|---|
| **PyTorch fr\_trace / Flight Recorder** | Meta (PyTorch) | Per-rank NCCL timeline, cross-rank desync detection, heuristic root-cause classification | None — local pickle files, no signing |
| **NVIDIA NCCL Inspector** | NVIDIA | Real-time NCCL performance monitoring, bandwidth/utilisation dashboards | None — observability only |
| **wasmagent-train-replay** | WasmAgent | Cross-rank PROV-DM causal graph, Ed25519-signed evidence bundles, deterministic replay | Ed25519 (DSSE-style envelope), SHA-256 digests per bundle |

## Comparison with fr_trace

`fr_trace` and NCCL Inspector excel at real-time observability and per-rank
timeline debugging.  This project occupies a **complementary layer above** those
tools — cross-rank causal provenance backed by cryptographic evidence.
The detailed differentiation is covered in the two sections below and the
landscape table above.

## Two gaps official tooling does not cover

### 1. Tamper-proof evidence chain

`fr_trace` emits local pickle files with no integrity guarantees. An operator who
needs to **prove fault attribution to an external auditor** — for example, showing
that rank 3 desynced and corrupted a shared gradient — has no cryptographic assurance
that the trace was not modified after collection.

`wasmagent-train-replay` fills this gap:

- **EpochEvidenceBundle** is canonicalised (fields sorted, signature stripped) and
  Ed25519-signed into a DSSE-style envelope.
- **SHA-256 digests** are computed from canonical JSON (`sort_keys=True`) of
  the full graph structure (nodes, edges, attributes).  Stability depends on
  deterministic event ordering during collection and preservation of the
  original backend string via ``collective_type_raw`` — unknown operations
  are tagged as ``CollectiveOp.UNKNOWN`` rather than silently remapped to a
  known type.  Any post-hoc edit to the graph invalidates the digest.
- **Recording policy** (`validation → delta → full`) ensures cost-aware evidence
  capture: cheap by default, escalates on risk signals.

This makes the evidence suitable for:
- Post-incident audits where fault attribution must be non-repudiable
- Multi-party training (federated learning) where participants need tamper-evident logs
- Compliance scenarios requiring immutable training records

### 2. Framework-agnostic causal graph abstraction

`fr_trace` currently targets NCCL only. The PyTorch March 2026 Flight Recorder blog
named **MTIA** and **Gloo** as planned future backend targets, but the tooling is
backend-coupled today.

`wasmagent-train-replay` builds the causal graph on **backend-neutral abstractions**:

- **`CollectiveOp`** enumerates operations common across NCCL, Gloo, and MTIA
  (`all_reduce`, `all_gather`, `broadcast`, `reduce_scatter`, `send`, `recv`,
  `barrier`, `all_to_all`, `reduce`, `gather`, `scatter`).
- **`Backend`** tags each event with its communication backend
  (`NCCL`, `GLOO`, `MTIA`, `CUSTOM`).
- **`OpSpec`** describes a collective in backend-neutral terms (op type, rank,
  process group, sizes, timestamps) — the graph layer never inspects backend-specific
  fields.
- **`CollisionDetector`** protocol abstracts cross-rank desync detection so new
  backends plug in without modifying the graph core.

This means:
- Adding Gloo or MTIA support requires only a new collector adapter, not changes to
  the graph, recording, or replay layers.
- RecSys, recommendation, or other non-NLP training workloads that use Gloo get the
  same causal provenance as NCCL workloads.

## What we do NOT claim

- We do **not** replace real-time monitoring — NCCL Inspector is better for live
  bandwidth/utilisation dashboards.
- We do **not** replace per-rank timeline analysis — `fr_trace` is faster for
  single-rank debugging.
- We do **not** generate Flight Recorder dumps — we consume them (along with any
  future collector adapters).

## Positioning summary

```
                    Real-time monitoring
                    NCCL Inspector ◄────────────────── LIVE dashboards
                           │
                           ▼
              Per-rank timeline analysis
              fr_trace ◄────────────────── NCCL traces, desync heuristics
                           │
                           ▼
        Cross-rank causal provenance + tamper evidence
        wasmagent-train-replay ◄────── PROV-DM graph, Ed25519 bundles, replay
                           │
                           ▼
              External audit / compliance
              (this is the gap we fill)
```

## Roadmap

Concrete abstractions still needed (tracked in follow-up issues):

1. **Gloo collector adapter** — parse Gloo trace format into `CollectiveEvent`. (#56)
2. **MTIA collector adapter** — parse MTIA profiler output into `CollectiveEvent`. (#57)
3. **`CollisionDetector` backend implementations** — one per backend, plugged into
   the abstract protocol defined in `train_replay/graph/collision.py`. (#58)
4. **Bundle persistence** — serialise `EpochEvidenceBundle` to JSON/CBOR for
   long-term storage and cross-system transfer. (#59)
