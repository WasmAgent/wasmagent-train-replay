# Architecture

> How `wasmagent-train-replay` turns per-rank Flight Recorder logs into a
> cross-rank causal evidence graph with tamper-evident, Ed25519-signed bundles.

This document is the system-level reference. For wire formats and field-by-field
schemas see [protocol.md](protocol.md). For wiring the profiler into a real
training loop see [integration.md](integration.md). For command-line usage see
[cli-reference.md](cli-reference.md). For the planned auditor evidence export
package see [export-command-design.md](export-command-design.md).

## Design goals

1. **Cross-rank causality.** PyTorch Flight Recorder gives a linear, per-rank
   timeline. This project stitches every rank into one PROV-DM graph so a bad
   tensor on rank *N* can be traced back to the collective (and rank) that
   produced it.
2. **Cost-aware evidence.** Not every collective needs a full tensor snapshot.
   The recording policy (`validation → delta → full`) records cheaply by default
   and escalates only on risk signals.
3. **Tamper-evidence.** Each epoch's evidence is canonicalised and Ed25519-signed
   into a DSSE-style envelope (Delegate Signing for Secure Environments) so
   audits can prove the record was not modified.

## System flow

```
PyTorch Flight Recorder dump (.pkl)        tensor-level events (profiler hook)
   torch._C._distributed_c10d                EvidenceProfilerHook
   ._dump_nccl_trace()                         record_tensor()
        │                                            │
        ▼                                            ▼
┌──────────────────────────────────────────────────────────────┐
│  collector                                                   │
│  ├── flight_recorder.py   load_flight_recorder()             │
│  │                         → list[CollectiveEvent]           │
│  └── profiler_hook.py     EvidenceProfilerHook.events        │
│                              → list[TensorEvent]             │
└──────────────────────┬───────────────────────────────────────┘
                       │ CollectiveEvent / TensorEvent
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  graph                  build_from_events(events)            │
│  ├── builder.py         cross-rank PROV-DM construction      │
│  └── prov_graph.py      ProvGraph (networkx.DiGraph)         │
│                          ancestors_of() / causal_subgraph()  │
└──────────────┬───────────────────────┬───────────────────────┘
               │                       │
               ▼                       ▼
┌──────────────────────────┐   ┌───────────────────────────────┐
│  recording               │   │  replay                        │
│  ├── modes.py            │   │  └── replayer.py               │
│  │   compile_recording_  │   │     EpochReplayer              │
│  │   policy()            │   │     find_root_cause()          │
│  ├── evidence.py         │   │     suspicious_actions()       │
│  │   EpochEvidenceBundle │   │     replay_rank() → ReplayResult│
│  └── recorder.py         │   └───────────────────────────────┘
│      EpochRecorder       │
└────────────┬─────────────┘
             │ EpochEvidenceBundle
             ▼
┌──────────────────────────────────────────────────────────────┐
│  signing               BundleSigner.sign(bundle)              │
│  └── signer.py         verify_bundle(bundle, pubkey)          │
│  Ed25519 / DSSE-style envelope                                 │
└──────────────────────────────────────────────────────────────┘
```

The flow is one-directional for ingestion, but `replay` reads back from the
graph *and* a recorded bundle to answer "which collectives caused this tensor,
and which of them were high-risk?"

In shorthand, the end-to-end evidence path is Flight Recorder (FR) ->
Collector -> PROV-DM Graph -> AEP Recorder -> Signing.

The current implementation keeps these stages as small, importable Python
modules. There is no background daemon or remote service in the ingestion path:
the CLI and tests call the same collector, graph, recording, replay, and signing
APIs documented below. A useful boundary to keep in mind is that the graph and
recording layers never parse raw PyTorch data directly; they receive
`CollectiveEvent` and `TensorEvent` dataclasses from the collector.

## Component responsibilities

| Component | Module | Responsibility |
|---|---|---|
| **collector** | `train_replay/collector/` | Parses external inputs into typed events: Flight Recorder pickle dumps (`CollectiveEvent`) and tensor-level autograd hook events (`TensorEvent`). |
| **graph** | `train_replay/graph/` | Builds the PROV-DM causal graph (`ProvGraph`) from events and answers causal queries (`ancestors_of`, `causal_subgraph`). |
| **recording** | `train_replay/recording/` | Decides *what* to record per collective via the recording policy, and accumulates it into an `EpochEvidenceBundle`. |
| **replay** | `train_replay/replay/` | Traces a tensor back to its causal ancestors and flags suspicious (FULL-mode) actions. |
| **signing** | `train_replay/signing/` | Ed25519-signs a bundle into a DSSE-style envelope and verifies signatures. |
| **anomaly** | `train_replay/anomaly/` | Builds a `TrainingProfile` from normal runs, detects statistical anomalies via Z-score/Isolation Forest, and produces `AnomalySignal` records. |
| **alerting** | `train_replay/alerting/` | Delivers formatted anomaly reports through pluggable notifiers (Slack, email). |
| **cli** | `train_replay/cli/` | `ingest`, `trace`, `record`, `anomaly` subcommands (entry point `train-replay`); the planned `export` subcommand is specified separately before implementation. |

### collector

`load_flight_recorder(path)` reads a pickle dump produced by
`torch._C._distributed_c10d._dump_nccl_trace()` and returns a list of
`CollectiveEvent` records — one per NCCL collective. The
`EvidenceProfilerHook` is the live-training counterpart: attach it to a loop,
call `record_tensor()` on each tensor of interest, and read back `TensorEvent`
records. The collector is the only component that touches external PyTorch
formats; everything downstream is pure Python dataclasses.

### graph

`build_from_events(events)` is where per-rank events become one graph. Each
`CollectiveEvent` becomes a PROV-DM **Activity**; its input/output tensors
become **Entities**; each (rank, process group) pair becomes an **Agent**.
Edges follow PROV-DM semantics (see [PROV-DM data model](#prov-dm-data-model)
below). The resulting `ProvGraph` is a thin wrapper over `networkx.DiGraph` and
exposes `ancestors_of(entity_id)` and `causal_subgraph(entity_id)` for
root-cause queries.

### recording

`EpochRecorder` accumulates one `EpochEvidenceBundle` per epoch. For every
collective it runs `compile_recording_policy(RiskContext)` to decide the
`RecordingMode` (`validation`, `delta`, or `full`), appends a
`AEPRecord` action, and can later bulk-escalate a rank via
`escalate_rank(rank)` when an anomaly is detected.

### replay

`EpochReplayer` couples the graph and a bundle. `find_root_cause(entity_id)`
walks the graph; `suspicious_actions(bundle)` filters the bundle to FULL-mode
actions; `replay_rank(bundle, rank, entity_id)` returns a `ReplayResult`
combining both. This is the layer an operator queries during a post-mortem.

### signing

`BundleSigner.sign(bundle)` canonicalises the bundle (signature field stripped,
fields sorted) and attaches an Ed25519 signature in a DSSE-style envelope stored
as `bundle.signature`. `verify_bundle(bundle, public_key)` recomputes the
canonical bytes and checks the signature. See
[Ed25519 signing](#ed25519-signing-dsse-envelope) below.

### cli

The `train-replay` entry point (defined in `pyproject.toml` as
`train-replay = "train_replay.cli.main:cli"`) wires the above together for
interactive use. Full flag reference: [cli-reference.md](cli-reference.md).
The future `export` command must keep the same component boundaries: collectors
normalize backend-specific traces into event records, `EpochRecorder` creates
or preserves the `EpochEvidenceBundle`, and export writes JSON/CBOR bundle
artifacts plus a manifest. Its full design contract is
[export-command-design.md](export-command-design.md).

## PROV-DM data model

The graph uses the W3C PROV-DM core types. Nodes carry a `kind` attribute
(`activity` | `entity` | `agent`); edges carry a `rel` attribute
(`used` | `wasGeneratedBy` | `wasAssociatedWith`).

### Node types

| Type | Dataclass | Key fields | Meaning |
|---|---|---|---|
| **Activity** | `ProvActivity` | `id`, `label`, `rank`, `process_group`, `timestamp_ns`, `collective_type` | One NCCL collective or kernel execution. Activity IDs produced by the builder: `act:{rank}:{collective_type}:{sequence_id}` |
| **Entity** | `ProvEntity` | `id`, `digest`, `rank`, `step` | One tensor/gradient at a specific (rank, step). Entity IDs produced by the builder: `tensor:{rank}:{sequence_id}:in` (input) and `tensor:{rank}:{sequence_id}:out` (output) |
| **Agent** | `ProvAgent` | `id`, `rank`, `process_group` | One rank / process group. Agent IDs produced by the builder: `rank:{rank}:pg:{process_group}` |

### Edge types

| Relation | Direction | Method | Semantics |
|---|---|---|---|
| `used` | activity → entity | `graph.used(act, ent)` | The activity consumed the entity as input. |
| `wasGeneratedBy` | activity → entity | `graph.was_generated_by(ent, act)` | The activity produced the entity. Stored internally as an activity→entity edge so that *predecessors* of an entity are its generating activities. |
| `wasAssociatedWith` | activity → agent | `graph.was_associated_with(act, agent)` | The activity ran under the given rank/process group. |

### Causal traversal

`ancestors_of(entity_id)` returns the activity IDs that causally contributed to
an entity. It traverses **only** `wasGeneratedBy` edges — `used` edges mark
*inputs*, not ancestry, so a bare input entity (consumed but never produced by a
known activity) correctly returns `[]`. `causal_subgraph(entity_id)` returns a
new `ProvGraph` containing just the entity and its ancestors, which is the unit
of evidence handed off for replay.

## Recording policy

The default recording stance is `validation`: keep enough ordering and digest
metadata to prove that a collective was observed without materialising the full
tensor. Validation checks and risk signals then escalate evidence capture to
`delta` or `full`:

- `delta` is selected for low-risk local mutation, where a statistical diff is
  enough to explain what changed.
- `full` is selected for unknown or external side effects, tainted inputs,
  consent/vetting anomalies, and post-hoc rank escalation.

In code, those checks are represented by `RiskContext` and compiled by
`compile_recording_policy(ctx)`.

The recording policy mirrors the `compileToRecordingPolicy` logic from
[@wasmagent/capability-compiler](https://github.com/WasmAgent/wasmagent-js/tree/main/packages/capability-compiler).
Given a `RiskContext`, `compile_recording_policy(ctx)` returns a
`RecordingPolicy(mode, reason)` evaluated in a fixed priority order:

| Priority | Condition | Mode | Reason string |
|---|---|---|---|
| 1 | `was_vetted` | `full` | `tool flagged by vetting` |
| 2 | `has_consent_anomaly` | `full` | `consent anomaly recorded` |
| 3 | `taint_chain_length > 0` **and** side effect ≠ `read` | `full` | `tainted input reaching state-changing call` |
| 4 | side effect == `unknown` | `full` | `unknown side-effect class` |
| 5 | side effect ∈ {`mutate-external`, `network-egress`} | `full` | `external mutation` |
| 6 | side effect == `mutate-local` | `delta` | `local mutation, low risk` |
| 7 | otherwise (side effect == `read`, no anomaly) | `validation` | `read-only, no anomaly` |

### What each mode stores

| Mode | Trigger | Stored |
|---|---|---|
| `validation` | read-only collectives, no anomaly | tensor hash + ordering metadata |
| `delta` | local mutations, low risk | statistical diff (mean/var/percentiles) |
| `full` | tainted input, external mutation, vetting flag, or anomaly | full tensor snapshot (sampled) |

### Side-effect classification for collectives

`EpochRecorder` classifies a collective's side effect via
`_collective_side_effect(ctype)`: collective types `recv` and `barrier` are
treated as `read`; everything else is treated as `mutate-external`. An explicit
`RiskContext` passed to `record_collective(evt, risk_override=...)` overrides
this default — this is how the fault-injection demo injects a `was_vetted`
signal. Escalation is also possible after the fact: `escalate_rank(rank)`
rewrites every existing action on a rank to `full` mode, which is how a
detected anomaly retroactively upgrades the evidence for the suspect rank.

## Ed25519 signing (DSSE-style envelope)

Every `EpochEvidenceBundle` can be signed into a DSSE-style envelope by
`BundleSigner`. DSSE stands for Delegate Signing for Secure Environments; in
this repository the envelope is implemented as the `signature` dictionary on
the bundle rather than as a separate protobuf or JSON document.

### Canonicalisation

Before signing, `bundle.canonical_bytes()` strips the `signature` field and
serialises the rest with `json.dumps(..., sort_keys=True, default=str)`. The
bundle digest is `sha256(canonical_bytes())` and is stable across runs as long
as the actions are unchanged — so any post-hoc edit of a signed bundle changes
the digest and invalidates the signature.

### Signature envelope

`BundleSigner.sign(bundle)` attaches:

```json
{
  "alg": "ed25519",
  "key_id": "<signer key_id, default 'dev-key'>",
  "sig": "<base64 Ed25519 signature of canonical_bytes()>"
}
```

### Verification

`verify_bundle(bundle, public_key)` returns `False` if there is no signature or
the signature does not verify against `canonical_bytes()`; otherwise `True`.
`BundleSigner.generate(key_id)` is the convenience constructor for tests and
development: it generates a fresh Ed25519 keypair and returns the signer plus
the matching public key.

## Anomaly Detection Pipeline

Milestone 5 adds an automated anomaly detection and alerting path that sits
between the PROV-DM graph layer and the replay/recording layers. The pipeline
operates in three stages:

```

CollectiveEvent / TensorEvent timeline
            │
            ▼
┌──────────────────────────────────────────────────────────────────┐
│  anomaly                                                         │
│  └── profile.py    TrainingProfile.fit_on_normal_run(events)     │
│                       → TrainingProfile (baseline statistics)    │
└──────────────────────────┬───────────────────────────────────────┘
                           │ TrainingProfile
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  anomaly                                                         │
│  └── detector.py   StatisticalAnomalyDetector.detect(events,     │
│                       profile)                                    │
│                       → list[AnomalySignal]                      │
└──────────────────────────┬───────────────────────────────────────┘
                           │ AnomalySignal
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  alerting                                                        │
│  └── notifier.py   AlertNotifier.send_alert(anomaly)             │
│                       SlackAlertNotifier / EmailAlertNotifier    │
└──────────────────────────────────────────────────────────────────┘
```

### Stage 1 — Baseline profiling

`TrainingProfile.fit_on_normal_run(events)` consumes a representative event
timeline from a known-good training run and computes baseline statistics: event
intervals, tensor value distributions (mean, variance, percentiles), and
per-collective-type operation patterns. The resulting `TrainingProfile` is
serialised to disk and reused for comparison against subsequent runs.

### Stage 2 — Statistical detection

`StatisticalAnomalyDetector.detect(events, profile)` compares a live (or
replayed) event timeline against the baseline profile using Z-score or
Isolation Forest methods on event timing and tensor statistics. It returns a
ranked `list[AnomalySignal]`, each carrying an anomaly score, the offending
event, and a human-readable description. An anomaly signal can also be passed
directly to `compile_recording_policy()` as the `anomaly_signal` parameter to
force escalation to `RecordingMode.FULL`.

### Stage 3 — Alerting

`AlertNotifier.send_alert(anomaly)` delivers a formatted anomaly report. Two
implementations are provided: `SlackAlertNotifier` (posts to a Slack webhook)
and `EmailAlertNotifier` (sends via SMTP). The notifier is a pluggable
interface so additional backends (PagerDuty, webhooks, etc.) can be added
without modifying the detection logic.

### Integration with existing components

- **Replay layer**: `EpochReplayer.anomaly_scan()` runs the statistical
detector over an event timeline and returns ranked anomalies with confidence
scores, complementing the existing `find_root_cause()` and
`suspicious_actions()` methods.
- **Recording layer**: An `AnomalySignal` feeds into `compile_recording_policy()`
to escalate evidence capture for anomalous collectives.
- **CLI**: The `train-replay anomaly` subcommand orchestrates the full
pipeline end-to-end: load a profile, scan a dump, optionally notify.

## Cross-environment compatibility

The PROV-DM graph format and the `RecordingMode` semantics are shared with
[@wasmagent/aep](https://github.com/WasmAgent/wasmagent-js/tree/main/packages/aep).
Cross-environment causal chains (for example gateway → agent process → training
job) can be joined by a shared `trace_id`, so a tensor anomaly traced here can
be correlated with agent-layer evidence recorded elsewhere.
