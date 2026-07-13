# Strategy

## Competitive landscape (2026)

### PyTorch Flight Recorder + fr_trace
PyTorch's official tool (production use at Meta since early 2026):
- Cross-rank collective alignment by sequence ID
- Rank desync detection and heuristic root-cause classification (4 categories)
- Automated root-cause scripts bundled in `fr_trace`
- **Explicit roadmap**: extend from NCCL to MTIA, Gloo, and other backends

What it does NOT do:
- Cryptographic signing or integrity proof of evidence
- Support for non-LLM training workloads (RecSys, recommendation systems)
- Framework-agnostic abstraction layer for non-NCCL backends (yet)
- Automated, agent-driven root-cause reasoning via LLM

### NVIDIA NCCL Inspector
Released May 2026:
- Real-time performance monitoring with Prometheus/Grafana integration
- Bandwidth attribution for AllGather/ReduceScatter collectives
- Positioned as **performance** tooling, not causal forensics

### At Scale Conference 2026
Emerging theme: "Leveraging Agents to Debug NCCL Watchdog Timeouts" — the
consensus is that manual debugging (hours to weeks) is the bottleneck, and
agent-automated root-cause reasoning is the next frontier.

## Our differentiation

**Do not compete on data collection.** `fr_trace` already covers NCCL ingestion
and heuristic analysis well. Instead, own the three things it cannot do:

### 1. Tamper-evident evidence chain
`fr_trace` outputs are local pickle files — no integrity guarantee. When a
cloud customer needs to prove to an infrastructure vendor that a training
failure was caused by the vendor's hardware (not the user's code), they need
a cryptographically signed, externally verifiable record.

`EpochEvidenceBundle` + Ed25519 + DSSE envelope is the forensic layer
`fr_trace` will never build (it's not in Meta's use case).

**Actionable**: get the auditor guide and bundle serialization done so this
capability can be demonstrated to an actual external party.

### 2. Framework-agnostic causal graph
PyTorch's `fr_trace` NCCL-extension roadmap is public. Before they ship it,
the PROV-DM abstraction layer should be able to ingest Gloo/MTIA events using
the same `CollectiveEvent` dataclass interface — making wasmagent-train-replay
the only tool that covers heterogeneous training infrastructure.

Target workloads not covered by official tools: RecSys (recommendation system
training), Gloo-based CPU collectives, MTIA accelerators.

### 3. Agent-automated root-cause reasoning
The causal graph already supports `ancestors_of()` and `causal_subgraph()`.
Wrapping this in an LLM-callable tool interface would let an agent walk the
graph and generate hypotheses — directly addressing the "hours to weeks of
manual debugging" problem discussed at At Scale Conference 2026.

This is the highest-leverage Phase 5 investment: it turns the causal graph
from a developer library into an autonomous debugging agent.

## What to avoid

- **Real-time performance dashboards**: NCCL Inspector has Prometheus/Grafana.
  Do not rebuild this. The recording policy's auto-escalation feature
  (`validation → delta → full`) is the correct complement: consume NCCL
  Inspector anomaly signals as escalation triggers, don't duplicate the UI.
- **NCCL-only root-cause heuristics**: `fr_trace` owns this space. Our
  heuristics should be framework-agnostic and feed into the agent reasoning
  layer, not duplicate fr_trace's four-category classification.

## Priority order
1. CLI replay subcommand + missing tests (Phase 2 — trust prerequisites)
2. SAFE_MODE circuit-breaker (production safety prerequisite)
3. Bundle serialization + auditor guide (make the tamper-evidence claim real)
4. Framework-agnostic collector interface (Gloo/MTIA)
5. Agent reasoning layer (highest long-term value, needs 1-4 first)
