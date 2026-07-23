# wasmagent-train-replay — CLAUDE.md

## Project overview
Python package that builds a cross-rank causal provenance graph for distributed GPU training,
records layered AEP (Agent Evidence Protocol) evidence, and supports deterministic replay
from any epoch. Consumes PyTorch Flight Recorder dumps and profiler hooks.


## Repository maturity

| | |
|---|---|
| **Status** | Experimental |
| **Contract stability** | Evolving |
| **Recommended for** | Distributed training audit; tamper-evident epoch evidence |
| **Not recommended for** | General PyTorch profiling (use `fr_trace`); real-time monitoring |

## Repository Boundaries

### This repository owns
- Cross-rank causal provenance graph (PROV-DM) for distributed GPU training
- `EpochEvidenceBundle` — Ed25519-signed, tamper-evident evidence bundle per epoch
- Recording policy (`validation → delta → full`) for training workloads
- CLI: `ingest`, `trace`, `record`, `replay`
- PyTorch Flight Recorder parser and profiler hooks
- Agent-automated root-cause reasoning layer (Phase 5)

### Other repositories own — do not duplicate here

| Capability | Owner |
|---|---|
| AEP + compliance **schema definitions** | `wasmagent-protocol` (`@wasmagent/protocol`) |
| AEP evidence **emission** at runtime | `wasmagent-js` (`@wasmagent/aep`) |
| Runtime MCP firewall / process-level evidence | `wasmagent-js` |
| Gateway-level HTTP evidence (Proxy-Wasm) | `wasmagent-proxy` |
| Enterprise audit report, regulatory mapping | `open-agent-audit` |
| Trust Passport specification and product | `open-agent-audit` (`@openagentaudit/passport`) |
| AgentBOM / MCP Posture specifications | `agent-trust-infra` |
| Evidence admission score, training-data pipeline | `trace-pipeline` |
| Dynamic evaluation protocol | `fresharena` |

### Allowed cross-repo patterns
- `EpochEvidenceBundle` emits records in AEP format — the AEP schema is defined in `wasmagent-protocol` (`@wasmagent/protocol`); never redefine the schema locally.
- Signed bundles are downstream input for `open-agent-audit` and `trace-pipeline`; keep serialization format (JSON/CBOR) stable and versioned.
- Do not duplicate general Flight Recorder collection already covered by `fr_trace` or NVIDIA NCCL Inspector — focus on tamper-evident signing and causal graph layers.
- LLM-callable root-cause tool (Phase 5) wraps existing `find_root_cause()` output; LLM routing goes via `wasmagent-js` smartrouter, not a local provider selection.

## Key concepts
- **PROV-DM**: W3C provenance model (Activity/Entity/Agent) used for causal graph
- **EpochEvidenceBundle**: Ed25519-signed, tamper-evident evidence bundle per training epoch
- **Recording policy**: validation → delta → full, auto-escalates on risk signals
- **AEP evidence**: Records emitted in Agent Evidence Protocol format for audit

## Tech stack
- Python 3.10+
- Package managed with hatchling (`pyproject.toml`)
- Tests: pytest (`tests/`)
- Lint: ruff, typecheck: mypy

## Build and test
```bash
pip install -e ".[dev]"
pytest tests/
```

## Code structure
```
train_replay/      — main package
  collector/       — Flight Recorder parser, profiler hooks
  graph/           — PROV-DM causal graph builder
  evidence/        — AEP evidence recording and signing
  cli/             — CLI for ingest, trace, replay
docs/              — architecture, protocol, integration, CLI reference, strategy
tests/             — pytest test suite
examples/          — example Flight Recorder dumps and usage
```

## Bot instructions
- All new code must have corresponding tests in `tests/`
- Use type annotations throughout (mypy strict mode)
- Follow ruff lint rules (no unused imports, consistent style)
- Do not add dependencies without updating `pyproject.toml`
- The verify command is: `pytest tests/`
- Never use real GPU/torch in tests — use mocks and fixtures
- Keep each function small and single-purpose

## Strategic positioning

**Read `docs/strategy.md` before opening new issues or designing new features.**

This project's differentiation is NOT "another Flight Recorder reader." PyTorch's
official `fr_trace` and NVIDIA NCCL Inspector already cover data collection and
real-time performance monitoring. Our defensible niche is:

1. **Tamper-evident evidence chain** — Ed25519-signed `EpochEvidenceBundle` is
   forensic-grade; `fr_trace` outputs are local pickle files with no integrity
   guarantee.
2. **Framework-agnostic causal graph** — PROV-DM abstraction layer can cover
   non-NCCL backends (Gloo, MTIA) and non-LLM training (RecSys) that official
   tools have explicitly deferred.
3. **Agent-automated root-cause reasoning** — causal ancestor traversal feeding
   into an LLM-based hypothesis layer.

Do NOT duplicate what `fr_trace` already does well.

**SAFE_MODE** is a prerequisite for production use.

## Current implementation status

### Completed (all tests passing)
- `collector/flight_recorder.py` — parses PyTorch Flight Recorder pickle dumps
- `collector/profiler_hook.py` — EvidenceProfilerHook for tensor-level events (NO TESTS YET)
- `graph/prov_graph.py` — PROV-DM graph with ancestor traversal and importance scoring
- `graph/builder.py` — builds ProvGraph from CollectiveEvent lists across ranks
- `recording/modes.py` — recording policy (validation/delta/full escalation)
- `recording/recorder.py` — EpochRecorder writes AEP bundles
- `recording/evidence.py` — EpochEvidenceBundle dataclass
- `replay/replayer.py` — EpochReplayer.find_root_cause() and suspicious_actions()
- `signing/signer.py` — Ed25519 signing of bundles (NO TESTS YET)
- CLI: `ingest`, `trace`, `record` commands
- `docs/` — architecture, protocol, integration, CLI reference

### Missing (open issues)
- CLI `replay` subcommand — issue #35 (PRs #37, #41 open)
- Tests for `profiler_hook` and `signing` — issue #11
- Multi-rank integration test — issue #12
- SAFE_MODE circuit-breaker — issue #34 (PR #36 open)
- Differentiation strategy doc in `docs/strategy.md` — issue #45 (PR #47, #48 open)
- Agent root-cause reasoning layer — issue #44

## Key references

| Reference | What it covers |
|-----------|---------------|
| `README.md` | Architecture, quick start, CLI commands |
| `docs/strategy.md` | **Strategic positioning, competitive landscape, differentiation** |
| `docs/architecture.md` | System flow, component responsibilities, PROV-DM data model, recording policy, Ed25519 signing |
| `docs/protocol.md` | Field-by-field schemas for `EpochEvidenceBundle`, `AEPRecord`, `CollectiveEvent`, and `TensorEvent` |
| `docs/integration.md` | Wiring `EvidenceProfilerHook` into a training loop, collecting Flight Recorder dumps, end-to-end trace example |
| `docs/cli-reference.md` | Complete reference for the `ingest`, `trace`, and `record` subcommands |
| `train_replay/graph/prov_graph.py` | PROV-DM graph — the core data model |
| `train_replay/recording/evidence.py` | EpochEvidenceBundle — the signed record format |
| `train_replay/collector/flight_recorder.py` | How PyTorch FR dumps are parsed |
| `tests/test_prov_graph.py` | Shows how the graph is built and queried |
| `tests/test_recording.py` | Shows recording policy (validation/delta/full) |

Read README.md first. For any new feature, read the relevant source file
and its test to understand the existing contract before modifying.

## Roadmap

See `docs/15-milestones.md` — that file is the single authoritative milestone list.
Patrol reads `docs/15-milestones.md` directly for bullet-to-issue conversion.

Do not maintain a parallel roadmap here; any inline bullets here would cause
duplicate issues alongside the milestone-status pipeline.

## How patrol sweep discovers new issues
The patrol sweep reads `docs/15-milestones.md` for this project.
When an issue is closed and its checkbox can be ticked, patrol will:
1. Tick the checkbox in `docs/15-milestones.md`
2. Open the next unchecked bullet as a new issue
This creates a self-driving development loop.
