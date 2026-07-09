# wasmagent-train-replay — CLAUDE.md

## Project overview
Python package that builds a cross-rank causal provenance graph for distributed GPU training,
records layered AEP (Agent Evidence Protocol) evidence, and supports deterministic replay
from any epoch. Consumes PyTorch Flight Recorder dumps and profiler hooks.

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

### Missing (open issues)
- CLI `replay` subcommand (issue #10) — EpochReplayer exists, just needs CLI wiring
- Tests for `profiler_hook` and `signing` (issue #11)
- Multi-rank integration test (issue #12)


## Key references — no docs/ directory

| Reference | What it covers |
|-----------|---------------|
| `README.md` | Architecture, quick start, CLI commands |
| `train_replay/graph/prov_graph.py` | PROV-DM graph — the core data model |
| `train_replay/recording/evidence.py` | EpochEvidenceBundle — the signed record format |
| `train_replay/collector/flight_recorder.py` | How PyTorch FR dumps are parsed |
| `tests/test_prov_graph.py` | Shows how the graph is built and queried |
| `tests/test_recording.py` | Shows recording policy (validation/delta/full) |

Read README.md first. For any new feature, read the relevant source file
and its test to understand the existing contract before modifying.

## Roadmap

### Phase 2: Complete CLI and test coverage (issues #10-#12)
- [x] #1 test coverage for all modules
- [x] #2 train-replay record CLI command
- [x] #3 CONTRIBUTING.md
- [x] #5 fix causal graph traversal
- [x] #8 node importance scoring
- [ ] #10 feat: replay CLI subcommand
- [ ] #11 test: EvidenceProfilerHook + Ed25519 signing tests
- [ ] #12 test: multi-rank integration test

### Phase 3: Real PyTorch integration
- [ ] feat: profiler_hook integration with torch.autograd register_hook
- [ ] feat: multi-dump ingestion (list of .pkl files, one per rank)
- [ ] feat: cli ingest-multi for cross-rank dumps with automatic rank detection

### Phase 4: Production readiness
- [ ] feat: EpochEvidenceBundle serialization to JSON/CBOR for persistence
- [ ] feat: replay --output flag writes causal report to file
- [ ] perf: streaming parser for large Flight Recorder dumps (>1GB)
- [ ] feat: anomaly detection — flag tensors with abnormal gradients automatically

## How patrol sweep discovers new issues
The patrol sweep reads this CLAUDE.md roadmap section.
When an issue is closed and its checkbox can be ticked, patrol will:
1. Tick the checkbox in this file
2. Open the next unchecked issue in the roadmap
This creates a self-driving development loop.
