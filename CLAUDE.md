# wasmagent-train-replay

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

## Roadmap / next issues

The following features are planned. Bot should implement these in order.
When an issue is closed, patrol sweep will check this list and open the next one.

### Phase 2: Complete CLI and test coverage
- [ ] #10 feat: replay CLI subcommand
- [ ] #11 test: EvidenceProfilerHook + Ed25519 signing tests
- [ ] #12 test: multi-rank integration test

### Phase 3: Real PyTorch integration
- [ ] feat: profiler_hook integration with torch.autograd (register_hook)
- [ ] feat: multi-dump ingestion (accept list of .pkl files, one per rank)
- [ ] feat: cli ingest-multi command for cross-rank dumps

### Phase 4: Production readiness
- [ ] feat: EpochEvidenceBundle serialization to JSON/CBOR
- [ ] feat: replay --output flag writes causal report to file
- [ ] perf: streaming parser for large Flight Recorder dumps (>1GB)
