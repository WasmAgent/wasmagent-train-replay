# Milestones

## Milestone 0 — Foundation (Completed Core)

> Core architecture implemented and functional. These items are already shipped.

- [x] `train_replay/collector/flight_recorder.py` — PyTorch Flight Recorder dump parser (`CollectiveEvent`, `load_flight_recorder`)
- [x] `train_replay/collector/profiler_hook.py` — Autograd/profiler hooks for tensor-level evidence (`EvidenceProfilerHook`, `TensorEvent`)
- [x] `train_replay/graph/prov_graph.py` — PROV-DM causal graph (`ProvGraph`, `ProvActivity`, `ProvEntity`, `ProvAgent`)
- [x] `train_replay/graph/builder.py` — Cross-rank graph construction from collective events (`build_from_events`)
- [x] `train_replay/recording/modes.py` — AEP recording modes + policy compiler (`RecordingMode`, `compile_recording_policy`)
- [x] `train_replay/recording/evidence.py` — Evidence data types (`TrainActionEvidence`, `EpochEvidenceBundle`)
- [x] `train_replay/recording/recorder.py` — Per-epoch evidence accumulation (`EpochRecorder`)
- [x] `train_replay/replay/replayer.py` — Causal ancestor traversal and suspicious action detection (`EpochReplayer`)
- [x] `train_replay/signing/signer.py` — Ed25519 DSSE-style signing (`BundleSigner`, `verify_bundle`)
- [x] `train_replay/cli/main.py` — CLI entry points: `ingest`, `trace`, `record`
- [x] `tests/test_prov_graph.py` — PROV-DM graph unit tests (ancestor traversal, subgraph, multi-hop chains)
- [x] `tests/test_recording.py` — Recording mode policy unit tests (validation/delta/full escalation logic)
- [x] `examples/fault_injection_demo.py` — End-to-end demo with synthetic 4-rank trace, corruption injection, causal tracing
- [x] `make demo` runs `examples/fault_injection_demo.py` end-to-end and exits 0

## Milestone 1 — Core Hardening & Test Coverage

> Expand test coverage to every module, add synthetic fixture data, and harden parsing against edge cases.

- [ ] `tests/test_collector.py` covers `CollectiveEvent` parsing edge cases and malformed pickle inputs with fixture data
- [ ] `tests/test_profiler_hook.py` covers `TensorEvent` collection, SHA-256 digest computation, and large-tensor truncation
- [ ] `tests/test_recorder.py` covers `EpochRecorder.record_collective`, `escalate_rank`, bundle serialization, and end-to-end sequences with `RiskContext` overrides
- [ ] `tests/test_signing.py` covers `sign` → `verify` round-trip, tamper detection on modified bundles, and missing signature rejection
- [ ] `tests/test_replayer.py` covers `find_root_cause`, `suspicious_actions` across mixed-mode bundles, and `replay_rank` output
- [ ] `tests/test_graph_builder.py` validates cross-rank edge correctness and node identity on synthetic multi-rank traces
- [ ] `examples/synthetic_trace.pkl` contains fixture Flight Recorder dump usable by all test suites without GPU

## Milestone 2 — Cross-Rank Auto-Escalation & Risk Signals

> Implement automated risk signal detection and multi-rank orchestration.

- [ ] `train_replay/recording/recorder.py` escalates to `FULL` mode on loss spike detection (configurable threshold)
- [ ] `train_replay/recording/recorder.py` escalates to `FULL` mode on gradient norm anomalies (configurable threshold)
- [ ] `train_replay/recording/recorder.py` escalates to `FULL` mode on DCGM/GCM XID events via callback hook
- [ ] `train_replay/recording/recorder.py` tracks per-mode action counts and logs escalation trigger reasons
- [ ] CLI `ingest` command supports multi-rank dump merging with `--ranks` option for selective rank filtering (currently accepts only `--rank` singular)
- [ ] CLI `trace` command outputs causal chain as structured JSON (`--format json`) alongside the rich table
- [ ] `tests/test_escalation.py` validates each risk signal trigger independently with mock metrics
- [ ] `docs/recording-modes.md` documents all escalation rules with concrete examples per trigger type

## Milestone 3 — AEP Protocol Compatibility & OpenTelemetry

> Ensure cross-environment audit chain interoperability with the WasmAgent AEP protocol.

- [ ] `train_replay/recording/evidence.py` `EpochEvidenceBundle` schema matches `@wasmagent/aep` protocol for cross-environment join
- [ ] `train_replay/recording/evidence.py` supports shared `trace_id` field for gateway → agent process → training job causal chains (currently uses `causal_chain_id`)
- [ ] `train_replay/` emits OpenTelemetry spans for graph construction, evidence recording, and signing when `[otel]` extras installed
- [ ] `tests/test_aep_compat.py` validates bundle serialization against AEP spec with known-good fixtures
- [ ] `tests/test_otel.py` validates span export and attribute correctness when otel extras available
- [ ] `docs/integration.md` documents how wasmagent-js consumers join training evidence chains via shared `trace_id`

## Milestone 4 — Production Readiness & Visualization

> Ship a production-quality package with comprehensive documentation, performance guarantees, and observability.

- [ ] `train_replay/replay/` supports deterministic replay from any epoch by loading serialized `EpochEvidenceBundle` files
- [ ] CLI `replay` command reconstructs training state, reports suspicious actions per rank, and renders causal subgraph
- [ ] `train_replay/` visualizes causal subgraph via `[viz]` extras (matplotlib/plotly) with rank-colored nodes
- [ ] `docs/architecture.md` documents full pipeline with data flow diagrams and PROV-DM node/edge semantics
- [ ] `docs/api.md` documents public API surface (`ProvGraph`, `EpochRecorder`, `BundleSigner`, `EpochReplayer`) for programmatic use
- [ ] Performance benchmarks: graph construction and replay on 1000+ event traces complete under 1 second
- [ ] README.md includes "Reproducing These Results" section with one-command `make test && make demo` verification
- [ ] `docs/future-work.md` outlines federation with open-agent-audit and trace-pipeline integration
