# Milestones

## Milestone 1 — Collision-Aware Replayer

- [x] Fix `train_replay/replay/replayer.py` `suspicious_actions()`: when `self._detector is not None`, call `self._detector.detect()` and merge desync events as synthetic `AEPRecord` entries (with `collective_type="desync"`) into the returned list
- [x] Fix `replay_rank()` in `train_replay/replay/replayer.py`: populate `ReplayResult.collision_report` by calling `self.check_collisions({rank: events})` instead of leaving it `None`
- [x] Add test in `tests/test_collision.py`: instantiate `EpochReplayer` with a Gloo `CollisionDetector`, inject a desync timeline, assert that `suspicious_actions()` includes the synthetic desync record
- [x] Add test in `tests/test_collision.py`: assert `replay_rank()` result has `collision_report` populated (not None) when detector fires
- [ ] Wire `GlooCollisionDetector` into `examples/fault_injection_demo.py` and print collision report to stdout
- [ ] Update `docs/architecture.md`: add "Collision Detection" section describing `CollisionDetector.detect()` to `ReplayResult.collision_report` data path

## Milestone 2 — Auditor Evidence Export

- [ ] Add `train_replay/cli/main.py` `export` subcommand: `train-replay export <dump_path> --format json|cbor --output <path> --sign-key <hex>` that calls `EpochRecorder`, `BundleSigner`, and writes the signed bundle to file
- [x] Add `train_replay/signing/signer.py` `load_private_key_hex(hex_str: str) -> Ed25519PrivateKey` helper so the CLI can accept a raw hex key without callers constructing cryptography objects
- [x] Add `EpochEvidenceBundle._from_dict()` version check: raise `ValueError` with message `"unsupported schema_version: {v}"` when `schema_version` is not in the supported set
- [ ] Add `tests/test_export.py`: full round-trip test — sign a bundle, write to JSON, re-read via `from_json()`, call `verify_bundle()`, assert True; repeat for CBOR
- [x] Add `docs/auditor-guide.md`: worked example showing `to_json()` and `to_cbor()`, verifying a bundle with `verify_bundle()` and a PEM public key, and interpreting the DSSE envelope fields
- [ ] Add `docs/cli-reference.md` entry for the new `export` subcommand with all flags documented

## Milestone 3 — NCCL Inspector Escalation Bridge

- [x] Add `train_replay/recording/escalation.py`: `EscalationSignal` dataclass (`source: str`, `severity: float`, `metric_name: str`) and `PrometheusAnomalySource` that polls a Prometheus query endpoint and yields `EscalationSignal` when alert value exceeds threshold
- [x] Modify `train_replay/recording/modes.py` `compile_recording_policy()`: add optional `escalation: EscalationSignal | None = None` parameter; return `RecordingMode.FULL` with reason `"external escalation signal"` when non-None
- [x] Add `EpochRecorder.record_with_escalation(event, escalation)` method in `train_replay/recording/recorder.py` that passes the signal through to `compile_recording_policy()`
- [x] Add `tests/test_escalation.py`: assert that passing a non-None `EscalationSignal` to `compile_recording_policy()` always returns `RecordingMode.FULL` regardless of side-effect class
- [x] Add `tests/test_escalation.py`: assert that `PrometheusAnomalySource` yields `None` when metric value is below threshold and `EscalationSignal` when above
- [ ] Update `docs/integration.md`: add "NCCL Inspector Escalation" section with a code snippet showing `PrometheusAnomalySource` wired to `EpochRecorder.record_with_escalation()`

## Milestone 4 — LLM Tool Interface

- [ ] Add `train_replay/agent/tools.py`: three JSON-Schema-described tool functions: `trace_tensor(entity_id, dump_path)` wrapping `EpochReplayer.find_root_cause()`, `list_suspicious_actions(dump_path, run_id, epoch)` wrapping `suspicious_actions()`, and `summarize_epoch(dump_path, run_id, epoch)` returning bundle stats dict
- [x] Add `train_replay/agent/__init__.py` and `train_replay/agent/schema.py`: `TypedDict` definitions for each tool's input and output matching the JSON Schema in `tools.py`
- [x] Add `train_replay/cli/main.py` `agent-query` subcommand: `train-replay agent-query <dump_path> --tool trace_tensor --args '{"entity_id": "tensor:0:1:out"}'` dispatching to `tools.py` and printing JSON output
- [ ] Add `tests/test_agent_tools.py`: call each tool function directly with `examples/generate_sample_trace.py` output and assert return types match the `TypedDict` schema
- [x] Add `docs/agent-integration.md`: worked example of calling the tool interface from a `tool_use` message with JSON Schema definitions, and a sample `tool_result` showing root-cause output

## Milestone 5 — Automated Anomaly Detection and Alerting

- [x] Add `train_replay/anomaly/detector.py`: `AnomalyDetector` abstract base class with `detect(events: List[AEPRecord]) -> List[AnomalySignal]` method and `StatisticalAnomalyDetector` implementation using Z-score/Isolation Forest on event timing and tensor statistics
- [x] Add `train_replay/anomaly/profile.py`: `TrainingProfile` dataclass that captures baseline statistics (event intervals, tensor distributions, collective operation patterns) from `fit_on_normal_run(events)` method
- [x] Modify `train_replay/recording/modes.py` `compile_recording_policy()`: add optional `anomaly_signal: AnomalySignal | None = None` parameter; return `RecordingMode.FULL` with reason `"statistical anomaly detected"` when anomaly score exceeds threshold
- [x] Add `EpochReplayer.anomaly_scan()` method in `train_replay/replay/replayer.py` that runs `StatisticalAnomalyDetector` over the event timeline and returns ranked anomalies with confidence scores
- [ ] Add `train_replay/alerting/notifier.py`: `AlertNotifier` interface with `send_alert(anomaly: AnomalySignal)` method and `SlackAlertNotifier`/`EmailAlertNotifier` implementations delivering formatted anomaly reports
- [x] Add `train-replay anomaly` CLI subcommand in `train_replay/cli/main.py`: `train-replay anomaly <dump_path> --profile <baseline_path> --threshold <z_score> --notify slack:webhook_url` for batch anomaly scanning
- [x] Add `tests/test_anomaly.py`: inject synthetic timing anomalies (delayed all-reduce, outlier gradient values) into normal event timeline, assert detector flags them with correct confidence scores
- [x] Add `docs/anomaly-guide.md`: explain profile creation from normal training runs, detector configuration (threshold tuning), interpreting anomaly signals, and integrating with alerting pipelines
- [x] Update `docs/architecture.md`: add "Anomaly Detection Pipeline" section describing `TrainingProfile.fit_on_normal_run()` → `StatisticalAnomalyDetector.detect()` → `AlertNotifier.send_alert()` data flow

## Milestone 6 — Differential Divergence Replay

- [x] Add `train_replay/replay/diff.py` with `DivergenceReplayer`: accepts two `ReplayResult` objects (baseline, candidate), walks rank-aligned action streams, and emits the first step where actions disagree along with an N-step context window on either side
- [x] Add `DivergenceReport` dataclass to `train_replay/replay/types.py` with fields `rank`, `first_divergence_step`, `baseline_action`, `candidate_action`, `context_window: list[AEPRecord]`, `correlated_collision: bool`, and `correlated_escalation: str | None`
- [x] Add `EpochReplayer.replay_diff(baseline_dump: str, candidate_dump: str) -> dict[int, DivergenceReport]` in `train_replay/replay/replayer.py`: loads both dumps, calls `replay_rank()` per rank for each, then delegates pairwise comparison to `DivergenceReplayer`
- [x] Correlate divergence with collisions: when a rank's `first_divergence_step` lands inside a desync window reported by that rank's `ReplayResult.collision_report`, set `correlated_collision=True` and attach the matching desync `AEPRecord`
- [x] Correlate divergence with escalation: when an `EscalationSignal` timestamp overlaps the divergence step, populate `correlated_escalation` with the signal's `metric_name` and `severity`
- [x] Add `train_replay/cli/main.py` `diff` subcommand: `train-replay diff <baseline_dump> <candidate_dump> [--rank N] [--context 10] [--output path]` that prints the report as JSON and exits non-zero when any divergence is found (CI-friendly)
- [x] Add `tests/test_diff.py`: construct two dumps that diverge at a known step on rank 1, assert `first_divergence_step`, `baseline_action`, `candidate_action`, and `correlated_collision` are correct
- [x] Add `tests/test_diff.py`: assert byte-identical dumps yield an empty report (`divergences == {}`) and that the `diff` CLI exits 0
- [x] Add `examples/divergence_demo.py`: load two pre-recorded dumps, invoke `replay_diff()`, and print the per-rank divergence report to stdout
- [x] Update `docs/architecture.md`: add a "Differential Replay" subsection describing the baseline/candidate comparison pipeline and how divergence points correlate with collision and escalation signals
- [x] Add `docs/auditor-guide.md` worked example: use `train-replay diff` to root-cause a divergent run from two signed evidence bundles produced via the Milestone 2 `export` subcommand

## Milestone 6 — Cross-Run Regression & Divergence Analysis

- [ ] Add `train_replay/replay/differ.py` `ReplayDiffer` class with `diff(left: ReplayResult, right: ReplayResult) -> DivergenceReport` that walks paired event streams, locates the first divergence step per rank, and classifies it (`tensor_value`, `collective_order`, `grad_norm`, `shape`)
- [x] Add `train_replay/replay/types.py` `DivergenceReport` dataclass (`first_divergence_step: int`, `divergences: list[Divergence]`, `per_rank_similarity: dict[int, float]`, `summary: str`) and a `to_json()`/`to_cbor()` pair mirroring `EpochEvidenceBundle`
- [ ] Extend `EpochRecorder` to capture per-step determinism anchors (RNG seed snapshot, collective-order hash, reduce-scatter checksum) into the evidence bundle so two replays are comparable without re-reading raw tensors
- [ ] Add `train_replay/replay/replayer.py` `ReplayResult.__eq__` and `__hash__` based on determinism anchors so `diff` can short-circuit identical runs without per-event walks
- [x] Add `train_replay/cli/main.py` `diff` subcommand: `train-replay diff <dump_a> <dump_b> [--rank N] [--format json|md|cbor] [--output PATH]` that loads both dumps, replays each, and emits the `DivergenceReport`
- [x] Extend `train_replay/signing/signer.py` `BundleSigner.sign_divergence_report()` so auditor-facing comparison outputs carry a DSSE envelope over both source bundles' digests
- [x] Add `tests/test_diff.py`: inject a single tensor mutation at step K across two dumps, assert `ReplayDiffer.diff()` returns `first_divergence_step == K` and `per_rank_similarity < 1.0`; assert identical replays yield `first_divergence_step == None` and similarity `1.0`
- [x] Add `tests/test_diff.py`: round-trip a `DivergenceReport` through `to_cbor()` / `verify_divergence_report()` and assert the dual-bundle signature validates only against the originating digests
- [x] Wire `ReplayDiffer` into `examples/fault_injection_demo.py` to print a before/after divergence report alongside the existing collision output
- [x] Add `docs/regression-analysis.md`: worked example comparing two training dumps, interpreting the divergence report fields, and using the `diff` CLI to attribute a regression to a specific rank/step
- [x] Update `docs/cli-reference.md` with the `diff` subcommand entry (all flags) and update `docs/architecture.md` with a "Cross-Run Divergence Analysis" section mapping the `ReplayResult → ReplayDiffer → DivergenceReport → signed report` data path
