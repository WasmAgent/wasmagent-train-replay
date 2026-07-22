# Milestones

## Milestone 1 — Collision-Aware Replayer

- [ ] Fix `train_replay/replay/replayer.py` `suspicious_actions()`: when `self._detector is not None`, call `self._detector.detect()` and merge desync events as synthetic `AEPRecord` entries (with `collective_type="desync"`) into the returned list
- [ ] Fix `replay_rank()` in `train_replay/replay/replayer.py`: populate `ReplayResult.collision_report` by calling `self.check_collisions({rank: events})` instead of leaving it `None`
- [ ] Add test in `tests/test_collision.py`: instantiate `EpochReplayer` with a Gloo `CollisionDetector`, inject a desync timeline, assert that `suspicious_actions()` includes the synthetic desync record
- [ ] Add test in `tests/test_collision.py`: assert `replay_rank()` result has `collision_report` populated (not None) when detector fires
- [ ] Wire `GlooCollisionDetector` into `examples/fault_injection_demo.py` and print collision report to stdout
- [ ] Update `docs/architecture.md`: add "Collision Detection" section describing `CollisionDetector.detect()` to `ReplayResult.collision_report` data path

## Milestone 2 — Auditor Evidence Export

- [ ] Add `train_replay/cli/main.py` `export` subcommand: `train-replay export <dump_path> --format json|cbor --output <path> --sign-key <hex>` that calls `EpochRecorder`, `BundleSigner`, and writes the signed bundle to file
- [ ] Add `train_replay/signing/signer.py` `load_private_key_hex(hex_str: str) -> Ed25519PrivateKey` helper so the CLI can accept a raw hex key without callers constructing cryptography objects
- [ ] Add `EpochEvidenceBundle._from_dict()` version check: raise `ValueError` with message `"unsupported schema_version: {v}"` when `schema_version` is not in the supported set
- [ ] Add `tests/test_export.py`: full round-trip test — sign a bundle, write to JSON, re-read via `from_json()`, call `verify_bundle()`, assert True; repeat for CBOR
- [ ] Add `docs/auditor-guide.md`: worked example showing `to_json()` and `to_cbor()`, verifying a bundle with `verify_bundle()` and a PEM public key, and interpreting the DSSE envelope fields
- [ ] Add `docs/cli-reference.md` entry for the new `export` subcommand with all flags documented

## Milestone 3 — NCCL Inspector Escalation Bridge

- [ ] Add `train_replay/recording/escalation.py`: `EscalationSignal` dataclass (`source: str`, `severity: float`, `metric_name: str`) and `PrometheusAnomalySource` that polls a Prometheus query endpoint and yields `EscalationSignal` when alert value exceeds threshold
- [ ] Modify `train_replay/recording/modes.py` `compile_recording_policy()`: add optional `escalation: EscalationSignal | None = None` parameter; return `RecordingMode.FULL` with reason `"external escalation signal"` when non-None
- [ ] Add `EpochRecorder.record_with_escalation(event, escalation)` method in `train_replay/recording/recorder.py` that passes the signal through to `compile_recording_policy()`
- [ ] Add `tests/test_escalation.py`: assert that passing a non-None `EscalationSignal` to `compile_recording_policy()` always returns `RecordingMode.FULL` regardless of side-effect class
- [ ] Add `tests/test_escalation.py`: assert that `PrometheusAnomalySource` yields `None` when metric value is below threshold and `EscalationSignal` when above
- [ ] Update `docs/integration.md`: add "NCCL Inspector Escalation" section with a code snippet showing `PrometheusAnomalySource` wired to `EpochRecorder.record_with_escalation()`

## Milestone 4 — LLM Tool Interface

- [ ] Add `train_replay/agent/tools.py`: three JSON-Schema-described tool functions: `trace_tensor(entity_id, dump_path)` wrapping `EpochReplayer.find_root_cause()`, `list_suspicious_actions(dump_path, run_id, epoch)` wrapping `suspicious_actions()`, and `summarize_epoch(dump_path, run_id, epoch)` returning bundle stats dict
- [ ] Add `train_replay/agent/__init__.py` and `train_replay/agent/schema.py`: `TypedDict` definitions for each tool's input and output matching the JSON Schema in `tools.py`
- [ ] Add `train_replay/cli/main.py` `agent-query` subcommand: `train-replay agent-query <dump_path> --tool trace_tensor --args '{"entity_id": "tensor:0:1:out"}'` dispatching to `tools.py` and printing JSON output
- [ ] Add `tests/test_agent_tools.py`: call each tool function directly with `examples/generate_sample_trace.py` output and assert return types match the `TypedDict` schema
- [ ] Add `docs/agent-integration.md`: worked example of calling the tool interface from a `tool_use` message with JSON Schema definitions, and a sample `tool_result` showing root-cause output

## Milestone 5 — Automated Anomaly Detection and Alerting

- [ ] Add `train_replay/anomaly/detector.py`: `AnomalyDetector` abstract base class with `detect(events: List[AEPRecord]) -> List[AnomalySignal]` method and `StatisticalAnomalyDetector` implementation using Z-score/Isolation Forest on event timing and tensor statistics
- [ ] Add `train_replay/anomaly/profile.py`: `TrainingProfile` dataclass that captures baseline statistics (event intervals, tensor distributions, collective operation patterns) from `fit_on_normal_run(events)` method
- [ ] Modify `train_replay/recording/modes.py` `compile_recording_policy()`: add optional `anomaly_signal: AnomalySignal | None = None` parameter; return `RecordingMode.FULL` with reason `"statistical anomaly detected"` when anomaly score exceeds threshold
- [ ] Add `EpochReplayer.anomaly_scan()` method in `train_replay/replay/replayer.py` that runs `StatisticalAnomalyDetector` over the event timeline and returns ranked anomalies with confidence scores
- [ ] Add `train_replay/alerting/notifier.py`: `AlertNotifier` interface with `send_alert(anomaly: AnomalySignal)` method and `SlackAlertNotifier`/`EmailAlertNotifier` implementations delivering formatted anomaly reports
- [ ] Add `train-replay anomaly` CLI subcommand in `train_replay/cli/main.py`: `train-replay anomaly <dump_path> --profile <baseline_path> --threshold <z_score> --notify slack:webhook_url` for batch anomaly scanning
- [ ] Add `tests/test_anomaly.py`: inject synthetic timing anomalies (delayed all-reduce, outlier gradient values) into normal event timeline, assert detector flags them with correct confidence scores
- [ ] Add `docs/anomaly-guide.md`: explain profile creation from normal training runs, detector configuration (threshold tuning), interpreting anomaly signals, and integrating with alerting pipelines
- [ ] Update `docs/architecture.md`: add "Anomaly Detection Pipeline" section describing `TrainingProfile.fit_on_normal_run()` → `StatisticalAnomalyDetector.detect()` → `AlertNotifier.send_alert()` data flow
