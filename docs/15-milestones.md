# Milestones

## Milestone 1 — Collector & Flight Recorder Ingestion

- [ ] Implement `collector/flight_recorder.py` parsing PyTorch Flight Recorder `.pkl` dumps into a normalized per-rank collective event stream (op type, rank, timestamp, tensor shape/dtype, comm group)
- [ ] Implement `collector/profiler_hook.py` autograd hooks capturing tensor-level evidence (gradient norm, NaN/Inf flag) on the backward pass
- [ ] Define a shared normalized-event schema (dataclass/pydantic) used as the contract between `collector` and `graph`
- [ ] Add unit tests for `flight_recorder` parsing against a committed fixture `.pkl` dump with a golden expected event list
- [ ] Add unit tests for `profiler_hook` NaN/Inf detection and gradient-norm capture on synthetic tensors
- [ ] Add `ingest` CLI subcommand that reads a directory of per-rank `.pkl` dumps and writes normalized events to JSON/Parquet
- [ ] Add integration test: ingest a multi-rank sample dump, assert every rank parsed with a consistent event count and aligned global step

## Milestone 2 — Cross-Rank PROV-DM Causal Graph

- [ ] Implement `graph/` module modeling PROV-DM `Activity` (collective ops), `Entity` (tensors), and `Agent` (ranks)
- [ ] Build cross-rank edges linking all-reduce outputs to contributing per-rank inputs across rank boundaries
- [ ] Implement causal-traversal API: given an anomalous tensor `Entity`, return origin rank, collective op, and upstream `Activity` chain
- [ ] Add unit tests for graph construction from a normalized event stream asserting expected `Activity`/`Entity`/`Agent` nodes and edges
- [ ] Add unit tests for causal traversal on a synthetic desync scenario, verifying the traced origin rank matches the planted anomaly
- [ ] Implement stable on-disk graph serialization (JSON-LD or protobuf) with round-trip equality test
- [ ] Add integration test: ingest multi-rank dump → build graph → trace a planted anomaly → verify correct origin rank and op

## Milestone 3 — Layered Evidence Recording & Signed Bundles

- [ ] Implement three-tier evidence recorder: `validation` (cheap checksums/stats), `delta` (diff since last checkpoint), `full` (complete tensor snapshot)
- [ ] Implement risk-signal-driven auto-escalation that triggers `full` capture on NaN/Inf, divergence, or gradient-norm delta above threshold
- [ ] Define `EpochEvidenceBundle` schema aggregating per-epoch layered evidence keyed by tensor/collective op
- [ ] Implement Ed25519 signing of `EpochEvidenceBundle` plus a standalone signature-verification utility
- [ ] Add unit tests for each evidence tier asserting capture correctness and size ordering `validation` < `delta` < `full`
- [ ] Add unit tests for escalation triggers firing `full` capture on injected anomalies and skipping it on healthy runs
- [ ] Add unit tests for sign/verify round-trip and tamper detection (mutated bundle fails verification, original passes)

## Milestone 4 — CLI, Deterministic Replay & End-to-End Tracing

- [ ] Implement `replay` CLI subcommand reconstructing collective execution from a saved graph + evidence bundle starting at any user-specified epoch
- [ ] Implement `trace` CLI subcommand that, given a tensor id/anomaly, prints origin rank, collective op, and the evidence chain to stdout
- [ ] Implement `verify` CLI subcommand validating the Ed25519 signature of a stored `EpochEvidenceBundle`
- [ ] Add replay-determinism test: replay the same epoch twice and assert byte-identical reproduced tensor states
- [ ] Add end-to-end test exercising the full pipeline (ingest → graph → evidence → replay → trace) on a synthetic 8-rank desync scenario
- [ ] Add a CLI smoke test asserting every subcommand runs with `--help` and exits 0
- [ ] Record a baseline ingestion + graph-build throughput benchmark on a large multi-rank dump in the README