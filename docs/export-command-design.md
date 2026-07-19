# Export command design

> Architecture and file-format specification for the future
> `train-replay export` auditor evidence export command.

This document is a design contract, not an implementation guide for a shipped
command. It defines what the implementation must produce so verify-first,
auditors, and downstream tooling have a stable target before export code is
added to `train_replay/cli/main.py`.

## Goals

1. Export tamper-evident auditor evidence from existing recording and collector
   primitives without introducing a second evidence schema.
2. Preserve enough source metadata for an auditor to reproduce the exported
   bundle digest from the input trace.
3. Support a human-readable JSON path and a compact CBOR path from the same
   `EpochEvidenceBundle` dataclass.
4. Keep collector integration backend-agnostic: Flight Recorder, Gloo, and MTIA
   inputs all normalize into collective events before recording.

## Source data

`train-replay export` operates on one logical epoch. The source may be either a
pre-recorded evidence bundle or raw collector trace input:

| Source kind | Flag | Required data | Loader | Result |
|---|---|---|---|---|
| Recorded bundle | `--bundle PATH` | JSON or CBOR produced from `EpochEvidenceBundle.to_json()` or `to_cbor()` | `EpochEvidenceBundle.from_json()` or `from_cbor()` selected by file extension or `--input-format` | Existing bundle is validated and repackaged. |
| Flight Recorder trace | `--trace PATH --source flight-recorder` | PyTorch Flight Recorder pickle dump | `load_flight_recorder(Path)` | `list[CollectiveEvent]` recorded into a bundle. |
| Gloo trace | `--trace PATH --source gloo` | Gloo JSON trace object or array | `load_gloo_trace(Path)` | `list[CollectiveEvent]` recorded into a bundle. |
| MTIA trace | `--trace PATH --source mtia` | MTIA profiler JSON trace | `parse_mtia_trace(Path)` plus adapter to the shared collective-event schema | `list[CollectiveEvent]` recorded into a bundle. |

Exactly one of `--bundle` or `--trace` must be supplied. Raw collector export
requires `--run-id` and `--epoch` because those fields are part of the
`EpochEvidenceBundle` identity. Bundle export reads `run_id` and `epoch` from
the bundle unless the implementation later adds an explicit metadata override;
this design does not define such an override.

Raw trace export is equivalent to:

```python
events = load_backend_events(trace_path, source)
recorder = EpochRecorder(run_id=run_id, epoch=epoch)
for event in events:
    recorder.record_collective(event)
bundle = recorder.bundle()
```

If `--signing-key-id` and a signing key source are provided by a later signing
issue, the bundle may be signed before serialization. Without signing flags, the
exported bundle's `signature` field is whatever the source bundle already had,
or `null` for a bundle created from raw traces.

## Command flags

Planned usage:

```text
train-replay export [OPTIONS]
```

| Flag | Type | Required | Default | Description |
|---|---|---:|---|---|
| `--bundle` | path, existing file | conditional | none | Existing `EpochEvidenceBundle` JSON or CBOR input. Mutually exclusive with `--trace`. |
| `--trace` | path, existing file | conditional | none | Raw collector trace input. Mutually exclusive with `--bundle`. |
| `--source` | choice: `flight-recorder`, `gloo`, `mtia` | no | `flight-recorder` | Selects the collector used to parse `--trace`. Ignored for `--bundle`. |
| `--input-format` | choice: `auto`, `json`, `cbor`, `pickle` | no | `auto` | Interprets `--bundle` or `--trace`. `auto` uses the file extension and source: `.json` for JSON, `.cbor` for CBOR, `.pkl`/`.pickle` for Flight Recorder pickle. |
| `--format`, `-f` | choice: `json`, `cbor`, `both` | no | `json` | Evidence payload encoding to write. `both` writes JSON and CBOR bundle files that represent the same bundle. |
| `--output`, `-o` | path | yes | none | Destination file when `--single-file` is set, otherwise destination directory for the export package. |
| `--single-file` | bool flag | no | `false` | Write only the bundle payload to `--output`. Disallows `--format both`, `--include-raw`, and `--manifest-name`. |
| `--manifest-name` | string | no | `manifest.json` | Manifest filename inside a directory export. Must be a relative filename with no path separators. |
| `--run-id` | string | required with `--trace` | none | Run identifier written into a bundle created from raw collector events. |
| `--epoch` | int | required with `--trace` | none | Epoch index written into a bundle created from raw collector events. Must be non-negative. |
| `--rank` | int, repeatable | no | all ranks | Include only selected ranks while recording from `--trace`. Repeating the flag includes multiple ranks. |
| `--include-raw` | bool flag | no | `false` | Copy the raw input trace or source bundle into `raw/` for chain-of-custody review. Directory exports only. |
| `--overwrite` | bool flag | no | `false` | Allow replacing an existing output file or export directory. Without this flag, existing outputs are an error. |
| `--strict/--no-strict` | bool flag pair | no | `true` | In strict mode, reject malformed inputs, empty event lists, unknown schema versions, unsupported recording modes, and inconsistent bundle metadata. In non-strict mode, recoverable validation findings are written to the manifest warnings list. |

Required-condition summary:

- `--output` is always required.
- Exactly one of `--bundle` or `--trace` is required.
- `--run-id` and `--epoch` are required when `--trace` is used.
- `--source` is meaningful only with `--trace`; when omitted for `--trace`,
  Flight Recorder is assumed.
- `--single-file` permits only `--format json` or `--format cbor`.

## Output formats

### Bundle JSON

JSON output is the exact text returned by `EpochEvidenceBundle.to_json()` with a
trailing newline added by the file writer. It is canonical for this project:
keys are sorted, enum values are serialized as strings, and the `signature`
field is included as an object or `null`.

Filename:

```text
bundle.json
```

Media type:

```text
application/vnd.wasmagent.train-aep.bundle+json
```

Schema:

```json
{
  "schema_version": "train-aep/v0.1",
  "run_id": "run-42",
  "epoch": 5,
  "actions": [
    {
      "action_id": "r2:seq3",
      "rank": 2,
      "step": 3,
      "collective_type": "all_reduce",
      "recording_mode": "full",
      "tensor_input_digest": null,
      "tensor_output_digest": null,
      "delta_stats": null,
      "timestamp_ns": 3100000,
      "causal_chain_id": null,
      "parent_action_id": null
    }
  ],
  "signature": {
    "alg": "ed25519",
    "key_id": "ci-key",
    "sig": "base64-ed25519-signature"
  }
}
```

The authoritative field definitions are in
[protocol.md](protocol.md#epochevidencebundle). The export command must not add
ad-hoc fields to the bundle object; package-level metadata belongs in the
manifest.

Required JSON object fields:

| Field | Type | Required | Source |
|---|---|---:|---|
| `schema_version` | string | yes | `EpochEvidenceBundle.schema_version` |
| `run_id` | string | yes | Existing bundle, or `--run-id` for raw trace export. |
| `epoch` | integer | yes | Existing bundle, or `--epoch` for raw trace export. |
| `actions` | array of `AEPRecord` objects | yes | Existing bundle, or actions recorded by `EpochRecorder`. |
| `signature` | object or null | yes | Existing bundle signature, future signer output, or `null`. |

Each `actions[]` object must contain only the fields defined by `AEPRecord`:
`action_id`, `rank`, `step`, `collective_type`, `recording_mode`,
`tensor_input_digest`, `tensor_output_digest`, `delta_stats`, `timestamp_ns`,
`causal_chain_id`, and `parent_action_id`.

### Bundle CBOR

CBOR output is the exact bytes returned by `EpochEvidenceBundle.to_cbor()`.
CBOR stores the same map keys and values as JSON, including `signature`.

Filename:

```text
bundle.cbor
```

Media type:

```text
application/vnd.wasmagent.train-aep.bundle+cbor
```

Use CBOR when the export package is for machine-to-machine exchange, storage
size matters, or the auditor has a verifier that already consumes CBOR. Use
JSON when the package will be inspected directly in review tools, attached to
human audit tickets, or diffed in source control.

CBOR schema rule: the decoded CBOR top-level value is a map with the same keys
and value types as the Bundle JSON schema above. It must round-trip through
`EpochEvidenceBundle.from_cbor()` without changing `bundle.digest()`.

### Manifest JSON

Directory exports include a manifest. Single-file exports do not.

Filename defaults to:

```text
manifest.json
```

Schema:

```json
{
  "schema_version": "train-replay-export/v0.1",
  "created_by": "wasmagent-train-replay",
  "export_command": "train-replay export",
  "source": {
    "kind": "trace",
    "backend": "flight-recorder",
    "path": "input/nccl_trace.pkl",
    "input_format": "pickle",
    "sha256": "hex-digest-of-input-file"
  },
  "bundle": {
    "schema_version": "train-aep/v0.1",
    "run_id": "run-42",
    "epoch": 5,
    "action_count": 128,
    "rank_count": 8,
    "ranks": [0, 1, 2, 3, 4, 5, 6, 7],
    "recording_modes": {
      "validation": 12,
      "delta": 0,
      "full": 116
    },
    "digest": "sha256-of-bundle-canonical-bytes",
    "signature_present": true,
    "signature_key_id": "ci-key"
  },
  "outputs": [
    {
      "path": "bundle.json",
      "format": "json",
      "media_type": "application/vnd.wasmagent.train-aep.bundle+json",
      "sha256": "hex-digest-of-bundle-json-file",
      "bytes": 12045
    },
    {
      "path": "bundle.cbor",
      "format": "cbor",
      "media_type": "application/vnd.wasmagent.train-aep.bundle+cbor",
      "sha256": "hex-digest-of-bundle-cbor-file",
      "bytes": 6910
    }
  ],
  "validation": {
    "strict": true,
    "status": "passed",
    "warnings": []
  }
}
```

Rules:

- `bundle.digest` is `EpochEvidenceBundle.digest()`, the SHA-256 digest of
  `canonical_bytes()`, not the hash of `bundle.json` or `bundle.cbor`.
- Each `outputs[*].sha256` is the file hash of that serialized artifact.
- Manifest paths are relative to the export directory.
- `source.path` is a relative path only when `--include-raw` copied the source
  into the package. Otherwise it is the basename of the input path so the
  manifest does not leak machine-specific absolute paths.
- `created_by` is stable text; implementation version may be added later as
  `tool_version` without changing `schema_version`.

Required manifest fields:

| Field | Type | Required | Description |
|---|---|---:|---|
| `schema_version` | string | yes | Export package schema version. Initial value: `train-replay-export/v0.1`. |
| `created_by` | string | yes | Stable producer name: `wasmagent-train-replay`. |
| `export_command` | string | yes | Command family that produced the package: `train-replay export`. |
| `source` | object | yes | Source kind, backend, path, input format, and input SHA-256. |
| `bundle` | object | yes | Bundle schema version, identity, counts, canonical digest, and signature summary. |
| `outputs` | array | yes | One entry per written payload artifact. |
| `validation` | object | yes | Strictness, final status, and warning strings. |

Required `source` fields:

| Field | Type | Required | Description |
|---|---|---:|---|
| `kind` | choice: `bundle`, `trace` | yes | Whether export read a pre-recorded bundle or raw collector trace. |
| `backend` | string or null | yes | `flight-recorder`, `gloo`, or `mtia` for raw traces; `null` for bundle input. |
| `path` | string | yes | Basename or package-relative raw copy path. |
| `input_format` | string | yes | Resolved input format after `auto` detection. |
| `sha256` | lowercase hex string | yes | SHA-256 of the original input file bytes. |

Required `bundle` fields:

| Field | Type | Required | Description |
|---|---|---:|---|
| `schema_version` | string | yes | Bundle schema version. |
| `run_id` | string | yes | Bundle run id. |
| `epoch` | integer | yes | Bundle epoch. |
| `action_count` | integer | yes | `len(bundle.actions)`. |
| `rank_count` | integer | yes | Count of unique action ranks. |
| `ranks` | array of integers | yes | Sorted unique action ranks. |
| `recording_modes` | object | yes | Counts for `validation`, `delta`, and `full`, including zero counts. |
| `digest` | lowercase hex string | yes | `bundle.digest()`. |
| `signature_present` | boolean | yes | Whether `bundle.signature` is not `null`. |
| `signature_key_id` | string or null | yes | Signature key id when present. |

## File structure

Directory export with JSON:

```text
audit-export/
  manifest.json
  bundle.json
```

Directory export with both formats and raw source:

```text
audit-export/
  manifest.json
  bundle.json
  bundle.cbor
  raw/
    nccl_trace.pkl
```

Single-file JSON export:

```text
run-42-epoch-5.bundle.json
```

Single-file CBOR export:

```text
run-42-epoch-5.bundle.cbor
```

Naming conventions:

- Directory exports use fixed artifact names: `bundle.json`, `bundle.cbor`, and
  the configured manifest name.
- Single-file exports use exactly the path supplied by `--output`; the command
  does not rewrite the basename.
- Raw source copies keep the input basename under `raw/`.
- The implementation must reject a raw source basename that would collide with
  an existing file in `raw/` unless `--overwrite` is set.

File layout matrix:

| `--single-file` | `--format` | Required paths |
|---:|---|---|
| `false` | `json` | `<output>/<manifest-name>`, `<output>/bundle.json` |
| `false` | `cbor` | `<output>/<manifest-name>`, `<output>/bundle.cbor` |
| `false` | `both` | `<output>/<manifest-name>`, `<output>/bundle.json`, `<output>/bundle.cbor` |
| `true` | `json` | `<output>` containing Bundle JSON |
| `true` | `cbor` | `<output>` containing Bundle CBOR |

## Integration pattern

Export should share the same loading and recording path as existing CLI
commands:

```text
--bundle
  read bytes
  select from_json/from_cbor
  validate EpochEvidenceBundle
  write selected output format(s)
  write manifest for directory export

--trace --source flight-recorder
  load_flight_recorder(path)
  filter ranks
  EpochRecorder.record_collective(event) for each event
  write selected output format(s)
  write manifest for directory export

--trace --source gloo
  load_gloo_trace(path)
  filter ranks
  EpochRecorder.record_collective(event) for each event
  write selected output format(s)
  write manifest for directory export

--trace --source mtia
  parse_mtia_trace(path)
  convert MtiaEvent to CollectiveEvent-compatible records
  filter ranks
  EpochRecorder.record_collective(event) for each event
  write selected output format(s)
  write manifest for directory export
```

The collector boundary is intentionally before recording: collectors parse
backend-specific files and return normalized event records; export never parses
graph nodes or replay results directly. If an implementation needs a
backend-neutral adapter, it should live in `train_replay/collector/` and return
the shared event shape consumed by `EpochRecorder`.

MTIA adapter mapping:

| `MtiaEvent` field | `CollectiveEvent` field |
|---|---|
| `rank` | `rank` |
| `process_group` | `process_group` |
| `op_type` | `collective_type` |
| `src_rank` | `src_rank` |
| `dst_rank` | `dst_rank` |
| `tensor_size` | `tensor_size` |
| `start_time_ns` | `start_time_ns` |
| `end_time_ns` | `end_time_ns` |
| `sequence_id` | `sequence_id` |
| `call_stack` | `call_stack` |
| absent | `enqueue_time_ns = start_time_ns` |

## Validation rules

Input validation:

- The input path must exist and be a regular file.
- `--bundle` and `--trace` are mutually exclusive.
- `--trace` requires `--run-id`, `--epoch`, and a supported `--source`.
- `--epoch` must be `>= 0`.
- `--rank` values must be `>= 0`.
- `--input-format auto` must resolve to a supported format for the selected
  source.
- Bundle input must use a supported `schema_version`.
- Bundle actions must have unique `action_id` values.
- Bundle action `recording_mode` values must be one of `validation`, `delta`,
  or `full`.
- Raw trace input must produce at least one event in strict mode.
- Rank filtering that removes all events is an error in strict mode and a
  warning in non-strict mode.

Output validation:

- `--output` must not exist unless `--overwrite` is set.
- Directory exports must be created atomically enough that a failed command does
  not leave a manifest claiming missing files.
- `--single-file` must not be combined with `--format both`.
- `--single-file` must not be combined with `--include-raw`.
- `--manifest-name` must not contain `/`, `\`, or `..`.
- Manifest `outputs[*].sha256` values must be computed after writing files.
- If both JSON and CBOR are written, deserializing both must reconstruct bundles
  with the same `digest()`.

Error handling:

- CLI argument and validation failures raise `click.ClickException` and exit
  non-zero.
- Parser exceptions from Flight Recorder, Gloo, or MTIA loaders are wrapped with
  the source name and input path.
- Serialization errors include the selected output format and destination path.
- Existing output errors include the path and recommend `--overwrite`.

## Examples

Export a Flight Recorder trace to a JSON directory package:

```bash
train-replay export \
  --trace traces/nccl_trace.pkl \
  --source flight-recorder \
  --run-id run-42 \
  --epoch 5 \
  --format json \
  --output audit/run-42-epoch-5
```

Expected output:

```text
audit/run-42-epoch-5/
  manifest.json
  bundle.json
```

Export the same epoch in both auditor-friendly and compact encodings:

```bash
train-replay export \
  --trace traces/nccl_trace.pkl \
  --source flight-recorder \
  --run-id run-42 \
  --epoch 5 \
  --format both \
  --include-raw \
  --output audit/run-42-epoch-5
```

Expected output:

```text
audit/run-42-epoch-5/
  manifest.json
  bundle.json
  bundle.cbor
  raw/
    nccl_trace.pkl
```

Repackage an existing signed bundle as single-file CBOR:

```bash
train-replay export \
  --bundle audit/run-42-epoch-5/bundle.json \
  --input-format json \
  --format cbor \
  --single-file \
  --output audit/run-42-epoch-5.bundle.cbor
```

Expected output:

```text
audit/run-42-epoch-5.bundle.cbor
```

Export only ranks 0 and 2 from a Gloo trace:

```bash
train-replay export \
  --trace traces/gloo.json \
  --source gloo \
  --run-id cpu-run-7 \
  --epoch 3 \
  --rank 0 \
  --rank 2 \
  --output audit/cpu-run-7-epoch-3
```

## Decision record: JSON vs CBOR

Decision: export supports JSON, CBOR, and `both`.

JSON is the default because it is inspectable with standard audit tools, easy to
attach to tickets, stable under text diffs, and already matches
`EpochEvidenceBundle.to_json()`. CBOR is included because the evidence bundle
already exposes `to_cbor()`/`from_cbor()`, it is smaller for large action lists,
and binary verifier pipelines can consume it without lossy conversion. The
command must write both formats from the same in-memory bundle when `--format
both` is selected; it must not generate JSON from CBOR or CBOR from JSON after
the first serialization step.

Consequences:

- JSON remains the compatibility and review format.
- CBOR remains the compact exchange format.
- The manifest is always JSON so every directory export has a human-readable
  inventory even when the bundle payload is CBOR-only.
- A future verifier can compare `bundle.digest` across JSON and CBOR packages
  because the digest is over `canonical_bytes()`, not either file encoding.

## Testable acceptance criteria for implementation

An implementation of this design is acceptable when automated tests verify:

1. `train-replay export --help` lists every flag in the [Command flags](#command-flags)
   table with the documented defaults or required conditions.
2. Supplying neither `--bundle` nor `--trace` exits non-zero.
3. Supplying both `--bundle` and `--trace` exits non-zero.
4. Raw Flight Recorder export calls `load_flight_recorder()`, records events
   through `EpochRecorder`, and writes a valid `bundle.json`.
5. Raw Gloo export calls `load_gloo_trace()`, records events through
   `EpochRecorder`, and writes a valid `bundle.json`.
6. Raw MTIA export calls `parse_mtia_trace()`, adapts events to the shared
   collective-event shape, records through `EpochRecorder`, and writes a valid
   `bundle.json`.
7. `--format cbor` writes `bundle.cbor` that round-trips through
   `EpochEvidenceBundle.from_cbor()`.
8. `--format both` writes JSON and CBOR bundles whose `digest()` values match
   after deserialization.
9. Directory export writes `manifest.json` with source metadata, bundle digest,
   output file hashes, action counts, rank counts, and recording-mode counts.
10. `--single-file` writes only the requested bundle payload and no manifest.
11. Existing output paths fail unless `--overwrite` is set.
12. Strict mode rejects empty raw event lists and malformed bundle metadata.
13. Non-strict mode records validation warnings in the manifest.
14. The bundle object contains only fields defined by
    [protocol.md](protocol.md#epochevidencebundle); package metadata appears
    only in the manifest.
