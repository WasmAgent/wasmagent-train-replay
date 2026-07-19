# CLI reference

> Complete reference for the `train-replay` command-line interface.

The CLI is built with [click](https://click.palletsprojects.com/) and rendered
with [rich](https://rich.readthedocs.io/). It is installed as the
`train-replay` entry point declared in `pyproject.toml`:

```
[project.scripts]
train-replay = "train_replay.cli.main:cli"
```

Source: `train_replay/cli/main.py`.

All path arguments are validated by click before command handlers convert them
to `pathlib.Path` objects for the collector.

The CLI is intentionally read-mostly: `ingest` and `trace` build graphs in
memory, and `record` prints a digest for the generated bundle. `export` is the
one command that writes to disk: it materialises a signed `EpochEvidenceBundle`
to a file (JSON or CBOR) for auditor evidence.

Contents:

- [Global](#global)
- [`train-replay ingest`](#train-replay-ingest)
- [`train-replay trace`](#train-replay-trace)
- [`train-replay record`](#train-replay-record)
- [`train-replay export`](#train-replay-export)

## Global

```
train-replay [--version] COMMAND ...
```

`train-replay` is a command group. `--version` prints the installed package
version and exits.

```bash
pip install -e ".[dev]"     # installs the train-replay entry point
train-replay --version
```

## `train-replay ingest`

Ingest a PyTorch Flight Recorder dump and build the causal graph.

### Usage

```
train-replay ingest [OPTIONS] DUMP_PATH
```

### Arguments

| Argument | Type | Required | Description |
|---|---|---|---|
| `DUMP_PATH` | path (must exist) | yes | Flight Recorder pickle dump produced by `torch._C._distributed_c10d._dump_nccl_trace()`. |

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--rank`, `-r` | `int` | *(all ranks)* | Filter to a specific rank before building the graph. |

### What it does

1. `load_flight_recorder(DUMP_PATH)` → `list[CollectiveEvent]`.
2. If `--rank` is given, keeps only events with that `rank`.
3. `build_from_events(events)` → `ProvGraph`.
4. Prints the number of loaded collective events and the resulting node count.

### Example

```bash
# Ingest the full dump
train-replay ingest path/to/nccl_trace.pkl

# Ingest only rank 2
train-replay ingest path/to/nccl_trace.pkl --rank 2
```

## `train-replay trace`

Trace the causal ancestors of a tensor entity.

### Usage

```
train-replay trace [OPTIONS] ENTITY_ID DUMP_PATH
```

### Arguments

| Argument | Type | Required | Description |
|---|---|---|---|
| `ENTITY_ID` | string | yes | The entity to trace, e.g. `tensor:2:3:out`. See entity-id conventions below. |
| `DUMP_PATH` | path (must exist) | yes | Flight Recorder pickle dump to build the graph from. |

### What it does

1. Builds the graph from `DUMP_PATH` (same as `ingest`).
2. `EpochReplayer(graph).find_root_cause(ENTITY_ID)` → list of contributing activity IDs.
3. Prints a `rich` table titled `Causal ancestors of <ENTITY_ID>`.

### Entity-id conventions

Entity ids follow the pattern produced by `build_from_events()`
(see [architecture.md § PROV-DM data model](architecture.md#prov-dm-data-model)):

- Input tensor: `tensor:{rank}:{sequence_id}:in`
- Output tensor: `tensor:{rank}:{sequence_id}:out`

### Example

```bash
train-replay trace "tensor:2:3:out" path/to/nccl_trace.pkl
```

## `train-replay record`

Record AEP evidence for all collectives in a Flight Recorder dump.

### Usage

```
train-replay record [OPTIONS] DUMP_PATH
```

### Arguments

| Argument | Type | Required | Description |
|---|---|---|---|
| `DUMP_PATH` | path (must exist) | yes | Flight Recorder pickle dump to record evidence from. |

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--run-id` | string | `dev-run` | Training run identifier written to the bundle. |
| `--epoch` | int | `0` | Epoch index written to the bundle. |

### What it does

1. `load_flight_recorder(DUMP_PATH)` → `list[CollectiveEvent]`.
2. Creates `EpochRecorder(run_id=..., epoch=...)` and calls
   `record_collective(evt)` for every event, each classified through the
   recording policy.
3. Prints the number of recorded actions and the bundle digest
   (`bundle.digest()` = `sha256` of `canonical_bytes()`).

`record` does not sign or persist the bundle by itself. Use
`train_replay.signing.BundleSigner` from Python when a workflow needs the
DSSE-style Ed25519 signature envelope.

### Example

```bash
train-replay record path/to/nccl_trace.pkl --run-id my-run --epoch 5
```

## `train-replay export`

Export a signed `EpochEvidenceBundle` to a file — the write-side companion to
`record`. Where `record` only prints a digest, `export` materialises the
tamper-evident bundle to disk in either canonical JSON or compact CBOR and
attaches a DSSE-style Ed25519 signature before writing.

This is the command an auditor-facing workflow runs to produce the artefact that
`verify_bundle()` later checks against an Ed25519 public key. See
[protocol.md](protocol.md) for the `EpochEvidenceBundle` and signature-envelope
field schemas.

### Usage

```
train-replay export [OPTIONS] DUMP_PATH
```

### Arguments

| Argument | Type | Required | Description |
|---|---|---|---|
| `DUMP_PATH` | path (must exist) | yes | Flight Recorder pickle dump to record evidence from (same loader as `ingest`/`record`). |

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--format` | choice: `json` \| `cbor` | `json` | Serialisation format of the written bundle. `json` is canonical, sorted-key text (`to_json()`); `cbor` is the compact binary equivalent (`to_cbor()`). Both round-trip through `from_json()` / `from_cbor()`. |
| `--output` | path | *(required)* | Destination file path. The signed bundle is written here, overwriting any existing file. |
| `--sign-key` | string (hex) | *(required)* | Ed25519 private key as a raw hex string (64 hex chars / 32 bytes). Decoded with `Ed25519PrivateKey.from_private_bytes()` and wrapped in a `BundleSigner` that signs the bundle. |
| `--run-id` | string | `dev-run` | Training run identifier written to the bundle (same semantics as `record`). |
| `--epoch` | int | `0` | Epoch index written to the bundle (same semantics as `record`). |

### What it does

1. `load_flight_recorder(DUMP_PATH)` → `list[CollectiveEvent]`.
2. Creates `EpochRecorder(run_id=..., epoch=...)` and calls
   `record_collective(evt)` for every event — the same recording path as
   `record`, classifying each collective through the recording policy.
3. `recorder.bundle()` → unsigned `EpochEvidenceBundle`.
4. `--sign-key` is decoded to an `Ed25519PrivateKey`. The command derives a
   stable `key_id` from the public key bytes, wraps the key in a `BundleSigner`,
   and calls `signer.sign(bundle)` to set `bundle.signature` to the DSSE-style
   envelope (`alg`, `key_id`, base64-encoded `sig`).
5. Writes the signed bundle to `--output`: `bundle.to_json()` when
   `--format json`, or `bundle.to_cbor()` when `--format cbor`.
6. Prints the output path, the bundle digest (`bundle.digest()`, the sha256 of
   `canonical_bytes()`), and the signature `key_id`.

The written file is tamper-evident and round-trippable: re-reading it with
`EpochEvidenceBundle.from_json()` / `from_cbor()` and calling
`verify_bundle(bundle, public_key)` with the matching Ed25519 public key
confirms the signature.

### Example

```bash
# Export a canonical-JSON bundle signed with a hex Ed25519 private key
train-replay export path/to/nccl_trace.pkl \
  --format json \
  --output evidence/epoch5.json \
  --sign-key 9d2f4a1b...e1c \
  --run-id my-run --epoch 5

# Compact CBOR for storage-constrained audit archives
train-replay export path/to/nccl_trace.pkl \
  --format cbor \
  --output evidence/epoch5.cbor \
  --sign-key 9d2f4a1b...e1c
```

## Exit codes & errors

The CLI relies on click's default behaviour: argument validation errors (for
example a `DUMP_PATH` that does not exist, or a missing required argument) exit
non-zero with a usage message. Successful commands exit `0`.

## See also

- [architecture.md](architecture.md) — system design and the PROV-DM data model.
- [protocol.md](protocol.md) — record schemas (`CollectiveEvent`,
  `AEPRecord`, `EpochEvidenceBundle`, `TensorEvent`).
- [integration.md](integration.md) — wiring the profiler hook into a training
  loop and an end-to-end trace example.
- `README.md` — quick start and recording-mode overview.
