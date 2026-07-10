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

All path arguments are click paths with `path_type=Path`, so command handlers
receive `pathlib.Path` objects before passing them to the collector.

Contents:

- [Global](#global)
- [`train-replay ingest`](#train-replay-ingest)
- [`train-replay trace`](#train-replay-trace)
- [`train-replay record`](#train-replay-record)

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

### Example

```bash
train-replay record path/to/nccl_trace.pkl --run-id my-run --epoch 5
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
