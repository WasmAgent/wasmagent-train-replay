# Protocol & data formats

> Field-by-field reference for every record type that flows through the pipeline.
> Types are Python type annotations as they appear in the source dataclasses.

This is the canonical schema reference. For how these records are produced and
consumed see [architecture.md](architecture.md); for the surrounding CLI see
[cli-reference.md](cli-reference.md).

Contents:

- [EpochEvidenceBundle](#epochevidencebundle) — the signed, per-epoch envelope
- [AEPRecord (TrainActionEvidence)](#aeprecord-trainactionevidence) — one recorded collective
- [CollectiveEvent](#collectiveevent) — one row from a Flight Recorder dump
- [TensorEvent](#tensorevent) — one tensor-level event from the profiler hook
- [Control enums](#control-enums) — shared `RecordingMode` / `SideEffectClass` values
- [Signature envelope](#signature-envelope) — the DSSE-style signing block

---

## EpochEvidenceBundle

The top-level, signable record. One bundle covers one epoch across all ranks.
Defined in `train_replay/recording/evidence.py`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `schema_version` | `str` | `"train-aep/v0.1"` | Wire-format version. Bumped on incompatible changes. |
| `run_id` | `str` | `""` | Training run identifier (e.g. `"dev-run"`). |
| `epoch` | `int` | `0` | Epoch index this bundle covers. |
| `actions` | `list[TrainActionEvidence]` | `[]` | One entry per recorded collective, in append order. |
| `signature` | `dict[str, str] \| None` | `None` | Signature envelope; `None` until `BundleSigner.sign()` runs. See [Signature envelope](#signature-envelope). |

### Canonicalisation & digest

- `canonical_bytes() -> bytes` — strips `signature`, serialises the rest with
  `json.dumps(..., sort_keys=True, default=str)`. The output is deterministic:
  field order and action order fully determine the bytes.
- `digest() -> str` — `sha256(canonical_bytes()).hexdigest()`. Any edit to a
  signed bundle changes this value and breaks verification.

Because canonicalisation drops `signature`, a bundle can be re-signed after an
edit without the old signature contaminating the new one — but the *digest*
will differ, which is exactly what an auditor compares against a recorded
reference.

---

## AEPRecord (TrainActionEvidence)

One Agent Evidence Protocol record per collective. Defined in
`train_replay/recording/evidence.py`. This is the per-action **AEPRecord** emitted
inside an `EpochEvidenceBundle.actions`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `action_id` | `str` | *(required)* | Stable action identifier. `EpochRecorder` produces `r{rank}:seq{sequence_id}`. |
| `rank` | `int` | *(required)* | Rank the collective ran on. |
| `step` | `int` | *(required)* | Optimizer step / sequence id. |
| `collective_type` | `str` | *(required)* | NCCL collective type, e.g. `all_reduce`, `recv`, `barrier`. |
| `recording_mode` | `RecordingMode` | *(required)* | Mode chosen by the policy: `validation`, `delta`, or `full`. |
| `tensor_input_digest` | `str \| None` | `None` | Hash of the input tensor (populated in `validation`/`full`). |
| `tensor_output_digest` | `str \| None` | `None` | Hash of the output tensor (populated in `validation`/`full`). |
| `delta_stats` | `dict[str, float] \| None` | `None` | Statistical diff (mean/var/percentiles) for `delta` mode. |
| `timestamp_ns` | `int` | `0` | Collective start time in nanoseconds (`CollectiveEvent.start_time_ns`). |
| `causal_chain_id` | `str \| None` | `None` | Optional causal-chain correlation id (cross-environment join key). |
| `parent_action_id` | `str \| None` | `None` | Optional link to the producing parent action for chain reconstruction. |

> **Note on population.** `EpochRecorder.record_collective()` currently records
> the policy-derived `recording_mode`, `action_id`, `rank`, `step`,
> `collective_type`, and `timestamp_ns`. The digest/delta/chain fields are part
> of the schema so richer evidence can be attached without a wire-format change
> as the profiler integration deepens.

---

## CollectiveEvent

One parsed row from a PyTorch Flight Recorder dump. Defined in
`train_replay/collector/flight_recorder.py`. Produced by
`load_flight_recorder(path)`, which reads a pickle dump emitted by
`torch._C._distributed_c10d._dump_nccl_trace()`.

| Field | Type | Default | Dump field it is read from |
|---|---|---|---|
| `rank` | `int` | *(from dump)* | `rank` (default `0` if absent) |
| `process_group` | `str` | *(from dump)* | `pg_name` (default `"default"`) |
| `collective_type` | `str` | *(from dump)* | `collective_seq` (default `"unknown"`) |
| `src_rank` | `int \| None` | *(from dump)* | `p2p_src` |
| `dst_rank` | `int \| None` | *(from dump)* | `p2p_dst` |
| `tensor_size` | `int` | *(from dump)* | `input_sizes[0][0]` (first input's size; `0` if absent) |
| `enqueue_time_ns` | `int` | *(from dump)* | `time_created_ns` |
| `start_time_ns` | `int` | *(from dump)* | `time_started_ns` |
| `end_time_ns` | `int` | *(from dump)* | `time_finished_ns` |
| `call_stack` | `list[str]` | `[]` | `frames` |
| `sequence_id` | `int` | `0` | `seq_id` |

The `sequence_id` is the join key used to name graph nodes (e.g.
`tensor:{rank}:{sequence_id}:out`) and to construct action ids
(`r{rank}:seq{sequence_id}`).

---

## TensorEvent

One tensor-level event captured live by the profiler hook. Defined in
`train_replay/collector/profiler_hook.py`. Produced by
`EvidenceProfilerHook.record_tensor()`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `tensor_id` | `str` | *(computed)* | `r{rank}:s{step}:{op_name}`. Uniquely names a tensor at a (rank, step, op). |
| `op_name` | `str` | *(required)* | The operation name passed to `record_tensor()`. |
| `rank` | `int` | *(required)* | Rank owning this tensor. |
| `step` | `int` | *(required)* | Current training step, incremented by `on_step_begin()`. |
| `shape` | `list[int]` | `[]` | `list(tensor.shape)`. |
| `digest` | `str \| None` | *(computed)* | `sha256` of the first 4096 bytes of the flattened tensor, truncated to 16 hex chars; `None` if the tensor could not be read. |

The digest is intentionally cheap (first 4096 bytes, 16 hex chars) so it can be
recorded per-tensor without dominating step time; it is sufficient for
ordering/identity checks, not cryptographic integrity of the full tensor.

---

## Control enums

Shared by the recording layer. Defined in `train_replay/recording/modes.py`.

### `RecordingMode`

| Name | Value | Stores |
|---|---|---|
| `VALIDATION` | `"validation"` | tensor hash + ordering metadata |
| `DELTA` | `"delta"` | statistical diff (mean/var/percentiles) |
| `FULL` | `"full"` | full tensor snapshot (sampled) |

### `SideEffectClass`

| Name | Value |
|---|---|
| `READ` | `"read"` |
| `MUTATE_LOCAL` | `"mutate-local"` |
| `MUTATE_EXTERNAL` | `"mutate-external"` |
| `NETWORK_EGRESS` | `"network-egress"` |
| `UNKNOWN` | `"unknown"` |

### `RiskContext`

| Field | Type | Default |
|---|---|---|
| `was_vetted` | `bool` | `False` |
| `has_consent_anomaly` | `bool` | `False` |
| `taint_chain_length` | `int` | `0` |
| `side_effect_class` | `SideEffectClass` | `SideEffectClass.UNKNOWN` |

### `RecordingPolicy`

| Field | Type |
|---|---|
| `mode` | `RecordingMode` |
| `reason` | `str` |

The mapping from `RiskContext` → `RecordingPolicy` is documented in
[architecture.md § Recording policy](architecture.md#recording-policy).

---

## Signature envelope

Attached to `EpochEvidenceBundle.signature` by `BundleSigner.sign(bundle)`.
Defined in `train_replay/signing/signer.py`.

| Key | Value |
|---|---|
| `alg` | `"ed25519"` |
| `key_id` | The signer's `key_id` (default `"dev-key"` from `BundleSigner.generate()`). |
| `sig` | Base64-encoded Ed25519 signature of `bundle.canonical_bytes()`. |

Verification (`verify_bundle(bundle, public_key)`) recomputes
`canonical_bytes()` and checks the signature with the supplied Ed25519 public
key, returning `False` on any missing signature or verification failure.
