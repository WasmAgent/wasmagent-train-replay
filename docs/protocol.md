# Protocol & data formats

> Field-by-field reference for every record type that flows through the pipeline.
> Types are Python type annotations as they appear in the source dataclasses.

This is the canonical schema and wire-format reference. The implementation does
not define a protobuf, msgpack, or custom binary frame: PyTorch Flight Recorder
input arrives as a Python pickle byte stream, tensor events are in-process
dataclasses, and AEP evidence is converted to deterministic UTF-8 JSON bytes by
`EpochEvidenceBundle.canonical_bytes()` before hashing and signing. Bundle
persistence uses `EpochEvidenceBundle.to_json()` and `to_cbor()`; the planned
auditor export package is specified in
[export-command-design.md](export-command-design.md). For how these records are
produced and consumed see [architecture.md](architecture.md); for the
surrounding CLI see [cli-reference.md](cli-reference.md).

All structs in this document are Python dataclasses in the `train_replay`
package. Field names below are source-level names, and any on-disk
representation is derived from those dataclasses rather than a separate schema
compiler.

Contents:

- [EpochEvidenceBundle](#epochevidencebundle) — the signed, per-epoch envelope
- [AEPRecord](#aeprecord) — one recorded collective
- [CollectiveEvent](#collectiveevent) — one row from a Flight Recorder dump
- [TensorEvent](#tensorevent) — one tensor-level event from the profiler hook
- [Control enums](#control-enums) — shared `RecordingMode` / `SideEffectClass` values
- [Signature envelope](#signature-envelope) — the DSSE-style signing block
- [Wire encodings](#wire-encodings) — byte-level source and signing formats

---

## Wire encodings

| Record | Source / sink | Byte representation |
|---|---|---|
| `CollectiveEvent` | Read from PyTorch Flight Recorder dumps by `load_flight_recorder(path)`. | Pickle bytes containing a dict with an `entries` list. The loader maps each entry to the `CollectiveEvent` dataclass. |
| `TensorEvent` | Produced in-process by `EvidenceProfilerHook.record_tensor()`. | Python dataclass values in memory; tensor identity is represented by `tensor_id`, shape metadata, and an optional digest. |
| `EpochEvidenceBundle` | Produced by `EpochRecorder.bundle()` and signed by `BundleSigner.sign()`. | `canonical_bytes()` returns UTF-8 JSON bytes with keys sorted and `signature` omitted. These bytes are the hash and signature payload. |
| `AEPRecord` | Embedded in `EpochEvidenceBundle.actions`. | JSON object inside the bundle canonical payload; `RecordingMode` values serialize as their string values (`"validation"`, `"delta"`, `"full"`). |
| Signing wrapper | Stored in `EpochEvidenceBundle.signature`. | DSSE-style (Delegate Signing for Secure Environments) JSON object with `alg`, `key_id`, and base64 `sig`, where `sig` is the Ed25519 signature over `canonical_bytes()`. |
| Export package manifest | Planned `train-replay export` directory output. | JSON inventory with source metadata, bundle digest, serialized artifact hashes, and validation status. See [export-command-design.md § Manifest JSON](export-command-design.md#manifest-json). |

The "binary" boundary for evidence is therefore the canonical byte string
returned by `canonical_bytes()`. That byte string, not the pretty-printed JSON
shown in examples, is what `digest()`, `BundleSigner.sign()`, and
`verify_bundle()` operate on.

When persisted outside the current CLI, callers should write either the raw
Flight Recorder pickle dump or a JSON/CBOR representation derived from the
dataclass fields below. The repository does not currently ship a bundle writer
command; `train-replay record` prints the action count and bundle digest. The
future export command must not add fields to `EpochEvidenceBundle`; package
metadata belongs in the export manifest.

---

## EpochEvidenceBundle

The top-level, signable record. One bundle covers one epoch across all ranks.
Defined in `train_replay/recording/evidence.py`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `schema_version` | `str` | `"train-aep/v0.1"` | Wire-format version. Bumped on incompatible changes. |
| `run_id` | `str` | `""` | Training run identifier (e.g. `"dev-run"`). |
| `epoch` | `int` | `0` | Epoch index this bundle covers. |
| `actions` | `list[AEPRecord]` | `[]` | One entry per recorded collective, in append order. |
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

## AEPRecord

One Agent Evidence Protocol (AEP) record per collective — the per-action
evidence emitted inside `EpochEvidenceBundle.actions`. Defined in
`train_replay/recording/evidence.py` as the `AEPRecord` dataclass.

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
| `digest` | `str \| None` | *(computed)* | `sha256` of the first 4096 bytes of the tensor after `.detach().cpu().float()` normalisation (row-major bytes — the first 1024 float32 values), truncated to 16 hex chars; `None` if the tensor could not be read. |

The digest is intentionally cheap (first 4096 bytes, 16 hex chars) so it can be
recorded per-tensor without dominating step time; it is sufficient for
ordering/identity checks, not cryptographic integrity of the full tensor. Note
the `.float()` cast: the digest is over the **float32-normalised** byte buffer,
so a `float16`/`bfloat16`/integer tensor is hashed as its upcast `float32` view,
not its raw storage.

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

---

## Worked example: a signed bundle on the wire

This ties the schemas above to the exact bytes that hit disk. The bundle below
holds a single recorded collective — an `all_reduce` on rank 2, sequence 3 —
recorded in `full` mode because `all_reduce` classifies as `mutate-external`
(see [architecture.md § Recording policy](architecture.md#recording-policy)).

### Canonical payload (`canonical_bytes()`)

`canonical_bytes()` strips `signature` and serialises everything else with
`json.dumps(..., sort_keys=True, default=str)`. On the wire this is a single
line; it is pretty-printed here for readability:

```json
{
  "actions": [
    {
      "action_id": "r2:seq3",
      "causal_chain_id": null,
      "collective_type": "all_reduce",
      "delta_stats": null,
      "parent_action_id": null,
      "rank": 2,
      "recording_mode": "full",
      "step": 3,
      "tensor_input_digest": null,
      "tensor_output_digest": null,
      "timestamp_ns": 3100000
    }
  ],
  "epoch": 5,
  "run_id": "run-42",
  "schema_version": "train-aep/v0.1"
}
```

Because keys are sorted and `RecordingMode` is a `str`-subclass enum (so it
serialises as its value, `"full"`), this byte string is stable for a given
action set.

### Digest

```
sha256(canonical_bytes()) = f45ba1a42da0d349d8aa8b4c2e4e73e738807b58b099fe9954ed23e6247b0779
```

An auditor records this digest out-of-band; any post-hoc edit of a field above
changes it, which is exactly what `verify_bundle()` detects.

### Signature envelope (`bundle.signature` after `BundleSigner.sign()`)

```json
{
  "alg": "ed25519",
  "key_id": "ci-key",
  "sig": "n6RDjXLF9z8FLEmo8IbMUI/xhORNI/QVt8PUpjbzJOAMi75BQbZSVE3ISpcHIAg6RWgDiYetxt/Vl451VtM6AQ=="
}
```

`sig` is the base64-encoded Ed25519 signature of the single-line
`canonical_bytes()`; it is specific to the key generated for this example
(`BundleSigner.generate(key_id="ci-key")`). `verify_bundle(bundle, public_key)`
returns `True` for the matching public key and `False` for any other key — or
for a bundle whose canonical bytes have been altered after signing.
