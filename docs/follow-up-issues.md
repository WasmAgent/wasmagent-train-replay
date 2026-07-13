# Follow-up issues: concrete abstractions needed

> Identified during the framework-agnostic audit of `train_replay/graph/`.
> Each item is a concrete abstraction or interface that must be introduced
> before `wasmagent-train-replay` can claim full backend-neutrality for
> Gloo, MTIA, or custom backends.

---

## Issue A: Abstract `CollectiveEvent` source interface

**Problem.** `build_from_events()` and all downstream code accept
`list[CollectiveEvent]`. The `CollectiveEvent` dataclass is defined in
`train_replay/collector/flight_recorder.py` — a module whose name implies
PyTorch-specific origin. Any new backend (Gloo, MTIA) must either reuse
`CollectiveEvent` directly (which ties it to the PyTorch Flight Recorder
module) or force a conversion step outside the graph layer.

**Proposed abstraction.** Define a `CollectiveEvent`-like protocol or
abstract base class (ABC) in `train_replay/graph/event.py` (or
`train_replay/events.py`) that decouples the graph builder from the
collector module. The builder accepts any iterable of objects conforming
to this protocol. The existing `CollectiveEvent` dataclass in
`flight_recorder.py` implements the protocol.

**Checklist:**
- [ ] Move/alias `CollectiveEvent` fields into a protocol in `train_replay/graph/event.py`.
- [ ] Update `build_from_events()` signature to accept `Iterable[CollectiveEventProtocol]`.
- [ ] Keep the existing `CollectiveEvent` dataclass in `flight_recorder.py` as an implementation.
- [ ] Add a test that builds a graph from synthetic events using a plain dict or a
      different backend's event class (e.g. `GlooEvent`) that also conforms.

---

## Issue B: Backend-agnostic collective type taxonomy

**Problem.** `collective_type` is a free-form `str` (e.g. `"all_reduce"`,
`"barrier"`, `"all_gather"`, `"broadcast"`). The recording layer in
`recorder.py` uses `_collective_side_effect()` to classify side effects
based on hardcoded string values:

```python
def _collective_side_effect(ctype: str) -> SideEffectClass:
    reads = {"recv", "barrier"}
    return SideEffectClass.READ if ctype.lower() in reads else SideEffectClass.MUTATE_EXTERNAL
```

This is fragile: a backend that names its barrier `"sync"` instead of
`"barrier"` would misclassify. Moreover, the mapping from collective type
to side-effect class is currently implicit and hardcoded.

**Proposed abstraction.** Introduce a `CollectiveType` enum or a
`CollectiveClassifier` interface that backends can register against:

```python
class CollectiveType(str, enum.Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    BROADCAST = "broadcast"
    BARRIER = "barrier"
    RECV = "recv"
    SEND = "send"
    REDUCE_SCATTER = "reduce_scatter"
    # ... plus a UNKNOWN for extension

class CollectiveClassifier(Protocol):
    def side_effect_class(self, collective_type: str) -> SideEffectClass: ...
```

The default classifier covers NCCL/Gloo common types. Backends (MTIA,
custom) provide their own classifier.

**Checklist:**
- [ ] Define `CollectiveType` canonical enum in `train_replay/graph/event.py`.
- [ ] Define `CollectiveClassifier` protocol.
- [ ] Replace `_collective_side_effect()` with the protocol in `EpochRecorder`.
- [ ] Wire a `--collective-classifier` option or accept a classifier instance
      in `EpochRecorder.__init__()`.

---

## Issue C: Multi-backend node ID scheme

**Problem.** Node IDs in the graph follow the pattern
`act:{rank}:{collective_type}:{sequence_id}`. The `collective_type` string
comes directly from the source event, so a backend that emits
`"AllReduce"` (capitalised) would produce IDs inconsistent with NCCL's
`"all_reduce"`. While case-insensitive comparison works at the graph level,
the ID string alone is not namespace-safe across backends.

**Proposed abstraction.** Prefix node IDs with a backend namespace:

```
act:nccl:0:all_reduce:1
act:gloo:0:allgather:1
act:mtia:0:barrier:1
```

or use a separate `backend` attribute on the node data.

**Checklist:**
- [ ] Add `backend` field to `ProvActivity` and use it in the ID prefix.
- [ ] Update `build_from_events()` to accept an optional `backend` parameter
      (default `"nccl"`).
- [ ] Update `CollectiveEvent` (or the protocol) to carry a `backend` field.
- [ ] Update all node-ID formatting in `builder.py` and `prov_graph.py`.

---

## Issue D: Parametric process group resolution

**Problem.** `build_from_events()` creates one `ProvAgent` per unique
`(rank, process_group)` tuple. The `process_group` string comes directly
from the Flight Recorder dump (`pg_name` field). For multi-backend runs
where different backends use the same process group name, the agent
uniqueness is lost.

**Proposed abstraction.** The agent key should include the backend
identifier (see Issue C), so agents are identified by
`(backend, rank, process_group)`.

**Checklist:**
- [ ] Update `ProvAgent` (or the agent ID) to include backend.
- [ ] Update agent-key creation in `build_from_events()`.

---

## Issue E: Recording policy extensibility for non-NCCL backends

**Problem.** `compile_recording_policy()` currently maps a small set of
`SideEffectClass` values to `RecordingMode`. The `RiskContext` fields
(`was_vetted`, `has_consent_anomaly`, `taint_chain_length`,
`side_effect_class`) are generic enough for any backend, but the
*default* side-effect classification (Issue B) is NCCL-centric.

**Proposed abstraction.** `EpochRecorder` should accept a
`CollectiveClassifier` (or equivalent) so that a Gloo or MTIA integrator
can supply their own classification without editing the recording module.

**Checklist:**
- [ ] Add `classifier: CollectiveClassifier` parameter to `EpochRecorder.__init__()`.
- [ ] Use the classifier in `record_collective()` instead of the private
      `_collective_side_effect()` function.

---

## Issue F: Cross-backend causal chain joining

**Problem.** The current `causal_chain_id` and `parent_action_id` fields on
`AEPRecord` support linking actions across environments (gateway → agent →
training job), but only if both sides agree on the trace-id format and
namespace. A backend-agnostic joining mechanism is needed for heterogenous
pipelines (e.g. Gloo data-loading collectives → NCCL training collectives).

**Proposed abstraction.** Define a `TraceId` schema (for example a
`trace_id: str` and `span_id: str` pair) that backends can propagate via
metadata, and a `ChainJoiner` that merges two `ProvGraph` instances on
shared trace/span ids.

**Checklist:**
- [ ] Define `TraceId` dataclass or protocol.
- [ ] Implement `ProvGraph.merge(other, on=...)` for cross-graph joining.
- [ ] Document trace-id propagation for backend integrators.

---

## Priority order

For the next development cycle, the recommended order is:

1. **Issue A + B** (interface definitions) — these unblock all downstream work.
2. **Issue C** (backend namespace in node IDs) — requires Issue A.
3. **Issue D** (parametric agent resolution) — requires Issue C.
4. **Issue E** (extensible recording policy) — requires Issue B.
5. **Issue F** (cross-backend joining) — standalone, lower priority.

---

### See also

- [`docs/positioning.md`](positioning.md) — why framework-agnostic coverage matters.
- [`train_replay/graph/builder.py`](../train_replay/graph/builder.py) — the current NCCL-centric builder.
- [`train_replay/collector/flight_recorder.py`](../train_replay/collector/flight_recorder.py) — the `CollectiveEvent` dataclass.
