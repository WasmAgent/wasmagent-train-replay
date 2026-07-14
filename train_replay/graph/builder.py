"""Build a ProvGraph from CollectiveEvent or OpSpec lists across all ranks."""

from __future__ import annotations

from ..collector.flight_recorder import CollectiveEvent
from .ops import OpSpec
from .prov_graph import ProvActivity, ProvAgent, ProvEntity, ProvGraph


def build_from_specs(specs: list[OpSpec]) -> ProvGraph:
    """Construct a cross-rank causal graph from backend-neutral OpSpecs.

    This is the recommended entry point for building a ProvGraph from any
    communication backend (NCCL, Gloo, MTIA, etc.).  The collector layer
    translates backend-specific trace events into ``OpSpec`` instances; the
    graph layer consumes ``OpSpec`` without inspecting backend-specific
    fields.

    Each collective is an Activity; input/output tensors are Entities.
    Ranks are Agents. Edges follow PROV-DM semantics.
    """
    graph = ProvGraph()
    agents: dict[tuple[int, str], str] = {}

    for spec in specs:
        agent_key = (spec.rank, spec.process_group)
        if agent_key not in agents:
            agent_id = f"rank:{spec.rank}:pg:{spec.process_group}"
            graph.add_agent(
                ProvAgent(id=agent_id, rank=spec.rank, process_group=spec.process_group),
            )
            agents[agent_key] = agent_id

        act_id = f"act:{spec.rank}:{spec.collective_type}:{spec.sequence_id}"
        graph.add_activity(ProvActivity(
            id=act_id,
            label=spec.collective_type,
            rank=spec.rank,
            process_group=spec.process_group,
            timestamp_ns=spec.start_time_ns,
            collective_type=spec.collective_type,
        ))
        graph.was_associated_with(act_id, agents[agent_key])

        # Input entity (tensor before collective)
        in_ent_id = f"tensor:{spec.rank}:{spec.sequence_id}:in"
        graph.add_entity(
            ProvEntity(id=in_ent_id, digest=None, rank=spec.rank, step=spec.sequence_id),
        )
        graph.used(act_id, in_ent_id)

        # Output entity (tensor after collective)
        out_ent_id = f"tensor:{spec.rank}:{spec.sequence_id}:out"
        graph.add_entity(
            ProvEntity(id=out_ent_id, digest=None, rank=spec.rank, step=spec.sequence_id),
        )
        graph.was_generated_by(out_ent_id, act_id)

    return graph


def build_from_events(events: list[CollectiveEvent]) -> ProvGraph:
    """Construct a cross-rank causal graph from NCCL Flight Recorder events.

    Convenience wrapper that converts ``CollectiveEvent`` records to
    backend-neutral ``OpSpec`` instances and delegates to ``build_from_specs``.

    Prefer ``build_from_specs`` for non-NCCL backends.
    """
    from .ops import Backend, CollectiveOp

    specs: list[OpSpec] = []
    for evt in events:
        # Map collective_type string to CollectiveOp enum; fall back to
        # UNKNOWN and preserve the raw string for round-trip fidelity.
        # This ensures the OpSpec.op field never silently misattributes an
        # unknown backend operation to a known one (which would violate
        # the determinism guarantee needed for tamper-evidence).
        try:
            op = CollectiveOp(evt.collective_type)
        except ValueError:
            op = CollectiveOp.UNKNOWN

        specs.append(OpSpec(
            op=op,
            backend=Backend.NCCL,
            rank=evt.rank,
            process_group=evt.process_group,
            sequence_id=evt.sequence_id,
            src_rank=evt.src_rank,
            dst_rank=evt.dst_rank,
            tensor_size=evt.tensor_size,
            start_time_ns=evt.start_time_ns,
            end_time_ns=evt.end_time_ns,
            collective_type_raw=evt.collective_type,
        ))

    return build_from_specs(specs)
