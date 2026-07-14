"""Build a ProvGraph from CollectiveEvent or MtiaEvent lists across all ranks."""

from __future__ import annotations

from ..collector.flight_recorder import CollectiveEvent
from ..collector.mtia import MtiaEvent
from .prov_graph import ProvActivity, ProvAgent, ProvEntity, ProvGraph


def build_from_events(events: list[CollectiveEvent]) -> ProvGraph:
    """Construct a cross-rank causal graph from Flight Recorder events.

    Each collective is an Activity; input/output tensors are Entities.
    Ranks are Agents. Edges follow PROV-DM semantics.
    """
    graph = ProvGraph()
    agents: dict[tuple[int, str], str] = {}

    for evt in events:
        agent_key = (evt.rank, evt.process_group)
        if agent_key not in agents:
            agent_id = f"rank:{evt.rank}:pg:{evt.process_group}"
            graph.add_agent(ProvAgent(id=agent_id, rank=evt.rank, process_group=evt.process_group))
            agents[agent_key] = agent_id

        act_id = f"act:{evt.rank}:{evt.collective_type}:{evt.sequence_id}"
        graph.add_activity(ProvActivity(
            id=act_id,
            label=evt.collective_type,
            rank=evt.rank,
            process_group=evt.process_group,
            timestamp_ns=evt.start_time_ns,
            collective_type=evt.collective_type,
        ))
        graph.was_associated_with(act_id, agents[agent_key])

        # Input entity (tensor before collective)
        in_ent_id = f"tensor:{evt.rank}:{evt.sequence_id}:in"
        graph.add_entity(ProvEntity(id=in_ent_id, digest=None, rank=evt.rank, step=evt.sequence_id))
        graph.used(act_id, in_ent_id)

        # Output entity (tensor after collective)
        out_ent_id = f"tensor:{evt.rank}:{evt.sequence_id}:out"
        graph.add_entity(
            ProvEntity(id=out_ent_id, digest=None, rank=evt.rank, step=evt.sequence_id),
        )
        graph.was_generated_by(out_ent_id, act_id)

    return graph


def build_from_mtia_events(events: list[MtiaEvent]) -> ProvGraph:
    """Construct a cross-rank causal graph from MTIA profiler events.

    Each collective is an Activity; input/output tensors are Entities.
    Ranks are Agents. Edges follow PROV-DM semantics.

    This is the MTIA-specific convenience wrapper around the same graph
    construction logic used by :func:`build_from_events`, accepting
    :class:`MtiaEvent` records instead of NCCL :class:`CollectiveEvent`.
    """
    graph = ProvGraph()
    agents: dict[tuple[int, str], str] = {}

    for evt in events:
        agent_key = (evt.rank, evt.process_group)
        if agent_key not in agents:
            agent_id = f"rank:{evt.rank}:pg:{evt.process_group}"
            graph.add_agent(
                ProvAgent(id=agent_id, rank=evt.rank, process_group=evt.process_group)
            )
            agents[agent_key] = agent_id

        act_id = f"act:{evt.rank}:{evt.op_type}:{evt.sequence_id}"
        graph.add_activity(
            ProvActivity(
                id=act_id,
                label=evt.op_type,
                rank=evt.rank,
                process_group=evt.process_group,
                timestamp_ns=evt.start_time_ns,
                collective_type=evt.op_type,
            )
        )
        graph.was_associated_with(act_id, agents[agent_key])

        # Input entity (tensor before collective)
        in_ent_id = f"tensor:{evt.rank}:{evt.sequence_id}:in"
        graph.add_entity(
            ProvEntity(id=in_ent_id, digest=None, rank=evt.rank, step=evt.sequence_id)
        )
        graph.used(act_id, in_ent_id)

        # Output entity (tensor after collective)
        out_ent_id = f"tensor:{evt.rank}:{evt.sequence_id}:out"
        graph.add_entity(
            ProvEntity(id=out_ent_id, digest=None, rank=evt.rank, step=evt.sequence_id),
        )
        graph.was_generated_by(out_ent_id, act_id)

    return graph
