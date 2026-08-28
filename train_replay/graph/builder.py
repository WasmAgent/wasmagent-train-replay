"""Build a ProvGraph from CollectiveEvent or MtiaEvent lists across all ranks."""

from __future__ import annotations

from collections.abc import Iterable

from ..collector.flight_recorder import CollectiveEvent
from ..collector.mtia import MtiaEvent
from .prov_graph import ProvActivity, ProvAgent, ProvEntity, ProvGraph


def _build_graph(rows: Iterable[tuple[int, str, str, int, int]]) -> ProvGraph:
    """Construct a PROV-DM graph from normalized event rows.

    Each row is ``(rank, process_group, collective_type, sequence_id,
    start_time_ns)``.  Each collective becomes an Activity; its input/output
    tensors become Entities and its rank an Agent, wired with PROV-DM edges.
    """
    graph = ProvGraph()
    agents: dict[tuple[int, str], str] = {}

    for rank, process_group, ctype, sequence_id, start_time_ns in rows:
        agent_key = (rank, process_group)
        if agent_key not in agents:
            agent_id = f"rank:{rank}:pg:{process_group}"
            graph.add_agent(
                ProvAgent(id=agent_id, rank=rank, process_group=process_group)
            )
            agents[agent_key] = agent_id

        act_id = f"act:{rank}:{ctype}:{sequence_id}"
        graph.add_activity(ProvActivity(
            id=act_id,
            label=ctype,
            rank=rank,
            process_group=process_group,
            timestamp_ns=start_time_ns,
            collective_type=ctype,
        ))
        graph.was_associated_with(act_id, agents[agent_key])

        # Input entity (tensor before collective) and output entity (after).
        graph.add_entity(ProvEntity(
            id=f"tensor:{rank}:{sequence_id}:in", digest=None, rank=rank, step=sequence_id,
        ))
        graph.used(act_id, f"tensor:{rank}:{sequence_id}:in")
        graph.add_entity(ProvEntity(
            id=f"tensor:{rank}:{sequence_id}:out", digest=None, rank=rank, step=sequence_id,
        ))
        graph.was_generated_by(f"tensor:{rank}:{sequence_id}:out", act_id)

    return graph


def build_from_events(events: list[CollectiveEvent]) -> ProvGraph:
    """Construct a cross-rank causal graph from Flight Recorder events.

    Each collective is an Activity; input/output tensors are Entities.
    Ranks are Agents. Edges follow PROV-DM semantics.
    """
    return _build_graph(
        (evt.rank, evt.process_group, evt.collective_type, evt.sequence_id, evt.start_time_ns)
        for evt in events
    )


def build_from_mtia_events(events: list[MtiaEvent]) -> ProvGraph:
    """Construct a cross-rank causal graph from MTIA profiler events.

    MTIA-specific convenience wrapper around the same graph construction logic
    used by :func:`build_from_events`, accepting :class:`MtiaEvent` records
    instead of NCCL :class:`CollectiveEvent`.
    """
    return _build_graph(
        (evt.rank, evt.process_group, evt.op_type, evt.sequence_id, evt.start_time_ns)
        for evt in events
    )
