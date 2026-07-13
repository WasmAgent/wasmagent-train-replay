"""Build a ProvGraph from CollectiveEvent lists across all ranks.

The standalone ``build_from_events`` function is retained for backward
compatibility.  ``NCCLGraphBuilder`` implements the ``GraphBuilder`` protocol
defined in ``base.py`` and can be used polymorphically alongside future
Gloo/MTIA builders.
"""

from __future__ import annotations

from ..collector.flight_recorder import CollectiveEvent
from .base import BackendEvent
from .prov_graph import ProvActivity, ProvAgent, ProvEntity, ProvGraph


class NCCLGraphBuilder:
    """Concrete ``GraphBuilder`` for NCCL ``CollectiveEvent`` records.

    Implements the ``GraphBuilder`` protocol from ``base.py``.  Each collective
    becomes a PROV-DM Activity; input/output tensors become Entities; each
    (rank, process group) pair becomes an Agent.  Edge semantics follow PROV-DM.
    """

    def build(self, events: list[BackendEvent]) -> ProvGraph:
        """Construct a cross-rank causal graph from NCCL Flight Recorder events."""
        graph = ProvGraph()
        agents: dict[tuple[int, str], str] = {}

        for evt in events:
            agent_key = (evt.rank, evt.group_id)
            if agent_key not in agents:
                agent_id = f"rank:{evt.rank}:pg:{evt.group_id}"
                graph.add_agent(ProvAgent(id=agent_id, rank=evt.rank, process_group=evt.group_id))
                agents[agent_key] = agent_id

            act_id = f"act:{evt.rank}:{evt.operation_type}:{evt.sequence_id}"
            graph.add_activity(ProvActivity(
                id=act_id,
                label=evt.operation_type,
                rank=evt.rank,
                process_group=evt.group_id,
                timestamp_ns=evt.timestamp_ns,
                collective_type=evt.operation_type,
            ))
            graph.was_associated_with(act_id, agents[agent_key])

            # Input entity (tensor before operation)
            in_ent_id = f"tensor:{evt.rank}:{evt.sequence_id}:in"
            graph.add_entity(
                ProvEntity(id=in_ent_id, digest=None, rank=evt.rank, step=evt.sequence_id),
            )
            graph.used(act_id, in_ent_id)

            # Output entity (tensor after operation)
            out_ent_id = f"tensor:{evt.rank}:{evt.sequence_id}:out"
            graph.add_entity(
                ProvEntity(id=out_ent_id, digest=None, rank=evt.rank, step=evt.sequence_id),
            )
            graph.was_generated_by(out_ent_id, act_id)

        return graph


def build_from_events(events: list[CollectiveEvent]) -> ProvGraph:
    """Construct a cross-rank causal graph from Flight Recorder events.

    Each collective is an Activity; input/output tensors are Entities.
    Ranks are Agents. Edges follow PROV-DM semantics.

    .. deprecated::
        Prefer ``NCCLGraphBuilder().build(events)`` for protocol-compatible
        usage.  This function is retained for backward compatibility.
    """
    return NCCLGraphBuilder().build(events)  # type: ignore[arg-type]
