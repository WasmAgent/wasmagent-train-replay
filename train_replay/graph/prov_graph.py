"""PROV-DM causal graph for cross-rank distributed training provenance."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import networkx as nx


@dataclass
class ProvActivity:
    """One NCCL collective or kernel execution."""
    id: str
    label: str
    rank: int
    process_group: str
    timestamp_ns: int
    collective_type: str


@dataclass
class ProvEntity:
    """One tensor/gradient at a specific (rank, step)."""
    id: str
    digest: str | None
    rank: int
    step: int


@dataclass
class ProvAgent:
    """One rank / process group."""
    id: str
    rank: int
    process_group: str


class ProvGraph:
    """Directed PROV-DM graph backed by networkx.DiGraph.

    Nodes carry ``kind`` attribute: 'activity' | 'entity' | 'agent'.
    Edges carry ``rel`` attribute: 'used' | 'wasGeneratedBy' | 'wasAssociatedWith'
    | 'wasDerivedFrom'.
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()

    def add_activity(self, act: ProvActivity) -> None:
        self._g.add_node(act.id, kind="activity", data=act)

    def add_entity(self, ent: ProvEntity) -> None:
        self._g.add_node(ent.id, kind="entity", data=ent)

    def add_agent(self, agent: ProvAgent) -> None:
        self._g.add_node(agent.id, kind="agent", data=agent)

    def used(self, activity_id: str, entity_id: str) -> None:
        self._g.add_edge(activity_id, entity_id, rel="used")

    def was_generated_by(self, entity_id: str, activity_id: str) -> None:
        # activity -> entity: traversing predecessors of entity reveals its generating activities
        self._g.add_edge(activity_id, entity_id, rel="wasGeneratedBy")

    def was_associated_with(self, activity_id: str, agent_id: str) -> None:
        self._g.add_edge(activity_id, agent_id, rel="wasAssociatedWith")

    def was_derived_from(self, entity_id: str, predecessor_id: str) -> None:
        """entity_id was derived from predecessor_id (a ``wasDerivedFrom`` edge).

        The edge direction is predecessor_id -> entity_id, so that
        ``in_edges(entity_id)`` returns predecessor entities.
        """
        self._g.add_edge(predecessor_id, entity_id, rel="wasDerivedFrom")

    def ancestors_of(self, entity_id: str) -> list[str]:
        """Return all activity IDs that causally contributed to entity_id.

        Traverses the graph bidirectionally:

        * From **entities** follows ``wasGeneratedBy`` edges to find the
          activity that produced the entity, and ``wasDerivedFrom`` edges to
          find predecessor entities that the entity was derived from.
        * From **activities** follows ``used`` edges to find the input entities
          consumed by that activity, so that the chain can continue backward
          through earlier activities.

        Input entities that are consumed but never produced by any activity
        will therefore halt the chain (they have no ``wasGeneratedBy``
        predecessor), which is the desired behaviour — a leaf input has no
        causal ancestors within the recorded trace.
        """
        if entity_id not in self._g:
            return []

        visited: list[str] = []
        queue = [entity_id]
        seen: set[str] = set()
        while queue:
            node = queue.pop()
            if node in seen:
                continue
            seen.add(node)
            node_kind: str | None = self._g.nodes[node].get("kind")

            # --- Incoming edges: what generated / produced this node? ---
            for pred, _, edge_data in self._g.in_edges(node, data=True):
                rel = edge_data.get("rel")
                if rel == "wasGeneratedBy" and node_kind == "entity":
                    # The predecessor is an activity that generated this entity.
                    if self._g.nodes[pred].get("kind") == "activity":
                        visited.append(pred)
                    queue.append(pred)
                elif rel == "wasDerivedFrom":
                    # The predecessor is an entity that this one was derived from.
                    queue.append(pred)
                # ``used`` edges point from activity -> entity; when seen as
                # an incoming edge on an entity they represent consumption,
                # not ancestry — skip.

            # --- Outgoing edges from activities: what did this activity use? ---
            if node_kind == "activity":
                for _, succ, edge_data in self._g.out_edges(node, data=True):
                    rel = edge_data.get("rel")
                    if rel == "used":
                        # The successor is an entity consumed by this activity.
                        queue.append(succ)
                    # ``wasGeneratedBy`` / ``wasAssociatedWith`` edges on
                    # activities are output-side relations and do not lead to
                    # further ancestors.

        return visited

    def causal_subgraph(self, entity_id: str) -> ProvGraph:
        """Return a new ProvGraph containing only the causal ancestors of entity_id."""
        ancestor_ids = set(self.ancestors_of(entity_id)) | {entity_id}
        sub = nx.subgraph(self._g, ancestor_ids)
        new = ProvGraph()
        new._g = nx.DiGraph(sub)
        return new

    def nodes(self) -> Iterator[tuple[str, dict[str, Any]]]:
        return self._g.nodes(data=True)  # type: ignore[no-any-return]

    def to_dict(self) -> dict[str, Any]:
        return nx.node_link_data(self._g)  # type: ignore[no-any-return]
