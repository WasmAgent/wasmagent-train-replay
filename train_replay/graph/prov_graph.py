"""PROV-DM causal graph for cross-rank distributed training provenance."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from dataclasses import dataclass

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
    Edges carry ``rel`` attribute: 'used' | 'wasGeneratedBy' | 'wasAssociatedWith'.
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
        self._g.add_edge(entity_id, activity_id, rel="wasGeneratedBy")

    def was_associated_with(self, activity_id: str, agent_id: str) -> None:
        self._g.add_edge(activity_id, agent_id, rel="wasAssociatedWith")

    def ancestors_of(self, entity_id: str) -> list[str]:
        """Return all activity IDs that causally contributed to entity_id."""
        visited: list[str] = []
        queue = [entity_id]
        seen: set[str] = set()
        while queue:
            node = queue.pop()
            if node in seen:
                continue
            seen.add(node)
            for pred in self._g.predecessors(node):
                data = self._g.nodes[pred]
                if data.get("kind") == "activity":
                    visited.append(pred)
                queue.append(pred)
        return visited

    def causal_subgraph(self, entity_id: str) -> ProvGraph:
        """Return a new ProvGraph containing only the causal ancestors of entity_id."""
        ancestor_ids = set(self.ancestors_of(entity_id)) | {entity_id}
        sub = nx.subgraph(self._g, ancestor_ids)
        new = ProvGraph()
        new._g = nx.DiGraph(sub)
        return new

    def nodes(self) -> Iterator[tuple[str, dict[str, Any]]]:
        return self._g.nodes(data=True)  # type: ignore[return-value,no-any-return]

    def to_dict(self) -> dict[str, Any]:
        return nx.node_link_data(self._g)  # type: ignore[return-value,no-any-return]
