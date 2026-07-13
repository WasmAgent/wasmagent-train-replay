"""PROV-DM causal graph for cross-rank distributed training provenance."""

from __future__ import annotations

import hashlib
import json
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
        # activity -> entity: traversing predecessors of entity reveals its generating activities
        self._g.add_edge(activity_id, entity_id, rel="wasGeneratedBy")

    def was_associated_with(self, activity_id: str, agent_id: str) -> None:
        self._g.add_edge(activity_id, agent_id, rel="wasAssociatedWith")

    def ancestors_of(self, entity_id: str) -> list[str]:
        """Return all activity IDs that causally contributed to entity_id.
        Only traverses wasGeneratedBy edges (not used edges), so input
        entities (consumed but not produced by an activity) return [].
        """
        visited: list[str] = []
        queue = [entity_id]
        seen: set[str] = set()
        while queue:
            node = queue.pop()
            if node in seen:
                continue
            seen.add(node)
            for pred, _, edge_data in self._g.in_edges(node, data=True):
                if edge_data.get("rel") != "wasGeneratedBy":
                    continue  # skip "used" edges — those are inputs, not ancestry
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
        return self._g.nodes(data=True)  # type: ignore[no-any-return]

    def to_dict(self) -> dict[str, Any]:
        return nx.node_link_data(self._g)  # type: ignore[no-any-return]

    def digest(self) -> str:
        """Return a SHA-256 hex digest of the graph's canonical structure.

        The digest covers all nodes, edges, and their attributes.  It is
        stable across runs as long as the graph topology and attribute
        values are identical.

        Uses the networkx node-link format (sorted by node/edge keys) to
        produce a deterministic JSON representation, then SHA-256 hashes it.
        """
        data = nx.node_link_data(self._g)
        # Ensure deterministic ordering by sorting all lists and dicts
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()
