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
    importance_score: float = 0.0


@dataclass
class ProvEntity:
    """One tensor/gradient at a specific (rank, step)."""
    id: str
    digest: str | None
    rank: int
    step: int
    importance_score: float = 0.0


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
            # Edges are stored in PROV-DM statement direction (subject -> object),
            # i.e. effect -> cause for both 'used' (activity -> entity) and
            # 'wasGeneratedBy' (entity -> activity). A causal ancestor (the cause)
            # is therefore reached by following successors, not predecessors.
            for succ in self._g.successors(node):
                data = self._g.nodes[succ]
                if data.get("kind") == "activity":
                    visited.append(succ)
                queue.append(succ)
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

    def compute_importance_scores(
        self,
        anomalous_node_ids: set[str] | None = None,
        anomaly_boost: float = 0.5,
    ) -> None:
        """Compute MemRL-style importance scores for all nodes.

        Base score = number of causal descendants / total nodes (connectivity ratio).
        Nodes that are ancestors of anomalous events receive a boost.

        Mirrors the Intent-Experience-Utility triple from MemRL (arXiv:2601.03192).
        """
        if not self._g.nodes():
            return

        total_nodes = len(self._g.nodes())
        anomalous_set = anomalous_node_ids or set()

        # Precompute causal descendants for all nodes.
        # PROV-DM edges go effect -> cause, so causal descendants (cause -> effect)
        # are reached by following predecessors in the graph (nx.ancestors).
        descendants_map: dict[str, set[str]] = {}
        for node_id in self._g.nodes():
            descendants_map[node_id] = set(nx.ancestors(self._g, node_id))
            descendants_map[node_id].add(node_id)

        # Compute base connectivity score.
        for node_id in self._g.nodes():
            desc_count = len(descendants_map[node_id])
            base_score = desc_count / total_nodes if total_nodes > 0 else 0.0
            node_data = self._g.nodes[node_id]["data"]
            if hasattr(node_data, "importance_score"):
                node_data.importance_score = base_score

        # Boost ancestors of anomalous nodes.
        if anomalous_set:
            for anom_id in anomalous_set:
                if anom_id not in self._g:
                    continue
                # Ancestors of the anomalous node (causes).
                for ancestor_id in self.ancestors_of(anom_id):
                    if ancestor_id in self._g:
                        ancestor_data = self._g.nodes[ancestor_id]["data"]
                        if hasattr(ancestor_data, "importance_score"):
                            ancestor_data.importance_score = min(
                                ancestor_data.importance_score + anomaly_boost, 1.0
                            )
                # The anomalous node itself also gets a boost.
                if anom_id in self._g:
                    anom_data = self._g.nodes[anom_id]["data"]
                    if hasattr(anom_data, "importance_score"):
                        anom_data.importance_score = min(
                            anom_data.importance_score + anomaly_boost, 1.0
                        )

    def get_high_importance_nodes(
        self, threshold: float = 0.5
    ) -> list[tuple[str, float]]:
        """Return nodes with importance_score >= threshold, sorted descending.

        Returns list of (node_id, score) tuples.
        """
        results: list[tuple[str, float]] = []
        for node_id, attrs in self._g.nodes(data=True):
            data = attrs["data"]
            if hasattr(data, "importance_score") and data.importance_score >= threshold:
                results.append((node_id, data.importance_score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
