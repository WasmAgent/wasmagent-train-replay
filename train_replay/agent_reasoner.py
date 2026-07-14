"""LLM-assisted root-cause hypothesis layer for causal evidence graphs.

Provides the ``AgentReasoner`` class that:
1. Takes an ``EpochEvidenceBundle`` and a ``ProvGraph`` (or builds one)
2. Runs causal ancestor traversal to surface earliest anomalous nodes
3. Formats traversal result as a structured prompt
4. Calls an LLM API (OpenAI-compatible, configurable endpoint)
5. Returns a structured ``RootCauseReport`` Pydantic model
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from pydantic import BaseModel, Field

from .graph.builder import build_from_specs
from .graph.ops import Backend, CollectiveOp, OpSpec
from .graph.prov_graph import ProvGraph
from .recording.evidence import AEPRecord, EpochEvidenceBundle
from .recording.modes import RecordingMode


class RootCauseReport(BaseModel):
    """Structured root-cause hypothesis report returned by the reasoner."""

    root_cause_activity_ids: list[str] = Field(
        description="Activity IDs identified as likely root causes",
    )
    root_cause_description: str = Field(
        description="Human-readable description of the hypothesized root cause",
    )
    confidence: str = Field(
        default="medium",
        description="Confidence level: high, medium, low",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence snippets supporting the hypothesis",
    )
    recommended_action: str = Field(
        default="",
        description="Recommended remediation action",
    )


class AgentReasoner:
    """Generate root-cause hypotheses from causal evidence bundles.

    The reasoner uses causal ancestor traversal (PROV-DM graph) and an
    optional LLM call to produce structured ``RootCauseReport`` instances.

    Args:
        graph: A pre-built ``ProvGraph`` or ``None`` to build from bundle.
        llm_endpoint: Base URL for an OpenAI-compatible chat completions API.
            If empty, the reasoner will still produce a report using graph
            analysis alone (no LLM call).
    """

    def __init__(
        self,
        graph: ProvGraph | None = None,
        llm_endpoint: str = "",
    ) -> None:
        self._graph = graph
        self._llm_endpoint = llm_endpoint

    def analyze(
        self,
        bundle: EpochEvidenceBundle,
        entity_id: str | None = None,
        api_key: str | None = None,
    ) -> RootCauseReport:
        """Analyze a bundle and produce a root-cause hypothesis.

        Args:
            bundle: The evidence bundle to analyze.
            entity_id: Specific entity to trace.  If None, uses the last
                anomalous action's output entity.
            api_key: API key for the LLM endpoint.  If not provided, reads
                from ``LLM_API_KEY`` environment variable.

        Returns:
            A structured ``RootCauseReport``.
        """
        # Build graph if not provided
        graph = self._graph
        if graph is None:
            graph = self._build_graph_from_bundle(bundle)

        # Determine target entity
        if entity_id is None:
            entity_id = self._select_target_entity(bundle)

        # Traverse causal ancestors
        ancestors = graph.ancestors_of(entity_id)
        suspicious = self._find_suspicious(bundle)

        # Summarise the graph for prompt context
        summary = self._summarise_graph(graph, ancestors, suspicious)

        # Try LLM call if endpoint configured
        resolved_key = api_key or os.environ.get("LLM_API_KEY")
        if self._llm_endpoint and resolved_key:
            report = self._call_llm(summary, resolved_key)
        else:
            # Fallback: graph-only analysis
            report = self._fallback_report(summary, ancestors, suspicious)

        return report

    def _build_graph_from_bundle(self, bundle: EpochEvidenceBundle) -> ProvGraph:
        """Rebuild a ProvGraph from the actions recorded in a bundle."""
        specs: list[OpSpec] = []
        for i, action in enumerate(bundle.actions):
            try:
                op = CollectiveOp(action.collective_type)
            except ValueError:
                op = CollectiveOp.UNKNOWN
            specs.append(
                OpSpec(
                    op=op,
                    backend=Backend.CUSTOM,
                    rank=action.rank,
                    process_group="default",
                    sequence_id=action.step or i,
                    start_time_ns=action.timestamp_ns,
                    end_time_ns=action.timestamp_ns,
                    collective_type_raw=action.collective_type,
                ),
            )
        return build_from_specs(specs)

    def _select_target_entity(self, bundle: EpochEvidenceBundle) -> str:
        """Select the most anomalous entity to trace.

        Picks the last FULL-mode action's output entity.  If no FULL-mode
        action exists, falls back to the last action overall.
        """
        for action in reversed(bundle.actions):
            if action.recording_mode == RecordingMode.FULL:
                return f"tensor:{action.rank}:{action.step}:out"
        if bundle.actions:
            last = bundle.actions[-1]
            return f"tensor:{last.rank}:{last.step}:out"
        return "tensor:0:0:out"

    @staticmethod
    def _find_suspicious(bundle: EpochEvidenceBundle) -> list[AEPRecord]:
        """Return actions recorded in FULL mode — highest-risk signals."""
        return [a for a in bundle.actions if a.recording_mode == RecordingMode.FULL]

    @staticmethod
    def _summarise_graph(
        graph: ProvGraph,
        ancestors: list[str],
        suspicious: list[AEPRecord],
    ) -> dict[str, Any]:
        """Produce a structured summary of the graph for LLM context."""
        nodes = list(graph.nodes())
        activities: list[dict[str, Any]] = []
        entities: list[dict[str, Any]] = []
        for nid, data in nodes:
            kind = data.get("kind")
            if kind == "activity":
                act_data = data.get("data")
                activities.append(
                    {
                        "id": nid,
                        "label": act_data.label if act_data else "",
                        "rank": act_data.rank if act_data else 0,
                    },
                )
            elif kind == "entity":
                ent_data = data.get("data")
                entities.append(
                    {
                        "id": nid,
                        "rank": ent_data.rank if ent_data else 0,
                        "step": ent_data.step if ent_data else 0,
                    },
                )

        return {
            "total_nodes": len(nodes),
            "activities": activities,
            "entities": entities,
            "causal_ancestors": ancestors,
            "suspicious_actions": [
                {
                    "action_id": a.action_id,
                    "rank": a.rank,
                    "step": a.step,
                    "collective_type": a.collective_type,
                }
                for a in suspicious
            ],
        }

    @staticmethod
    def _build_prompt(summary: dict[str, Any]) -> str:
        """Build a structured prompt for the LLM."""
        return (
            "You are a root-cause analysis expert for distributed GPU"
            " training.\n\n"
            "Given the following causal graph summary, identify the most likely"
            " root cause.\n\n"
            "Graph summary:\n"
            f"- Total nodes: {summary['total_nodes']}\n"
            f"- Activities: {json.dumps(summary['activities'], indent=2)}\n"
            f"- Causal ancestors (traced):"
            f" {json.dumps(summary['causal_ancestors'], indent=2)}\n"
            f"- Suspicious actions (FULL recording mode):"
            f" {json.dumps(summary['suspicious_actions'], indent=2)}\n\n"
            "Respond with a JSON object following this schema:\n"
            '{\n'
            '  "root_cause_activity_ids": ["list of activity IDs"],\n'
            '  "root_cause_description": "string description",\n'
            '  "confidence": "high|medium|low",\n'
            '  "supporting_evidence": ["evidence strings"],\n'
            '  "recommended_action": "string"\n'
            '}\n'
        )

    def _call_llm(
        self,
        summary: dict[str, Any],
        api_key: str,
    ) -> RootCauseReport:
        """Call the OpenAI-compatible LLM endpoint.

        Falls back to graph-only analysis on any API error.
        """
        prompt = self._build_prompt(summary)

        body = json.dumps(
            {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a distributed training debug"
                        " assistant.  Always respond with valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            },
        ).encode()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        endpoint = self._llm_endpoint.rstrip("/") + "/chat/completions"

        req = urllib.request.Request(
            endpoint,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
            content = result["choices"][0]["message"]["content"]
            return RootCauseReport.model_validate_json(content)
        except (
            urllib.error.URLError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
            OSError,
        ):
            # Fallback on API error
            return self._fallback_report(
                summary,
                summary.get("causal_ancestors", []),
                [],
            )

    @staticmethod
    def _fallback_report(
        summary: dict[str, Any],
        ancestors: list[str],
        suspicious: list[AEPRecord],
    ) -> RootCauseReport:
        """Generate a report using graph analysis alone (no LLM)."""
        desc_parts: list[str] = []
        if ancestors:
            parts = (
                f"Anomaly traced to {len(ancestors)} causal ancestor(s)."
            )
            desc_parts.append(parts)
        if suspicious:
            parts = (
                f"{len(suspicious)} suspicious action(s) with FULL"
                " recording mode."
            )
            desc_parts.append(parts)
        if not ancestors and not suspicious:
            desc_parts.append(
                "No causal ancestors or suspicious actions identified.",
            )

        return RootCauseReport(
            root_cause_activity_ids=ancestors[:3] if ancestors else [],
            root_cause_description=" ".join(desc_parts)
            or "No root cause identified.",
            confidence="low" if not ancestors else "medium",
            supporting_evidence=[
                f"Causal ancestor: {a}" for a in ancestors[:5]
            ],
            recommended_action=(
                "Investigate the identified causal ancestors for anomalies."
            ),
        )
