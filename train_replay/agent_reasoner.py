"""LLM-assisted root-cause hypothesis layer for distributed training evidence.

Takes an EpochEvidenceBundle, runs causal ancestor traversal via ProvGraph
to surface earliest anomalous nodes, formats the traversal result as a
structured prompt, calls an LLM API, and returns a structured RootCauseReport.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from train_replay.graph.prov_graph import ProvGraph
from train_replay.recording.evidence import AEPRecord, EpochEvidenceBundle
from train_replay.replay.replayer import EpochReplayer


# ── Data models ─────────────────────────────────────────────────────
@dataclass
class CausalContext:
    """Aggregated causal context for one anomalous tensor entity.

    Attributes:
        entity_id: The anomalous tensor entity ID to analyze.
        ancestor_activity_ids: Activity IDs that causally contributed to
            the entity, ordered from earliest to latest.
        epoch: Training epoch number from the evidence bundle.
        run_id: Run identifier from the evidence bundle.
        graph_summary: Human-readable summary of the causal graph
            (e.g. number of nodes, ranks, collectives).
        suspicious_actions: Actions recorded in FULL mode (highest risk).
    """

    entity_id: str = ""
    ancestor_activity_ids: list[str] = field(default_factory=list)
    epoch: int = 0
    run_id: str = ""
    graph_summary: str = ""
    suspicious_actions: list[AEPRecord] = field(default_factory=list)


class RootCauseHypothesis(BaseModel):  # type: ignore[misc]
    """One hypothesized root cause with confidence and supporting evidence."""

    description: str = Field(
        ..., description="Natural-language description of the hypothesized root cause."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1.")
    affected_ranks: list[int] | None = Field(None, description="Ranks affected by this root cause.")
    evidence_activity_ids: list[str] | None = Field(
        None, description="Activity IDs that support this hypothesis."
    )


class RootCauseReport(BaseModel):  # type: ignore[misc]
    """Structured report of root-cause analysis."""

    summary: str = Field(..., description="High-level summary of the analysis.")
    anomaly_type: str = Field(
        ..., description="Type of anomaly detected (e.g. deadlock, timeout, desync)."
    )
    hypotheses: list[RootCauseHypothesis] = Field(
        ..., description="Ordered list of root-cause hypotheses, highest confidence first."
    )
    raw_llm_response: str | None = Field(None, description="Raw LLM response text, if available.")


# ── Causal traversal ────────────────────────────────────────────────


def _summarise_graph(graph: ProvGraph) -> str:
    """Return a human-readable summary of a ProvGraph."""
    activities = 0
    entities = 0
    agents = 0
    ranks: set[int] = set()
    for _nid, data in graph.nodes():
        kind = data.get("kind")
        if kind == "activity":
            activities += 1
            d = data.get("data")
            if d is not None:
                ranks.add(d.rank)
        elif kind == "entity":
            entities += 1
        elif kind == "agent":
            agents += 1
            d = data.get("data")
            if d is not None:
                ranks.add(d.rank)
    return (
        f"{activities} activities, {entities} entities, "
        f"{agents} agents across {len(ranks)} rank(s)"
    )


def assemble_causal_context(
    bundle: EpochEvidenceBundle,
    graph: ProvGraph,
    entity_id: str,
    rank: int | None = None,
) -> CausalContext:
    """Gather causal context for an anomalous tensor entity.

    Uses the graph to find causal ancestors and the bundle to surface
    suspicious (FULL-mode) actions, optionally filtered by rank.

    Args:
        bundle: Signed evidence bundle for the epoch.
        graph: Causal PROV-DM graph.
        entity_id: The anomalous tensor entity ID to trace.
        rank: If set, only include suspicious actions for this rank.

    Returns:
        A CausalContext populated with ancestry and suspicious actions.
    """
    replayer = EpochReplayer(graph)
    ancestor_ids = replayer.find_root_cause(entity_id)

    suspicious = replayer.suspicious_actions(bundle)
    if rank is not None:
        suspicious = [a for a in suspicious if a.rank == rank]

    return CausalContext(
        entity_id=entity_id,
        ancestor_activity_ids=ancestor_ids,
        epoch=bundle.epoch,
        run_id=bundle.run_id,
        graph_summary=_summarise_graph(graph),
        suspicious_actions=suspicious,
    )


# ── Prompt assembly ─────────────────────────────────────────────────


def format_causal_prompt(ctx: CausalContext) -> str:
    """Format a structured prompt for the LLM from the causal context.

    The prompt asks the LLM to analyse the causal chain and return a
    JSON object conforming to the RootCauseReport schema.
    """
    ancestors_str = ", ".join(ctx.ancestor_activity_ids) if ctx.ancestor_activity_ids else "(none)"

    suspicious_lines: list[str] = []
    for a in ctx.suspicious_actions:
        suspicious_lines.append(
            f"  - action_id={a.action_id} rank={a.rank} step={a.step} "
            f"collective_type={a.collective_type} recording_mode={a.recording_mode.value}"
        )
    suspicious_str = "\n".join(suspicious_lines) if suspicious_lines else "  (none)"

    prompt = f"""You are a distributed-training root-cause analysis assistant.

Causal Analysis Request
=======================
Entity ID:          {ctx.entity_id}
Ancestor Activities: {ancestors_str}
Epoch:              {ctx.epoch}
Run ID:             {ctx.run_id}
Graph Summary:      {ctx.graph_summary}

Suspicious Actions (FULL recording mode):
{suspicious_str}

Instructions
------------
Analyse the causal chain above. Identify likely root causes for the
anomalous entity. Return ONLY valid JSON with the following schema:

{{
  "summary": "High-level summary of the analysis.",
  "anomaly_type": "Type of anomaly (e.g. deadlock, timeout, desync).",
  "hypotheses": [
    {{
      "description": "Description of the hypothesized root cause.",
      "confidence": 0.0-1.0,
      "affected_ranks": [list of rank ints],
      "evidence_activity_ids": ["activity ID strings"]
    }}
  ]
}}

Do NOT include markdown fences, explanations, or extra text outside the JSON.
"""
    return prompt


# ── LLM caller ──────────────────────────────────────────────────────


def call_llm(
    prompt: str,
    llm_endpoint: str = "http://localhost:8000/v1/chat/completions",
    model: str = "gpt-4o-mini",
    api_key: str = "",
) -> str:
    """Call an OpenAI-compatible chat completions endpoint.

    Args:
        prompt: The formatted prompt string.
        llm_endpoint: Full URL of the OpenAI-compatible endpoint.
        model: Model identifier to use.
        api_key: Bearer token for authentication (empty = no auth header).

    Returns:
        The content string from the first choice in the response.

    Raises:
        ValueError: If the response contains no choices.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = Request(llm_endpoint, data=body, headers=headers, method="POST")
    with urlopen(req) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    choices = raw.get("choices", [])
    if not choices:
        raise ValueError("LLM response contains no choices")

    return str(choices[0]["message"]["content"])


# ── Response parsing ────────────────────────────────────────────────


def parse_root_cause_report(response_text: str) -> RootCauseReport:
    """Parse an LLM response into a RootCauseReport.

    Handles optional markdown code fences (```json ... ```) around the
    JSON payload.

    Args:
        response_text: Raw LLM response string.

    Returns:
        A validated RootCauseReport instance.

    Raises:
        json.JSONDecodeError: If the response does not contain valid JSON.
        pydantic.ValidationError: If the parsed JSON does not match the schema.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    fence_pattern = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)
    match = fence_pattern.match(text)
    if match:
        text = match.group(1).strip()

    data = json.loads(text)
    return RootCauseReport(**data)


# ── Top-level analysis entry point ──────────────────────────────────


def analyze_bundle(
    bundle: EpochEvidenceBundle,
    graph: ProvGraph,
    entity_id: str,
    rank: int | None = None,
    llm_endpoint: str = "http://localhost:8000/v1/chat/completions",
    model: str = "gpt-4o-mini",
    api_key: str = "",
) -> RootCauseReport:
    """Run the full root-cause analysis pipeline.

    1. Assemble causal context from the bundle and graph.
    2. Format a structured prompt for the LLM.
    3. Call the LLM and parse the response.
    4. Return a validated RootCauseReport.

    Args:
        bundle: Signed evidence bundle for the epoch.
        graph: Causal PROV-DM graph.
        entity_id: The anomalous tensor entity ID to trace.
        rank: Optional rank filter for suspicious actions.
        llm_endpoint: OpenAI-compatible LLM endpoint URL.
        model: LLM model name.
        api_key: API key for the LLM endpoint.

    Returns:
        A RootCauseReport with hypotheses and raw LLM response.
    """
    ctx = assemble_causal_context(bundle, graph, entity_id, rank=rank)
    prompt = format_causal_prompt(ctx)
    raw_response = call_llm(prompt, llm_endpoint=llm_endpoint, model=model, api_key=api_key)
    report = parse_root_cause_report(raw_response)
    report.raw_llm_response = raw_response
    return report
