"""LLM-assisted root-cause hypothesis layer.

Takes an EpochEvidenceBundle, runs causal ancestor traversal on the
PROV-DM graph, formats the traversal as a structured prompt, calls an
OpenAI-compatible LLM endpoint, and returns a structured RootCauseReport.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen

import pydantic

from .graph.prov_graph import ProvGraph
from .recording.evidence import AEPRecord, EpochEvidenceBundle
from .replay.replayer import EpochReplayer

# ── Structured output model ─────────────────────────────────────────


class RootCauseHypothesis(pydantic.BaseModel):  # type: ignore[misc]
    """One hypothesis returned by the LLM."""

    description: str
    confidence: float  # 0.0–1.0
    affected_ranks: list[int] = pydantic.Field(default_factory=list)
    evidence_activity_ids: list[str] = pydantic.Field(default_factory=list)


class RootCauseReport(pydantic.BaseModel):  # type: ignore[misc]
    """Structured root-cause analysis produced by the LLM."""

    summary: str
    anomaly_type: str
    hypotheses: list[RootCauseHypothesis] = pydantic.Field(default_factory=list)
    raw_llm_response: str | None = None


# ── Causal context assembled from traversal ─────────────────────────


@dataclass
class CausalContext:
    """Intermediate representation: traversal result + suspicious actions."""

    entity_id: str
    ancestor_activity_ids: list[str] = field(default_factory=list)
    suspicious_actions: list[AEPRecord] = field(default_factory=list)
    epoch: int = 0
    run_id: str = ""
    graph_summary: str = ""


# ── Prompt formatting ──────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You are a distributed GPU training debug analyst. Given a causal provenance
trace from an NCCL training run, identify the most likely root cause of the
anomaly and return a JSON object matching this schema:

{
  "summary": "one-paragraph explanation of the anomaly",
  "anomaly_type": "category label (e.g. deadlock, gradient_overflow, staleness, topology_mismatch)",
  "hypotheses": [
    {
      "description": "detailed hypothesis",
      "confidence": 0.85,
      "affected_ranks": [0, 1],
      "evidence_activity_ids": ["act:0:all_reduce:1"]
    }
  ]
}

Return ONLY valid JSON. No markdown fences, no extra text."""


def format_causal_prompt(ctx: CausalContext) -> str:
    """Format a CausalContext into the user message sent to the LLM."""

    ancestor_lines = "\n".join(f"  - {aid}" for aid in ctx.ancestor_activity_ids) or "  (none)"
    suspicious_lines = "\n".join(
        f"  - rank={a.rank} step={a.step} type={a.collective_type} mode={a.recording_mode.value}"
        for a in ctx.suspicious_actions
    ) or "  (none)"

    return (
        f"## Causal Analysis Request\n\n"
        f"Run ID: {ctx.run_id}\n"
        f"Epoch: {ctx.epoch}\n"
        f"Anomalous entity: {ctx.entity_id}\n\n"
        f"### Causal ancestor activities (earliest → latest)\n"
        f"{ancestor_lines}\n\n"
        f"### Suspicious actions (recorded in FULL mode)\n"
        f"{suspicious_lines}\n\n"
        f"### Graph summary\n"
        f"{ctx.graph_summary}\n"
    )


# ── Traversal → CausalContext assembly ──────────────────────────────


def assemble_causal_context(
    bundle: EpochEvidenceBundle,
    graph: ProvGraph,
    entity_id: str,
    rank: int | None = None,
) -> CausalContext:
    """Run causal ancestor traversal and assemble a CausalContext."""
    replayer = EpochReplayer(graph)
    ancestor_ids = replayer.find_root_cause(entity_id)

    suspicious = replayer.suspicious_actions(bundle)
    if rank is not None:
        suspicious = [a for a in suspicious if a.rank == rank]

    # Summarize the graph
    node_count = sum(1 for _ in graph.nodes())
    graph_summary = (
        f"Graph contains {node_count} nodes. "
        f"{len(ancestor_ids)} causal ancestors found."
    )

    return CausalContext(
        entity_id=entity_id,
        ancestor_activity_ids=ancestor_ids,
        suspicious_actions=suspicious,
        epoch=bundle.epoch,
        run_id=bundle.run_id,
        graph_summary=graph_summary,
    )


# ── LLM client ─────────────────────────────────────────────────────


def call_llm(
    prompt: str,
    llm_endpoint: str = "http://localhost:8000/v1/chat/completions",
    model: str = "gpt-4o-mini",
    api_key: str = "",
    timeout: int = 30,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint and return the content string.

    Uses only the stdlib so we don't add an httpx/openai dependency.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    data = json.dumps(payload).encode()
    req = Request(llm_endpoint, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())

    choices: list[dict[str, Any]] = body.get("choices", [])
    if not choices:
        raise ValueError("LLM returned no choices")
    return str(choices[0].get("message", {}).get("content", ""))


# ── Output parsing ─────────────────────────────────────────────────


def parse_root_cause_report(raw: str) -> RootCauseReport:
    """Parse the LLM JSON response into a RootCauseReport.

    Strips markdown fences if present, then validates against the Pydantic model.
    """
    text = raw.strip()
    # Strip markdown code fences if the LLM wrapped the JSON
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fence markers)
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)

    data = json.loads(text)
    return RootCauseReport.model_validate(data)


# ── Top-level orchestrator ──────────────────────────────────────────


def analyze_bundle(
    bundle: EpochEvidenceBundle,
    graph: ProvGraph,
    entity_id: str,
    *,
    rank: int | None = None,
    llm_endpoint: str = "http://localhost:8000/v1/chat/completions",
    model: str = "gpt-4o-mini",
    api_key: str = "",
    timeout: int = 30,
) -> RootCauseReport:
    """End-to-end: traverse → format → call LLM → parse → return report."""
    ctx = assemble_causal_context(bundle, graph, entity_id, rank=rank)
    prompt = format_causal_prompt(ctx)
    raw_response = call_llm(
        prompt,
        llm_endpoint=llm_endpoint,
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
    report = parse_root_cause_report(raw_response)
    report.raw_llm_response = raw_response
    return report
