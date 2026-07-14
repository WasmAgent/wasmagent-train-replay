"""Tests for LLM-assisted root-cause hypothesis layer (agent_reasoner.py)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from train_replay.agent_reasoner import AgentReasoner, RootCauseReport
from train_replay.graph.builder import build_from_specs
from train_replay.graph.ops import Backend, CollectiveOp, OpSpec
from train_replay.graph.prov_graph import ProvGraph
from train_replay.recording.evidence import AEPRecord, EpochEvidenceBundle
from train_replay.recording.modes import RecordingMode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_bundle(
    actions: list[tuple[str, int, int, str, RecordingMode]] | None = None,
) -> EpochEvidenceBundle:
    """Build an EpochEvidenceBundle from compact action tuples.

    Each tuple: (action_id, rank, step, collective_type, recording_mode)
    """
    if actions is None:
        actions = [
            ("r0:seq1", 0, 1, "all_reduce", RecordingMode.FULL),
            ("r1:seq1", 1, 1, "all_reduce", RecordingMode.VALIDATION),
        ]
    return EpochEvidenceBundle(
        run_id="test-run",
        epoch=0,
        actions=[
            AEPRecord(
                action_id=aid,
                rank=rank,
                step=step,
                collective_type=ctype,
                recording_mode=mode,
                timestamp_ns=1000,
            )
            for aid, rank, step, ctype, mode in actions
        ],
    )


def _make_graph() -> ProvGraph:
    """Build a simple two-rank graph for testing."""
    specs = [
        OpSpec(
            op=CollectiveOp.ALL_REDUCE,
            backend=Backend.NCCL,
            rank=0,
            process_group="default",
            sequence_id=1,
            start_time_ns=1000,
            end_time_ns=1200,
        ),
        OpSpec(
            op=CollectiveOp.ALL_REDUCE,
            backend=Backend.NCCL,
            rank=1,
            process_group="default",
            sequence_id=1,
            start_time_ns=1000,
            end_time_ns=1200,
        ),
    ]
    return build_from_specs(specs)


# ---------------------------------------------------------------------------
# RootCauseReport model
# ---------------------------------------------------------------------------

class TestRootCauseReport:
    def test_defaults(self) -> None:
        report = RootCauseReport(
            root_cause_activity_ids=["act:0:all_reduce:1"],
            root_cause_description="NCCL timeout on rank 0",
        )
        assert report.confidence == "medium"
        assert report.supporting_evidence == []
        assert report.recommended_action == ""

    def test_json_roundtrip(self) -> None:
        report = RootCauseReport(
            root_cause_activity_ids=["act:0:all_reduce:1"],
            root_cause_description="desc",
            confidence="high",
            supporting_evidence=["evidence1"],
            recommended_action="fix it",
        )
        data = report.model_dump()
        restored = RootCauseReport.model_validate(data)
        assert restored == report

    def test_model_validate_json(self) -> None:
        raw = json.dumps({
            "root_cause_activity_ids": ["act:0:all_reduce:1"],
            "root_cause_description": "NCCL timeout on rank 0",
            "confidence": "high",
            "supporting_evidence": ["Slow interconnect detected"],
            "recommended_action": "Check NVLink bandwidth",
        })
        report = RootCauseReport.model_validate_json(raw)
        assert report.root_cause_activity_ids == ["act:0:all_reduce:1"]
        assert report.confidence == "high"


# ---------------------------------------------------------------------------
# AgentReasoner — unit tests
# ---------------------------------------------------------------------------

class TestAgentReasoner:
    def test_analyze_with_prebuilt_graph(self) -> None:
        """Analyze with a pre-built graph and bundle."""
        graph = _make_graph()
        bundle = _make_bundle()
        reasoner = AgentReasoner(graph=graph)
        report = reasoner.analyze(bundle=bundle, entity_id="tensor:0:1:out")
        assert isinstance(report, RootCauseReport)
        assert "act:0:all_reduce:1" in report.root_cause_activity_ids
        assert report.confidence in ("high", "medium", "low")

    def test_analyze_builds_graph_from_bundle(self) -> None:
        """When no graph is provided, the reasoner builds one from the bundle."""
        bundle = _make_bundle()
        reasoner = AgentReasoner()
        report = reasoner.analyze(bundle=bundle)
        assert isinstance(report, RootCauseReport)
        # The bundle has FULL mode on rank 0, so entity_id should target that
        assert report.root_cause_description != ""

    def test_analyze_selects_last_full_mode_entity(self) -> None:
        """_select_target_entity should pick the last FULL-mode action."""
        bundle = _make_bundle([
            ("r0:seq1", 0, 1, "all_reduce", RecordingMode.VALIDATION),
            ("r1:seq1", 1, 1, "all_reduce", RecordingMode.FULL),
        ])
        reasoner = AgentReasoner()
        entity = reasoner._select_target_entity(bundle)
        assert entity == "tensor:1:1:out"

    def test_analyze_selects_last_action_when_no_full(self) -> None:
        """When no FULL-mode actions exist, fall back to the last action."""
        bundle = _make_bundle([
            ("r0:seq1", 0, 1, "all_reduce", RecordingMode.VALIDATION),
            ("r1:seq1", 1, 2, "all_reduce", RecordingMode.VALIDATION),
        ])
        reasoner = AgentReasoner()
        entity = reasoner._select_target_entity(bundle)
        assert entity == "tensor:1:2:out"

    def test_analyze_fallback_empty_bundle(self) -> None:
        """An empty bundle should not crash; returns a low-confidence report."""
        bundle = _make_bundle([])
        reasoner = AgentReasoner()
        report = reasoner.analyze(bundle=bundle)
        assert isinstance(report, RootCauseReport)
        assert report.confidence == "low"

    def test_summarise_graph_structure(self) -> None:
        """_summarise_graph should return expected keys and types."""
        graph = _make_graph()
        bundle = _make_bundle()
        ancestors = graph.ancestors_of("tensor:0:1:out")
        suspicious = AgentReasoner._find_suspicious(bundle)
        summary = AgentReasoner._summarise_graph(graph, ancestors, suspicious)

        assert "total_nodes" in summary
        assert "activities" in summary
        assert "entities" in summary
        assert "causal_ancestors" in summary
        assert "suspicious_actions" in summary
        assert isinstance(summary["total_nodes"], int)
        assert summary["total_nodes"] > 0
        # activities should contain the generating activity
        assert any(
            a["id"] == "act:0:all_reduce:1"
            for a in summary["activities"]
        )

    def test_build_prompt_contains_summary_data(self) -> None:
        """_build_prompt should incorporate summary fields."""
        summary = {
            "total_nodes": 5,
            "activities": [{"id": "act:0:all_reduce:1", "label": "all_reduce", "rank": 0}],
            "entities": [],
            "causal_ancestors": ["act:0:all_reduce:1"],
            "suspicious_actions": [],
        }
        prompt = AgentReasoner._build_prompt(summary)
        assert "act:0:all_reduce:1" in prompt
        assert "total_nodes" in prompt
        assert "root_cause_activity_ids" in prompt

    def test_fallback_report_with_ancestors(self) -> None:
        """_fallback_report should include ancestors in the report."""
        summary = {"causal_ancestors": ["act:0:all_reduce:1", "act:0:all_reduce:2"]}
        report = AgentReasoner._fallback_report(summary, ["act:0:all_reduce:1"], [])
        assert "act:0:all_reduce:1" in report.root_cause_activity_ids
        assert "2 causal ancestor(s)" in report.root_cause_description
        assert report.confidence == "medium"

    def test_fallback_report_without_ancestors(self) -> None:
        """_fallback_report with no ancestors should return low confidence."""
        report = AgentReasoner._fallback_report(
            {"causal_ancestors": []}, [], []
        )
        assert report.root_cause_activity_ids == []
        assert report.confidence == "low"
        assert "No causal ancestors" in report.root_cause_description

    def test_find_suspicious_returns_full_mode_only(self) -> None:
        """_find_suspicious should only return FULL-mode actions."""
        bundle = _make_bundle([
            ("r0:seq1", 0, 1, "all_reduce", RecordingMode.FULL),
            ("r1:seq1", 1, 1, "all_reduce", RecordingMode.DELTA),
            ("r2:seq1", 2, 1, "all_reduce", RecordingMode.VALIDATION),
        ])
        suspicious = AgentReasoner._find_suspicious(bundle)
        assert len(suspicious) == 1
        assert suspicious[0].action_id == "r0:seq1"

    def test_analyze_no_llm_falls_back(self) -> None:
        """When no LLM endpoint is configured, analyze uses fallback."""
        bundle = _make_bundle()
        reasoner = AgentReasoner()
        report = reasoner.analyze(bundle=bundle)
        assert isinstance(report, RootCauseReport)
        # Should have a description from fallback logic
        assert len(report.root_cause_description) > 0

    def test_analyze_calls_llm_when_configured(self) -> None:
        """When LLM endpoint and key are provided, _call_llm is invoked."""
        bundle = _make_bundle()
        graph = _make_graph()

        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "root_cause_activity_ids": ["act:0:all_reduce:1"],
                            "root_cause_description": "NCCL timeout",
                            "confidence": "high",
                            "supporting_evidence": ["Slow rank 0"],
                            "recommended_action": "Check NVLink",
                        }),
                    },
                },
            ],
        }

        with patch(
            "train_replay.agent_reasoner.urllib.request.urlopen",
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_response).encode()
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            reasoner = AgentReasoner(
                graph=graph,
                llm_endpoint="http://fake-llm.local:8000/v1",
            )
            report = reasoner.analyze(
                bundle=bundle,
                entity_id="tensor:0:1:out",
                api_key="test-key-123",
            )

        assert isinstance(report, RootCauseReport)
        assert report.root_cause_activity_ids == ["act:0:all_reduce:1"]
        assert report.root_cause_description == "NCCL timeout"
        assert report.confidence == "high"
        assert report.recommended_action == "Check NVLink"

    def test_analyze_llm_fallback_on_api_error(self) -> None:
        """When the LLM call fails, the reasoner falls back gracefully."""
        bundle = _make_bundle()
        graph = _make_graph()

        with patch(
            "train_replay.agent_reasoner.urllib.request.urlopen",
            side_effect=ConnectionError("API unreachable"),
        ):
            reasoner = AgentReasoner(
                graph=graph,
                llm_endpoint="http://fake-llm.local:8000/v1",
            )
            report = reasoner.analyze(
                bundle=bundle,
                entity_id="tensor:0:1:out",
                api_key="test-key-123",
            )

        assert isinstance(report, RootCauseReport)
        # Should still have content (fallback)
        assert len(report.root_cause_description) > 0

    def test_build_graph_from_bundle_reconstructs_structure(self) -> None:
        """_build_graph_from_bundle should produce a traversable graph."""
        bundle = _make_bundle([
            ("r0:seq1", 0, 1, "all_reduce", RecordingMode.FULL),
        ])
        reasoner = AgentReasoner()
        graph = reasoner._build_graph_from_bundle(bundle)
        node_ids = [n for n, _ in graph.nodes()]
        # Should have at least one activity and entity
        assert any("act:" in n for n in node_ids)
        assert any("tensor:" in n for n in node_ids)
        # Should be traversable
        ancestors = graph.ancestors_of("tensor:0:1:out")
        assert len(ancestors) > 0

    def test_build_prompt_is_valid_json(self) -> None:
        """The prompt should contain valid JSON schema for the response."""
        summary = {
            "total_nodes": 2,
            "activities": [],
            "entities": [],
            "causal_ancestors": [],
            "suspicious_actions": [],
        }
        prompt = AgentReasoner._build_prompt(summary)
        # The prompt should reference the expected JSON keys
        assert "root_cause_activity_ids" in prompt
        assert "root_cause_description" in prompt

    def test_report_serializes_to_json(self) -> None:
        """RootCauseReport should serialize to JSON cleanly."""
        report = RootCauseReport(
            root_cause_activity_ids=["act:0:all_reduce:1"],
            root_cause_description="desc",
        )
        raw = report.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["root_cause_activity_ids"] == ["act:0:all_reduce:1"]
        assert parsed["root_cause_description"] == "desc"
        assert parsed["confidence"] == "medium"
