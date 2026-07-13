"""Tests for LLM-assisted root-cause hypothesis layer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from train_replay.agent_reasoner import (
    CausalContext,
    RootCauseReport,
    _validate_llm_endpoint,
    analyze_bundle,
    assemble_causal_context,
    call_llm,
    format_causal_prompt,
    parse_root_cause_report,
)
from train_replay.graph.prov_graph import (
    ProvActivity,
    ProvAgent,
    ProvEntity,
    ProvGraph,
)
from train_replay.recording.evidence import AEPRecord, EpochEvidenceBundle
from train_replay.recording.modes import RecordingMode

# ── Fixtures ───────────────────────────────────────────────────────


def _make_graph() -> ProvGraph:
    """Build a small causal graph with two collectives on two ranks."""
    g = ProvGraph()
    g.add_agent(ProvAgent(id="rank:0:pg:default", rank=0, process_group="default"))
    g.add_agent(ProvAgent(id="rank:1:pg:default", rank=1, process_group="default"))

    # Rank 0: all_reduce at seq 1
    g.add_activity(ProvActivity(
        id="act:0:all_reduce:1", label="all_reduce",
        rank=0, process_group="default",
        timestamp_ns=1000, collective_type="all_reduce",
    ))
    g.add_entity(ProvEntity(id="tensor:0:1:in", digest=None, rank=0, step=1))
    g.add_entity(ProvEntity(id="tensor:0:1:out", digest=None, rank=0, step=1))
    g.used("act:0:all_reduce:1", "tensor:0:1:in")
    g.was_generated_by("tensor:0:1:out", "act:0:all_reduce:1")
    g.was_associated_with("act:0:all_reduce:1", "rank:0:pg:default")

    # Rank 1: all_reduce at seq 1
    g.add_activity(ProvActivity(
        id="act:1:all_reduce:1", label="all_reduce",
        rank=1, process_group="default",
        timestamp_ns=1100, collective_type="all_reduce",
    ))
    g.add_entity(ProvEntity(id="tensor:1:1:in", digest=None, rank=1, step=1))
    g.add_entity(ProvEntity(id="tensor:1:1:out", digest=None, rank=1, step=1))
    g.used("act:1:all_reduce:1", "tensor:1:1:in")
    g.was_generated_by("tensor:1:1:out", "act:1:all_reduce:1")
    g.was_associated_with("act:1:all_reduce:1", "rank:1:pg:default")

    return g


def _make_bundle() -> EpochEvidenceBundle:
    return EpochEvidenceBundle(
        run_id="test-run",
        epoch=3,
        actions=[
            AEPRecord(
                action_id="r0:seq1",
                rank=0,
                step=1,
                collective_type="all_reduce",
                recording_mode=RecordingMode.FULL,
                timestamp_ns=1000,
            ),
            AEPRecord(
                action_id="r1:seq1",
                rank=1,
                step=1,
                collective_type="all_reduce",
                recording_mode=RecordingMode.DELTA,
                timestamp_ns=1100,
            ),
        ],
    )


_SAMPLE_LLM_JSON = json.dumps({
    "summary": "Rank 0 all_reduce timed out, causing cascade failure on rank 1.",
    "anomaly_type": "deadlock",
    "hypotheses": [
        {
            "description": "NCCL watchdog timeout on rank 0 due to missing barrier sync.",
            "confidence": 0.9,
            "affected_ranks": [0, 1],
            "evidence_activity_ids": ["act:0:all_reduce:1"],
        }
    ],
})


# ── assemble_causal_context tests ───────────────────────────────────


def test_assemble_context_finds_ancestors():
    graph = _make_graph()
    bundle = _make_bundle()
    ctx = assemble_causal_context(bundle, graph, "tensor:0:1:out")
    assert "act:0:all_reduce:1" in ctx.ancestor_activity_ids
    assert ctx.entity_id == "tensor:0:1:out"
    assert ctx.epoch == 3
    assert ctx.run_id == "test-run"


def test_assemble_context_filters_suspicious_by_rank():
    graph = _make_graph()
    bundle = _make_bundle()
    ctx = assemble_causal_context(bundle, graph, "tensor:0:1:out", rank=0)
    # Only the FULL-mode action on rank 0
    assert len(ctx.suspicious_actions) == 1
    assert ctx.suspicious_actions[0].rank == 0


def test_assemble_context_no_rank_filter():
    graph = _make_graph()
    bundle = _make_bundle()
    ctx = assemble_causal_context(bundle, graph, "tensor:0:1:out")
    # All FULL-mode actions (rank 0 only is FULL)
    assert len(ctx.suspicious_actions) == 1


def test_assemble_context_leaf_entity_no_ancestors():
    graph = _make_graph()
    bundle = _make_bundle()
    ctx = assemble_causal_context(bundle, graph, "tensor:0:1:in")
    assert ctx.ancestor_activity_ids == []


# ── format_causal_prompt tests ───────────────────────────────────────


def test_prompt_contains_entity_id():
    ctx = CausalContext(
        entity_id="tensor:0:1:out",
        ancestor_activity_ids=["act:0:all_reduce:1"],
        epoch=3,
        run_id="test-run",
        graph_summary="5 nodes",
    )
    prompt = format_causal_prompt(ctx)
    assert "tensor:0:1:out" in prompt
    assert "act:0:all_reduce:1" in prompt
    assert "test-run" in prompt


def test_prompt_contains_suspicious_actions():
    action = AEPRecord(
        action_id="r0:seq1", rank=0, step=1,
        collective_type="all_reduce", recording_mode=RecordingMode.FULL,
    )
    ctx = CausalContext(
        entity_id="tensor:0:1:out",
        suspicious_actions=[action],
    )
    prompt = format_causal_prompt(ctx)
    assert "rank=0" in prompt
    assert "full" in prompt


def test_prompt_empty_lists():
    ctx = CausalContext(entity_id="tensor:X", graph_summary="empty")
    prompt = format_causal_prompt(ctx)
    assert "(none)" in prompt


# ── parse_root_cause_report tests ───────────────────────────────────


def test_parse_valid_json():
    report = parse_root_cause_report(_SAMPLE_LLM_JSON)
    assert report.anomaly_type == "deadlock"
    assert len(report.hypotheses) == 1
    assert report.hypotheses[0].confidence == 0.9


def test_parse_strips_markdown_fences():
    fenced = f"```json\n{_SAMPLE_LLM_JSON}\n```"
    report = parse_root_cause_report(fenced)
    assert report.anomaly_type == "deadlock"


def test_parse_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_root_cause_report("not json at all")


def test_parse_missing_fields_raises():
    with pytest.raises(Exception):  # pydantic ValidationError
        parse_root_cause_report('{"summary": "test"}')


# ── call_llm test (mocked) ─────────────────────────────────────────


@patch("train_replay.agent_reasoner.urlopen")
def test_call_llm_returns_content(mock_urlopen: MagicMock):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "hello"}}],
    }).encode()
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    result = call_llm("test prompt", llm_endpoint="http://example.com/v1/chat/completions")
    assert result == "hello"
    # Verify request was made
    mock_urlopen.assert_called_once()
    call_args = mock_urlopen.call_args
    req = call_args[0][0]
    assert req.method == "POST"


@patch("train_replay.agent_reasoner.urlopen")
def test_call_llm_sends_api_key(mock_urlopen: MagicMock):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "ok"}}],
    }).encode()
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    call_llm("prompt", llm_endpoint="http://example.com/v1/chat/completions", api_key="sk-test-123")
    req = mock_urlopen.call_args[0][0]
    assert req.headers["Authorization"] == "Bearer sk-test-123"


@patch("train_replay.agent_reasoner.urlopen")
def test_call_llm_no_choices_raises(mock_urlopen: MagicMock):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"choices": []}).encode()
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    with pytest.raises(ValueError, match="no choices"):
        call_llm("prompt", llm_endpoint="http://example.com/v1/chat/completions")


# ── analyze_bundle end-to-end (mocked LLM) ──────────────────────────


@patch("train_replay.agent_reasoner.call_llm")
def test_analyze_bundle_returns_report(mock_call_llm: MagicMock):
    mock_call_llm.return_value = _SAMPLE_LLM_JSON
    graph = _make_graph()
    bundle = _make_bundle()

    report = analyze_bundle(
        bundle, graph, "tensor:0:1:out",
        llm_endpoint="http://example.com/v1/chat/completions",
    )
    assert isinstance(report, RootCauseReport)
    assert report.anomaly_type == "deadlock"
    assert report.raw_llm_response is not None
    assert len(report.hypotheses) == 1
    mock_call_llm.assert_called_once()


@patch("train_replay.agent_reasoner.call_llm")
def test_analyze_bundle_passes_rank_filter(mock_call_llm: MagicMock):
    mock_call_llm.return_value = _SAMPLE_LLM_JSON
    graph = _make_graph()
    bundle = _make_bundle()

    analyze_bundle(
        bundle, graph, "tensor:0:1:out",
        llm_endpoint="http://example.com/v1/chat/completions",
        rank=1,
    )
    prompt_arg = mock_call_llm.call_args[0][0]
    # The prompt should mention suspicious actions from rank 1
    # (there are no FULL-mode actions on rank 1, so prompt shows "(none)")
    assert "Causal Analysis Request" in prompt_arg


# ── _validate_llm_endpoint tests ──────────────────────────────────


def test_validate_endpoint_accepts_http():
    _validate_llm_endpoint("http://localhost:8000/v1/chat/completions")


def test_validate_endpoint_accepts_https():
    _validate_llm_endpoint("https://api.openai.com/v1/chat/completions")


@pytest.mark.parametrize("bad_url", [
    "ftp://localhost:8000/v1/chat/completions",
    "file:///etc/passwd",
    "data:text/plain,hello",
    "javascript:alert(1)",
    "",
])
def test_validate_endpoint_rejects_bad_schemes(bad_url: str):
    with pytest.raises(ValueError, match="http or https scheme"):
        _validate_llm_endpoint(bad_url)


def test_validate_endpoint_rejects_no_hostname():
    with pytest.raises(ValueError, match="hostname"):
        _validate_llm_endpoint("http:///v1/chat/completions")


# ── call_llm validation tests ─────────────────────────────────────


def test_call_llm_rejects_bad_scheme():
    with pytest.raises(ValueError, match="http or https scheme"):
        call_llm("prompt", llm_endpoint="ftp://localhost/call")


def test_call_llm_rejects_empty_model():
    with pytest.raises(ValueError, match="non-empty string"):
        call_llm("prompt", llm_endpoint="http://example.com/v1/chat/completions", model="")


@patch("train_replay.agent_reasoner.urlopen")
def test_call_llm_passes_timeout(mock_urlopen: MagicMock):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": "ok"}}],
    }).encode()
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    call_llm("prompt", llm_endpoint="http://example.com/v1/chat/completions")
    call_args = mock_urlopen.call_args
    # urlopen(req, timeout=...) passes timeout as a keyword argument
    assert call_args[1]["timeout"] == 120


# ── Security hardening (PR review findings) ────────────────────────


def test_call_llm_requires_explicit_endpoint() -> None:
    """Finding 3: no insecure localhost default — endpoint must be supplied."""
    with pytest.raises(TypeError, match="llm_endpoint"):
        call_llm("prompt")  # type: ignore[call-arg]


def test_analyze_bundle_requires_explicit_endpoint() -> None:
    """Finding 3: analyze_bundle has no localhost default either."""
    graph = _make_graph()
    bundle = _make_bundle()
    with pytest.raises(TypeError, match="llm_endpoint"):
        analyze_bundle(bundle, graph, "tensor:0:1:out")  # type: ignore[call-arg]


def _invoke_cli(args: list[str]) -> object:
    """Invoke the CLI group in a runner, returning the click Result."""
    from click.testing import CliRunner

    from train_replay.cli.main import cli

    runner = CliRunner()
    return runner.invoke(cli, args)


def test_cli_analyze_requires_llm_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding 3: --llm-endpoint is a required CLI option (no localhost default)."""
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")
    dump = tmp_path / "dump.pkl"
    dump.write_text("")

    result = _invoke_cli([
        "analyze", str(bundle),
        "--entity-id", "tensor:0:1:out",
        "--dump-path", str(dump),
    ])
    assert getattr(result, "exit_code") != 0
    assert "--llm-endpoint" in getattr(result, "output")


def test_cli_analyze_rejects_api_key_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding 1: the --api-key flag has been removed (shell-history leak vector)."""
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")
    dump = tmp_path / "dump.pkl"
    dump.write_text("")

    result = _invoke_cli([
        "analyze", str(bundle),
        "--entity-id", "tensor:0:1:out",
        "--dump-path", str(dump),
        "--llm-endpoint", "https://api.openai.com/v1/chat/completions",
        "--api-key", "should-not-be-accepted",
    ])
    assert getattr(result, "exit_code") != 0
    assert "no such option" in getattr(result, "output").lower()


def test_cli_analyze_requires_llm_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding 2: missing LLM_API_KEY fails fast before any network request."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")
    dump = tmp_path / "dump.pkl"
    dump.write_text("")

    result = _invoke_cli([
        "analyze", str(bundle),
        "--entity-id", "tensor:0:1:out",
        "--dump-path", str(dump),
        "--llm-endpoint", "https://api.openai.com/v1/chat/completions",
    ])
    assert getattr(result, "exit_code") != 0
    assert "LLM_API_KEY" in getattr(result, "output")


def test_cli_analyze_rejects_whitespace_only_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding 2: a whitespace-only LLM_API_KEY is treated as missing/empty."""
    monkeypatch.setenv("LLM_API_KEY", "   ")
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}")
    dump = tmp_path / "dump.pkl"
    dump.write_text("")

    result = _invoke_cli([
        "analyze", str(bundle),
        "--entity-id", "tensor:0:1:out",
        "--dump-path", str(dump),
        "--llm-endpoint", "https://api.openai.com/v1/chat/completions",
    ])
    assert getattr(result, "exit_code") != 0
    assert "LLM_API_KEY" in getattr(result, "output")
