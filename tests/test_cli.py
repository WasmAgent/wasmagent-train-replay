"""Tests for the train_replay CLI."""

from __future__ import annotations

import binascii
import json
import pickle
from pathlib import Path

import pytest
from click.testing import CliRunner
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from train_replay.cli.main import cli
from train_replay.cli.safemode import SafeMode


@pytest.fixture
def safe_mode() -> SafeMode:
    """A fresh SafeMode instance per test.

    Each test injects this into the CLI via ``CliRunner.invoke(...,
    obj={"safe_mode": safe_mode})`` so the click context carries it through the
    command tree. This gives every test full isolation — no module-level global
    is mutated, so there is nothing to reset and no risk of cross-test
    pollution under pytest-xdist or any shared-interpreter runner.
    """
    return SafeMode()


def _write_sample_trace(path: Path) -> None:
    """Write a minimal Flight Recorder pickle file for testing."""
    data = {
        "entries": [
            {
                "rank": 0,
                "pg_name": "default",
                "collective_seq": "all_reduce",
                "p2p_src": None,
                "p2p_dst": None,
                "input_sizes": [[4096]],
                "time_created_ns": 1000,
                "time_started_ns": 1100,
                "time_finished_ns": 1200,
                "frames": [],
                "seq_id": 1,
            },
            {
                "rank": 1,
                "pg_name": "default",
                "collective_seq": "all_reduce",
                "p2p_src": None,
                "p2p_dst": None,
                "input_sizes": [[4096]],
                "time_created_ns": 1000,
                "time_started_ns": 1100,
                "time_finished_ns": 1200,
                "frames": [],
                "seq_id": 1,
            },
            {
                "rank": 0,
                "pg_name": "default",
                "collective_seq": "all_reduce",
                "p2p_src": None,
                "p2p_dst": None,
                "input_sizes": [[8192]],
                "time_created_ns": 2000,
                "time_started_ns": 2100,
                "time_finished_ns": 2200,
                "frames": [],
                "seq_id": 2,
            },
            {
                "rank": 1,
                "pg_name": "default",
                "collective_seq": "all_reduce",
                "p2p_src": None,
                "p2p_dst": None,
                "input_sizes": [[8192]],
                "time_created_ns": 2000,
                "time_started_ns": 2100,
                "time_finished_ns": 2200,
                "frames": [],
                "seq_id": 2,
            },
        ]
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def test_ingest_command(tmp_path: Path) -> None:
    """ingest command prints graph node count."""
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", str(trace_path)])
    assert result.exit_code == 0
    assert "Loaded" in result.output
    assert "Built causal graph" in result.output


def test_trace_command(tmp_path: Path) -> None:
    """trace command prints causal ancestors."""
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["trace", "tensor:0:1:out", str(trace_path)])
    assert result.exit_code == 0
    assert "Causal ancestors" in result.output
    assert "act:0:all_reduce:1" in result.output


def test_record_command(tmp_path: Path) -> None:
    """record command prints action count and digest."""
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["record", str(trace_path)])
    assert result.exit_code == 0
    assert "Recorded" in result.output
    assert "Bundle digest" in result.output


def _hex_key() -> str:
    """A fresh Ed25519 private key encoded as 64-char hex."""
    return binascii.hexlify(Ed25519PrivateKey.generate().private_bytes_raw()).decode()


def test_record_command_signs_bundle_with_hex_key(tmp_path: Path) -> None:
    """record --signing-key-hex signs the bundle via load_private_key_hex.

    The CLI must accept a raw hex key without the caller constructing any
    cryptography object: pass the hex string, get a signed bundle back.
    """
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["record", str(trace_path), "--signing-key-hex", _hex_key()],
    )
    assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
    assert "Signed bundle with key_id" in result.output
    assert "Signature:" in result.output


def test_record_command_rejects_invalid_hex_key(tmp_path: Path) -> None:
    """record --signing-key-hex with non-hex input exits non-zero."""
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["record", str(trace_path), "--signing-key-hex", "not-valid-hex-zz"],
    )
    assert result.exit_code != 0
    assert "hex" in result.output.lower()


def test_agent_query_trace_tensor_command(tmp_path: Path) -> None:
    """agent-query dispatches trace_tensor and prints JSON."""
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "agent-query",
            str(trace_path),
            "--tool", "trace_tensor",
            "--args", '{"entity_id": "tensor:0:1:out"}',
        ],
    )
    assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
    payload = json.loads(result.output)
    assert payload == {
        "tool": "trace_tensor",
        "entity_id": "tensor:0:1:out",
        "causal_ancestors": ["act:0:all_reduce:1"],
    }


def test_agent_query_rejects_invalid_args_json(tmp_path: Path) -> None:
    """agent-query reports invalid JSON in --args."""
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["agent-query", str(trace_path), "--tool", "trace_tensor", "--args", "{"],
    )
    assert result.exit_code != 0
    assert "Invalid JSON for --args" in result.output


def test_agent_query_rejects_unknown_tool(tmp_path: Path) -> None:
    """agent-query reports unsupported tools."""
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["agent-query", str(trace_path), "--tool", "missing"])
    assert result.exit_code != 0
    assert "Unknown agent tool: missing" in result.output


def test_replay_command(tmp_path: Path) -> None:
    """replay command prints replay result with causal ancestors."""
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "replay",
            str(trace_path),
            "tensor:0:2:out",
            "--rank", "0",
            "--run-id", "test-run",
            "--epoch", "1",
        ],
    )
    assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
    assert "Replay Result" in result.output
    assert "Causal ancestors" in result.output
    assert "Suspicious actions" in result.output
    # tensor:0:2:out was generated by act:0:all_reduce:2
    assert "act:0:all_reduce:2" in result.output


def test_replay_command_default_rank(tmp_path: Path) -> None:
    """replay command uses default rank 0."""
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["replay", str(trace_path), "tensor:0:1:out"])
    assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
    assert "Replay Result" in result.output
    assert "rank 0" in result.output


def test_replay_command_missing_entity(tmp_path: Path) -> None:
    """replay command handles entity with no ancestors."""
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    # "tensor:99:99:out" does not exist in the graph
    result = runner.invoke(cli, ["replay", str(trace_path), "tensor:99:99:out"])
    # Should exit cleanly with empty ancestors list
    assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
    assert "Replay Result" in result.output


def test_replay_with_suspicious_actions(tmp_path: Path) -> None:
    """replay command shows suspicious (FULL mode) actions."""
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    # Add a risk override by using --rank; the all_reduce collective triggers MUTATE_EXTERNAL
    # which results in FULL recording mode per compile_recording_policy
    result = runner.invoke(
        cli,
        [
            "replay",
            str(trace_path),
            "tensor:0:1:out",
            "--rank", "0",
        ],
    )
    assert result.exit_code == 0, f"Exit code {result.exit_code}: {result.output}"
    assert "Suspicious actions" in result.output


# ---------------------------------------------------------------------------
# admin safe-mode subcommand tests
#
# Every test injects a fresh ``safe_mode`` fixture into the CLI via
# ``CliRunner.invoke(..., obj={"safe_mode": safe_mode})``. Because the click
# context carries the instance (rather than reaching for a module-level
# singleton), each test starts from a clean state with nothing to reset and no
# possibility of cross-test pollution under any parallel runner.
# ---------------------------------------------------------------------------


def test_admin_safe_mode_status_default_off(safe_mode: SafeMode) -> None:
    """admin safe-mode --status reports OFF by default."""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["admin", "safe-mode", "--status"], obj={"safe_mode": safe_mode}
    )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "OFF" in result.output


def test_admin_safe_mode_on(safe_mode: SafeMode) -> None:
    """admin safe-mode --on activates safe mode."""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["admin", "safe-mode", "--on"], obj={"safe_mode": safe_mode}
    )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "activated" in result.output.lower()
    assert safe_mode.status() is True


def test_admin_safe_mode_off(safe_mode: SafeMode) -> None:
    """admin safe-mode --off deactivates safe mode."""
    safe_mode.trigger()
    runner = CliRunner()
    result = runner.invoke(
        cli, ["admin", "safe-mode", "--off"], obj={"safe_mode": safe_mode}
    )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "deactivated" in result.output.lower()
    assert safe_mode.status() is False


def test_admin_safe_mode_mutually_exclusive(safe_mode: SafeMode) -> None:
    """admin safe-mode --on --off aborts with error."""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["admin", "safe-mode", "--on", "--off"], obj={"safe_mode": safe_mode}
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_admin_safe_mode_no_flags_shows_status(safe_mode: SafeMode) -> None:
    """admin safe-mode with no flags defaults to showing status."""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["admin", "safe-mode"], obj={"safe_mode": safe_mode}
    )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "OFF" in result.output


def test_admin_safe_mode_status_when_active(safe_mode: SafeMode) -> None:
    """admin safe-mode --status reports ON when active."""
    safe_mode.trigger()
    runner = CliRunner()
    result = runner.invoke(
        cli, ["admin", "safe-mode", "--status"], obj={"safe_mode": safe_mode}
    )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "ON" in result.output


# ---------------------------------------------------------------------------
# resume command tests
# ---------------------------------------------------------------------------


def test_resume_clears_safe_mode(safe_mode: SafeMode) -> None:
    """resume command clears safe mode."""
    safe_mode.trigger()
    assert safe_mode.status() is True
    runner = CliRunner()
    result = runner.invoke(cli, ["resume"], obj={"safe_mode": safe_mode})
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "Safe mode cleared" in result.output
    assert safe_mode.status() is False


def test_resume_when_not_in_safe_mode(safe_mode: SafeMode) -> None:
    """resume is a no-op when safe mode is already off."""
    runner = CliRunner()
    result = runner.invoke(cli, ["resume"], obj={"safe_mode": safe_mode})
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "Safe mode cleared" in result.output


# ---------------------------------------------------------------------------
# Safe mode enforcement tests
# ---------------------------------------------------------------------------


def test_record_blocked_by_safe_mode(tmp_path: Path, safe_mode: SafeMode) -> None:
    """record command is blocked when safe mode is active."""
    safe_mode.trigger()
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["record", str(trace_path)], obj={"safe_mode": safe_mode}
    )
    assert result.exit_code != 0, f"Expected non-zero exit: {result.output}"
    assert "blocked by safe mode" in result.output.lower()


def test_record_succeeds_when_safe_mode_off(tmp_path: Path, safe_mode: SafeMode) -> None:
    """record command succeeds when safe mode is inactive."""
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["record", str(trace_path)], obj={"safe_mode": safe_mode}
    )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "Recorded" in result.output


def test_replay_blocked_by_safe_mode(tmp_path: Path, safe_mode: SafeMode) -> None:
    """replay command is blocked when safe mode is active."""
    safe_mode.trigger()
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["replay", str(trace_path), "tensor:0:1:out", "--rank", "0"],
        obj={"safe_mode": safe_mode},
    )
    assert result.exit_code != 0, f"Expected non-zero exit: {result.output}"
    assert "blocked by safe mode" in result.output.lower()


def test_replay_succeeds_when_safe_mode_off(tmp_path: Path, safe_mode: SafeMode) -> None:
    """replay command succeeds when safe mode is inactive."""
    trace_path = tmp_path / "trace.pkl"
    _write_sample_trace(trace_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["replay", str(trace_path), "tensor:0:1:out", "--rank", "0"],
        obj={"safe_mode": safe_mode},
    )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "Replay Result" in result.output
