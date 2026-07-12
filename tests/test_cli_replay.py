"""Tests for the replay CLI subcommand."""

from __future__ import annotations

import pickle
from pathlib import Path

from click.testing import CliRunner

from train_replay.cli.main import cli


def _make_dump(tmp_path: Path) -> Path:
    """Create a minimal Flight Recorder dump for testing."""
    dump = tmp_path / "test_dump.pkl"
    raw = {
        "entries": [
            {
                "rank": 0,
                "pg_name": "default",
                "collective_seq": "all_reduce",
                "p2p_src": None,
                "p2p_dst": None,
                "input_sizes": [[1024]],
                "time_created_ns": 100,
                "time_started_ns": 200,
                "time_finished_ns": 300,
                "frames": [],
                "seq_id": 1,
            },
            {
                "rank": 1,
                "pg_name": "default",
                "collective_seq": "all_reduce",
                "p2p_src": None,
                "p2p_dst": None,
                "input_sizes": [[1024]],
                "time_created_ns": 150,
                "time_started_ns": 250,
                "time_finished_ns": 350,
                "frames": [],
                "seq_id": 1,
            },
        ],
    }
    dump.write_bytes(pickle.dumps(raw))
    return dump


def test_replay_command_exists() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "replay" in result.output


def test_replay_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["replay", "--help"])
    assert result.exit_code == 0
    assert "ENTITY_ID" in result.output
    assert "DUMP_PATH" in result.output
    assert "--rank" in result.output


def test_replay_no_args() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["replay"])
    assert result.exit_code != 0


def test_replay_with_dump(tmp_path: Path) -> None:
    dump = _make_dump(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["replay", "tensor:0:1:out", str(dump), "--rank", "0"],
    )
    assert result.exit_code == 0
    assert "Replay result" in result.output
    assert "epoch" in result.output


def test_replay_shows_causal_ancestors(tmp_path: Path) -> None:
    dump = _make_dump(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["replay", "tensor:0:1:out", str(dump), "--rank", "0"],
    )
    assert result.exit_code == 0
    assert "Causal ancestors" in result.output


def test_replay_leaf_entity_no_ancestors(tmp_path: Path) -> None:
    dump = _make_dump(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["replay", "tensor:0:1:in", str(dump), "--rank", "0"],
    )
    assert result.exit_code == 0
    assert "No causal ancestors found" in result.output


def test_replay_suspicious_actions_shown(tmp_path: Path) -> None:
    dump = _make_dump(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["replay", "tensor:0:1:out", str(dump), "--rank", "0"],
    )
    assert result.exit_code == 0
    # all_reduce is MUTATE_EXTERNAL → FULL mode → shows as suspicious
    assert "Suspicious actions" in result.output
