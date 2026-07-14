"""Tests for the replay CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from train_replay.cli.main import cli
from train_replay.collector.flight_recorder import CollectiveEvent


def _make_events() -> list[CollectiveEvent]:
    """Mock collective events — no pickle, no file I/O."""
    return [
        CollectiveEvent(
            rank=0,
            process_group="default",
            collective_type="all_reduce",
            src_rank=None,
            dst_rank=None,
            tensor_size=4096,
            enqueue_time_ns=1000,
            start_time_ns=1100,
            end_time_ns=1200,
            sequence_id=1,
        ),
        CollectiveEvent(
            rank=1,
            process_group="default",
            collective_type="all_reduce",
            src_rank=None,
            dst_rank=None,
            tensor_size=4096,
            enqueue_time_ns=1000,
            start_time_ns=1100,
            end_time_ns=1200,
            sequence_id=1,
        ),
    ]


def _write_bundle_json(tmp_path: Path) -> Path:
    """Write a minimal evidence bundle as JSON (safe, human-readable format)."""
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps({
        "schema_version": "train-aep/v0.1",
        "run_id": "test-run",
        "epoch": 0,
        "actions": [
            {
                "action_id": "r0:seq1",
                "rank": 0,
                "step": 1,
                "collective_type": "all_reduce",
                "recording_mode": "full",
                "timestamp_ns": 1100,
            },
            {
                "action_id": "r1:seq1",
                "rank": 1,
                "step": 1,
                "collective_type": "all_reduce",
                "recording_mode": "validation",
                "timestamp_ns": 1100,
            },
        ],
    }))
    return bundle_path


def _touch_dump(tmp_path: Path) -> Path:
    """Create an empty file that satisfies click.Path(exists=True)."""
    dump = tmp_path / "dump.pkl"
    dump.touch()
    return dump


# CI checks reference _make_dump — keep as an alias so both names resolve.
_make_dump = _touch_dump


@patch("train_replay.collector.flight_recorder.load_flight_recorder")
class TestReplayCommand:
    def test_replay_entity_id(self, mock_load: MagicMock, tmp_path: Path) -> None:
        mock_load.return_value = _make_events()
        dump = _touch_dump(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "replay", str(dump), "--entity-id", "tensor:0:1:out",
        ])
        assert result.exit_code == 0
        assert "Causal ancestors" in result.output

    def test_replay_with_bundle(self, mock_load: MagicMock, tmp_path: Path) -> None:
        mock_load.return_value = _make_events()
        dump = _touch_dump(tmp_path)
        bundle_path = _write_bundle_json(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "replay", str(dump), "--bundle-path", str(bundle_path),
        ])
        assert result.exit_code == 0
        assert "Suspicious" in result.output

    def test_replay_no_flags_warns(self, mock_load: MagicMock, tmp_path: Path) -> None:
        mock_load.return_value = _make_events()
        dump = _touch_dump(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["replay", str(dump)])
        assert result.exit_code == 0
        assert "Provide" in result.output

    def test_replay_rank_filter(self, mock_load: MagicMock, tmp_path: Path) -> None:
        mock_load.return_value = _make_events()
        dump = _touch_dump(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "replay", str(dump), "--rank", "1",
        ])
        assert result.exit_code == 0
        assert "1 collective events" in result.output

    def test_replay_with_bundle_no_suspicious(self, mock_load: MagicMock, tmp_path: Path) -> None:
        mock_load.return_value = _make_events()
        dump = _touch_dump(tmp_path)
        bundle_path = tmp_path / "bundle.json"
        bundle_path.write_text(json.dumps({
            "run_id": "test-run",
            "epoch": 0,
            "actions": [
                {
                    "action_id": "r0:seq1",
                    "rank": 0,
                    "step": 1,
                    "collective_type": "all_reduce",
                    "recording_mode": "validation",
                    "timestamp_ns": 1100,
                },
            ],
        }))
        runner = CliRunner()
        result = runner.invoke(cli, [
            "replay", str(dump), "--bundle-path", str(bundle_path),
        ])
        assert result.exit_code == 0
        assert "No suspicious actions found" in result.output

    def test_replay_command_exists(self, mock_load: MagicMock, tmp_path: Path) -> None:
        """The replay subcommand must be registered in the CLI group."""
        runner = CliRunner()
        # Invoke with --help to verify the command is recognised.
        result = runner.invoke(cli, ["replay", "--help"])
        assert result.exit_code == 0
        assert "Replay an epoch" in result.output
