"""Tests for the train_replay CLI."""

from __future__ import annotations

import binascii
import json
import pickle
from pathlib import Path
from urllib.request import Request

import pytest
from click.testing import CliRunner
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from train_replay.anomaly.profile import TrainingProfile
from train_replay.cli.main import _profile_to_dict, _send_slack_notification, cli
from train_replay.cli.safemode import SafeMode
from train_replay.collector.flight_recorder import CollectiveEvent


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


# ---------------------------------------------------------------------------
# `train-replay anomaly` subcommand tests
# ---------------------------------------------------------------------------


def _anomaly_entry(rank: int, size: int, started: int, seq: int) -> dict[str, object]:
    """One Flight Recorder entry for an anomaly-scan dump."""
    return {
        "rank": rank,
        "pg_name": "default",
        "collective_seq": "all_reduce",
        "p2p_src": None,
        "p2p_dst": None,
        "input_sizes": [[size]],
        "time_created_ns": started,
        "time_started_ns": started,
        "time_finished_ns": started + 100,
        "frames": [],
        "seq_id": seq,
    }


def _write_anomaly_trace(path: Path, entries: list[dict[str, object]]) -> None:
    with open(path, "wb") as f:
        pickle.dump({"entries": entries}, f)


def _baseline_event(rank: int, size: int, started: int, seq: int) -> CollectiveEvent:
    return CollectiveEvent(
        rank=rank,
        process_group="default",
        collective_type="all_reduce",
        src_rank=None,
        dst_rank=None,
        tensor_size=size,
        enqueue_time_ns=started,
        start_time_ns=started,
        end_time_ns=started + 100,
        sequence_id=seq,
    )


def test_anomaly_command_flags_outlier_tensor_size(tmp_path: Path) -> None:
    """anomaly --profile flags a tensor-size outlier against the baseline."""
    normal_sizes = [4000, 4100, 3900, 4200]
    baseline = TrainingProfile.fit_on_normal_run(
        [_baseline_event(0, s, 1000 + i * 1000, i) for i, s in enumerate(normal_sizes)]
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile_to_dict(baseline)), encoding="utf-8")

    entries = [
        _anomaly_entry(0, s, 1000 + i * 1000, i) for i, s in enumerate(normal_sizes)
    ]
    entries.append(
        _anomaly_entry(0, 1_000_000, 1000 + len(normal_sizes) * 1000, len(normal_sizes))
    )
    trace_path = tmp_path / "trace.pkl"
    _write_anomaly_trace(trace_path, entries)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["anomaly", str(trace_path), "--profile", str(profile_path), "--threshold", "3.0"],
    )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "Baseline profile loaded" in result.output
    assert "Found" in result.output
    assert "1 anomalies" in result.output
    assert "tensor_size_zscore" in result.output


def test_anomaly_command_self_referential_baseline(tmp_path: Path) -> None:
    """anomaly without --profile derives a baseline from the dump itself."""
    entries = [_anomaly_entry(0, 4096, 1000 + i * 1000, i) for i in range(5)]
    trace_path = tmp_path / "trace.pkl"
    _write_anomaly_trace(trace_path, entries)

    runner = CliRunner()
    result = runner.invoke(cli, ["anomaly", str(trace_path)])
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "Derived self-referential baseline" in result.output
    # Equal tensor sizes (std 0) and constant spacing (std 0) -> no anomalies.
    assert "0 anomalies" in result.output


def test_anomaly_command_rejects_invalid_profile_json(tmp_path: Path) -> None:
    """anomaly --profile with malformed JSON exits non-zero."""
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("{not json", encoding="utf-8")
    trace_path = tmp_path / "trace.pkl"
    _write_anomaly_trace(trace_path, [_anomaly_entry(0, 4096, 1000, 0)])

    runner = CliRunner()
    result = runner.invoke(
        cli, ["anomaly", str(trace_path), "--profile", str(profile_path)]
    )
    assert result.exit_code != 0
    assert "not valid JSON" in result.output


def test_anomaly_command_rejects_unsupported_notify_channel(tmp_path: Path) -> None:
    """anomaly --notify with a non-slack channel exits non-zero."""
    trace_path = tmp_path / "trace.pkl"
    _write_anomaly_trace(trace_path, [_anomaly_entry(0, 4096, 1000, 0)])

    runner = CliRunner()
    result = runner.invoke(
        cli, ["anomaly", str(trace_path), "--notify", "email:foo@example.com"]
    )
    assert result.exit_code != 0
    assert "Unsupported --notify target" in result.output


class _FakeSlackResponse:
    def __enter__(self) -> _FakeSlackResponse:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None

    def read(self) -> bytes:
        return b"ok"


def test_send_slack_notification_posts_json_payload() -> None:
    """_send_slack_notification POSTs a Slack webhook JSON body via the opener."""
    seen: dict[str, object] = {}

    def opener(request: Request, _timeout: float) -> _FakeSlackResponse:
        seen["url"] = request.full_url
        data = request.data
        assert data is not None
        seen["body"] = json.loads(data)
        return _FakeSlackResponse()

    _send_slack_notification(
        "https://hooks.slack.example/services/T/B/xxx", "hello", opener=opener
    )
    assert seen["url"] == "https://hooks.slack.example/services/T/B/xxx"
    assert seen["body"] == {"text": "hello"}


def test_anomaly_command_full_signature_notifies_slack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full bullet signature end-to-end: anomaly <dump> --profile --threshold --notify slack:<url>.

    Exercises the ``--notify slack:<webhook_url>`` dispatch path through the
    CLI by stubbing the module-level notifier, so no network is touched and the
    dispatched target + message can be asserted.
    """
    normal_sizes = [4000, 4100, 3900, 4200]
    baseline = TrainingProfile.fit_on_normal_run(
        [_baseline_event(0, s, 1000 + i * 1000, i) for i, s in enumerate(normal_sizes)]
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile_to_dict(baseline)), encoding="utf-8")

    entries = [
        _anomaly_entry(0, s, 1000 + i * 1000, i) for i, s in enumerate(normal_sizes)
    ]
    entries.append(
        _anomaly_entry(0, 1_000_000, 1000 + len(normal_sizes) * 1000, len(normal_sizes))
    )
    trace_path = tmp_path / "trace.pkl"
    _write_anomaly_trace(trace_path, entries)

    dispatched: list[tuple[str, str]] = []

    def fake_notify(webhook_url: str, message: str, **_kwargs: object) -> None:
        dispatched.append((webhook_url, message))

    monkeypatch.setattr(
        "train_replay.cli.main._send_slack_notification", fake_notify
    )

    webhook = "https://hooks.slack.example/services/T/B/xxx"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "anomaly",
            str(trace_path),
            "--profile",
            str(profile_path),
            "--threshold",
            "3.0",
            "--notify",
            f"slack:{webhook}",
        ],
    )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "1 anomalies" in result.output
    assert "Sent Slack alert" in result.output
    # The webhook URL after the ``slack:`` prefix is forwarded verbatim.
    assert dispatched == [(webhook, dispatched[0][1])]
    assert "1 anomalies" in dispatched[0][1]


def test_anomaly_command_skips_notify_when_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--notify slack:...`` is a no-op (no dispatch) when no anomalies are found."""
    entries = [_anomaly_entry(0, 4096, 1000 + i * 1000, i) for i in range(5)]
    trace_path = tmp_path / "trace.pkl"
    _write_anomaly_trace(trace_path, entries)

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("notifier must not run when there are no anomalies")

    monkeypatch.setattr("train_replay.cli.main._send_slack_notification", fail_if_called)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["anomaly", str(trace_path), "--notify", "slack:https://hooks.example/x"],
    )
    assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
    assert "0 anomalies" in result.output
    assert "Slack notification skipped" in result.output
