"""Tests for the replay CLI subcommand.

Pickle usage justification
--------------------------
PyTorch's Flight Recorder — ``torch._C._distributed_c10d._dump_nccl_trace()`` —
persists its dumps as a *pickle* of a plain ``{"entries": [...]}`` dict, and
``load_flight_recorder`` therefore reads them back with ``pickle.load`` (see
``train_replay/collector/flight_recorder.py``).  These tests must mirror that
exact on-disk format, so ``_make_dump`` writes a pickle too.

The bytes written by ``_make_dump`` are produced from a *plain dict literal*
containing only inert Python primitives: ints, strings, ``None`` and nested
lists/dicts.  There are no custom classes, ``__reduce__``/``__setstate__`` hooks,
callables or any other executable payload, so the round-trip through ``pickle``
is inert: the test only ever deserializes data it constructed itself from a
hard-coded literal.  In production the dump is emitted by the trusted local
training process (not an external attacker), which is why ``load_flight_recorder``
is documented as the consumer of PyTorch-produced dumps.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from click.testing import CliRunner

from train_replay.cli.main import cli

# Fixture constants. Entity/activity IDs queried through the CLI are *derived*
# from these (see ``build_from_events``) so the assertions track the dump rather
# than relying on magic strings that could drift out of sync.
_RANK = 0
_SEQ = 1
_EPOCH = 0  # matches the replay CLI's ``--epoch`` default
# Input tensors carry only a ``used`` (consumption) edge; output tensors carry a
# ``wasGeneratedBy`` edge — see ``build_from_events`` and ``ancestors_of``.
_IN_ENTITY = f"tensor:{_RANK}:{_SEQ}:in"
_OUT_ENTITY = f"tensor:{_RANK}:{_SEQ}:out"
_EXPECTED_ACTIVITY = f"act:{_RANK}:all_reduce:{_SEQ}"


def _make_dump(tmp_path: Path) -> Path:
    """Write a minimal Flight Recorder dump in PyTorch's native pickle format.

    ``raw`` is a plain dict literal built only from ints, strings, ``None`` and
    lists/dicts.  ``pickle.dumps`` serializes that trusted literal — no classes
    or callables are involved, so the round-trip cannot turn into code execution.
    The structure matches what ``load_flight_recorder`` parses via ``pickle.load``
    followed by ``.get(...)`` on string keys.
    """
    dump = tmp_path / "test_dump.pkl"
    raw = {
        "entries": [
            {
                "rank": _RANK,
                "pg_name": "default",
                "collective_seq": "all_reduce",
                "p2p_src": None,
                "p2p_dst": None,
                "input_sizes": [[1024]],
                "time_created_ns": 100,
                "time_started_ns": 200,
                "time_finished_ns": 300,
                "frames": [],
                "seq_id": _SEQ,
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
                "seq_id": _SEQ,
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
        ["replay", _OUT_ENTITY, str(dump), "--rank", str(_RANK)],
    )
    assert result.exit_code == 0
    assert "Replay result" in result.output
    # The banner reports the bundle's ``epoch``/``rank`` fields, not a derived
    # timestamp; assert on the rendered values so the test pins real behaviour.
    assert f"epoch {_EPOCH}" in result.output
    assert f"rank {_RANK}" in result.output


def test_replay_shows_causal_ancestors(tmp_path: Path) -> None:
    """Output tensors have a ``wasGeneratedBy`` edge, so their ancestors render."""
    dump = _make_dump(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["replay", _OUT_ENTITY, str(dump), "--rank", str(_RANK)],
    )
    assert result.exit_code == 0
    assert "Causal ancestors" in result.output
    # The generating activity ID is derived from the fixture constants above.
    assert _EXPECTED_ACTIVITY in result.output


def test_replay_leaf_entity_no_ancestors(tmp_path: Path) -> None:
    """Input tensors report no causal ancestors by graph contract, not coincidence.

    ``ancestors_of`` skips ``used`` edges (see ``prov_graph.py``) and input
    entities never receive a ``wasGeneratedBy`` edge in ``build_from_events``.
    Therefore ``_IN_ENTITY`` deterministically maps to an empty ancestor set, and
    the CLI prints the "No causal ancestors found" branch.
    """
    dump = _make_dump(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["replay", _IN_ENTITY, str(dump), "--rank", str(_RANK)],
    )
    assert result.exit_code == 0
    assert "No causal ancestors found" in result.output


def test_replay_suspicious_actions_shown(tmp_path: Path) -> None:
    """``all_reduce`` is MUTATE_EXTERNAL → FULL mode → ranked under suspicious.

    ``replay_rank`` filters suspicious actions to the requested rank, so the
    rank-0 action ``r0:seq1`` is present while the rank-1 action is excluded.
    """
    dump = _make_dump(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["replay", _OUT_ENTITY, str(dump), "--rank", str(_RANK)],
    )
    assert result.exit_code == 0
    assert "Suspicious actions" in result.output
    # Action ID format is ``r{rank}:seq{seq}`` (see ``EpochRecorder``).
    assert f"r{_RANK}:seq{_SEQ}" in result.output
    # The rank filter excludes the rank-1 entry present in the fixture.
    assert "r1:seq1" not in result.output
