"""Smoke tests verifying that top-level package imports succeed.

These mirror the ``python-import`` gate in ``.claude-bot/verify.yml`` so that
import breakage is caught by the test suite in addition to the CI gate.
"""

from __future__ import annotations


def test_import_train_replay() -> None:
    """Root package is importable."""
    import train_replay

    assert train_replay.__file__ is not None


def test_import_agent() -> None:
    """train_replay.agent subpackage is importable."""
    import train_replay.agent  # noqa: F401


def test_import_replay() -> None:
    """train_replay.replay subpackage is importable."""
    import train_replay.replay  # noqa: F401


def test_import_anomaly() -> None:
    """train_replay.anomaly subpackage is importable."""
    import train_replay.anomaly  # noqa: F401


def test_import_recording() -> None:
    """train_replay.recording subpackage is importable."""
    import train_replay.recording  # noqa: F401
