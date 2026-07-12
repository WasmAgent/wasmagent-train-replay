"""Tests for SafeMode machinery."""

import threading

import pytest

from train_replay.cli.safemode import SafeMode, SafeModeError


def test_initial_state_is_off() -> None:
    safe = SafeMode()
    assert safe.status() is False


def test_trigger_activates() -> None:
    safe = SafeMode()
    safe.trigger()
    assert safe.status() is True


def test_clear_deactivates() -> None:
    safe = SafeMode()
    safe.trigger()
    safe.clear()
    assert safe.status() is False


def test_double_trigger_remains_active() -> None:
    safe = SafeMode()
    safe.trigger()
    safe.trigger()
    assert safe.status() is True


def test_clear_when_inactive_is_noop() -> None:
    safe = SafeMode()
    safe.clear()  # should not raise
    assert safe.status() is False


def test_check_raises_when_active() -> None:
    safe = SafeMode()
    safe.trigger()
    with pytest.raises(SafeModeError, match="blocked by safe mode"):
        safe.check("record")


def test_check_passes_when_inactive() -> None:
    safe = SafeMode()
    safe.check("record")  # should not raise


def test_check_after_clear() -> None:
    safe = SafeMode()
    safe.trigger()
    safe.clear()
    safe.check("record")  # should not raise


def test_instances_are_independent() -> None:
    """Each SafeMode instance holds its own state (no shared/global state).

    The machinery deliberately exposes no module-level singleton; callers
    construct their own instance (the CLI carries one per invocation through
    the click context). Mutating one instance must not affect another.
    """
    a = SafeMode()
    b = SafeMode()
    a.trigger()
    assert a.status() is True
    assert b.status() is False, "SafeMode instances must not share state"
    b.clear()
    assert a.status() is True, "clearing b must not clear a"


def test_thread_safety() -> None:
    """Concurrent trigger/clear/status calls should not corrupt state or raise."""
    safe = SafeMode()
    errors: list[Exception] = []

    def toggle() -> None:
        for _ in range(100):
            try:
                safe.trigger()
                safe.status()
                safe.clear()
                safe.status()
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=toggle) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread-safety errors: {errors}"
    # The storm above leaves the final state dependent on thread interleaving,
    # so settle to a known state before asserting — the point of this test is
    # that concurrent access neither corrupts state nor raises, not that any
    # particular op happened to run last.
    safe.clear()
    assert safe.status() is False


def test_check_with_operation_name_in_message() -> None:
    """check(operation) includes the operation name in the error."""
    safe = SafeMode()
    safe.trigger()
    with pytest.raises(SafeModeError, match="'replay' blocked by safe mode"):
        safe.check("replay")
