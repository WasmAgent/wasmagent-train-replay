"""Safe mode machinery — prevents operations when the system is in a locked state.

Mirrors the SAFE_MODE concept from sibling Go projects: when safe mode is
active, the system declines to perform side-effecting operations (recording,
replaying, etc.) until an operator explicitly clears it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


class SafeModeError(RuntimeError):
    """Raised when an operation is attempted while safe mode is active."""


@dataclass
class SafeMode:
    """Thread-safe safe-mode state.

    Usage:
        safe = SafeMode()
        safe.trigger()          # lock
        safe.trigger()          # no-op (already locked)
        safe.status()           # True
        safe.check("record")    # raises SafeModeError
        safe.clear()            # unlock
    """

    _active: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def trigger(self) -> None:
        """Activate safe mode."""
        with self._lock:
            self._active = True

    def clear(self) -> None:
        """Deactivate safe mode."""
        with self._lock:
            self._active = False

    def status(self) -> bool:
        """Return True if safe mode is active."""
        with self._lock:
            return self._active

    def check(self, operation: str = "") -> None:
        """Raise SafeModeError if safe mode is active.

        Args:
            operation: Optional human-readable operation name for the error
                       message.
        """
        with self._lock:
            if self._active:
                if operation:
                    msg = f"Operation '{operation}' blocked by safe mode"
                else:
                    msg = "Operation blocked by safe mode"
                raise SafeModeError(msg)


# NOTE: SafeMode is intentionally NOT instantiated at module import time.
# A module-level mutable singleton leaks state across tests that share an
# interpreter (and across any parallel/async runner), producing
# non-deterministic failures. Callers construct their own instance, and the
# CLI carries one per invocation through the click ``Context.obj``
# (see :func:`train_replay.cli.main.cli`).
