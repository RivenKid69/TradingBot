"""Time provider abstractions for trading components.

The default implementation returns the current wall-clock time while also
providing a simple testing hook for deterministic time sources.
"""
from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class TimeProvider(Protocol):
    """Protocol describing objects that can provide monotonic timestamps."""

    def time(self) -> float:
        """Return the current time in seconds."""

    def time_ms(self) -> int:
        """Return the current time in milliseconds."""

    def sleep(self, seconds: float) -> None:
        """Sleep for ``seconds`` seconds."""


class RealTimeProvider:
    """Concrete :class:`TimeProvider` using :mod:`time` functions."""

    def time(self) -> float:
        return time.time()

    def time_ms(self) -> int:
        return int(self.time() * 1000.0)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


__all__ = ["TimeProvider", "RealTimeProvider"]
