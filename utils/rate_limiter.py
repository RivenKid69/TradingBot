from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass
class SignalRateLimiter:
    """Simple rate limiter with exponential backoff.

    Parameters
    ----------
    max_per_sec:
        Maximum number of allowed signals per second.  ``0`` disables
        limiting.
    backoff_base:
        Base for the exponential backoff when the limit is exceeded.
    max_backoff:
        Maximum backoff delay in seconds.
    """

    max_per_sec: float
    backoff_base: float = 2.0
    max_backoff: float = 60.0
    _last_reset: float = field(default_factory=lambda: 0.0, init=False)
    _count: int = field(default=0, init=False)
    _cooldown_until: float = field(default_factory=lambda: 0.0, init=False)
    _current_backoff: float = field(default_factory=lambda: 0.0, init=False)

    def can_send(self, now: float | None = None) -> bool:
        """Return ``True`` if a new signal can be sent at ``now``.

        When the limit is exceeded the next allowed time is delayed using
        exponential backoff.
        """
        if self.max_per_sec <= 0:
            return True

        ts = float(time.time() if now is None else now)
        if ts < self._cooldown_until:
            return False

        if ts - self._last_reset >= 1.0:
            self._last_reset = ts
            self._count = 0

        if self._count < self.max_per_sec:
            self._count += 1
            self._current_backoff = 0.0
            return True

        # limit exceeded -> backoff
        if self._current_backoff == 0.0:
            self._current_backoff = 1.0 / max(self.max_per_sec, 1.0)
        else:
            self._current_backoff = min(self._current_backoff * self.backoff_base, self.max_backoff)
        self._cooldown_until = ts + self._current_backoff
        return False

    def reset(self) -> None:
        """Reset internal counters and timers."""
        self._last_reset = 0.0
        self._count = 0
        self._cooldown_until = 0.0
        self._current_backoff = 0.0
