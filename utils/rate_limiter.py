from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Literal, Tuple


@dataclass
class TokenBucket:
    """Simple token bucket rate limiter."""

    rps: float
    burst: float
    tokens: float = 0.0
    last_ts: float = field(default_factory=time.monotonic)

    def consume(self, tokens: float = 1, now: float | None = None) -> bool:
        ts = time.monotonic() if now is None else now
        self.tokens = min(self.burst, self.tokens + (ts - self.last_ts) * self.rps)
        self.last_ts = ts
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


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

    def can_send(self, now: float | None = None) -> Tuple[bool, Literal["ok", "delayed", "rejected"]]:
        """Return status whether a new signal can be sent at ``now``.

        Returns
        -------
        allowed : bool
            ``True`` if a signal may be sent immediately.
        status : {"ok", "delayed", "rejected"}
            Additional status describing the limiter decision. ``"delayed"``
            means the call happened during a cooldown period, while
            ``"rejected"`` indicates the rate limit has just been exceeded and
            a new cooldown is started.
        """
        if self.max_per_sec <= 0:
            return True, "ok"

        ts = float(time.time() if now is None else now)
        if ts < self._cooldown_until:
            return False, "delayed"

        if ts - self._last_reset >= 1.0:
            self._last_reset = ts
            self._count = 0

        if self._count < self.max_per_sec:
            self._count += 1
            self._current_backoff = 0.0
            return True, "ok"

        # limit exceeded -> backoff
        if self._current_backoff == 0.0:
            self._current_backoff = 1.0 / self.max_per_sec
        else:
            self._current_backoff = min(self._current_backoff * self.backoff_base, self.max_backoff)
        self._cooldown_until = ts + self._current_backoff
        return False, "rejected"

    def reset(self) -> None:
        """Reset internal counters and timers."""
        self._last_reset = 0.0
        self._count = 0
        self._cooldown_until = 0.0
        self._current_backoff = 0.0
