from __future__ import annotations

"""Utilities for basic pipeline time-to-live checks."""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Tuple

from utils_time import next_bar_open_ms


class Stage(Enum):
    """Pipeline stages for decision making."""

    CLOSED_BAR = auto()
    WINDOWS = auto()
    ANOMALY = auto()
    EXTREME = auto()
    POLICY = auto()
    RISK = auto()
    PUBLISH = auto()


class Reason(Enum):
    """Reasons for halting or skipping pipeline stages."""

    INCOMPLETE_BAR = auto()
    MAINTENANCE = auto()
    WINDOW = auto()
    ANOMALY_RET = auto()
    ANOMALY_SPREAD = auto()
    EXTREME_VOL = auto()
    EXTREME_SPREAD = auto()
    RISK_POSITION = auto()
    OTHER = auto()


@dataclass
class PipelineResult:
    """Result produced by each pipeline stage."""

    action: str
    stage: Stage
    reason: Reason | None = None
    decision: Any | None = None


def compute_expires_at(bar_close_ms: int, timeframe_ms: int) -> int:
    """Compute expiration timestamp for a bar.

    Parameters
    ----------
    bar_close_ms : int
        Close timestamp of the bar in milliseconds since epoch.
    timeframe_ms : int
        Timeframe of the bar in milliseconds.

    Returns
    -------
    int
        The timestamp (ms since epoch) when the next bar opens.
    """
    return next_bar_open_ms(bar_close_ms, timeframe_ms)


def check_ttl(
    bar_close_ms: int, now_ms: int, timeframe_ms: int
) -> Tuple[bool, int, str]:
    """Validate that a bar has not exceeded its time-to-live.

    The TTL for a bar is one full timeframe after its close. This function
    checks the absolute age of the bar against that limit.

    Parameters
    ----------
    bar_close_ms : int
        Close timestamp of the bar.
    now_ms : int
        Current time in milliseconds since epoch.
    timeframe_ms : int
        Bar timeframe in milliseconds.

    Returns
    -------
    Tuple[bool, int, str]
        A tuple of ``(valid, expires_at_ms, reason)`` where ``valid`` indicates
        whether the bar is still within its TTL, ``expires_at_ms`` is the
        absolute expiration timestamp, and ``reason`` provides context when the
        bar is no longer valid.
    """
    expires_at_ms = compute_expires_at(bar_close_ms, timeframe_ms)
    age_ms = now_ms - bar_close_ms
    if now_ms <= expires_at_ms:
        return True, expires_at_ms, ""
    return False, expires_at_ms, f"age {age_ms}ms exceeds {timeframe_ms}ms"


__all__ = [
    "Stage",
    "Reason",
    "PipelineResult",
    "compute_expires_at",
    "check_ttl",
]
