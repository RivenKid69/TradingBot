"""Prometheus-backed monitoring helpers."""
from __future__ import annotations

import time
from typing import Union

from utils.prometheus import Counter

try:  # pragma: no cover - optional dependency
    from prometheus_client import Gauge
except Exception:  # pragma: no cover - fallback when prometheus_client is missing
    class _DummyGauge:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def labels(self, *args, **kwargs) -> "_DummyGauge":
            return self

        def set(self, *args, **kwargs) -> None:
            pass

    Gauge = _DummyGauge  # type: ignore

# Gauges for latest clock sync measurements
_CLOCK_SYNC_DRIFT_MS = Gauge(
    "clock_sync_drift_ms",
    "Latest measured clock drift in milliseconds",
)
_CLOCK_SYNC_RTT_MS = Gauge(
    "clock_sync_rtt_ms",
    "Latest measured clock sync round-trip time in milliseconds",
)
_CLOCK_SYNC_LAST_TS = Gauge(
    "clock_sync_last_sync_ts",
    "Timestamp of last successful clock sync in milliseconds since epoch",
)

# Counters for sync attempts
clock_sync_success = Counter(
    "clock_sync_success_total",
    "Total number of successful clock synchronizations",
)
clock_sync_fail = Counter(
    "clock_sync_fail_total",
    "Total number of failed clock synchronization attempts",
)

_last_sync_ts_ms: float = 0.0


def report_clock_sync(
    drift_ms: Union[int, float],
    rtt_ms: Union[int, float],
    success: bool,
    sync_ts: Union[int, float],
) -> None:
    """Report outcome of a clock synchronization attempt.

    Parameters
    ----------
    drift_ms : Union[int, float]
        Estimated clock drift in milliseconds.
    rtt_ms : Union[int, float]
        Round-trip time of the sync request in milliseconds.
    success : bool
        Whether the synchronization succeeded.
    sync_ts : Union[int, float]
        Timestamp of the sync (milliseconds since epoch).
    """
    global _last_sync_ts_ms

    try:
        if success:
            clock_sync_success.inc()
        else:
            clock_sync_fail.inc()
    except Exception:
        pass

    try:
        _CLOCK_SYNC_DRIFT_MS.set(float(drift_ms))
        _CLOCK_SYNC_RTT_MS.set(float(rtt_ms))
        if success:
            _last_sync_ts_ms = float(sync_ts)
            _CLOCK_SYNC_LAST_TS.set(float(sync_ts))
    except Exception:
        pass


def clock_sync_age_seconds() -> float:
    """Return seconds elapsed since the last successful clock sync."""
    if _last_sync_ts_ms <= 0:
        return float("inf")
    return max(0.0, time.time() - _last_sync_ts_ms / 1000.0)


__all__ = [
    "clock_sync_fail",
    "clock_sync_success",
    "report_clock_sync",
    "clock_sync_age_seconds",
]
