"""Prometheus-backed monitoring helpers."""
from __future__ import annotations

import time
from typing import Union

from utils.prometheus import Counter, Histogram

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
clock_sync_drift_ms = Gauge(
    "clock_sync_drift_ms",
    "Latest measured clock drift in milliseconds",
)
clock_sync_rtt_ms = Gauge(
    "clock_sync_rtt_ms",
    "Latest measured clock sync round-trip time in milliseconds",
)
clock_sync_last_sync_ts = Gauge(
    "clock_sync_last_sync_ts",
    "Timestamp of last successful clock sync in milliseconds since epoch",
)

# Length of throttling queue
queue_len = Gauge(
    "throttle_queue_len",
    "Current number of queued signals awaiting tokens",
)

# Throttling outcomes
throttle_dropped_count = Counter(
    "throttle_dropped_count",
    "Signals dropped due to throttling",
    ["symbol", "reason"],
)
throttle_enqueued_count = Counter(
    "throttle_enqueued_count",
    "Signals enqueued due to throttling",
    ["symbol", "reason"],
)
throttle_queue_expired_count = Counter(
    "throttle_queue_expired_count",
    "Queued signals expired before tokens became available",
    ["symbol"],
)

# Event bus metrics
queue_depth = Gauge(
    "event_bus_queue_depth",
    "Current depth of the event bus queue",
)
events_in = Counter(
    "event_bus_events_in_total",
    "Total number of events enqueued to the event bus",
)
dropped_bp = Counter(
    "event_bus_dropped_backpressure_total",
    "Events dropped due to backpressure in the event bus",
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

# Bars dropped because they were not fully closed
skipped_incomplete_bars = Counter(
    "skipped_incomplete_bars",
    "Bars dropped because not closed",
    ["symbol"],
)

# Websocket duplicates skipped
ws_dup_skipped_count = Counter(
    "ws_dup_skipped_count",
    "WS duplicates skipped",
    ["symbol"],
)

# Websocket bars dropped due to event bus backpressure
ws_backpressure_drop_count = Counter(
    "ws_backpressure_drop_count",
    "WS bars dropped due to event bus backpressure",
    ["symbol"],
)

# Orders dropped because their originating bar exceeded TTL boundary
ttl_expired_boundary_count = Counter(
    "ttl_expired_boundary_count",
    "Orders dropped due to bar TTL expiration before processing",
    ["symbol"],
)

# Signals dropped or published
signal_boundary_count = Counter(
    "signal_boundary_count",
    "Signals dropped due to TTL boundary expiration",
    ["symbol"],
)
signal_absolute_count = Counter(
    "signal_absolute_count",
    "Signals dropped due to absolute TTL expiration",
    ["symbol"],
)
signal_published_count = Counter(
    "signal_published_count",
    "Signals successfully published",
    ["symbol"],
)

# Age of signals at publish time
age_at_publish_ms = Histogram(
    "age_at_publish_ms",
    "Age of signals when published in milliseconds",
    ["symbol"],
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
        clock_sync_drift_ms.set(float(drift_ms))
        clock_sync_rtt_ms.set(float(rtt_ms))
        if success:
            _last_sync_ts_ms = float(sync_ts)
            clock_sync_last_sync_ts.set(float(sync_ts))
    except Exception:
        pass


def clock_sync_age_seconds() -> float:
    """Return seconds elapsed since the last successful clock sync."""
    if _last_sync_ts_ms <= 0:
        return float("inf")
    return max(0.0, time.time() - _last_sync_ts_ms / 1000.0)


__all__ = [
    "skipped_incomplete_bars",
    "ws_dup_skipped_count",
    "ws_backpressure_drop_count",
    "ttl_expired_boundary_count",
    "signal_boundary_count",
    "signal_absolute_count",
    "signal_published_count",
    "age_at_publish_ms",
    "clock_sync_fail",
    "clock_sync_success",
    "clock_sync_drift_ms",
    "clock_sync_rtt_ms",
    "clock_sync_last_sync_ts",
    "queue_len",
    "throttle_dropped_count",
    "throttle_enqueued_count",
    "throttle_queue_expired_count",
    "queue_depth",
    "events_in",
    "dropped_bp",
    "report_clock_sync",
    "clock_sync_age_seconds",
]
