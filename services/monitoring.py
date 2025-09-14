"""Prometheus-backed monitoring helpers.

This module defines lightweight wrappers around Prometheus metrics so that
basic statistics remain available even if the ``prometheus_client`` package is
missing.  In addition to individual counters and gauges, a helper
``snapshot_metrics`` function summarises the most problematic trading symbols
based on feed lag, websocket failures and signal error rates.
"""
from __future__ import annotations

import json
import time
from typing import Dict, Tuple, Union, Optional, Any

from utils.prometheus import Counter, Histogram

from core_config import KillSwitchConfig

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

# Additional per-symbol metrics
feed_lag_max_ms = Gauge(
    "feed_lag_max_ms",
    "Maximum observed feed lag for each symbol in milliseconds",
    ["symbol"],
)
ws_failure_count = Counter(
    "ws_failure_count",
    "Websocket message failures per symbol",
    ["symbol"],
)
signal_error_rate = Gauge(
    "signal_error_rate",
    "Fraction of dropped signals per symbol",
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
_feed_lag_max: Dict[str, float] = {}
_kill_cfg: Optional[KillSwitchConfig] = None
_kill_triggered: bool = False
_kill_reason: Dict[str, Any] = {}


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


def report_feed_lag(symbol: str, lag_ms: Union[int, float]) -> None:
    """Record feed lag for ``symbol`` and update maximum observed value."""
    try:
        lag = float(lag_ms)
    except Exception:
        return
    try:
        feed_lag_max_ms.labels(symbol).set(max(_feed_lag_max.get(symbol, 0.0), lag))
    except Exception:
        pass
    prev = _feed_lag_max.get(symbol, 0.0)
    if lag > prev:
        _feed_lag_max[symbol] = lag
        try:
            feed_lag_max_ms.labels(symbol).set(lag)
        except Exception:
            pass
    _check_kill_switch()


def report_ws_failure(symbol: str) -> None:
    """Record a websocket failure for ``symbol``."""
    try:
        ws_failure_count.labels(symbol).inc()
    except Exception:
        pass
    _check_kill_switch()


def configure_kill_switch(cfg: Optional[KillSwitchConfig]) -> None:
    """Configure kill switch thresholds."""
    global _kill_cfg, _kill_triggered, _kill_reason
    _kill_cfg = cfg
    _kill_triggered = False
    _kill_reason = {}
    _feed_lag_max.clear()
    _check_kill_switch()


def kill_switch_triggered() -> bool:
    """Return whether the kill switch has been triggered."""
    return _kill_triggered


def kill_switch_info() -> Dict[str, Any]:
    """Return details about the kill switch trigger."""
    return dict(_kill_reason)


def _check_kill_switch() -> None:
    """Evaluate metrics against thresholds and update kill switch state."""
    global _kill_triggered, _kill_reason
    cfg = _kill_cfg
    if cfg is None:
        return

    feed_lag = _feed_lag_max.copy()
    ws_fail = _collect(ws_failure_count)
    boundary = _collect(signal_boundary_count)
    absolute = _collect(signal_absolute_count)
    published = _collect(signal_published_count)

    error_rates: Dict[str, float] = {}
    for sym in set(boundary) | set(absolute) | set(published):
        errors = boundary.get(sym, 0.0) + absolute.get(sym, 0.0)
        total = errors + published.get(sym, 0.0)
        rate = errors / total if total > 0 else 0.0
        error_rates[sym] = rate

    worst_feed = max(feed_lag.items(), key=lambda x: x[1], default=(None, 0.0))
    worst_ws = max(ws_fail.items(), key=lambda x: x[1], default=(None, 0.0))
    worst_err = max(error_rates.items(), key=lambda x: x[1], default=(None, 0.0))

    if cfg.feed_lag_ms > 0 and worst_feed[1] > cfg.feed_lag_ms:
        _kill_triggered = True
        _kill_reason = {
            "metric": "feed_lag_ms",
            "symbol": worst_feed[0],
            "value": worst_feed[1],
        }
        return
    if cfg.ws_failures > 0 and worst_ws[1] > cfg.ws_failures:
        _kill_triggered = True
        _kill_reason = {
            "metric": "ws_failures",
            "symbol": worst_ws[0],
            "value": worst_ws[1],
        }
        return
    if cfg.error_rate > 0 and worst_err[1] > cfg.error_rate:
        _kill_triggered = True
        _kill_reason = {
            "metric": "error_rate",
            "symbol": worst_err[0],
            "value": worst_err[1],
        }
        return

    _kill_triggered = False
    _kill_reason = {}


def _collect(counter: Union[Counter, Gauge]) -> Dict[str, float]:
    """Collect labeled metric values as a mapping of symbol to value."""
    try:
        metric = counter.collect()[0]
        out: Dict[str, float] = {}
        for sample in metric.samples:
            if "symbol" in sample.labels and not sample.name.endswith("_created"):
                out[sample.labels["symbol"]] = float(sample.value)
        return out
    except Exception:
        return {}


def snapshot_metrics() -> Tuple[str, str]:
    """Return current metrics snapshot as ``(json, csv)`` strings."""
    feed_lag = _feed_lag_max.copy()
    ws_fail = _collect(ws_failure_count)
    boundary = _collect(signal_boundary_count)
    absolute = _collect(signal_absolute_count)
    published = _collect(signal_published_count)

    error_rates: Dict[str, float] = {}
    for sym in set(boundary) | set(absolute) | set(published):
        errors = boundary.get(sym, 0.0) + absolute.get(sym, 0.0)
        total = errors + published.get(sym, 0.0)
        rate = errors / total if total > 0 else 0.0
        error_rates[sym] = rate
        try:
            signal_error_rate.labels(sym).set(rate)
        except Exception:
            pass

    worst_feed = max(feed_lag.items(), key=lambda x: x[1], default=(None, 0.0))
    worst_ws = max(ws_fail.items(), key=lambda x: x[1], default=(None, 0.0))
    worst_err = max(error_rates.items(), key=lambda x: x[1], default=(None, 0.0))

    summary = {
        "worst_feed_lag": {"symbol": worst_feed[0], "feed_lag_ms": worst_feed[1]},
        "worst_ws_failures": {"symbol": worst_ws[0], "failures": worst_ws[1]},
        "worst_error_rate": {"symbol": worst_err[0], "error_rate": worst_err[1]},
    }

    json_str = json.dumps(summary, sort_keys=True)
    csv_lines = ["metric,symbol,value"]
    csv_lines.append(f"worst_feed_lag,{worst_feed[0] or ''},{worst_feed[1]}")
    csv_lines.append(f"worst_ws_failures,{worst_ws[0] or ''},{worst_ws[1]}")
    csv_lines.append(f"worst_error_rate,{worst_err[0] or ''},{worst_err[1]}")
    csv_str = "\n".join(csv_lines)
    return json_str, csv_str


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
    "feed_lag_max_ms",
    "ws_failure_count",
    "signal_error_rate",
    "report_feed_lag",
    "report_ws_failure",
    "configure_kill_switch",
    "kill_switch_triggered",
    "kill_switch_info",
    "snapshot_metrics",
]
