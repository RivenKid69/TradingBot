"""Prometheus-backed monitoring helpers.

This module defines lightweight wrappers around Prometheus metrics so that
basic statistics remain available even if the ``prometheus_client`` package is
missing.  In addition to individual counters and gauges, a helper
``snapshot_metrics`` function summarises the most problematic trading symbols
based on feed lag, websocket failures and signal error rates.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Tuple, Union, Optional, Any
from collections import deque

from enum import Enum
from utils.prometheus import Counter, Histogram
from .utils_app import atomic_write_with_retry
from .alerts import AlertManager

from core_config import KillSwitchConfig, MonitoringConfig

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

# HTTP request metrics
http_request_count = Counter(
    "http_request_count",
    "Total number of HTTP requests",
)
http_success_count = Counter(
    "http_success_count",
    "Total number of successful HTTP responses",
    ["status"],
)
http_error_count = Counter(
    "http_error_count",
    "Total number of HTTP request errors",
    ["code"],
)


def record_http_request() -> None:
    """Record an HTTP request attempt."""
    try:
        http_request_count.inc()
    except Exception:
        pass


def record_http_success(status: Union[int, str]) -> None:
    """Record successful HTTP response with ``status`` code."""
    try:
        http_success_count.labels(str(status)).inc()
    except Exception:
        pass


def record_http_error(code: Union[int, str]) -> None:
    """Record HTTP error with classification ``code``."""
    try:
        http_error_count.labels(str(code)).inc()
    except Exception:
        pass

# Pipeline stage drops
pipeline_stage_drop_count = Counter(
    "pipeline_stage_drop_count",
    "Pipeline drops per stage and reason",
    ["symbol", "stage", "reason"],
)

# Global pipeline stage and reason counters
pipeline_stage_count = Counter(
    "pipeline_stage_count",
    "Total number of processed pipeline stages",
    ["stage"],
)
pipeline_reason_count = Counter(
    "pipeline_reason_count",
    "Total number of pipeline drop reasons",
    ["reason"],
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

def _label(value: Enum | str) -> str:
    """Return the name of an Enum member or cast value to string."""
    try:
        return value.name  # type: ignore[attr-defined]
    except Exception:
        return str(value)


def inc_stage(stage: Enum | str) -> None:
    """Increment processed pipeline stage counter."""
    try:
        pipeline_stage_count.labels(_label(stage)).inc()
    except Exception:
        pass


def inc_reason(reason: Enum | str) -> None:
    """Increment pipeline drop reason counter."""
    try:
        pipeline_reason_count.labels(_label(reason)).inc()
    except Exception:
        pass

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


def reset_kill_switch_counters() -> None:
    """Reset kill switch metrics and internal state."""
    ws_failure_count._metrics.clear()
    signal_boundary_count._metrics.clear()
    signal_absolute_count._metrics.clear()
    signal_published_count._metrics.clear()
    _feed_lag_max.clear()
    global _kill_triggered, _kill_reason
    _kill_triggered = False
    _kill_reason = {}
    _check_kill_switch()


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


class MonitoringAggregator:
    """Aggregate runtime metrics and emit alerts.

    The aggregator collects various runtime statistics and periodically
    evaluates them against :class:`MonitoringConfig.thresholds`.  If any of the
    thresholds are exceeded an alert is sent via the provided
    :class:`AlertManager` instance.  Metrics are also periodically flushed to a
    JSON lines file for external processing.
    """

    def __init__(self, cfg: MonitoringConfig, alerts: AlertManager) -> None:
        self.cfg = cfg
        self.alerts = alerts
        self.enabled = bool(cfg.enabled)
        # Frequency of metrics flushes
        self.flush_interval_sec = int(getattr(cfg, "snapshot_metrics_sec", 60))
        self._last_flush = time.time()

        # Sliding window stores
        self._ws_events: deque[tuple[int, str]] = deque()
        self._http_events: deque[tuple[int, bool, Optional[int]]] = deque()
        self._signal_events: deque[tuple[int, str, int, int]] = deque()
        self._last_bar_close_ms: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Recording helpers
    def record_feed(self, symbol: str, close_ms: int) -> None:
        """Record receipt of latest bar for *symbol* with ``close_ms``."""
        if not self.enabled:
            return
        now_ms = int(time.time() * 1000)
        self._last_bar_close_ms[symbol] = int(close_ms)
        try:
            report_feed_lag(symbol, now_ms - int(close_ms))
        except Exception:
            pass

    def record_ws(self, event: str) -> None:
        """Record websocket ``event`` (e.g. ``reconnect`` or ``failure``)."""
        if not self.enabled:
            return
        self._ws_events.append((int(time.time() * 1000), str(event)))

    def record_http(self, success: bool, status: Optional[int]) -> None:
        """Record outcome of an HTTP request."""
        if not self.enabled:
            return
        self._http_events.append((int(time.time() * 1000), bool(success), status))

    def record_signals(self, symbol: str, emitted: int, duplicates: int) -> None:
        """Record signal counts for ``symbol``."""
        if not self.enabled:
            return
        self._signal_events.append((int(time.time() * 1000), symbol, int(emitted), int(duplicates)))

    # ------------------------------------------------------------------
    def _prune(self, dq: deque[tuple], cutoff_ms: int) -> None:
        """Drop events older than ``cutoff_ms`` from ``dq``."""
        while dq and dq[0][0] < cutoff_ms:
            dq.popleft()

    def tick(self, now_ms: int) -> None:
        """Aggregate recent data and emit alerts if thresholds exceeded."""
        if not self.enabled:
            return

        cutoff_1m = now_ms - 60_000
        cutoff_5m = now_ms - 5 * 60_000

        # Prune outdated events
        self._prune(self._ws_events, cutoff_5m)
        self._prune(self._http_events, cutoff_5m)
        self._prune(self._signal_events, cutoff_5m)

        th = self.cfg.thresholds

        # Feed lag checks
        feed_lags: Dict[str, int] = {}
        if th.feed_lag_ms > 0:
            for sym, close in self._last_bar_close_ms.items():
                lag = int(now_ms - close)
                feed_lags[sym] = lag
                if lag > th.feed_lag_ms:
                    try:
                        self.alerts.notify(
                            f"feed_lag_{sym}",
                            f"{sym} feed lag {lag}ms exceeds {th.feed_lag_ms}",
                        )
                    except Exception:
                        pass

        # Websocket events
        ws_fail_1m = sum(1 for ts, ev in self._ws_events if ts >= cutoff_1m and ev == "failure")
        ws_fail_5m = sum(1 for _, ev in self._ws_events if ev == "failure")
        ws_rec_1m = sum(1 for ts, ev in self._ws_events if ts >= cutoff_1m and ev == "reconnect")
        ws_rec_5m = sum(1 for _, ev in self._ws_events if ev == "reconnect")
        if th.ws_failures > 0 and ws_fail_1m > th.ws_failures:
            try:
                self.alerts.notify(
                    "ws_failures", f"websocket failures last minute: {ws_fail_1m}"
                )
            except Exception:
                pass

        # HTTP statistics
        http_total_1m = sum(1 for ts, *_ in self._http_events if ts >= cutoff_1m)
        http_error_1m = sum(1 for ts, ok, _ in self._http_events if ts >= cutoff_1m and not ok)
        http_total_5m = len(self._http_events)
        http_error_5m = sum(1 for _, ok, _ in self._http_events if not ok)

        # Signal statistics
        emitted_1m: Dict[str, int] = {}
        dup_1m: Dict[str, int] = {}
        for ts, sym, em, du in self._signal_events:
            if ts >= cutoff_1m:
                emitted_1m[sym] = emitted_1m.get(sym, 0) + em
                dup_1m[sym] = dup_1m.get(sym, 0) + du
        emitted_5m: Dict[str, int] = {}
        dup_5m: Dict[str, int] = {}
        for _, sym, em, du in self._signal_events:
            emitted_5m[sym] = emitted_5m.get(sym, 0) + em
            dup_5m[sym] = dup_5m.get(sym, 0) + du

        worst_sym: Optional[str] = None
        worst_rate = 0.0
        for sym in set(emitted_1m) | set(dup_1m):
            em = emitted_1m.get(sym, 0)
            du = dup_1m.get(sym, 0)
            rate = du / em if em > 0 else 0.0
            if rate > worst_rate:
                worst_rate = rate
                worst_sym = sym
        if th.error_rate > 0 and worst_rate > th.error_rate and worst_sym is not None:
            try:
                self.alerts.notify(
                    "signal_error_rate",
                    f"{worst_sym} duplicate rate {worst_rate:.3f} exceeds {th.error_rate}",
                )
            except Exception:
                pass

        # Periodic flush to metrics file
        now_sec = now_ms / 1000.0
        if now_sec - self._last_flush >= self.flush_interval_sec:
            metrics = {
                "ts_ms": now_ms,
                "feed_lag_ms": feed_lags,
                "ws": {
                    "reconnects_1m": ws_rec_1m,
                    "reconnects_5m": ws_rec_5m,
                    "failures_1m": ws_fail_1m,
                    "failures_5m": ws_fail_5m,
                },
                "http": {
                    "total_1m": http_total_1m,
                    "errors_1m": http_error_1m,
                    "total_5m": http_total_5m,
                    "errors_5m": http_error_5m,
                },
                "signals": {
                    "emitted_1m": sum(emitted_1m.values()),
                    "duplicates_1m": sum(dup_1m.values()),
                    "emitted_5m": sum(emitted_5m.values()),
                    "duplicates_5m": sum(dup_5m.values()),
                },
            }
            line = json.dumps(metrics, sort_keys=True)
            try:
                atomic_write_with_retry(
                    os.path.join("logs", "metrics.jsonl"), line + "\n", retries=3, backoff=0.1
                )
            except Exception:
                pass
            self._last_flush = now_sec


def snapshot_metrics(json_path: str, csv_path: str) -> Tuple[Dict[str, Any], str, str]:
    """Persist current metrics snapshot to ``json_path`` and ``csv_path``.

    Parameters
    ----------
    json_path : str
        Destination for JSON summary.
    csv_path : str
        Destination for CSV summary.

    Returns
    -------
    Tuple[Dict[str, Any], str, str]
        The structured summary along with its JSON and CSV representations.
    """
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

    try:
        atomic_write_with_retry(json_path, json_str, retries=3, backoff=0.1)
    except Exception:
        pass
    try:
        atomic_write_with_retry(csv_path, csv_str, retries=3, backoff=0.1)
    except Exception:
        pass

    return summary, json_str, csv_str


__all__ = [
    "skipped_incomplete_bars",
    "ws_dup_skipped_count",
    "ws_backpressure_drop_count",
    "http_request_count",
    "http_success_count",
    "http_error_count",
    "pipeline_stage_drop_count",
    "pipeline_stage_count",
    "pipeline_reason_count",
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
    "record_http_request",
    "record_http_success",
    "record_http_error",
    "report_clock_sync",
    "clock_sync_age_seconds",
    "feed_lag_max_ms",
    "ws_failure_count",
    "signal_error_rate",
    "report_feed_lag",
    "report_ws_failure",
    "configure_kill_switch",
    "inc_stage",
    "inc_reason",
    "kill_switch_triggered",
    "kill_switch_info",
    "reset_kill_switch_counters",
    "MonitoringAggregator",
    "snapshot_metrics",
]
