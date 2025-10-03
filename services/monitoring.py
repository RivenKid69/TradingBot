"""Prometheus-backed monitoring helpers.

This module defines lightweight wrappers around Prometheus metrics so that
basic statistics remain available even if the ``prometheus_client`` package is
missing.  In addition to individual counters and gauges, a helper
``snapshot_metrics`` function summarises the most problematic trading symbols
based on feed lag, websocket failures and signal error rates.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, Tuple, Union, Optional, Any, DefaultDict, Iterable, Mapping
from collections import deque, defaultdict

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

# Binance filters metadata freshness
filters_age_days = Gauge(
    "filters_age_days",
    "Age of binance_filters.json in days",
)

# Length of throttling queue
queue_len = Gauge(
    "throttle_queue_len",
    "Current number of queued signals awaiting tokens",
)

# Daily entry limiter blocks
entry_limiter_block_count = Counter(
    "entry_limiter_block_count",
    "Signals blocked by the daily entry limiter",
    ["symbol"],
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

# Consecutive zero-signal bars exceeding threshold
zero_signals_alert_count = Counter(
    "zero_signals_alert_count",
    "Zero-signal bar alerts",
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

_runtime_aggregator: "MonitoringAggregator | None" = None


def set_runtime_aggregator(agg: "MonitoringAggregator | None") -> None:
    """Register the currently active :class:`MonitoringAggregator`."""

    global _runtime_aggregator
    _runtime_aggregator = agg


def clear_runtime_aggregator() -> None:
    """Reset the registered runtime :class:`MonitoringAggregator`."""

    set_runtime_aggregator(None)


def get_runtime_aggregator() -> "MonitoringAggregator | None":
    return _runtime_aggregator


def record_http_request() -> None:
    """Record an HTTP request attempt."""
    agg = get_runtime_aggregator()
    if agg is not None:
        try:
            agg.record_http_attempt()
        except Exception:
            pass
    try:
        http_request_count.inc()
    except Exception:
        pass


def record_http_success(status: Union[int, str], *, timed_out: bool = False) -> None:
    """Record successful HTTP response with ``status`` code."""
    agg = get_runtime_aggregator()
    if agg is not None:
        try:
            agg.record_http(True, status, timed_out=bool(timed_out))
        except Exception:
            pass
    try:
        http_success_count.labels(str(status)).inc()
    except Exception:
        pass


def record_http_error(
    code: Union[int, str], *, timed_out: bool = False
) -> None:
    """Record HTTP error with classification ``code``."""
    agg = get_runtime_aggregator()
    if agg is not None:
        try:
            agg.record_http(False, code, timed_out=bool(timed_out))
        except Exception:
            pass
    try:
        http_error_count.labels(str(code)).inc()
    except Exception:
        pass


def record_signals(symbol: str, emitted: int, duplicates: int) -> None:
    """Record per-bar signal statistics for ``symbol``."""
    agg = get_runtime_aggregator()
    if agg is not None:
        try:
            agg.record_signals(symbol, emitted, duplicates)
        except Exception:
            pass
    total = int(emitted) + int(duplicates)
    if total <= 0:
        return
    try:
        signal_error_rate.labels(symbol).set(float(duplicates) / float(total))
    except Exception:
        pass


def alert_zero_signals(symbol: str) -> None:
    """Record an alert for consecutive zero-signal bars for ``symbol``."""
    try:
        zero_signals_alert_count.labels(symbol).inc()
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

signal_idempotency_skipped_count = Counter(
    "signal_idempotency_skipped_count",
    "Signals skipped because of idempotency cache",
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
_last_ws_failure_ts_ms: int = 0
_last_ws_reconnect_ts_ms: int = 0
_throttle_queue_depth_snapshot: Dict[str, int] = {"size": 0, "max": 0}
_cooldowns_active_snapshot: Dict[str, Any] = {
    "count": 0,
    "global": False,
    "symbols": [],
}
_zero_signal_streaks_snapshot: Dict[str, int] = {}


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
    """Aggregate runtime metrics, evaluate thresholds and emit alerts."""

    def __init__(self, cfg: MonitoringConfig, alerts: AlertManager) -> None:
        self.cfg = cfg
        self.alerts = alerts
        self.enabled = bool(getattr(cfg, "enabled", False))
        thresholds = getattr(cfg, "thresholds", None)
        if thresholds is None:
            thresholds = MonitoringConfig().thresholds
        self.thresholds = thresholds

        self._window_ms: Dict[str, int] = {"1m": 60_000, "5m": 5 * 60_000}
        self._ws_events: Dict[str, deque[tuple[int, str]]] = {
            key: deque() for key in self._window_ms
        }
        self._ws_counts: Dict[str, DefaultDict[str, int]] = {
            key: defaultdict(int) for key in self._window_ms
        }
        self._http_events: Dict[str, deque[tuple[int, bool, Optional[Union[int, str]], str]]] = {
            key: deque() for key in self._window_ms
        }
        self._http_counts: Dict[str, Dict[str, int]] = {
            key: {
                "total": 0,
                "success": 0,
                "error": 0,
                "429": 0,
                "5xx": 0,
                "timeout": 0,
                "other": 0,
            }
            for key in self._window_ms
        }
        self._http_attempts: Dict[str, deque[int]] = {
            key: deque() for key in self._window_ms
        }
        self._signal_events: Dict[str, deque[tuple[int, str, int, int]]] = {
            key: deque() for key in self._window_ms
        }
        self._signal_counts: Dict[str, Dict[str, DefaultDict[str, int]]] = {
            key: {
                "emitted": defaultdict(int),
                "duplicates": defaultdict(int),
            }
            for key in self._window_ms
        }
        self._last_bar_close_ms: Dict[str, int] = {}
        self._zero_signal_streaks: Dict[str, int] = {}
        self._zero_signal_alerted: set[str] = set()
        self._feed_alerted: set[str] = set()
        self._signal_alerted: set[str] = set()
        self._stale_alerted: set[str] = set()
        self._ws_alert_active = False
        self._fill_alert_active = False
        self._pnl_alert_active = False
        self._consecutive_ws_failures = 0
        self._metrics_path = os.path.join("logs", "metrics.jsonl")
        self._last_flush_ts = time.time()

        self.fill_ratio: Optional[float] = None
        self.daily_pnl: Optional[float] = None

        self._bar_interval_ms: Dict[str, int] = {}
        self._execution_mode: str = "bar"
        self._bar_events: Dict[str, deque[Dict[str, Any]]] = {
            key: deque() for key in self._window_ms
        }
        self._bar_totals: Dict[str, float] = {
            "decisions": 0.0,
            "act_now": 0.0,
            "turnover_usd": 0.0,
            "realized_cost_weight": 0.0,
            "realized_cost_wsum": 0.0,
            "modeled_cost_weight": 0.0,
            "modeled_cost_wsum": 0.0,
        }
        self._bar_caps_by_symbol: Dict[str, float] = {}
        self._bar_mode_totals: DefaultDict[str, float] = defaultdict(float)
        self.last_ws_reconnect_ms: Optional[int] = None
        self.last_ws_failure_ms: Optional[int] = None
        self.throttle_queue_depth: Dict[str, int] = {"size": 0, "max": 0}
        self.cooldowns_active: Dict[str, Any] = {
            "count": 0,
            "global": False,
            "symbols": [],
        }
        self.zero_signal_streaks: Dict[str, int] = {}
        self.daily_turnover: Dict[str, Any] = {}
        self._cost_bias_alerted: set[str] = set()
        global _throttle_queue_depth_snapshot, _cooldowns_active_snapshot, _zero_signal_streaks_snapshot
        _throttle_queue_depth_snapshot = dict(self.throttle_queue_depth)
        _cooldowns_active_snapshot = dict(self.cooldowns_active)
        _zero_signal_streaks_snapshot = {}

        # ``ServiceSignalRunner`` is responsible for registering the runtime
        # aggregator once the full runtime wiring is complete.  This avoids
        # transient registration in case initialisation fails midway through
        # ServiceSignalRunner construction.

    # ------------------------------------------------------------------
    # Properties and helpers
    @property
    def flush_interval_sec(self) -> int:
        """Flush interval in seconds derived from configuration."""

        raw = getattr(self.cfg, "snapshot_metrics_sec", 60)
        try:
            interval = int(raw)
        except Exception:
            interval = 60
        if interval <= 0:
            interval = 60
        return interval

    @staticmethod
    def _sanitize_daily_turnover(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): MonitoringAggregator._sanitize_daily_turnover(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [MonitoringAggregator._sanitize_daily_turnover(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            return float(value)
        except Exception:
            try:
                return str(value)
            except Exception:
                return None

    def register_feed_intervals(
        self, symbols: Iterable[str], interval_ms: int
    ) -> None:
        """Register the base bar interval for ``symbols`` in milliseconds."""

        if not self.enabled:
            return
        try:
            interval = int(interval_ms)
        except Exception:
            return
        if interval <= 0:
            return
        for sym in symbols:
            if sym is None:
                continue
            try:
                key = str(sym)
            except Exception:
                continue
            if not key:
                continue
            self._bar_interval_ms[key] = interval

    def _stale_threshold_for(self, symbol: str) -> int:
        """Return stale threshold for ``symbol`` in milliseconds."""

        feed_threshold = float(getattr(self.thresholds, "feed_lag_ms", 0.0) or 0.0)
        base_threshold = int(feed_threshold) if feed_threshold > 0 else 0
        base_threshold = max(base_threshold, 120_000)
        interval = self._bar_interval_ms.get(symbol)
        if interval is not None and interval > 0:
            if interval * 2 > base_threshold:
                base_threshold = interval * 2
        return base_threshold

    def _notify(self, key: str, message: str) -> None:
        if not self.alerts:
            return
        try:
            self.alerts.notify(key, message)
        except Exception:
            pass

    def _classify_http(self, success: bool, status: Optional[Union[int, str]]) -> str:
        if success:
            return "success"
        if status is None:
            return "timeout"
        if isinstance(status, str):
            code = status.strip().lower()
            if code == "timeout":
                return "timeout"
            if code.isdigit():
                try:
                    status = int(code)
                except Exception:
                    return "other"
            else:
                return "other"
        if isinstance(status, (int, float)):
            status_int = int(status)
            if status_int == 429:
                return "429"
            if 500 <= status_int <= 599:
                return "5xx"
            return "other"
        return "other"

    def _prune_ws_window(self, window: str, now_ms: int) -> None:
        cutoff = now_ms - self._window_ms[window]
        dq = self._ws_events[window]
        counts = self._ws_counts[window]
        while dq and dq[0][0] < cutoff:
            _, event = dq.popleft()
            counts[event] -= 1
            if counts[event] <= 0:
                counts.pop(event, None)

    def _prune_http_window(self, window: str, now_ms: int) -> None:
        cutoff = now_ms - self._window_ms[window]
        dq = self._http_events[window]
        counts = self._http_counts[window]
        while dq and dq[0][0] < cutoff:
            _, ok, _, classification = dq.popleft()
            counts["total"] = max(0, counts["total"] - 1)
            if ok:
                counts["success"] = max(0, counts["success"] - 1)
            else:
                counts["error"] = max(0, counts["error"] - 1)
                key = classification if classification in counts else "other"
                counts[key] = max(0, counts.get(key, 0) - 1)

    def _prune_http_attempts(self, window: str, now_ms: int) -> None:
        cutoff = now_ms - self._window_ms[window]
        dq = self._http_attempts[window]
        while dq and dq[0] < cutoff:
            dq.popleft()

    def _prune_bar_events(self, window: str, now_ms: int) -> None:
        dq = self._bar_events.get(window)
        if dq is None:
            return
        cutoff = now_ms - self._window_ms[window]
        while dq and int(dq[0].get("ts", 0)) < cutoff:
            dq.popleft()

    def _prune_signal_window(self, window: str, now_ms: int) -> None:
        cutoff = now_ms - self._window_ms[window]
        dq = self._signal_events[window]
        counts = self._signal_counts[window]
        emitted = counts["emitted"]
        duplicates = counts["duplicates"]
        while dq and dq[0][0] < cutoff:
            _, sym, em, du = dq.popleft()
            if em:
                new_val = emitted.get(sym, 0) - em
                if new_val > 0:
                    emitted[sym] = new_val
                else:
                    emitted.pop(sym, None)
            if du:
                new_val = duplicates.get(sym, 0) - du
                if new_val > 0:
                    duplicates[sym] = new_val
                else:
                    duplicates.pop(sym, None)

    def _http_snapshot(self, window: str) -> Dict[str, Any]:
        counts = self._http_counts[window]
        total = counts["total"]
        errors = counts["error"]
        snapshot = {
            "attempts": len(self._http_attempts[window]),
            "success": counts["success"],
            "errors": errors,
            "total": total,
            "error_rate": float(errors) / float(total) if total > 0 else 0.0,
            "by_code": {
                "429": counts.get("429", 0),
                "5xx": counts.get("5xx", 0),
                "timeout": counts.get("timeout", 0),
                "other": counts.get("other", 0),
            },
        }
        return snapshot

    def _signal_snapshot(self, window: str) -> Dict[str, Any]:
        emitted = self._signal_counts[window]["emitted"]
        duplicates = self._signal_counts[window]["duplicates"]
        totals = {
            "emitted": sum(emitted.values()),
            "duplicates": sum(duplicates.values()),
        }
        rates: Dict[str, float] = {}
        worst_symbol: Optional[str] = None
        worst_rate = 0.0
        for sym in set(emitted) | set(duplicates):
            em = emitted.get(sym, 0)
            du = duplicates.get(sym, 0)
            if em > 0:
                rate = float(du) / float(em)
            else:
                rate = 1.0 if du > 0 else 0.0
            rates[sym] = rate
            if rate > worst_rate:
                worst_rate = rate
                worst_symbol = sym
        return {
            "emitted": totals["emitted"],
            "duplicates": totals["duplicates"],
            "rates": rates,
            "worst_symbol": worst_symbol,
            "worst_rate": worst_rate,
        }

    def update_queue_depth(self, size: int, max_size: int | None = None) -> None:
        if not self.enabled:
            return
        try:
            queue_size = int(size)
        except Exception:
            queue_size = 0
        if max_size is None:
            queue_cap = 0
        else:
            try:
                queue_cap = int(max_size)
            except Exception:
                queue_cap = 0
        payload = {"size": max(0, queue_size), "max": max(0, queue_cap)}
        self.throttle_queue_depth = payload
        global _throttle_queue_depth_snapshot
        _throttle_queue_depth_snapshot = dict(payload)

    def update_cooldowns(self, payload: Mapping[str, Any] | None) -> None:
        if not self.enabled:
            return
        global_flag = False
        symbols: list[str] = []
        count_val: Optional[int] = None
        if isinstance(payload, Mapping):
            global_flag = bool(payload.get("global"))
            raw_symbols = payload.get("symbols")
            if isinstance(raw_symbols, Mapping):
                for sym, active in raw_symbols.items():
                    if not active:
                        continue
                    try:
                        sym_str = str(sym)
                    except Exception:
                        continue
                    if sym_str:
                        symbols.append(sym_str)
            elif isinstance(raw_symbols, (list, tuple, set)):
                for sym in raw_symbols:
                    try:
                        sym_str = str(sym)
                    except Exception:
                        continue
                    if sym_str:
                        symbols.append(sym_str)
            try:
                count_val = int(payload.get("count"))
            except Exception:
                count_val = None
        symbols = sorted({sym for sym in symbols if sym})
        count = count_val
        if count is None:
            count = len(symbols) + (1 if global_flag else 0)
        self.cooldowns_active = {
            "count": max(0, int(count)),
            "global": bool(global_flag),
            "symbols": symbols,
        }
        global _cooldowns_active_snapshot
        _cooldowns_active_snapshot = dict(self.cooldowns_active)

    def _update_zero_signal_snapshot(self) -> None:
        if not self.enabled:
            return
        active = {
            sym: int(streak)
            for sym, streak in self._zero_signal_streaks.items()
            if int(streak) > 0
        }
        self.zero_signal_streaks = active
        global _zero_signal_streaks_snapshot
        _zero_signal_streaks_snapshot = dict(active)

    def update_daily_turnover(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        cleaned = self._sanitize_daily_turnover(payload)
        if isinstance(cleaned, Mapping):
            self.daily_turnover = dict(cleaned)
        else:
            self.daily_turnover = cleaned if cleaned is not None else {}

    def set_execution_mode(self, mode: str) -> None:
        normalized = str(mode or "bar").lower()
        if normalized not in {"order", "bar"}:
            normalized = "bar"
        if normalized == self._execution_mode:
            return
        self._execution_mode = normalized
        if normalized != "bar":
            self._bar_events = {key: deque() for key in self._window_ms}
            self._bar_totals = {
                "decisions": 0.0,
                "act_now": 0.0,
                "turnover_usd": 0.0,
                "realized_cost_weight": 0.0,
                "realized_cost_wsum": 0.0,
                "modeled_cost_weight": 0.0,
                "modeled_cost_wsum": 0.0,
            }
            self._bar_caps_by_symbol = {}
            self._bar_mode_totals = defaultdict(float)
            self._cost_bias_alerted.clear()

    def _bar_window_snapshot(self, window: str) -> Dict[str, Any]:
        dq = self._bar_events.get(window, deque())

        def _as_float(value: Any) -> float | None:
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(result):
                return None
            return result

        def _entry_weight(entry: Mapping[str, Any]) -> float:
            for key in ("weight", "turnover_usd", "decisions"):
                candidate = _as_float(entry.get(key))
                if candidate is not None and candidate > 0:
                    return candidate
            return 0.0

        decisions_total = 0.0
        act_total = 0.0
        turnover_total = 0.0
        caps_by_symbol: Dict[str, float] = {}
        mode_counts: Dict[str, int] = {}
        realized_sum = 0.0
        realized_weight = 0.0
        modeled_sum = 0.0
        modeled_weight = 0.0
        bias_sum = 0.0
        bias_weight = 0.0

        for entry in dq:
            dec_val = _as_float(entry.get("decisions")) or 0.0
            act_val = _as_float(entry.get("act_now")) or 0.0
            turnover_val = _as_float(entry.get("turnover_usd")) or 0.0
            cap_val = _as_float(entry.get("cap_usd"))
            decisions_total += dec_val
            act_total += act_val
            turnover_total += turnover_val
            if cap_val is not None and cap_val > 0:
                symbol_val = entry.get("symbol")
                try:
                    symbol_key = str(symbol_val)
                except Exception:
                    symbol_key = None
                if symbol_key:
                    caps_by_symbol[symbol_key] = cap_val
            mode_value = entry.get("impact_mode")
            if mode_value:
                try:
                    mode_key = str(mode_value)
                except Exception:
                    mode_key = None
                if mode_key:
                    mode_counts[mode_key] = mode_counts.get(mode_key, 0) + int(dec_val)
            weight = _entry_weight(entry)
            realized_val = _as_float(entry.get("realized_slippage_bps"))
            if realized_val is not None and weight > 0:
                realized_sum += realized_val * weight
                realized_weight += weight
            modeled_val = _as_float(entry.get("modeled_cost_bps"))
            if modeled_val is not None and weight > 0:
                modeled_sum += modeled_val * weight
                modeled_weight += weight
            bias_val = _as_float(entry.get("cost_bias_bps"))
            if bias_val is None and realized_val is not None and modeled_val is not None:
                bias_val = realized_val - modeled_val
            if bias_val is not None and weight > 0:
                bias_sum += bias_val * weight
                bias_weight += weight

        cap_sum = float(sum(caps_by_symbol.values())) if caps_by_symbol else 0.0
        rate = float(act_total / decisions_total) if decisions_total > 0 else None
        ratio = float(turnover_total / cap_sum) if cap_sum > 0 else None
        realized_avg = float(realized_sum / realized_weight) if realized_weight > 0 else None
        modeled_avg = float(modeled_sum / modeled_weight) if modeled_weight > 0 else None
        bias_avg = (
            float(bias_sum / bias_weight) if bias_weight > 0 else (
                (realized_avg - modeled_avg)
                if realized_avg is not None and modeled_avg is not None
                else None
            )
        )

        def _maybe(value: float | None) -> float | None:
            if value is None:
                return None
            if not math.isfinite(value):
                return None
            return float(value)

        snapshot: Dict[str, Any] = {
            "decisions": int(decisions_total),
            "act_now": int(act_total),
            "act_now_rate": rate,
            "turnover_usd": float(turnover_total),
            "cap_usd": float(cap_sum) if cap_sum > 0 else None,
            "turnover_vs_cap": ratio,
            "impact_mode_counts": mode_counts,
        }
        snapshot["realized_slippage_bps"] = _maybe(realized_avg)
        snapshot["modeled_cost_bps"] = _maybe(modeled_avg)
        snapshot["cost_bias_bps"] = _maybe(bias_avg)
        return snapshot

    def _bar_execution_snapshot(self) -> Dict[str, Any]:
        cumulative_decisions = float(self._bar_totals.get("decisions", 0.0))
        cumulative_act = float(self._bar_totals.get("act_now", 0.0))
        cumulative_turnover = float(self._bar_totals.get("turnover_usd", 0.0))
        cumulative_cap = float(sum(self._bar_caps_by_symbol.values()))
        cumulative_rate = (
            float(cumulative_act / cumulative_decisions)
            if cumulative_decisions > 0
            else None
        )
        cumulative_ratio = (
            float(cumulative_turnover / cumulative_cap)
            if cumulative_cap > 0
            else None
        )
        realized_weight = float(self._bar_totals.get("realized_cost_weight", 0.0))
        modeled_weight = float(self._bar_totals.get("modeled_cost_weight", 0.0))
        realized_avg = (
            float(self._bar_totals.get("realized_cost_wsum", 0.0) / realized_weight)
            if realized_weight > 0
            else None
        )
        modeled_avg = (
            float(self._bar_totals.get("modeled_cost_wsum", 0.0) / modeled_weight)
            if modeled_weight > 0
            else None
        )
        bias_avg = (
            (realized_avg - modeled_avg)
            if realized_avg is not None and modeled_avg is not None
            else None
        )
        def _maybe(value: float | None) -> float | None:
            if value is None:
                return None
            if not math.isfinite(value):
                return None
            return float(value)
        return {
            "window_1m": self._bar_window_snapshot("1m"),
            "window_5m": self._bar_window_snapshot("5m"),
            "cumulative": {
                "decisions": int(cumulative_decisions),
                "act_now": int(cumulative_act),
                "act_now_rate": cumulative_rate,
                "turnover_usd": cumulative_turnover,
                "cap_usd": cumulative_cap if cumulative_cap > 0 else None,
                "turnover_vs_cap": cumulative_ratio,
                "impact_mode_counts": {
                    mode: int(count)
                    for mode, count in self._bar_mode_totals.items()
                    if count > 0
                },
                "realized_slippage_bps": _maybe(realized_avg),
                "modeled_cost_bps": _maybe(modeled_avg),
                "cost_bias_bps": _maybe(bias_avg),
            },
        }

    def record_bar_execution(
        self,
        symbol: str,
        *,
        decisions: int,
        act_now: int,
        turnover_usd: float,
        cap_usd: Optional[float] = None,
        impact_mode: Optional[str] = None,
        modeled_cost_bps: Optional[float] = None,
        realized_slippage_bps: Optional[float] = None,
        cost_bias_bps: Optional[float] = None,
        bar_ts: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            dec = max(0, int(decisions))
        except Exception:
            dec = 0
        if dec <= 0:
            return
        try:
            act = max(0, min(dec, int(act_now)))
        except Exception:
            act = 0
        try:
            turnover = float(turnover_usd)
        except Exception:
            turnover = 0.0
        else:
            if not math.isfinite(turnover) or turnover < 0:
                turnover = 0.0
        cap_value: Optional[float]
        try:
            cap_value = float(cap_usd) if cap_usd is not None else None
        except Exception:
            cap_value = None
        try:
            symbol_text = str(symbol)
        except Exception:
            symbol_text = ""
        ts_ms: int
        if bar_ts is not None:
            try:
                ts_candidate = int(bar_ts)
            except Exception:
                try:
                    ts_candidate = int(float(bar_ts))
                except Exception:
                    ts_candidate = int(time.time() * 1000)
            else:
                ts_candidate = int(ts_candidate)
            ts_ms = ts_candidate
        else:
            ts_ms = int(time.time() * 1000)
        self._execution_mode = "bar"
        mode_key: Optional[str]
        if impact_mode is None:
            mode_key = None
        else:
            try:
                mode_key = str(impact_mode)
            except Exception:
                mode_key = None
            if mode_key is not None:
                candidate = mode_key.strip()
                mode_key = candidate if candidate else None
        modeled_value: Optional[float]
        try:
            modeled_value = float(modeled_cost_bps) if modeled_cost_bps is not None else None
        except (TypeError, ValueError):
            modeled_value = None
        else:
            if modeled_value is not None and not math.isfinite(modeled_value):
                modeled_value = None
        realized_value: Optional[float]
        try:
            realized_value = (
                float(realized_slippage_bps)
                if realized_slippage_bps is not None
                else None
            )
        except (TypeError, ValueError):
            realized_value = None
        else:
            if realized_value is not None and not math.isfinite(realized_value):
                realized_value = None
        try:
            bias_value = (
                float(cost_bias_bps) if cost_bias_bps is not None else None
            )
        except (TypeError, ValueError):
            bias_value = None
        else:
            if bias_value is not None and not math.isfinite(bias_value):
                bias_value = None
        if bias_value is None and realized_value is not None and modeled_value is not None:
            bias_value = realized_value - modeled_value
        weight_value: Optional[float]
        if turnover > 0:
            weight_value = float(turnover)
        elif dec > 0:
            weight_value = float(dec)
        else:
            weight_value = None
        for window in self._window_ms:
            dq = self._bar_events.setdefault(window, deque())
            entry: Dict[str, Any] = {
                "ts": ts_ms,
                "symbol": symbol_text,
                "decisions": dec,
                "act_now": act,
                "turnover_usd": turnover,
                "cap_usd": cap_value if cap_value and cap_value > 0 else None,
                "impact_mode": mode_key,
            }
            if modeled_value is not None:
                entry["modeled_cost_bps"] = modeled_value
            if realized_value is not None:
                entry["realized_slippage_bps"] = realized_value
            if bias_value is not None:
                entry["cost_bias_bps"] = bias_value
            if weight_value is not None and weight_value > 0:
                entry["weight"] = float(weight_value)
            dq.append(entry)
            self._prune_bar_events(window, ts_ms)
        self._bar_totals["decisions"] += dec
        self._bar_totals["act_now"] += act
        self._bar_totals["turnover_usd"] += turnover
        if symbol_text:
            if cap_value is not None and cap_value > 0:
                self._bar_caps_by_symbol[symbol_text] = cap_value
            else:
                self._bar_caps_by_symbol.pop(symbol_text, None)
        if mode_key:
            self._bar_mode_totals[mode_key] += dec
        if weight_value is not None and weight_value > 0:
            if modeled_value is not None:
                self._bar_totals["modeled_cost_weight"] += weight_value
                self._bar_totals["modeled_cost_wsum"] += modeled_value * weight_value
            if realized_value is not None:
                self._bar_totals["realized_cost_weight"] += weight_value
                self._bar_totals["realized_cost_wsum"] += realized_value * weight_value

    def _build_metrics(self, now_ms: int, feed_lags: Dict[str, int], stale: list[str]) -> Dict[str, Any]:
        worst_feed = max(feed_lags.items(), key=lambda item: item[1], default=(None, 0))
        ws_snapshot = {
            "failures_1m": int(self._ws_counts["1m"].get("failure", 0)),
            "failures_5m": int(self._ws_counts["5m"].get("failure", 0)),
            "reconnects_1m": int(self._ws_counts["1m"].get("reconnect", 0)),
            "reconnects_5m": int(self._ws_counts["5m"].get("reconnect", 0)),
            "consecutive_failures": int(self._consecutive_ws_failures),
            "last_failure_ms": self.last_ws_failure_ms,
            "last_reconnect_ms": self.last_ws_reconnect_ms,
        }
        http_snapshot = {
            "window_1m": self._http_snapshot("1m"),
            "window_5m": self._http_snapshot("5m"),
        }
        signal_snapshot = {
            "window_1m": self._signal_snapshot("1m"),
            "window_5m": self._signal_snapshot("5m"),
            "zero_signal_symbols": sorted(
                [sym for sym, streak in self._zero_signal_streaks.items() if streak > 0]
            ),
            "zero_signal_streaks": dict(self.zero_signal_streaks),
        }
        metrics = {
            "ts_ms": now_ms,
            "feed_lag_ms": dict(feed_lags),
            "worst_feed_lag": {
                "symbol": worst_feed[0],
                "lag_ms": worst_feed[1],
            },
            "stale_symbols": stale,
            "fill_ratio": self.fill_ratio,
            "pnl": self.daily_pnl,
            "execution_mode": self._execution_mode,
            "bar_execution": self._bar_execution_snapshot() if self._execution_mode == "bar" else {},
            "ws": ws_snapshot,
            "http": http_snapshot,
            "signals": signal_snapshot,
            "throttle_queue": dict(self.throttle_queue_depth),
            "cooldowns_active": dict(self.cooldowns_active),
            "daily_turnover": self.daily_turnover,
        }
        return metrics

    # ------------------------------------------------------------------
    # Recording helpers
    def record_feed(self, symbol: str, close_ms: int) -> None:
        if not self.enabled:
            return
        try:
            sym = str(symbol)
        except Exception:
            return
        if not sym:
            return
        try:
            close = int(close_ms)
        except Exception:
            return
        self._last_bar_close_ms[sym] = close
        self._stale_alerted.discard(sym)
        try:
            now_ms = int(time.time() * 1000)
            report_feed_lag(sym, max(0, now_ms - close))
        except Exception:
            pass

    def record_stale(self, symbol: str, lag_ms: Optional[int] = None) -> None:
        """Record that ``symbol`` feed is stale for ``lag_ms`` milliseconds."""

        if not self.enabled:
            return
        try:
            sym = str(symbol)
        except Exception:
            return
        if not sym:
            return
        if lag_ms is None:
            close = self._last_bar_close_ms.get(sym)
            if close:
                lag = max(0, int(time.time() * 1000) - close)
            else:
                lag = 0
        else:
            try:
                lag = int(lag_ms)
            except Exception:
                lag = 0
        if lag <= 0:
            return
        threshold = self._stale_threshold_for(sym)
        if lag < threshold:
            return
        if sym not in self._stale_alerted:
            self._notify(
                f"feed_stale_{sym}",
                f"{sym} stale for {lag}ms exceeds {threshold}",
            )
        self._stale_alerted.add(sym)

    def record_ws(self, event: str, consecutive: Optional[int] = None) -> None:
        if not self.enabled:
            return
        ts_ms = int(time.time() * 1000)
        ev = str(event)
        global _last_ws_failure_ts_ms, _last_ws_reconnect_ts_ms
        if ev == "failure":
            self.last_ws_failure_ms = ts_ms
            _last_ws_failure_ts_ms = ts_ms
        elif ev == "reconnect":
            self.last_ws_reconnect_ms = ts_ms
            _last_ws_reconnect_ts_ms = ts_ms
        for window in self._window_ms:
            self._ws_events[window].append((ts_ms, ev))
            self._ws_counts[window][ev] += 1
            self._prune_ws_window(window, ts_ms)
        override: Optional[int] = None
        if consecutive is not None:
            try:
                override = int(consecutive)
            except Exception:
                override = None
            else:
                if override < 0:
                    override = 0
        if override is not None:
            self._consecutive_ws_failures = override
        elif ev == "failure":
            self._consecutive_ws_failures += 1
        elif ev == "reconnect":
            self._consecutive_ws_failures = 0

    def record_http_attempt(self) -> None:
        if not self.enabled:
            return
        ts_ms = int(time.time() * 1000)
        for window in self._window_ms:
            self._http_attempts[window].append(ts_ms)
            self._prune_http_attempts(window, ts_ms)

    def record_http(
        self,
        success: bool,
        status: Optional[Union[int, str]],
        *,
        timed_out: bool = False,
    ) -> None:
        if not self.enabled:
            return
        ts_ms = int(time.time() * 1000)
        classification = self._classify_http(success, status)
        if timed_out:
            classification = "timeout"
        for window in self._window_ms:
            self._http_events[window].append((ts_ms, bool(success), status, classification))
            counts = self._http_counts[window]
            counts["total"] += 1
            if success:
                counts["success"] += 1
            else:
                counts["error"] += 1
                key = classification if classification in counts else "other"
                counts[key] += 1
            self._prune_http_window(window, ts_ms)

    def record_signals(self, symbol: str, emitted: int, duplicates: int) -> None:
        if not self.enabled:
            return
        sym = str(symbol)
        em = int(emitted)
        du = int(duplicates)
        ts_ms = int(time.time() * 1000)
        if em > 0 or du > 0:
            for window in self._window_ms:
                self._signal_events[window].append((ts_ms, sym, em, du))
                if em:
                    self._signal_counts[window]["emitted"][sym] += em
                if du:
                    self._signal_counts[window]["duplicates"][sym] += du
                self._prune_signal_window(window, ts_ms)
        if em <= 0:
            self._zero_signal_streaks[sym] = self._zero_signal_streaks.get(sym, 0) + 1
        else:
            if sym in self._zero_signal_streaks:
                self._zero_signal_streaks.pop(sym, None)
            self._zero_signal_alerted.discard(sym)
        self._update_zero_signal_snapshot()

    def record_fill(self, requested: Optional[float], filled: Optional[float]) -> None:
        if not self.enabled:
            return
        if self._execution_mode == "bar":
            self.fill_ratio = None
            return
        if requested is None or filled is None:
            return
        try:
            req = float(requested)
            fil = float(filled)
        except (TypeError, ValueError):
            return
        if req <= 0:
            return
        ratio = fil / req
        self.fill_ratio = ratio
        threshold = float(getattr(self.thresholds, "fill_ratio_min", 0.0) or 0.0)
        if threshold > 0 and ratio < threshold:
            if not self._fill_alert_active:
                self._notify("fill_ratio", f"fill ratio {ratio:.3f} below {threshold}")
                self._fill_alert_active = True
        else:
            self._fill_alert_active = False

    def record_pnl(self, daily_pnl: Optional[float]) -> None:
        if not self.enabled:
            return
        if daily_pnl is None:
            return
        try:
            pnl = float(daily_pnl)
        except (TypeError, ValueError):
            return
        self.daily_pnl = pnl
        threshold = float(getattr(self.thresholds, "pnl_min", 0.0) or 0.0)
        if threshold > 0 and pnl < threshold:
            if not self._pnl_alert_active:
                self._notify("pnl", f"daily pnl {pnl:.2f} below {threshold}")
                self._pnl_alert_active = True
        else:
            self._pnl_alert_active = False

    # ------------------------------------------------------------------
    def tick(self, now_ms: int) -> None:
        if not self.enabled:
            return
        for window in self._window_ms:
            self._prune_ws_window(window, now_ms)
            self._prune_http_window(window, now_ms)
            self._prune_http_attempts(window, now_ms)
            self._prune_signal_window(window, now_ms)
            self._prune_bar_events(window, now_ms)

        th = self.thresholds

        feed_lags: Dict[str, int] = {
            sym: max(0, now_ms - close)
            for sym, close in self._last_bar_close_ms.items()
        }
        feed_threshold = float(getattr(th, "feed_lag_ms", 0.0) or 0.0)
        stale_symbols = sorted(
            [sym for sym, lag in feed_lags.items() if lag > self._stale_threshold_for(sym)]
        )

        if feed_threshold > 0:
            current_alerts: set[str] = set()
            for sym, lag in feed_lags.items():
                if lag > feed_threshold:
                    current_alerts.add(sym)
                    if sym not in self._feed_alerted:
                        self._notify(
                            f"feed_lag_{sym}",
                            f"{sym} feed lag {lag}ms exceeds {feed_threshold}",
                        )
                        self._feed_alerted.add(sym)
                elif sym in self._feed_alerted:
                    self._feed_alerted.discard(sym)
            for sym in list(self._feed_alerted):
                if sym not in current_alerts:
                    self._feed_alerted.discard(sym)
        else:
            self._feed_alerted.clear()

        for sym in stale_symbols:
            if sym not in self._stale_alerted:
                threshold = self._stale_threshold_for(sym)
                self._notify(
                    f"feed_stale_{sym}",
                    f"{sym} stale for {feed_lags.get(sym, 0)}ms exceeds {threshold}",
                )
                self._stale_alerted.add(sym)
        for sym in list(self._stale_alerted):
            if sym not in stale_symbols:
                self._stale_alerted.discard(sym)

        ws_fail_1m = int(self._ws_counts["1m"].get("failure", 0))
        ws_fail_5m = int(self._ws_counts["5m"].get("failure", 0))
        ws_rec_1m = int(self._ws_counts["1m"].get("reconnect", 0))
        ws_rec_5m = int(self._ws_counts["5m"].get("reconnect", 0))
        ws_threshold = float(getattr(th, "ws_failures", 0.0) or 0.0)
        triggered_ws = ws_threshold > 0 and (
            ws_fail_1m > ws_threshold or self._consecutive_ws_failures >= ws_threshold
        )
        if triggered_ws:
            if not self._ws_alert_active:
                self._notify(
                    "ws_failures",
                    (
                        f"websocket failures last minute: {ws_fail_1m} "
                        f"consecutive={self._consecutive_ws_failures}"
                    ),
                )
                self._ws_alert_active = True
        else:
            self._ws_alert_active = False

        error_threshold = float(getattr(th, "error_rate", 0.0) or 0.0)
        if error_threshold > 0:
            emitted = self._signal_counts["1m"]["emitted"]
            duplicates = self._signal_counts["1m"]["duplicates"]
            active: set[str] = set()
            for sym in set(emitted) | set(duplicates):
                em = emitted.get(sym, 0)
                du = duplicates.get(sym, 0)
                rate = float(du) / float(em) if em > 0 else (1.0 if du > 0 else 0.0)
                if rate > error_threshold:
                    active.add(sym)
                    if sym not in self._signal_alerted:
                        self._notify(
                            "signal_error_rate",
                            f"{sym} duplicate rate {rate:.3f} exceeds {error_threshold}",
                        )
                        self._signal_alerted.add(sym)
                elif sym in self._signal_alerted:
                    self._signal_alerted.discard(sym)
            for sym in list(self._signal_alerted):
                if sym not in active:
                    self._signal_alerted.discard(sym)
        else:
            self._signal_alerted.clear()

        cost_threshold = float(getattr(th, "cost_bias_bps", 0.0) or 0.0)
        if cost_threshold > 0:
            events = self._bar_events.get("5m", deque())
            symbol_bias: Dict[str, Dict[str, float]] = {}

            def _as_float(value: Any) -> float | None:
                try:
                    result = float(value)
                except (TypeError, ValueError):
                    return None
                if not math.isfinite(result):
                    return None
                return result

            for entry in events:
                weight = 0.0
                for key in ("weight", "turnover_usd", "decisions"):
                    candidate = _as_float(entry.get(key))
                    if candidate is not None and candidate > 0:
                        weight = candidate
                        break
                if weight <= 0:
                    continue
                bias_val = _as_float(entry.get("cost_bias_bps"))
                if bias_val is None:
                    realized_val = _as_float(entry.get("realized_slippage_bps"))
                    modeled_val = _as_float(entry.get("modeled_cost_bps"))
                    if realized_val is None or modeled_val is None:
                        continue
                    bias_val = realized_val - modeled_val
                if bias_val is None or not math.isfinite(bias_val):
                    continue
                sym_raw = entry.get("symbol")
                try:
                    sym = str(sym_raw) if sym_raw is not None else ""
                except Exception:
                    sym = ""
                sym = sym.strip().upper() or "UNKNOWN"
                stats = symbol_bias.setdefault(sym, {"sum": 0.0, "weight": 0.0})
                stats["sum"] += float(bias_val) * weight
                stats["weight"] += weight

            active_bias: set[str] = set()
            for sym, data in symbol_bias.items():
                weight_total = data.get("weight", 0.0)
                if weight_total <= 0:
                    continue
                avg_bias = data["sum"] / weight_total
                if abs(avg_bias) > cost_threshold:
                    active_bias.add(sym)
                    if sym not in self._cost_bias_alerted:
                        self._notify(
                            f"cost_bias_{sym}",
                            f"{sym} cost bias {avg_bias:.3f}bps exceeds {cost_threshold:.3f}",
                        )
                        self._cost_bias_alerted.add(sym)
                elif sym in self._cost_bias_alerted:
                    self._cost_bias_alerted.discard(sym)
            for sym in list(self._cost_bias_alerted):
                if sym not in active_bias:
                    self._cost_bias_alerted.discard(sym)
        else:
            self._cost_bias_alerted.clear()

        zero_threshold = int(getattr(th, "zero_signals", 0) or 0)
        if zero_threshold > 0:
            for sym, streak in list(self._zero_signal_streaks.items()):
                if streak >= zero_threshold:
                    if sym not in self._zero_signal_alerted:
                        self._notify(
                            "zero_signals",
                            f"{sym} {streak} consecutive zero-signal bars",
                        )
                        self._zero_signal_alerted.add(sym)
                else:
                    self._zero_signal_alerted.discard(sym)
                if streak <= 0:
                    self._zero_signal_streaks.pop(sym, None)
        else:
            self._zero_signal_alerted.clear()
            self._zero_signal_streaks = {
                sym: streak for sym, streak in self._zero_signal_streaks.items() if streak > 0
            }
        self._update_zero_signal_snapshot()

        now_sec = now_ms / 1000.0
        if now_sec - self._last_flush_ts >= self.flush_interval_sec:
            metrics = self._build_metrics(now_ms, feed_lags, stale_symbols)
            line = json.dumps(metrics, sort_keys=True)
            try:
                atomic_write_with_retry(
                    self._metrics_path,
                    line + "\n",
                    retries=3,
                    backoff=0.1,
                    mode="a",
                )
            except Exception:
                pass
            self._last_flush_ts = now_sec

    def flush(self) -> None:
        if not self.enabled:
            return
        try:
            self._last_flush_ts = 0.0
            self.tick(int(time.time() * 1000))
        except Exception:
            pass


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
    zero_streaks = dict(_zero_signal_streaks_snapshot)
    worst_zero = max(zero_streaks.items(), key=lambda x: x[1], default=(None, 0))
    queue_depth = dict(_throttle_queue_depth_snapshot)
    cooldowns = dict(_cooldowns_active_snapshot)

    summary = {
        "worst_feed_lag": {"symbol": worst_feed[0], "feed_lag_ms": worst_feed[1]},
        "worst_ws_failures": {"symbol": worst_ws[0], "failures": worst_ws[1]},
        "worst_error_rate": {"symbol": worst_err[0], "error_rate": worst_err[1]},
        "queue_depth": queue_depth,
        "cooldowns_active": cooldowns,
        "last_ws_failure_ms": _last_ws_failure_ts_ms or None,
        "last_ws_reconnect_ms": _last_ws_reconnect_ts_ms or None,
        "worst_zero_signal": {
            "symbol": worst_zero[0],
            "streak": worst_zero[1],
        },
        "zero_signal_streaks": zero_streaks,
    }

    json_str = json.dumps(summary, sort_keys=True)
    csv_lines = ["metric,symbol,value"]
    csv_lines.append(f"worst_feed_lag,{worst_feed[0] or ''},{worst_feed[1]}")
    csv_lines.append(f"worst_ws_failures,{worst_ws[0] or ''},{worst_ws[1]}")
    csv_lines.append(f"worst_error_rate,{worst_err[0] or ''},{worst_err[1]}")
    csv_lines.append(f"worst_zero_signal,{worst_zero[0] or ''},{worst_zero[1]}")
    csv_lines.append(f"queue_size,,{queue_depth.get('size', 0)}")
    csv_lines.append(f"queue_max,,{queue_depth.get('max', 0)}")
    csv_lines.append(f"cooldowns_total,,{cooldowns.get('count', 0)}")
    csv_lines.append(
        f"cooldowns_global,,{1 if cooldowns.get('global') else 0}"
    )
    csv_lines.append(f"last_ws_failure_ms,,{_last_ws_failure_ts_ms}")
    csv_lines.append(f"last_ws_reconnect_ms,,{_last_ws_reconnect_ts_ms}")
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
    "zero_signals_alert_count",
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
    "record_signals",
    "get_runtime_aggregator",
    "set_runtime_aggregator",
    "clear_runtime_aggregator",
    "alert_zero_signals",
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
