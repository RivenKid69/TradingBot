"""Utilities for caching latency volatility statistics.

This module exposes :class:`LatencyVolatilityCache` which maintains a
per-symbol rolling window of latency adjustment observations.  The cache is
responsible for collecting observations via
``update_latency_factor(symbol, ts_ms, value)`` and for returning a
volatility multiplier through :meth:`LatencyVolatilityCache.latency_multiplier`.

The implementation is intentionally thread-safe because the latency model is
often accessed from multiple worker threads.  Each symbol keeps a snapshot of
its recent values together with summary statistics which are used to compute
the mean, standard deviation and z-score of the most recent value.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import threading
from typing import Deque, Dict, Optional, Tuple


@dataclass
class _SymbolState:
    """Mutable statistics for a particular trading symbol.

    Attributes
    ----------
    values:
        A deque with the latest observed metric values.  Only the most recent
        ``window`` values are retained.
    sum:
        The running sum of ``values``.
    sum_sq:
        The running sum of squared ``values``.
    count:
        Number of retained observations for the symbol.
    last_ts:
        Timestamp in milliseconds of the latest update.
    """

    values: Deque[float] = field(default_factory=deque)
    sum: float = 0.0
    sum_sq: float = 0.0
    count: int = 0
    last_ts: Optional[int] = None


class LatencyVolatilityCache:
    """Cache used to derive latency multipliers from volatility metrics.

    Parameters
    ----------
    window:
        Maximum number of observations stored per symbol.  The value is
        normalised to be at least two in order to avoid degenerate variance
        calculations.
    min_ready:
        Optional minimum number of observations required before the cache is
        considered *ready*.  If omitted the window size is used.  The effective
        threshold never drops below two and never exceeds the configured
        ``window``.
    """

    __slots__ = ("_window", "_min_ready", "_lock", "_states")

    def __init__(self, window: int, min_ready: Optional[int] | None = None) -> None:
        window_int = int(window)
        if window_int < 2:
            window_int = 2
        self._window = window_int

        if min_ready is None:
            ready = self._window
        else:
            try:
                ready = int(min_ready)
            except (TypeError, ValueError):
                ready = self._window
        if ready < 2:
            ready = 2
        if ready > self._window:
            ready = self._window
        self._min_ready = ready

        self._lock = threading.Lock()
        self._states: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # Updating
    # ------------------------------------------------------------------
    def update_latency_factor(self, *, symbol: str, ts_ms: int, value: float) -> None:
        """Record a new volatility observation for ``symbol``.

        Parameters
        ----------
        symbol:
            Trading pair identifier.  The value is normalised to upper case.
        ts_ms:
            Timestamp of the observation in milliseconds.
        value:
            Numeric metric value to be aggregated.  Non-finite numbers are
            silently ignored.
        """

        try:
            sym = str(symbol).upper()
        except Exception:
            return

        try:
            ts = int(ts_ms)
        except (TypeError, ValueError):
            return

        try:
            val = float(value)
        except (TypeError, ValueError):
            return

        if not math.isfinite(val):
            return

        with self._lock:
            state = self._states.get(sym)
            if state is None:
                state = _SymbolState()
                self._states[sym] = state

            state.values.append(val)
            state.sum += val
            state.sum_sq += val * val
            state.count += 1
            state.last_ts = ts

            while state.count > self._window:
                old = state.values.popleft()
                state.sum -= old
                state.sum_sq -= old * old
                state.count -= 1

            # Guard against numerical drift and external mutations.
            if state.count != len(state.values):
                state.count = len(state.values)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def _normalize_min_count(self, min_count: Optional[int]) -> int:
        if min_count is None:
            threshold = self._min_ready
        else:
            try:
                threshold = int(min_count)
            except (TypeError, ValueError):
                threshold = self._min_ready
        if threshold < 2:
            threshold = 2
        if threshold > self._window:
            threshold = self._window
        return threshold

    def is_ready(
        self, symbol: Optional[str] = None, *, min_count: Optional[int] = None
    ) -> bool:
        """Return ``True`` if enough observations are available for ``symbol``.

        The ``symbol`` argument should correspond to a trading pair identifier.
        The cache needs at least ``min_count`` observations (or ``min_ready``
        configured during initialisation).  When ``symbol`` is missing the
        method conservatively returns ``False``.
        """

        if not symbol:
            return False
        sym = str(symbol).upper()
        threshold = self._normalize_min_count(min_count)
        with self._lock:
            state = self._states.get(sym)
            if state is None:
                return False
            return state.count >= threshold

    @property
    def ready(self) -> bool:
        """Return whether *any* tracked symbol has a ready window."""

        with self._lock:
            for state in self._states.values():
                if state.count >= self._min_ready:
                    return True
        return False

    def snapshot(self, symbol: str) -> Optional[Dict[str, object]]:
        """Return a shallow copy of cached statistics for ``symbol``.

        The snapshot is primarily intended for diagnostic logging and should
        not be mutated by the caller.
        """

        if not symbol:
            return None
        sym = str(symbol).upper()
        with self._lock:
            state = self._states.get(sym)
            if state is None:
                return None
            values_list = list(state.values)
            count = state.count
            mean = state.sum / count if count else 0.0
            variance = state.sum_sq / count - mean * mean if count else 0.0
            variance = max(variance, 0.0)
            std = math.sqrt(variance)
            return {
                "symbol": sym,
                "values": values_list,
                "sum": state.sum,
                "sum_sq": state.sum_sq,
                "count": count,
                "last_ts": state.last_ts,
                "mean": mean,
                "std": std,
            }

    def latency_multiplier(
        self,
        *,
        symbol: str,
        ts_ms: int,
        metric: str,
        window: int,
        gamma: float,
        clip: float,
    ) -> Tuple[float, Dict[str, object]]:
        """Return a volatility multiplier for ``symbol``.

        The method requires the cache to be ready for ``symbol``.  When the
        rolling window is not warm yet a multiplier of ``1.0`` is returned
        alongside a diagnostic payload explaining the reason.
        """

        try:
            gamma_value = float(gamma)
        except (TypeError, ValueError):
            gamma_value = 0.0

        try:
            clip_value = float(clip)
        except (TypeError, ValueError):
            clip_value = 0.0

        try:
            ts = int(ts_ms)
        except (TypeError, ValueError):
            debug = {
                "symbol": str(symbol).upper() if symbol else symbol,
                "metric": metric,
                "configured_window": self._window,
                "gamma": gamma_value,
                "clip": clip_value,
                "ts_ms": ts_ms,
            }
            debug["reason"] = "invalid_ts"
            return 1.0, debug

        try:
            requested_window = int(window)
        except (TypeError, ValueError):
            requested_window = self._window

        debug: Dict[str, object] = {
            "symbol": str(symbol).upper() if symbol else symbol,
            "metric": metric,
            "requested_window": requested_window,
            "configured_window": self._window,
            "gamma": gamma_value,
            "clip": clip_value,
            "ts_ms": ts,
        }

        sym = debug["symbol"]
        if sym is None:
            debug["reason"] = "missing_symbol"
            return 1.0, debug

        with self._lock:
            state = self._states.get(sym)
            required = self._normalize_min_count(None)
            if state is None:
                debug["reason"] = "not_ready"
                debug["count"] = 0
                debug["required"] = required
                return 1.0, debug

            count = state.count
            if count < required:
                debug["reason"] = "not_ready"
                debug["count"] = count
                debug["required"] = required
                return 1.0, debug

            last_value = state.values[-1]
            mean = state.sum / count if count else last_value
            variance = state.sum_sq / count - mean * mean if count else 0.0
            variance = max(variance, 0.0)
            std = math.sqrt(variance)
            last_ts = state.last_ts

        if not math.isfinite(clip_value) or clip_value < 0.0:
            clip_value = 0.0

        if std <= 0.0 or not math.isfinite(std):
            z_raw = 0.0
        else:
            z_raw = (last_value - mean) / std

        if clip_value > 0.0:
            zscore = max(-clip_value, min(clip_value, z_raw))
        else:
            zscore = 0.0 if clip_value == 0.0 else z_raw

        vol_mult = max(0.0, 1.0 + gamma_value * zscore)

        debug.update(
            {
                "count": count,
                "required": required,
                "last_value": last_value,
                "mean": mean,
                "std": std,
                "zscore_raw": z_raw,
                "zscore": zscore,
                "vol_mult": vol_mult,
                "last_ts": last_ts,
            }
        )

        return vol_mult, debug


__all__ = ["LatencyVolatilityCache"]

