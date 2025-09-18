"""Unified feature pipe for online and offline processing.

This module provides :class:`FeaturePipe` which implements the
``core_contracts.FeaturePipe`` protocol and exposes additional offline
helpers used during dataset preparation.  The class wraps
``OnlineFeatureTransformer`` for streaming feature computation and reuses the
deterministic ``apply_offline_features`` routine for batch processing.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from decimal import InvalidOperation, Decimal
from math import sqrt, isfinite
from typing import Iterable, Mapping, Optional, Any, Deque, Dict, Union
import copy

import pandas as pd

from transformers import FeatureSpec, OnlineFeatureTransformer, apply_offline_features
from core_models import Bar
from core_contracts import FeaturePipe as FeaturePipeProtocol  # noqa: F401


@dataclass(frozen=True)
class SignalQualitySnapshot:
    """Current quality metrics produced by :class:`SignalQualityMetrics`."""

    current_sigma: Optional[float]
    current_vol_median: Optional[float]
    window_ready: bool


@dataclass
class _SymbolMetricsState:
    prev_close: Optional[float] = None
    returns: Deque[float] = field(default_factory=deque)
    volumes: Deque[float] = field(default_factory=deque)
    sum_returns: float = 0.0
    sum_sq_returns: float = 0.0


@dataclass
class _FeatureStatsState:
    prev_close: Optional[float] = None
    returns: Deque[float] = field(default_factory=deque)
    tranges: Deque[float] = field(default_factory=deque)
    bar_count: int = 0
    ret_last: Optional[float] = None
    sigma: Optional[float] = None
    atr_pct: Optional[float] = None
    spread_bps: Optional[float] = None
    spread_ts_ms: Optional[int] = None
    spread_valid_until_ms: Optional[int] = None
    last_bar_ts: Optional[int] = None
    tr_sum: float = 0.0


@dataclass(frozen=True)
class MarketMetricsSnapshot:
    ret_last: Optional[float]
    sigma: Optional[float]
    atr_pct: Optional[float]
    spread_bps: Optional[float]
    window_ready: bool
    bar_count: int
    last_bar_ts: Optional[int]
    spread_ts: Optional[int]
    spread_valid_until: Optional[int]


class SignalQualityMetrics:
    """Maintain rolling statistics for bar quality checks.

    Parameters
    ----------
    sigma_window:
        Window size used to compute the standard deviation of returns.
    vol_median_window:
        Window size used to compute the rolling median of volumes.
    """

    def __init__(self, sigma_window: int, vol_median_window: int) -> None:
        if sigma_window <= 0:
            raise ValueError("sigma_window must be positive")
        if vol_median_window <= 0:
            raise ValueError("vol_median_window must be positive")

        self.sigma_window = int(sigma_window)
        self.vol_median_window = int(vol_median_window)
        self._state: Dict[str, _SymbolMetricsState] = {}
        self._latest: Dict[str, SignalQualitySnapshot] = {}

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for all symbols."""

        self._state.clear()
        self._latest.clear()

    def reset_symbol(self, symbol: str) -> None:
        """Reset state for a specific ``symbol`` if it exists."""

        sym = str(symbol).upper()
        self._state.pop(sym, None)
        self._latest.pop(sym, None)

    # ------------------------------------------------------------------
    def update(self, symbol: str, bar: Bar) -> SignalQualitySnapshot:
        """Update metrics with a new bar and return current values."""

        sym = str(symbol).upper()
        state = self._state.setdefault(sym, _SymbolMetricsState())

        try:
            close = float(bar.close)
        except (TypeError, ValueError, InvalidOperation):
            return self._snapshot(sym, state)

        if not isfinite(close):
            return self._snapshot(sym, state)

        if state.prev_close not in (None, 0.0):
            ret = close / state.prev_close - 1.0
            self._append_return(state, ret)

        state.prev_close = close

        volume_source = bar.volume_quote if bar.volume_quote is not None else bar.volume_base
        if volume_source is not None:
            try:
                volume_value = float(volume_source)
            except (TypeError, ValueError, InvalidOperation):
                volume_value = None
            if volume_value is not None and isfinite(volume_value):
                self._append_volume(state, float(volume_value))

        snapshot = self._snapshot(sym, state)
        self._latest[sym] = snapshot
        return snapshot

    # ------------------------------------------------------------------
    def _append_return(self, state: _SymbolMetricsState, value: float) -> None:
        if len(state.returns) == self.sigma_window:
            removed = state.returns.popleft()
            state.sum_returns -= removed
            state.sum_sq_returns -= removed * removed

        state.returns.append(value)
        state.sum_returns += value
        state.sum_sq_returns += value * value

    def _append_volume(self, state: _SymbolMetricsState, value: float) -> None:
        if len(state.volumes) == self.vol_median_window:
            state.volumes.popleft()

        state.volumes.append(value)

    def _compute_sigma(self, state: _SymbolMetricsState) -> Optional[float]:
        count = len(state.returns)
        if count == 0:
            return None

        mean = state.sum_returns / count
        variance = state.sum_sq_returns / count - mean * mean
        if variance < 0.0:
            variance = 0.0
        return sqrt(variance)

    def _compute_volume_median(self, state: _SymbolMetricsState) -> Optional[float]:
        if not state.volumes:
            return None

        sorted_volumes = sorted(state.volumes)
        mid = len(sorted_volumes) // 2
        if len(sorted_volumes) % 2:
            return sorted_volumes[mid]
        return 0.5 * (sorted_volumes[mid - 1] + sorted_volumes[mid])

    def _snapshot(self, symbol: str, state: _SymbolMetricsState) -> SignalQualitySnapshot:
        sigma = self._compute_sigma(state)
        vol_median = self._compute_volume_median(state)
        window_ready = (
            len(state.returns) >= self.sigma_window
            and len(state.volumes) >= self.vol_median_window
        )
        snapshot = SignalQualitySnapshot(
            current_sigma=sigma,
            current_vol_median=vol_median,
            window_ready=window_ready,
        )
        self._latest[symbol] = snapshot
        return snapshot

    @property
    def latest(self) -> Mapping[str, SignalQualitySnapshot]:
        """Expose latest computed snapshots for all symbols."""

        return dict(self._latest)




@dataclass
class FeaturePipe:
    """Feature pipe with both streaming and offline capabilities.

    Parameters
    ----------
    spec:
        Configuration for the feature calculations.
    price_col:
        Column name used for prices in offline dataframes.
    label_col:
        Optional name of an existing label column.  If provided, the
        :meth:`make_targets` method will simply return this column when it is
        present in the dataframe.
    """

    spec: FeatureSpec
    price_col: str = "price"
    label_col: Optional[str] = None
    metrics: Optional[SignalQualityMetrics] = None
    sigma_window: int = 120
    min_sigma_periods: int = 2
    spread_ttl_ms: int = 60_000
    signal_quality: Dict[str, SignalQualitySnapshot] = field(
        init=False, default_factory=dict
    )
    market_state: Dict[str, MarketMetricsSnapshot] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        # Initialize transformer for online mode.
        self._tr = OnlineFeatureTransformer(self.spec)
        self._read_only = False
        self._sigma_window = max(1, int(self.sigma_window))
        self._min_sigma_periods = max(2, int(self.min_sigma_periods))
        if self._min_sigma_periods > self._sigma_window:
            self._min_sigma_periods = self._sigma_window
        self._spread_ttl_ms = max(0, int(self.spread_ttl_ms))
        self._symbol_state: Dict[str, _FeatureStatsState] = {}

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state of the transformer."""
        self._tr = OnlineFeatureTransformer(self.spec)
        self.signal_quality.clear()
        self.market_state.clear()
        self._symbol_state.clear()
        if self.metrics is not None:
            try:
                self.metrics.reset()
            except Exception:
                pass

    def set_read_only(self, flag: bool = True) -> None:
        """Enable or disable read-only mode for normalization stats."""
        self._read_only = bool(flag)

    def warmup(self, bars: Iterable[Bar] = ()) -> None:
        """Warm up transformer with a sequence of historical bars."""
        self.reset()
        for b in bars:
            self.update(b)

    def update(self, bar: Bar, *, skip_metrics: bool = False) -> Mapping[str, Any]:
        """Process a single bar and return computed features."""
        try:
            close_value = float(bar.close)
        except (TypeError, ValueError, InvalidOperation):
            return {}

        if not isfinite(close_value):
            return {}

        symbol = bar.symbol.upper()
        ts_ms = int(bar.ts)
        state = self._get_symbol_state(symbol)
        self._update_market_metrics(symbol, state, close_value, ts_ms, bar)

        if not self._read_only:
            feats = self._tr.update(
                symbol=symbol,
                ts_ms=ts_ms,
                close=close_value,
            )
        else:
            state_backup = copy.deepcopy(self._tr._state)
            feats = self._tr.update(
                symbol=symbol,
                ts_ms=ts_ms,
                close=close_value,
            )
            self._tr._state = state_backup

        if self.metrics is not None and not skip_metrics:
            try:
                snapshot = self.metrics.update(symbol, bar)
            except Exception:
                snapshot = None
            if snapshot is not None:
                self.signal_quality[symbol] = snapshot

        return feats

    # ``core_contracts.FeaturePipe`` historically exposes ``on_bar``.  Keep
    # an alias for backward compatibility with existing call sites.
    on_bar = update

    # ------------------------------------------------------------------
    # Market metrics helpers
    # ------------------------------------------------------------------
    def get_market_metrics(self, symbol: str) -> Optional[MarketMetricsSnapshot]:
        sym = str(symbol).upper()
        return self.market_state.get(sym)

    def record_spread(
        self,
        symbol: str,
        *,
        spread_bps: Optional[Union[float, int, Decimal]] = None,
        bid: Optional[Union[float, int, Decimal]] = None,
        ask: Optional[Union[float, int, Decimal]] = None,
        ts_ms: Optional[int] = None,
        high: Optional[Union[float, int, Decimal]] = None,
        low: Optional[Union[float, int, Decimal]] = None,
        mid: Optional[Union[float, int, Decimal]] = None,
        ttl_ms: Optional[int] = None,
    ) -> None:
        sym = str(symbol).upper()
        if not sym:
            return
        state = self._get_symbol_state(sym)
        spread_value = self._coerce_float(spread_bps)
        if spread_value is None:
            spread_value = self._compute_spread_from_quotes(bid, ask)
        if spread_value is None:
            spread_value = self._compute_spread_from_range(high, low, mid)
        if spread_value is None or not isfinite(spread_value) or spread_value <= 0:
            return
        state.spread_bps = spread_value
        if ts_ms is not None:
            try:
                state.spread_ts_ms = int(ts_ms)
            except Exception:
                pass
        elif state.last_bar_ts is not None:
            state.spread_ts_ms = state.last_bar_ts
        ttl = self._spread_ttl_ms if ttl_ms is None else max(0, int(ttl_ms))
        if ttl > 0 and state.spread_ts_ms is not None:
            state.spread_valid_until_ms = state.spread_ts_ms + ttl
        else:
            state.spread_valid_until_ms = None
        self._publish_market_snapshot(sym, state, state.last_bar_ts)

    def _get_symbol_state(self, symbol: str) -> _FeatureStatsState:
        sym = str(symbol).upper()
        state = self._symbol_state.get(sym)
        if state is None:
            state = _FeatureStatsState(
                returns=deque(maxlen=self._sigma_window),
                tranges=deque(maxlen=self._sigma_window),
            )
            self._symbol_state[sym] = state
        else:
            if state.returns.maxlen != self._sigma_window:
                state.returns = deque(state.returns, maxlen=self._sigma_window)
            if state.tranges.maxlen != self._sigma_window:
                state.tranges = deque(state.tranges, maxlen=self._sigma_window)
            state.tr_sum = sum(state.tranges)
        return state

    def _update_market_metrics(
        self,
        symbol: str,
        state: _FeatureStatsState,
        close_value: float,
        ts_ms: int,
        bar: Bar,
    ) -> None:
        state.last_bar_ts = ts_ms
        state.bar_count += 1
        ret_last: Optional[float] = None
        prev = state.prev_close
        if prev not in (None, 0.0):
            if prev is not None and prev != 0.0:
                try:
                    ret_last = close_value / prev - 1.0
                except Exception:
                    ret_last = None
        if ret_last is not None and isfinite(ret_last):
            state.ret_last = ret_last
            state.returns.append(ret_last)
        atr_candidate: Optional[float] = None
        high_val = self._coerce_float(getattr(bar, "high", None))
        low_val = self._coerce_float(getattr(bar, "low", None))
        if (
            prev not in (None, 0.0)
            and high_val is not None
            and low_val is not None
        ):
            try:
                prev_close_val = float(prev) if prev is not None else None
                hi = float(high_val)
                lo = float(low_val)
            except (TypeError, ValueError):
                prev_close_val = None
                hi = lo = 0.0
            if (
                prev_close_val is not None
                and prev_close_val != 0.0
                and isfinite(prev_close_val)
                and isfinite(hi)
                and isfinite(lo)
            ):
                tr = max(hi - lo, abs(hi - prev_close_val), abs(lo - prev_close_val))
                if tr >= 0.0 and isfinite(tr):
                    atr_candidate = max(0.0, tr / abs(prev_close_val))
        if atr_candidate is not None and isfinite(atr_candidate):
            dq = state.tranges
            if dq.maxlen != self._sigma_window:
                dq = deque(dq, maxlen=self._sigma_window)
                state.tranges = dq
            if dq.maxlen is not None and len(dq) == dq.maxlen:
                removed = dq.popleft()
                state.tr_sum -= removed
            dq.append(atr_candidate)
            state.tr_sum += atr_candidate
        if state.tranges:
            count = len(state.tranges)
            if count > 0:
                state.atr_pct = max(0.0, state.tr_sum / count)
            else:
                state.atr_pct = None
        state.prev_close = close_value
        state.sigma = self._compute_sigma(state.returns)
        self._resolve_spread(symbol, state, bar, ts_ms)
        self._publish_market_snapshot(symbol, state, ts_ms)

    def _publish_market_snapshot(
        self,
        symbol: str,
        state: _FeatureStatsState,
        ts_ms: Optional[int],
    ) -> None:
        ret_last = state.ret_last if state.ret_last is not None and isfinite(state.ret_last) else None
        sigma = state.sigma if state.sigma is not None and isfinite(state.sigma) else None
        atr_pct = (
            state.atr_pct
            if state.atr_pct is not None
            and isfinite(state.atr_pct)
            and state.atr_pct >= 0.0
            else None
        )
        spread = (
            state.spread_bps
            if state.spread_bps is not None
            and isfinite(state.spread_bps)
            and state.spread_bps > 0
            else None
        )
        ready = state.bar_count >= self._sigma_window and sigma is not None
        state.last_bar_ts = ts_ms if ts_ms is not None else state.last_bar_ts
        spread_valid_until = state.spread_valid_until_ms
        if spread_valid_until is not None:
            try:
                spread_valid_until = int(spread_valid_until)
            except Exception:
                spread_valid_until = None
        snapshot = MarketMetricsSnapshot(
            ret_last=ret_last,
            sigma=sigma,
            atr_pct=atr_pct,
            spread_bps=spread,
            window_ready=ready,
            bar_count=state.bar_count,
            last_bar_ts=state.last_bar_ts,
            spread_ts=state.spread_ts_ms,
            spread_valid_until=spread_valid_until,
        )
        self.market_state[symbol] = snapshot

    def _compute_sigma(self, returns: Deque[float]) -> Optional[float]:
        if not returns:
            return None
        finite = [r for r in returns if isfinite(r)]
        if len(finite) < max(2, self._min_sigma_periods):
            return None
        mean = sum(finite) / len(finite)
        variance = sum((r - mean) ** 2 for r in finite)
        if len(finite) <= 1:
            return None
        var = variance / (len(finite) - 1)
        if var < 0.0:
            var = 0.0
        return sqrt(var)

    def _resolve_spread(
        self,
        symbol: str,
        state: _FeatureStatsState,
        bar: Bar,
        ts_ms: int,
    ) -> Optional[float]:
        spread = None
        if state.spread_bps is not None:
            if self._spread_ttl_ms <= 0 or state.spread_valid_until_ms is None:
                spread = state.spread_bps
            else:
                if ts_ms <= state.spread_valid_until_ms:
                    spread = state.spread_bps
                else:
                    state.spread_bps = None
                    state.spread_ts_ms = None
                    state.spread_valid_until_ms = None
        if spread is None:
            spread = self._extract_spread_from_bar(bar)
            if spread is not None and isfinite(spread) and spread > 0:
                state.spread_bps = spread
                state.spread_ts_ms = ts_ms
                if self._spread_ttl_ms > 0:
                    state.spread_valid_until_ms = ts_ms + self._spread_ttl_ms
                else:
                    state.spread_valid_until_ms = None
        return spread

    def _extract_spread_from_bar(self, bar: Bar) -> Optional[float]:
        for attr, multiplier in (("spread_bps", 1.0), ("half_spread_bps", 2.0)):
            raw = getattr(bar, attr, None)
            val = self._coerce_float(raw)
            if val is not None and isfinite(val):
                spread = val * multiplier
                if spread > 0:
                    return spread
        bid = getattr(bar, "bid", None)
        ask = getattr(bar, "ask", None)
        if bid is None or ask is None:
            bid = getattr(bar, "bid_price", None)
            ask = getattr(bar, "ask_price", None)
        spread = self._compute_spread_from_quotes(bid, ask)
        if spread is not None:
            return spread
        high = getattr(bar, "high", None)
        low = getattr(bar, "low", None)
        mid = None
        try:
            if high is not None and low is not None:
                high_val = float(high)
                low_val = float(low)
                if isfinite(high_val) and isfinite(low_val):
                    mid = (high_val + low_val) * 0.5
        except Exception:
            mid = None
        return self._compute_spread_from_range(high, low, mid)

    def _compute_spread_from_quotes(
        self,
        bid: Optional[Union[float, int, Decimal]],
        ask: Optional[Union[float, int, Decimal]],
    ) -> Optional[float]:
        bid_val = self._coerce_float(bid)
        ask_val = self._coerce_float(ask)
        if (
            bid_val is None
            or ask_val is None
            or not isfinite(bid_val)
            or not isfinite(ask_val)
            or bid_val <= 0
            or ask_val <= 0
        ):
            return None
        mid = (bid_val + ask_val) * 0.5
        if mid <= 0:
            return None
        spread = (ask_val - bid_val) / mid * 10000.0
        if spread <= 0 or not isfinite(spread):
            return None
        return spread

    def _compute_spread_from_range(
        self,
        high: Optional[Union[float, int, Decimal]],
        low: Optional[Union[float, int, Decimal]],
        mid: Optional[Union[float, int, Decimal]] = None,
    ) -> Optional[float]:
        high_val = self._coerce_float(high)
        low_val = self._coerce_float(low)
        mid_val = self._coerce_float(mid)
        if high_val is None or low_val is None:
            return None
        if not (isfinite(high_val) and isfinite(low_val)):
            return None
        span = high_val - low_val
        if span <= 0.0:
            return None
        if mid_val is None or not isfinite(mid_val) or mid_val <= 0.0:
            mid_val = (high_val + low_val) * 0.5
        if mid_val == 0.0 or not isfinite(mid_val):
            return None
        spread = span / abs(mid_val) * 10000.0
        if spread <= 0.0 or not isfinite(spread):
            return None
        return spread

    @staticmethod
    def _coerce_float(value: Optional[Union[float, int, Decimal]]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Offline helpers
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> None:  # noqa: D401 - protocol no-op
        """No fitting required for deterministic features."""
        # Stateless transformer; method exists for protocol compatibility.
        return None

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply offline feature computation to ``df``.

        The dataframe must contain ``ts_ms`` and ``symbol`` columns together
        with ``price_col``.
        """

        return apply_offline_features(
            df,
            spec=self.spec,
            ts_col="ts_ms",
            symbol_col="symbol",
            price_col=self.price_col,
        )

    def make_targets(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Build target series for training.

        If ``label_col`` is provided and present in ``df`` the existing
        column is returned.  Otherwise the method computes a next-step return
        based on ``price_col``.
        """

        if self.label_col and self.label_col in df.columns:
            return df[self.label_col]

        price = df[self.price_col].astype(float)
        future_price = df.groupby("symbol")[self.price_col].shift(-1)
        target = future_price.div(price) - 1.0
        return target.rename("target")


__all__ = [
    "FeaturePipe",
    "SignalQualityMetrics",
    "SignalQualitySnapshot",
    "MarketMetricsSnapshot",
]

