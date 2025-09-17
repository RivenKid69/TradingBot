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
from math import sqrt
from typing import Iterable, Mapping, Optional, Any, Deque, Dict
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

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for all symbols."""

        self._state.clear()

    # ------------------------------------------------------------------
    def update(self, bar: Bar) -> SignalQualitySnapshot:
        """Update metrics with a new bar and return current values."""

        symbol = bar.symbol.upper()
        state = self._state.setdefault(symbol, _SymbolMetricsState())

        close = float(bar.close)
        if state.prev_close not in (None, 0.0):
            ret = close / state.prev_close - 1.0
            self._append_return(state, ret)

        state.prev_close = close

        volume_source = bar.volume_quote if bar.volume_quote is not None else bar.volume_base
        if volume_source is not None:
            self._append_volume(state, float(volume_source))

        sigma = self._compute_sigma(state)
        vol_median = self._compute_volume_median(state)
        window_ready = (
            len(state.returns) >= self.sigma_window
            and len(state.volumes) >= self.vol_median_window
        )

        return SignalQualitySnapshot(
            current_sigma=sigma,
            current_vol_median=vol_median,
            window_ready=window_ready,
        )

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

    def __post_init__(self) -> None:
        # Initialize transformer for online mode.
        self._tr = OnlineFeatureTransformer(self.spec)
        self._read_only = False

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state of the transformer."""
        self._tr = OnlineFeatureTransformer(self.spec)

    def set_read_only(self, flag: bool = True) -> None:
        """Enable or disable read-only mode for normalization stats."""
        self._read_only = bool(flag)

    def warmup(self, bars: Iterable[Bar] = ()) -> None:
        """Warm up transformer with a sequence of historical bars."""
        self.reset()
        for b in bars:
            self.update(b)

    def update(self, bar: Bar) -> Mapping[str, Any]:
        """Process a single bar and return computed features."""
        if not self._read_only:
            feats = self._tr.update(
                symbol=bar.symbol.upper(),
                ts_ms=int(bar.ts),
                close=float(bar.close),
            )
            return feats

        state_backup = copy.deepcopy(self._tr._state)
        feats = self._tr.update(
            symbol=bar.symbol.upper(),
            ts_ms=int(bar.ts),
            close=float(bar.close),
        )
        self._tr._state = state_backup
        return feats

    # ``core_contracts.FeaturePipe`` historically exposes ``on_bar``.  Keep
    # an alias for backward compatibility with existing call sites.
    on_bar = update

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


__all__ = ["FeaturePipe", "SignalQualityMetrics", "SignalQualitySnapshot"]

