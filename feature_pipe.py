"""Unified feature pipe for online and offline processing.

This module provides :class:`FeaturePipe` which implements the
``core_contracts.FeaturePipe`` protocol and exposes additional offline
helpers used during dataset preparation.  The class wraps
``OnlineFeatureTransformer`` for streaming feature computation and reuses the
deterministic ``apply_offline_features`` routine for batch processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Any

import pandas as pd

from transformers import FeatureSpec, OnlineFeatureTransformer, apply_offline_features
from core_models import Bar
from core_contracts import FeaturePipe as FeaturePipeProtocol  # noqa: F401


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

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state of the transformer."""
        self._tr = OnlineFeatureTransformer(self.spec)

    def warmup(self, bars: Iterable[Bar] = ()) -> None:
        """Warm up transformer with a sequence of historical bars."""
        self.reset()
        for b in bars:
            self.update(b)

    def update(self, bar: Bar) -> Mapping[str, Any]:
        """Process a single bar and return computed features."""
        feats = self._tr.update(
            symbol=bar.symbol.upper(),
            ts_ms=int(bar.ts),
            close=float(bar.close),
        )
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


__all__ = ["FeaturePipe"]

