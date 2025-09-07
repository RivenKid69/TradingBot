"""Offline feature pipe for ServiceTrain.

Computes deterministic features using ``apply_offline_features`` and
optionally builds simple targets. The pipe conforms to the
``FeaturePipe`` protocol expected by :class:`ServiceTrain`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from transformers import FeatureSpec, apply_offline_features


@dataclass
class OfflineFeaturePipe:
    """Simple offline feature builder.

    Parameters
    ----------
    spec:
        Feature specification for SMA/returns and RSI parameters.
    price_col:
        Name of the column containing prices in the input dataframe.
    label_col:
        Optional column name for existing target values.  When provided and
        present in the dataframe, :meth:`make_targets` simply returns this
        column.  Otherwise a next-step return is computed from ``price_col``.
    """

    spec: FeatureSpec
    price_col: str = "price"
    label_col: Optional[str] = None

    # Protocol methods -------------------------------------------------
    def warmup(self) -> None:  # noqa: D401 - protocol no-op
        """No warmup needed for offline computation."""
        # nothing to warm up for deterministic offline features

    def fit(self, df: pd.DataFrame) -> None:  # noqa: D401 - protocol no-op
        """No fitting required."""
        # offline features are stateless; the method exists for protocol
        # compatibility

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature transformer to ``df``.

        The dataframe must contain ``ts_ms`` and ``symbol`` columns together
        with ``price_col``.  The resulting dataframe follows the same schema
        as produced by :func:`apply_offline_features`.
        """
        return apply_offline_features(
            df,
            spec=self.spec,
            ts_col="ts_ms",
            symbol_col="symbol",
            price_col=self.price_col,
        )

    def make_targets(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Build target series.

        If ``label_col`` is provided and present in ``df`` the existing column
        is returned.  Otherwise the method computes a one-step ahead return
        based on ``price_col``.
        """
        if self.label_col and self.label_col in df.columns:
            return df[self.label_col]

        # compute forward return within each symbol
        price = df[self.price_col].astype(float)
        future_price = df.groupby("symbol")[self.price_col].shift(-1)
        target = future_price.div(price) - 1.0
        return target.rename("target")
