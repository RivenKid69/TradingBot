"""Deterministic portfolio allocator used to derive target weights.

The allocator applies a sequence of deterministic filters to the latest score
vector and enforces a set of portfolio level constraints.  It does not submit
orders – the caller is responsible for mapping the resulting weights to any
execution layer.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping, MutableMapping

import pandas as pd


@dataclass(slots=True)
class PortfolioConstraints:
    """Constraints applied by :class:`DeterministicPortfolioAllocator`."""

    top_n: int | None = None
    threshold: float = 0.0
    max_weight_per_symbol: float = 1.0
    max_gross_exposure: float = 1.0
    realloc_threshold: float = 0.0


class DeterministicPortfolioAllocator:
    """Allocate deterministic weights from a score matrix.

    The allocator operates on a ``pandas.DataFrame`` containing model scores per
    symbol.  Only the most recent row of the frame is used – callers are free to
    pre-filter the input to select any desired timestamp.

    The allocation algorithm proceeds through the following stages:

    1. Scores below ``threshold`` are discarded.
    2. The remaining symbols are sorted by score and optionally truncated to the
       top ``N`` entries.
    3. Scores are normalised to sum to one.  If all scores are non-positive the
       allocator returns a zero vector.
    4. Position weights are clipped to ``max_weight_per_symbol`` and then scaled
       down uniformly when the gross exposure would otherwise exceed
       ``max_gross_exposure``.
    5. When ``realloc_threshold`` is positive, symbols whose weight change is
       smaller than the threshold keep their previous allocations.
    """

    def __init__(self, constraints: PortfolioConstraints | None = None) -> None:
        self._constraints = constraints or PortfolioConstraints()

    @property
    def constraints(self) -> PortfolioConstraints:
        return self._constraints

    def compute_weights(
        self,
        scores_df: pd.DataFrame,
        prev_weights: Mapping[str, float] | pd.Series | None = None,
        **override_params: float | int | None,
    ) -> pd.Series:
        """Return the target weights implied by ``scores_df``.

        Parameters
        ----------
        scores_df:
            DataFrame with per-symbol scores.  The allocator uses the most
            recent row (``iloc[-1]``) and ignores non-numeric columns.
        prev_weights:
            Optional mapping with previous portfolio weights.  When provided the
            resulting Series will contain all symbols from the previous
            allocation so that callers can compare deltas.  The mapping is also
            used when enforcing the reallocation threshold.
        override_params:
            Optional keyword arguments overriding the constraints stored on the
            allocator instance.  Supported keys match the fields of
            :class:`PortfolioConstraints`.
        """

        if scores_df is None or scores_df.empty:
            return self._empty_like(prev_weights)

        constraints = self._apply_overrides(override_params)

        latest_row = scores_df.select_dtypes(include=["number"]).tail(1)
        if latest_row.empty:
            return self._empty_like(prev_weights)

        scores = latest_row.iloc[0].dropna().astype(float)
        if scores.empty:
            return self._empty_like(prev_weights)

        filtered = scores[scores >= constraints.threshold]
        if filtered.empty:
            return self._maybe_keep_prev(prev_weights)

        filtered = filtered.sort_values(ascending=False)
        if constraints.top_n is not None and constraints.top_n > 0:
            filtered = filtered.head(constraints.top_n)
        if filtered.empty:
            return self._maybe_keep_prev(prev_weights)

        positive_mask = filtered > 0
        if not positive_mask.any():
            return self._maybe_keep_prev(prev_weights)

        filtered = filtered[positive_mask]
        weights = filtered / filtered.sum()
        weights = weights.clip(upper=constraints.max_weight_per_symbol)
        gross = weights.sum()
        if gross > 0:
            scale = min(1.0, constraints.max_gross_exposure / gross)
            weights = weights * scale

        prev_series = self._coerce_prev(prev_weights, weights.index)
        if prev_series is not None and constraints.realloc_threshold > 0:
            aligned_prev = prev_series.reindex(weights.index).fillna(0.0)
            delta = (weights - aligned_prev).abs()
            mask = delta < constraints.realloc_threshold
            if mask.any():
                weights.loc[mask] = aligned_prev.loc[mask]

        if prev_series is not None:
            weights = weights.reindex(prev_series.index).fillna(0.0)

        weights = weights[weights.abs() > 0]
        return weights.sort_index()

    def _apply_overrides(self, overrides: MutableMapping[str, float | int | None]) -> PortfolioConstraints:
        params = asdict(self.constraints)
        for key, value in overrides.items():
            if key not in params or value is None:
                continue
            params[key] = value
        return PortfolioConstraints(
            top_n=int(params["top_n"]) if params["top_n"] not in (None, False) else None,
            threshold=float(params["threshold"]),
            max_weight_per_symbol=float(params["max_weight_per_symbol"]),
            max_gross_exposure=float(params["max_gross_exposure"]),
            realloc_threshold=float(params["realloc_threshold"]),
        )

    @staticmethod
    def _coerce_prev(prev: Mapping[str, float] | pd.Series | None, new_index: pd.Index) -> pd.Series | None:
        if prev is None:
            return None
        if isinstance(prev, pd.Series):
            series = prev.astype(float)
        else:
            series = pd.Series(prev, dtype=float)
        series = series.reindex(series.index.union(new_index), fill_value=0.0)
        return series.sort_index()

    @staticmethod
    def _empty_like(prev: Mapping[str, float] | pd.Series | None) -> pd.Series:
        if prev is None:
            return pd.Series(dtype=float)
        if isinstance(prev, pd.Series):
            index = prev.index
        else:
            index = list(prev.keys())
        return pd.Series(0.0, index=index, dtype=float)

    def _maybe_keep_prev(
        self,
        prev: Mapping[str, float] | pd.Series | None,
    ) -> pd.Series:
        prev_series = self._coerce_prev(prev, pd.Index([]))
        if prev_series is None:
            return pd.Series(dtype=float)
        return prev_series
