"""Diagnostic helper for the deterministic portfolio allocator.

The script loads a table of model scores, evaluates the
:class:`~portfolio_allocator.DeterministicPortfolioAllocator` for each regime
present in the dataset and stores the resulting target weights.  No trading
environment is instantiated and no orders are emitted – the output represents
paper allocations useful for sanity checks.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_config import TrainConfig, load_config
from portfolio_allocator import DeterministicPortfolioAllocator


def _load_scores(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Scores file not found: {path}")
    if file_path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("Scores file is empty")
    return df


def _latest_slice(df: pd.DataFrame, regime: str | None) -> pd.DataFrame:
    subset = df if regime is None else df[df.get("regime") == regime]
    if subset.empty:
        raise ValueError(f"No score rows available for regime '{regime}'")
    if "timestamp" in subset.columns:
        ts = subset["timestamp"].max()
        subset = subset[subset["timestamp"] == ts]
    if {"symbol", "score"}.issubset(subset.columns):
        latest = subset.groupby("symbol")[["score"]].last()["score"].astype(float)
        return latest.to_frame().T
    numeric_cols = [
        c for c in subset.select_dtypes(include=["number"]).columns if c.lower() not in {"timestamp", "time", "ts"}
    ]
    if not numeric_cols:
        raise ValueError("Scores frame does not contain numeric columns for allocation")
    return subset[numeric_cols].tail(1)


def _extract_params(cfg: TrainConfig) -> Mapping[str, float | int | None]:
    portfolio_cfg = getattr(cfg, "portfolio", None)
    params: dict[str, float | int | None] = {
        "top_n": getattr(portfolio_cfg, "top_n", None) if portfolio_cfg else None,
        "threshold": float(getattr(portfolio_cfg, "threshold", 0.0)) if portfolio_cfg else 0.0,
        "max_weight_per_symbol": float(getattr(portfolio_cfg, "max_weight_per_symbol", 1.0)) if portfolio_cfg else 1.0,
        "max_gross_exposure": float(getattr(portfolio_cfg, "max_gross_exposure", 1.0)) if portfolio_cfg else 1.0,
        "realloc_threshold": float(getattr(portfolio_cfg, "realloc_threshold", 0.0)) if portfolio_cfg else 0.0,
    }
    return params


def _save_weights(weights: pd.Series, output_dir: Path, regime: str | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = regime if regime is not None else "all"
    path = output_dir / f"weights_{suffix}.csv"
    weights.to_frame("weight").to_csv(path, index_label="symbol")
    return path


def main(argv: list[str] | None = None) -> dict[str, pd.Series]:
    parser = argparse.ArgumentParser(description="Validate deterministic portfolio allocations")
    parser.add_argument("--config", default="configs/config_train.yaml", help="Training config with portfolio params")
    parser.add_argument("--scores", required=True, help="Input CSV/Parquet with model scores")
    parser.add_argument("--output-dir", default="reports/portfolio_weights", help="Directory for per-regime weight dumps")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if not isinstance(cfg, TrainConfig):
        raise TypeError("Expected training config with portfolio parameters")

    df = _load_scores(args.scores)
    params = _extract_params(cfg)
    allocator = DeterministicPortfolioAllocator()

    regimes = sorted(set(df["regime"].dropna().astype(str))) if "regime" in df.columns else [None]
    results: dict[str, pd.Series] = {}

    for regime in regimes:
        slice_df = _latest_slice(df, regime)
        weights = allocator.compute_weights(slice_df, **params)
        path = _save_weights(weights, Path(args.output_dir), regime)
        gross = float(weights.abs().sum())
        limit = params["max_gross_exposure"] or 0.0
        if limit and gross > limit + 1e-9:
            raise AssertionError(
                f"Gross exposure {gross:.4f} exceeds configured max_gross_exposure {limit:.4f} for regime {regime}."
            )
        if not weights.empty and np.any(weights.values > params["max_weight_per_symbol"] + 1e-9):
            raise AssertionError(
                f"Weight exceeds max_weight_per_symbol constraint for regime {regime}."
            )
        key = "all" if regime is None else regime
        results[key] = weights
        print(f"Regime '{key}': {len(weights)} symbols, gross {gross:.4f} → {path}")

    return results


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
