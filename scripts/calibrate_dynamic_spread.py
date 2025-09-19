#!/usr/bin/env python3
"""Calibrate dynamic spread parameters from historical bar data.

This utility estimates the coefficients used by the ``slippage.dynamic_spread``
configuration block.  The estimator works by regressing an observed (or
desired) spread in basis points against a volatility proxy that is computed
from the supplied bar data.

Example
-------
The example below calibrates using range based volatility, keeps the regression
inputs between the 5th and 95th percentiles, and writes a YAML fragment that
can be copy pasted into a configuration file::

    $ python scripts/calibrate_dynamic_spread.py data/bars.parquet \
          --symbol BTCUSDT --timeframe 1m \
          --volatility-metric range_ratio_bps \
          --clip-lower 5 --clip-upper 95 \
          --output calibration.yaml

The resulting YAML fragment will look similar to::

    slippage:
      dynamic_spread:
        alpha_bps: 4.732551
        beta_coef: 0.310742
        min_spread_bps: 3.281905
        max_spread_bps: 9.834627
        smoothing_alpha: null
        vol_metric: range_ratio_bps
        clip_percentiles: [5.0, 95.0]

Run the tool with ``--help`` to see the complete list of options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is an optional dependency
    yaml = None  # type: ignore[assignment]


def _read_table(path: Path) -> pd.DataFrame:
    """Load tabular data using :mod:`pandas` based on file suffix."""

    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".feather", ".ft"}:
        return pd.read_feather(path)

    raise ValueError(f"Unsupported file type for {path}")


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _compute_mid(df: pd.DataFrame) -> pd.Series:
    bid_col = _pick_column(df, ["bid", "best_bid", "bid_price", "bid_px"])
    ask_col = _pick_column(df, ["ask", "best_ask", "ask_price", "ask_px"])
    if bid_col and ask_col:
        return (df[ask_col] + df[bid_col]) / 2

    mid_col = _pick_column(df, ["mid", "mid_price", "mid_px", "price", "close"])
    if mid_col:
        return df[mid_col]

    raise KeyError(
        "Unable to infer mid price column. Provide bid/ask or a mid/close column."
    )


def _compute_spread_bps(df: pd.DataFrame, mid: pd.Series) -> Optional[pd.Series]:
    if "spread_bps" in df.columns:
        return df["spread_bps"]

    bid_col = _pick_column(df, ["bid", "best_bid", "bid_price", "bid_px"])
    ask_col = _pick_column(df, ["ask", "best_ask", "ask_price", "ask_px"])
    if bid_col and ask_col:
        spread = df[ask_col] - df[bid_col]
        return (spread / mid) * 1e4

    raw_spread_col = _pick_column(df, ["spread", "spread_abs"])
    if raw_spread_col:
        return (df[raw_spread_col] / mid) * 1e4

    return None


def _prepare_dataframe(
    df: pd.DataFrame,
    symbol: Optional[str],
    timeframe: Optional[str],
) -> pd.DataFrame:
    result = df.copy()
    if symbol and "symbol" in result.columns:
        result = result[result["symbol"] == symbol]
    if timeframe and _pick_column(result, ["interval", "timeframe", "resolution"]):
        tf_col = _pick_column(result, ["interval", "timeframe", "resolution"])
        if tf_col:
            result = result[result[tf_col] == timeframe]

    if "high" not in result.columns or "low" not in result.columns:
        raise KeyError("Bar data must contain 'high' and 'low' columns")

    mid = _compute_mid(result)
    result = result.assign(mid=mid)
    mask = (
        result["mid"].notna()
        & result["mid"].astype(float).replace([np.inf, -np.inf], np.nan).notna()
        & result["mid"] > 0
        & result["high"].notna()
        & result["low"].notna()
    )
    result = result.loc[mask]

    high = pd.to_numeric(result["high"], errors="coerce")
    low = pd.to_numeric(result["low"], errors="coerce")
    mid_clean = pd.to_numeric(result["mid"], errors="coerce")
    price_range = high - low
    price_range = price_range.where(price_range >= 0, price_range.abs())
    ratio = price_range / mid_clean
    ratio = ratio.where(mid_clean > 0.0)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio = ratio.where(ratio >= 0.0)
    result = result.assign(range_ratio_bps=ratio * 1e4)
    return result


def _select_volatility(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "range_ratio_bps":
        return df["range_ratio_bps"]
    if metric in df.columns:
        return df[metric]

    raise KeyError(
        f"Volatility metric '{metric}' not found. Available columns: {', '.join(df.columns)}"
    )


def _clip_percentiles(series: pd.Series, lower: float, upper: float) -> pd.Series:
    if series.empty:
        return series
    cleaned = series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return series
    try:
        lower_bound = np.nanpercentile(cleaned, lower)
        upper_bound = np.nanpercentile(cleaned, upper)
    except IndexError:
        return series
    if lower_bound == upper_bound:
        return series
    return series.clip(lower_bound, upper_bound)


def _linear_regression(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    coeffs = np.polyfit(x, y, 1)
    beta = float(coeffs[0])
    alpha = float(coeffs[1])
    return alpha, beta


def _fallback_parameters(volatility: pd.Series, spread: pd.Series) -> Tuple[float, float]:
    spread_median = float(np.nanmedian(spread)) if len(spread) else 0.0
    if len(volatility) == 0:
        return spread_median, 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = spread / volatility
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if len(ratio) == 0:
        beta = 0.0
    else:
        beta = float(np.nanmedian(ratio))
    alpha = spread_median
    return alpha, beta


def _derive_spread_bounds(
    spread: pd.Series, lower_pct: float, upper_pct: float
) -> Tuple[Optional[float], Optional[float]]:
    if spread.empty:
        return None, None
    cleaned = spread.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return None, None
    lower_val = float(np.nanpercentile(cleaned, lower_pct))
    upper_val = float(np.nanpercentile(cleaned, upper_pct))
    if upper_val < lower_val:
        upper_val = lower_val
    return lower_val, upper_val


def _normalise_smoothing_alpha(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value <= 0.0:
        return None
    if value >= 1.0:
        return 1.0
    return float(value)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate coefficients for the dynamic spread configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data_path", type=Path, help="Path to the bar data file (CSV/Parquet/Feather)")
    parser.add_argument("--symbol", help="Optional symbol to filter by")
    parser.add_argument("--timeframe", help="Optional timeframe/interval to filter by")
    parser.add_argument(
        "--volatility-metric",
        default="range_ratio_bps",
        help=(
            "Column name to use as the volatility proxy. Use 'range_ratio_bps' to "
            "compute it from high/low/mid prices."
        ),
    )
    parser.add_argument(
        "--clip-lower",
        type=float,
        default=5.0,
        help="Lower percentile for volatility clipping",
    )
    parser.add_argument(
        "--clip-upper",
        type=float,
        default=95.0,
        help="Upper percentile for volatility clipping",
    )
    parser.add_argument(
        "--target-spread-bps",
        type=float,
        help=(
            "Optional constant spread (in bps) to regress against when the dataset "
            "does not contain an observed spread column."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write a YAML fragment with slippage.dynamic_spread values",
    )
    parser.add_argument(
        "--min-spread-bps",
        type=float,
        default=None,
        help="Optional explicit lower bound for the dynamic spread (bps)",
    )
    parser.add_argument(
        "--max-spread-bps",
        type=float,
        default=None,
        help="Optional explicit upper bound for the dynamic spread (bps)",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=None,
        help="Optional EMA smoothing coefficient [0,1] for the resulting spread",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    if not args.data_path.exists():
        raise FileNotFoundError(args.data_path)

    if args.clip_lower < 0 or args.clip_upper > 100 or args.clip_lower >= args.clip_upper:
        raise ValueError("clip percentiles must satisfy 0 <= lower < upper <= 100")

    df = _read_table(args.data_path)
    df = _prepare_dataframe(df, args.symbol, args.timeframe)

    fallback_reason: Optional[str] = None
    if df.empty:
        fallback_reason = "No rows remaining after filtering"
        if args.target_spread_bps is not None:
            spread_series = pd.Series([args.target_spread_bps], dtype=float)
        else:
            spread_series = pd.Series(dtype=float)
        volatility = pd.Series(dtype=float)
    else:
        mid = df["mid"]
        spread_series = _compute_spread_bps(df, mid)
        if spread_series is None:
            if args.target_spread_bps is None:
                raise KeyError(
                    "Unable to infer spread from dataset. Provide bid/ask columns, "
                    "a spread column, or --target-spread-bps."
                )
            spread_series = pd.Series(args.target_spread_bps, index=df.index)

        volatility = _select_volatility(df, args.volatility_metric)

        valid_mask = spread_series.notna() & volatility.notna()
        spread_series = spread_series[valid_mask]
        volatility = volatility[valid_mask]

        if len(spread_series) < 2:
            fallback_reason = "Not enough samples after filtering"

    clipped_volatility = _clip_percentiles(volatility, args.clip_lower, args.clip_upper)

    success = False
    if fallback_reason is None:
        cleaned_vol = clipped_volatility.astype(float).replace([np.inf, -np.inf], np.nan)
        cleaned_vol = cleaned_vol.dropna()
        cleaned_spread = spread_series.astype(float).replace([np.inf, -np.inf], np.nan)
        cleaned_spread = cleaned_spread.dropna()
        has_variance = len(cleaned_vol) >= 2 and not np.isclose(
            float(cleaned_vol.max()), float(cleaned_vol.min())
        )
        if cleaned_spread.empty or not has_variance:
            fallback_reason = "Volatility samples degenerate after clipping"
        else:
            try:
                alpha, beta = _linear_regression(clipped_volatility, spread_series)
                success = True
            except (np.linalg.LinAlgError, ValueError) as exc:
                print(f"Regression failed: {exc}", file=sys.stderr)
                fallback_reason = "Regression failure"

    if not success:
        alpha, beta = _fallback_parameters(clipped_volatility, spread_series)

    print("Dynamic spread calibration")
    print("----------------------------")
    print(f"Samples used: {len(spread_series)}")
    print(f"Volatility metric: {args.volatility_metric}")
    print(f"Percentile clip: [{args.clip_lower}, {args.clip_upper}]")
    print(f"alpha (bps): {alpha:.6f}")
    print(f"beta: {beta:.6f}")
    if fallback_reason is not None:
        print(f"(values derived from fallback heuristics: {fallback_reason})")

    derived_min, derived_max = _derive_spread_bounds(
        spread_series, args.clip_lower, args.clip_upper
    )
    min_spread = (
        float(args.min_spread_bps)
        if args.min_spread_bps is not None
        else derived_min
    )
    max_spread = (
        float(args.max_spread_bps)
        if args.max_spread_bps is not None
        else derived_max
    )
    if min_spread is not None and max_spread is not None and max_spread < min_spread:
        max_spread = min_spread

    smoothing_alpha = _normalise_smoothing_alpha(args.smoothing_alpha)

    if min_spread is not None:
        print(f"min spread (bps): {min_spread:.6f}")
    if max_spread is not None:
        print(f"max spread (bps): {max_spread:.6f}")
    if smoothing_alpha is not None:
        print(f"smoothing alpha: {smoothing_alpha:.6f}")

    if args.output:
        fragment = {
            "slippage": {
                "dynamic_spread": {
                    "alpha_bps": float(alpha),
                    "beta_coef": float(beta),
                    "min_spread_bps": float(min_spread)
                    if min_spread is not None
                    else None,
                    "max_spread_bps": float(max_spread)
                    if max_spread is not None
                    else None,
                    "smoothing_alpha": smoothing_alpha,
                    "vol_metric": args.volatility_metric,
                    "clip_percentiles": [
                        float(args.clip_lower),
                        float(args.clip_upper),
                    ],
                }
            }
        }
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required for --output. Install it or omit the argument."
            )
        args.output.write_text(
            yaml.safe_dump(fragment, sort_keys=False, default_flow_style=False)
        )
        print(f"Wrote YAML fragment to {args.output}")

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    raise SystemExit(main())
