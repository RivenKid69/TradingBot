"""Build hour-of-week spread multipliers for the dynamic slippage module.

The script expects a CSV or Parquet file with historical bar data. It
computes the average bid-ask spread (either from an explicit spread column or
from high/low mid-prices) for every hour of the week, normalises the result so
that the mean multiplier is ``1.0`` and writes a JSON profile compatible with
``SlippageConfig.dynamic``.

Example
-------

.. code-block:: bash

    python scripts/build_spread_seasonality.py \
        --data data/seasonality_source/latest.parquet \
        --out data/slippage/hourly_profile.json \
        --window-days 90
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from utils.time import HOURS_IN_WEEK, DAY_MS, hour_of_week


@dataclass(frozen=True)
class SpreadComputationResult:
    multipliers: np.ndarray
    hourly_raw: np.ndarray
    counts: np.ndarray
    overall_mean: float


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _series_to_numeric(series: pd.Series) -> np.ndarray:
    if series.dtype.kind in {"f", "i", "u"}:
        return series.to_numpy(dtype=float)
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _to_timestamp_ms(series: pd.Series) -> np.ndarray:
    if np.issubdtype(series.dtype, np.datetime64):
        return series.view("int64") // 1_000_000
    if series.dtype.kind in {"i", "u"}:
        arr = series.to_numpy(dtype=np.int64)
        if arr.size == 0:
            return arr
        max_abs = np.nanmax(np.abs(arr))
        if max_abs < 10**11:  # treat as seconds
            arr = arr * 1000
        return arr
    if series.dtype.kind == "f":
        arr = series.to_numpy(dtype=np.float64)
        if arr.size == 0:
            return arr.astype(np.int64)
        max_abs = np.nanmax(np.abs(arr))
        if math.isnan(max_abs):
            return np.full_like(arr, fill_value=np.nan, dtype=np.float64).astype(np.int64)
        if max_abs < 10**8:
            arr = arr * 1000.0
        return np.asarray(arr, dtype=np.int64)
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.view("int64") // 1_000_000


def _infer_timestamp_column(df: pd.DataFrame, explicit: Optional[str]) -> str:
    candidates: Iterable[str]
    if explicit:
        candidates = [explicit]
    else:
        candidates = [
            "ts_ms",
            "timestamp_ms",
            "close_time",
            "open_time",
            "timestamp",
            "ts",
            "time",
        ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not locate timestamp column. Provide it explicitly via --ts-col."
    )


def _infer_spread_series(df: pd.DataFrame, explicit: Optional[str]) -> np.ndarray:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Requested spread column '{explicit}' is missing")
        return _series_to_numeric(df[explicit])

    for col in [
        "spread",
        "spread_bps",
        "avg_spread",
        "spread_abs",
        "spread_pct",
        "mid_spread",
    ]:
        if col in df.columns:
            return _series_to_numeric(df[col])

    high_col = next(
        (c for c in ["high", "high_price", "ask_high", "best_ask"] if c in df.columns),
        None,
    )
    low_col = next(
        (c for c in ["low", "low_price", "bid_low", "best_bid"] if c in df.columns),
        None,
    )
    if not high_col or not low_col:
        raise ValueError(
            "Spread columns not found. Provide --spread-column or include high/low prices."
        )
    high = _series_to_numeric(df[high_col])
    low = _series_to_numeric(df[low_col])
    if "mid" in df.columns:
        mid = _series_to_numeric(df["mid"])
    elif "mid_price" in df.columns:
        mid = _series_to_numeric(df["mid_price"])
    else:
        mid = (high + low) / 2.0
    spread = high - low
    with np.errstate(divide="ignore", invalid="ignore"):
        metric = np.where(mid > 0, spread / mid, np.nan)
    return metric


def _filter_window(ts_ms: np.ndarray, values: np.ndarray, window_days: int) -> tuple[np.ndarray, np.ndarray]:
    if window_days <= 0 or ts_ms.size == 0:
        return ts_ms, values
    cutoff = ts_ms.max() - int(window_days * DAY_MS)
    mask = ts_ms >= cutoff
    return ts_ms[mask], values[mask]


def compute_spread_multipliers(
    df: pd.DataFrame,
    *,
    ts_col: str,
    spread_column: Optional[str],
    window_days: int,
    symbol: Optional[str],
) -> SpreadComputationResult:
    if symbol is not None:
        if "symbol" not in df.columns:
            raise ValueError("Dataset does not contain a 'symbol' column")
        df = df[df["symbol"] == symbol]
        if df.empty:
            raise ValueError(f"No rows found for symbol '{symbol}'")

    ts_ms = _to_timestamp_ms(df[ts_col])
    spread_values = _infer_spread_series(df, spread_column)
    if ts_ms.shape[0] != spread_values.shape[0]:
        raise ValueError("Timestamp and spread arrays must align")

    mask = np.isfinite(ts_ms) & np.isfinite(spread_values)
    mask &= ts_ms > 0
    spread_values = spread_values[mask]
    ts_ms = ts_ms[mask]

    ts_ms, spread_values = _filter_window(ts_ms, spread_values, window_days)
    if ts_ms.size == 0:
        raise ValueError("No samples remain after applying filters")

    positive_mask = spread_values > 0
    spread_values = spread_values[positive_mask]
    ts_ms = ts_ms[positive_mask]
    if spread_values.size == 0:
        raise ValueError("Spread values must be positive")

    df = pd.DataFrame({
        "hour": hour_of_week(ts_ms),
        "spread": spread_values,
    })
    grouped = df.groupby("hour")["spread"].agg(["mean", "count"])
    hourly_mean = grouped["mean"].reindex(range(HOURS_IN_WEEK), fill_value=np.nan).to_numpy(dtype=float)
    counts = grouped["count"].reindex(range(HOURS_IN_WEEK), fill_value=0).to_numpy(dtype=int)
    overall_mean = float(np.nanmean(spread_values))

    filled = hourly_mean.copy()
    if overall_mean <= 0 or not np.isfinite(overall_mean):
        overall_mean = 1.0
    filled[np.isnan(filled)] = overall_mean

    multipliers = filled / overall_mean if overall_mean else np.ones_like(filled)
    mean_mult = float(np.nanmean(multipliers))
    if mean_mult and np.isfinite(mean_mult):
        multipliers = multipliers / mean_mult
    else:
        multipliers = np.ones_like(filled)

    return SpreadComputationResult(
        multipliers=multipliers,
        hourly_raw=filled,
        counts=counts,
        overall_mean=overall_mean,
    )


def _warn_if_stale(path: Path, refresh_warn_days: int) -> None:
    if refresh_warn_days <= 0 or not path.exists():
        return
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age = datetime.now(timezone.utc) - mtime
    if age > timedelta(days=refresh_warn_days):
        print(
            f"WARNING: {path} is {age.days} days old (>{refresh_warn_days} days).",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build hour-of-week spread multipliers for dynamic slippage",
    )
    parser.add_argument(
        "--data",
        default="data/seasonality_source/latest.parquet",
        help="Path to OHLC/quote data (CSV or Parquet)",
    )
    parser.add_argument(
        "--out",
        default="data/slippage/hourly_profile.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=0,
        help="Limit computation to the most recent N days (0 = use all data)",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Optional symbol filter when the dataset contains multiple instruments",
    )
    parser.add_argument(
        "--ts-col",
        default=None,
        help="Timestamp column name (auto-detected when omitted)",
    )
    parser.add_argument(
        "--spread-column",
        default=None,
        help="Column with spread metric. When omitted the script derives it from high/low prices.",
    )
    parser.add_argument(
        "--profile-kind",
        default="hourly",
        help="Metadata tag describing the resulting profile",
    )
    parser.add_argument(
        "--refresh-warn-days",
        type=int,
        default=7,
        help="Emit a warning if the source file is older than this many days",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Input dataset {data_path} not found")

    _warn_if_stale(data_path, int(args.refresh_warn_days))

    df = _load_table(data_path)
    if df.empty:
        raise ValueError("Input dataset is empty")

    ts_column = _infer_timestamp_column(df, args.ts_col)
    result = compute_spread_multipliers(
        df,
        ts_col=ts_column,
        spread_column=args.spread_column,
        window_days=int(args.window_days),
        symbol=args.symbol,
    )

    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": str(data_path),
            "profile_kind": args.profile_kind,
            "window_days": int(args.window_days),
            "normalization": "mean = 1.0",
            "symbol": args.symbol,
            "samples": int(result.counts.sum()),
            "overall_mean_spread": result.overall_mean,
        },
        "profile": [float(x) for x in result.multipliers],
        "hourly_raw": [float(x) for x in result.hourly_raw],
        "counts": [int(x) for x in result.counts],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote spread profile to {out_path}")


if __name__ == "__main__":
    main()
