#!/usr/bin/env python3
"""Calibrate live slippage curves from execution logs.

This utility scans trade fills produced by :class:`sim_logging.LogWriter`,
aggregates empirical impact-vs-notional curves and derives seasonal
adjustments that can be consumed by the runtime slippage model.  The
workflow is fully offline – it only depends on historical log files and
public data that may already be embedded in those logs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from glob import glob

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Ensure local imports work regardless of current working directory
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from service_calibrate_slippage import fit_k_closed_form
from utils.time import HOURS_IN_WEEK, hour_of_week


LOGGER = logging.getLogger("calibrate_live_slippage")


ColumnMap = Mapping[str, Sequence[str]]

DEFAULT_COLUMN_ALIASES: ColumnMap = {
    "timestamp": ("ts_ms", "ts", "timestamp", "time", "datetime"),
    "symbol": ("symbol", "instrument"),
    "slippage_bps": ("slippage_bps", "impact_bps", "observed_slip_bps"),
    "spread_bps": ("spread_bps", "half_spread_bps"),
    "notional": ("notional", "abs_notional", "trade_notional"),
    "size": ("size", "quantity", "qty", "base_qty", "trade_qty"),
    "liquidity": (
        "liquidity_notional",
        "book_liquidity",
        "liquidity",
        "adv",
        "participation_denom",
    ),
    "vol_factor": ("vol_factor", "volatility_factor", "vol_mult"),
    "execution_profile": ("execution_profile", "profile"),
    "market_regime": ("market_regime", "regime"),
}

_NULL_STRINGS = {"", "nan", "None", "NONE", "NaN", "NULL", "null"}


def _load_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if ext in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format for {path}")


def _resolve_column(df: pd.DataFrame, names: Iterable[str]) -> pd.Series | None:
    for name in names:
        if name in df.columns:
            return df[name]
    return None


def _normalise_timestamp(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="int64")
    if np.issubdtype(series.dtype, np.integer):
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if np.issubdtype(series.dtype, np.floating):
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    # Assume string-like timestamps
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return (ts.view("int64") // 1_000_000).astype("Int64")


def _normalise_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce").astype(float)


def _normalise_string(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=str)
    return series.astype(str)


def _expand_patterns(patterns: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(p).expanduser() for p in glob(pattern, recursive=True))
        if not matches:
            LOGGER.warning("pattern %s did not match any files", pattern)
        files.extend(matches)
    unique = sorted({path.resolve() for path in files})
    return unique


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    ts_series = _normalise_timestamp(_resolve_column(df, DEFAULT_COLUMN_ALIASES["timestamp"]))
    symbol_series = _resolve_column(df, DEFAULT_COLUMN_ALIASES["symbol"])
    slip_series = _normalise_numeric(_resolve_column(df, DEFAULT_COLUMN_ALIASES["slippage_bps"]))
    spread_series = _normalise_numeric(_resolve_column(df, DEFAULT_COLUMN_ALIASES["spread_bps"]))
    notional_series = _normalise_numeric(_resolve_column(df, DEFAULT_COLUMN_ALIASES["notional"]))
    size_series = _normalise_numeric(_resolve_column(df, DEFAULT_COLUMN_ALIASES["size"]))
    liquidity_series = _normalise_numeric(_resolve_column(df, DEFAULT_COLUMN_ALIASES["liquidity"]))
    vol_series = _normalise_numeric(_resolve_column(df, DEFAULT_COLUMN_ALIASES["vol_factor"]))
    exec_profile_series = _resolve_column(df, DEFAULT_COLUMN_ALIASES["execution_profile"])
    regime_series = _resolve_column(df, DEFAULT_COLUMN_ALIASES["market_regime"])
    if regime_series is not None:
        regime_payload = regime_series
    else:
        regime_payload = pd.Series([None] * len(df), index=df.index, dtype=object)

    out = pd.DataFrame({
        "ts_ms": ts_series,
        "symbol": symbol_series,
        "slippage_bps": slip_series,
        "spread_bps": spread_series,
        "notional": notional_series,
        "size": size_series,
        "liquidity": liquidity_series,
        "vol_factor": vol_series,
        "execution_profile": exec_profile_series,
        "market_regime": regime_payload,
    })

    # Drop rows without symbols or slippage values
    out = out.dropna(subset=["symbol", "slippage_bps"])
    out["symbol"] = out["symbol"].astype(str)
    out["ts_ms"] = pd.to_numeric(out["ts_ms"], errors="coerce").astype("Int64")
    out["notional"] = pd.to_numeric(out["notional"], errors="coerce").astype(float)
    out["size"] = pd.to_numeric(out["size"], errors="coerce").astype(float)
    out["liquidity"] = pd.to_numeric(out["liquidity"], errors="coerce").astype(float)
    out["vol_factor"] = pd.to_numeric(out["vol_factor"], errors="coerce").astype(float)
    if exec_profile_series is not None:
        out["execution_profile"] = out["execution_profile"].astype(str)
        out["execution_profile"] = out["execution_profile"].replace(
            {val: np.nan for val in _NULL_STRINGS}
        )
    if regime_series is not None:
        out["market_regime"] = out["market_regime"].astype(str)
        out["market_regime"] = out["market_regime"].replace(
            {val: np.nan for val in _NULL_STRINGS}
        )
    return out


def _compute_hourly_multipliers(df: pd.DataFrame, impact_col: str) -> dict[str, list]:
    if df["ts_ms"].isna().all():
        return {
            "hour_of_week": list(range(HOURS_IN_WEEK)),
            "multipliers": [1.0] * HOURS_IN_WEEK,
            "counts": [0] * HOURS_IN_WEEK,
        }

    ts_values = df["ts_ms"].dropna().astype(np.int64).to_numpy()
    hours = hour_of_week(ts_values)
    frame = df.loc[df["ts_ms"].notna(), [impact_col]].copy()
    frame["hour_of_week"] = hours
    grouped = frame.groupby("hour_of_week")[impact_col].agg(["mean", "count"])
    overall = frame[impact_col].mean()
    if overall is None or not np.isfinite(overall) or overall == 0.0:
        multipliers = np.ones(HOURS_IN_WEEK)
    else:
        ratios = grouped["mean"] / overall
        multipliers = ratios.reindex(range(HOURS_IN_WEEK)).fillna(1.0).to_numpy()
    counts = grouped["count"].reindex(range(HOURS_IN_WEEK)).fillna(0).astype(int).to_numpy()
    return {
        "hour_of_week": list(range(HOURS_IN_WEEK)),
        "multipliers": [float(x) for x in multipliers],
        "counts": [int(x) for x in counts],
    }


def _compute_tag_multipliers(
    df: pd.DataFrame,
    impact_col: str,
    tag_col: str | None,
) -> dict[str, dict[str, float | int]]:
    if not tag_col or tag_col not in df.columns:
        return {}
    data = df[[tag_col, impact_col]].dropna(subset=[tag_col])
    if data.empty:
        return {}
    data[tag_col] = data[tag_col].astype(str)
    grouped = data.groupby(tag_col)[impact_col].agg(["mean", "count"])
    overall = data[impact_col].mean()
    result: dict[str, dict[str, float | int]] = {}
    for tag, row in grouped.iterrows():
        multiplier = 1.0
        if overall and np.isfinite(overall) and overall != 0.0:
            multiplier = float(row["mean"]) / float(overall)
        result[str(tag)] = {
            "multiplier": float(multiplier),
            "impact_mean_bps": float(row["mean"]),
            "count": int(row["count"]),
        }
    return result


def _fit_symbol_params(
    df: pd.DataFrame,
    *,
    notional_bins: int,
    min_bucket_samples: int,
    regime_col: str | None,
) -> dict[str, Mapping[str, object]]:
    results: dict[str, Mapping[str, object]] = {}
    for symbol, group in df.groupby("symbol"):
        g = group.copy()
        g["abs_notional"] = g["notional"].abs()
        g = g[~g["slippage_bps"].isna()]
        g = g[g["abs_notional"] > 0]
        if g.empty:
            LOGGER.warning("symbol %s has no usable fills", symbol)
            continue

        g["spread_bps"] = g["spread_bps"].fillna(g["spread_bps"].median())
        g["half_spread_bps"] = g["spread_bps"] / 2.0
        g["impact_bps"] = g["slippage_bps"] - g["half_spread_bps"]
        g["impact_bps"] = g["impact_bps"].replace([np.inf, -np.inf], np.nan)
        g = g.dropna(subset=["impact_bps"])
        if g.empty:
            LOGGER.warning("symbol %s has no valid impact measurements", symbol)
            continue

        calc = g.copy()
        if calc["size"].isna().all():
            calc["size"] = calc["abs_notional"]
        if calc["liquidity"].isna().all():
            calc["liquidity"] = calc["abs_notional"]
        calc["vol_factor"] = calc["vol_factor"].fillna(1.0)
        calc = calc.dropna(subset=["size", "liquidity", "vol_factor", "impact_bps", "half_spread_bps"])
        calc = calc[(calc["size"] > 0) & (calc["liquidity"] > 0)]
        calc = calc.rename(columns={"impact_bps": "observed_slip_bps"})
        calc["observed_slip_bps"] = calc["observed_slip_bps"].astype(float)
        calc["half_spread_bps"] = calc["half_spread_bps"].astype(float)
        calc["size"] = calc["size"].astype(float)
        calc["liquidity"] = calc["liquidity"].astype(float)
        calc["vol_factor"] = calc["vol_factor"].astype(float)
        if calc.empty:
            LOGGER.warning("symbol %s has insufficient data for k fit", symbol)
            k_value = 0.8
        else:
            k_value = fit_k_closed_form(calc)

        default_spread = float(np.nanmedian(g["spread_bps"])) if not g["spread_bps"].isna().all() else 0.0
        half_spread = g["half_spread_bps"].dropna()
        min_half_spread = float(half_spread.quantile(0.1)) if not half_spread.empty else 0.0

        bucket_info: list[dict[str, object]] = []
        try:
            labels, bins = pd.qcut(
                g["abs_notional"],
                q=min(notional_bins, g["abs_notional"].nunique()),
                labels=False,
                retbins=True,
                duplicates="drop",
            )
        except ValueError:
            labels = pd.Series([-1] * len(g), index=g.index)
            bins = np.array([0.0, float(g["abs_notional"].max())])

        g = g.assign(_bucket=labels)
        if g["_bucket"].nunique(dropna=True) <= 1:
            # Fallback: simple quantiles are not available, build linear bins
            abs_values = g["abs_notional"].to_numpy()
            if len(abs_values) >= 2:
                bins = np.linspace(abs_values.min(), abs_values.max(), num=min(notional_bins, len(abs_values)) + 1)
                if np.all(np.isfinite(bins)) and len(np.unique(bins)) > 1:
                    g["_bucket"] = pd.cut(g["abs_notional"], bins=np.unique(bins), labels=False, include_lowest=True)

        grouped = g.groupby("_bucket")
        for bucket_id, bucket_df in grouped:
            if bucket_df.empty or len(bucket_df) < min_bucket_samples:
                continue
            bucket_lower = float(bucket_df["abs_notional"].min())
            bucket_upper = float(bucket_df["abs_notional"].max())
            bucket_info.append(
                {
                    "bucket": int(bucket_id) if bucket_id is not None and bucket_id != -1 else len(bucket_info),
                    "lower_notional": bucket_lower,
                    "upper_notional": bucket_upper,
                    "mean_notional": float(bucket_df["abs_notional"].mean()),
                    "median_notional": float(bucket_df["abs_notional"].median()),
                    "mean_slippage_bps": float(bucket_df["slippage_bps"].mean()),
                    "mean_impact_bps": float(bucket_df["impact_bps"].mean()),
                    "count": int(len(bucket_df)),
                }
            )

        hourly = _compute_hourly_multipliers(g, "impact_bps")
        tag_stats = _compute_tag_multipliers(g, "impact_bps", regime_col)

        exec_counts: MutableMapping[str, int] = defaultdict(int)
        if "execution_profile" in g.columns:
            for profile, cnt in g["execution_profile"].value_counts().items():
                if isinstance(profile, float) and math.isnan(profile):
                    continue
                exec_counts[str(profile)] = int(cnt)

        results[str(symbol)] = {
            "samples": int(len(g)),
            "k": float(k_value),
            "default_spread_bps": float(default_spread),
            "min_half_spread_bps": float(min_half_spread),
            "impact_mean_bps": float(g["impact_bps"].mean()),
            "impact_std_bps": float(g["impact_bps"].std(ddof=0)),
            "notional_curve": bucket_info,
            "hourly_multipliers": hourly,
            "regime_multipliers": {
                "column": regime_col,
                "values": tag_stats,
            }
            if regime_col
            else {},
            "execution_profile_counts": dict(exec_counts),
        }

    return results


def _detect_regime_column(df: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested regime column '{requested}' not found in dataset")
        if df[requested].notna().any():
            return requested
        LOGGER.warning("requested regime column '%s' has no non-null values", requested)
        return None
    for candidate in ("market_regime", "regime", "execution_profile"):
        if candidate in df.columns and df[candidate].notna().any():
            return candidate
    return None


def build_report(
    patterns: Sequence[str],
    *,
    notional_bins: int,
    min_bucket_samples: int,
    regime_column: str | None,
) -> dict[str, object]:
    files = _expand_patterns(patterns)
    if not files:
        raise SystemExit("no files matched input patterns")

    frames = []
    for path in files:
        LOGGER.info("loading %s", path)
        frames.append(_prepare_dataframe(_load_table(path)))

    if not frames:
        raise SystemExit("no data loaded from input files")

    df = pd.concat(frames, ignore_index=True, sort=False)
    regime_col = _detect_regime_column(df, regime_column)
    symbol_params = _fit_symbol_params(
        df,
        notional_bins=notional_bins,
        min_bucket_samples=min_bucket_samples,
        regime_col=regime_col,
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    return {
        "generated_at": generated_at,
        "source_files": [str(path) for path in files],
        "total_samples": int(len(df)),
        "regime_column": regime_col,
        "symbols": symbol_params,
        "notes": {
            "hour_of_week_definition": "0=Monday 00:00 UTC",
            "impact_definition": "slippage_bps - 0.5 * spread_bps",
        },
    }


def _dump_report(report: Mapping[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".yaml", ".yml"}:
        out_path.write_text(
            yaml.safe_dump(report, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    else:
        out_path.write_text(
            json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False), encoding="utf-8"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate slippage curves and seasonal multipliers from fill logs",
    )
    parser.add_argument(
        "--fills",
        nargs="+",
        required=True,
        help="Glob pattern(s) pointing to CSV/Parquet fill logs",
    )
    parser.add_argument(
        "--out",
        default="data/slippage/live_slippage_calibration.json",
        help="Output path for calibration artifact (JSON or YAML)",
    )
    parser.add_argument(
        "--notional-bins",
        type=int,
        default=8,
        help="Number of quantile buckets for notional curves",
    )
    parser.add_argument(
        "--min-bucket-samples",
        type=int,
        default=25,
        help="Minimum fills required per notional bucket",
    )
    parser.add_argument(
        "--regime-column",
        default=None,
        help="Optional name of column containing market regime labels",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write output file, only print summary",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    report = build_report(
        args.fills,
        notional_bins=max(1, args.notional_bins),
        min_bucket_samples=max(1, args.min_bucket_samples),
        regime_column=args.regime_column,
    )

    out_path = Path(args.out)
    if args.dry_run:
        LOGGER.info("dry-run requested, not writing %s", out_path)
        LOGGER.info("summary: %s", json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False))
        return

    if out_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Output file {out_path} already exists; use --overwrite to replace it"
        )

    _dump_report(report, out_path)
    LOGGER.info("wrote calibration artifact to %s", out_path)


if __name__ == "__main__":
    main()

