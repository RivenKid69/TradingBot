"""Validate hourly seasonality multipliers against historical data.

The hour-of-week index used throughout assumes ``0 = Monday 00:00 UTC``.
"""

import argparse
import json
import datetime as dt
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.time import hour_of_week
from execution_sim import ExecutionSimulator
from latency import LatencyModel, SeasonalLatencyModel


def _load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    ts_col = "ts_ms" if "ts_ms" in df.columns else "ts"
    if ts_col not in df.columns:
        raise ValueError("ts or ts_ms column required")
    ts_ms = df[ts_col].to_numpy(dtype=np.int64)
    # ``hour_of_week`` indexes 0=Monday 00:00 UTC
    df = df.assign(hour_of_week=hour_of_week(ts_ms))
    return df


def write_checksum(path: Path) -> Path:
    """Compute sha256 checksum for *path* and write `<path>.sha256`."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum_path = path.with_suffix(path.suffix + '.sha256')
    checksum_path.write_text(digest)
    return checksum_path


def _historical_multipliers(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    metrics: Dict[str, np.ndarray] = {}
    cols_map = {
        "liquidity": ["liquidity", "order_size", "qty", "quantity"],
        "spread_bps": ["spread_bps", "spread"],
        "latency_ms": ["latency_ms"],
    }
    for key, candidates in cols_map.items():
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            continue
        grouped = df.groupby("hour_of_week")[col].mean()
        overall = df[col].mean()
        if overall:
            mult = grouped / overall
        else:
            mult = grouped * 0.0 + 1.0
        metrics[key] = mult.reindex(range(168), fill_value=1.0).to_numpy(dtype=float)
    return metrics


def _simulate(multipliers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    base_dt = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    sim = ExecutionSimulator(
        liquidity_seasonality=multipliers.get("liquidity"),
        spread_seasonality=multipliers.get("spread"),
    )
    sim_liq, sim_spread = [], []
    for hour in range(168):
        ts = int(base_dt.timestamp() * 1000 + hour * 3_600_000)
        sim.set_market_snapshot(bid=100.0, ask=101.0, liquidity=1.0, spread_bps=1.0, ts_ms=ts)
        sim_liq.append(sim._last_liquidity or 0.0)
        sim_spread.append(sim._last_spread_bps or 0.0)
    result = {"liquidity": np.asarray(sim_liq), "spread_bps": np.asarray(sim_spread)}
    lat_mult = multipliers.get("latency")
    if lat_mult is not None:
        model = LatencyModel(base_ms=1, jitter_ms=0, spike_p=0.0, timeout_ms=10, seed=0)
        lat = SeasonalLatencyModel(model, lat_mult)
        sim_lat = []
        for hour in range(168):
            ts = int(base_dt.timestamp() * 1000 + hour * 3_600_000)
            sim_lat.append(lat.sample(ts)["total_ms"])
        result["latency_ms"] = np.asarray(sim_lat, dtype=float)
    return result


def _compare(hist: Dict[str, np.ndarray], sim: Dict[str, np.ndarray], threshold: float) -> Tuple[Dict[str, Dict[str, float]], bool]:
    stats: Dict[str, Dict[str, float]] = {}
    ok = True
    for key, hist_arr in hist.items():
        sim_arr = sim.get(key)
        if sim_arr is None:
            continue
        rel_diff = np.abs(hist_arr - sim_arr) / (np.abs(hist_arr) + 1e-9)
        max_diff = float(rel_diff.max())
        mean_diff = float(rel_diff.mean())
        stats[key] = {"max_rel_diff": max_diff, "mean_rel_diff": mean_diff}
        if max_diff > threshold:
            ok = False
    return stats, ok


def main(argv=None) -> bool:
    parser = argparse.ArgumentParser(description="Validate hourly seasonality multipliers")
    parser.add_argument(
        "--historical",
        default="data/seasonality_source/latest.parquet",
        help="Path to historical dataset (csv or parquet)",
    )
    parser.add_argument(
        "--multipliers",
        default="data/latency/liquidity_latency_seasonality.json",
        help="Path to multipliers JSON",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="If set, load multipliers for this instrument symbol",
    )
    parser.add_argument("--threshold", type=float, default=0.1, help="Max allowed relative difference")
    args = parser.parse_args(argv)

    hist_path = Path(args.historical)
    df = _load_dataset(hist_path)
    checksum_path = write_checksum(hist_path)
    print(f"Historical data checksum written to {checksum_path}")
    hist_mult = _historical_multipliers(df)
    loaded = json.loads(Path(args.multipliers).read_text())
    if args.symbol and isinstance(loaded, dict) and args.symbol in loaded:
        loaded = loaded[args.symbol]
    mult = {k: np.asarray(v, dtype=float) for k, v in loaded.items() if isinstance(v, list)}
    sim_mult = _simulate(mult)
    stats, ok = _compare(hist_mult, sim_mult, args.threshold)
    for name, vals in stats.items():
        print(f"Metric: {name}")
        print(f"  max_rel_diff: {vals['max_rel_diff']:.4f}")
        print(f"  mean_rel_diff: {vals['mean_rel_diff']:.4f}")
    if ok:
        print("✅ Seasonality validation passed")
    else:
        print("❌ Seasonality validation failed")
    return ok


if __name__ == "__main__":  # pragma: no cover - CLI
    import sys
    success = main()
    if not success:
        sys.exit(1)
