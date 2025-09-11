#!/usr/bin/env python3
"""Visualize deviations from historical liquidity/latency distributions.

The script aggregates simulator logs by hour of week and compares observed
liquidity and latency values against historical multipliers (``0 = Monday 00:00
UTC``). It then outputs a PNG chart for quick inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.time import hour_of_week


def main() -> None:
    parser = argparse.ArgumentParser(description="Seasonality deviation dashboard")
    parser.add_argument("--log", required=True, help="Path to simulator log (csv or parquet)")
    parser.add_argument(
        "--multipliers", required=True, help="Path to liquidity_latency_seasonality.json"
    )
    parser.add_argument("--out", default="seasonality.png", help="Output PNG file")
    args = parser.parse_args()

    log_path = Path(args.log)
    if log_path.suffix == ".parquet":
        df = pd.read_parquet(log_path)
    else:
        df = pd.read_csv(log_path)
    # ``hour_of_week`` indexes 0=Monday 00:00 UTC
    df["hour"] = hour_of_week(df["ts_ms"].to_numpy(dtype=np.int64))
    agg = df.groupby("hour").agg({"liquidity": "mean", "latency_ms": "mean"}).fillna(0)

    with open(args.multipliers, "r", encoding="utf-8") as f:
        hist = json.load(f)
    liq_mult = hist.get("liquidity", [1.0] * 168)
    lat_mult = hist.get("latency", [1.0] * 168)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(liq_mult, label="historical")
    if "liquidity" in agg:
        liq_ratio = agg["liquidity"] / max(agg["liquidity"].mean(), 1e-9)
        axes[0].plot(liq_ratio.to_list(), label="run")
    axes[0].set_ylabel("Liquidity multiplier")
    axes[0].legend()

    axes[1].plot(lat_mult, label="historical")
    if "latency_ms" in agg:
        lat_ratio = agg["latency_ms"] / max(agg["latency_ms"].mean(), 1e-9)
        axes[1].plot(lat_ratio.to_list(), label="run")
    axes[1].set_ylabel("Latency multiplier")
    axes[1].set_xlabel("Hour of week")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(args.out)


if __name__ == "__main__":
    main()

