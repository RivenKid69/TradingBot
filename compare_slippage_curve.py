#!/usr/bin/env python3
"""Compare slippage curves between historical and simulated trades.

The script loads two CSV files containing executed trades with at least the
following columns:

``order_size``
    Absolute size of the executed order.
``slippage_bps``
    Slippage in basis points relative to a reference price.

For a configurable number of quantiles the script computes the average
slippage for trades falling into each order size bucket.  The two resulting
curves are plotted for visual inspection and compared numerically.  If the
absolute difference between the curves exceeds the allowed tolerance the
script exits with a non‑zero code.
"""

from __future__ import annotations

import argparse
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_trades(path: str) -> pd.DataFrame:
    """Load trades from a CSV file."""
    df = pd.read_csv(path)
    missing = {"order_size", "slippage_bps"} - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    return df


def _slippage_curve(df: pd.DataFrame, quantiles: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return midpoints of order size buckets and average slippage."""
    labels, bins = pd.qcut(
        df["order_size"], quantiles, labels=False, retbins=True, duplicates="drop"
    )
    df = df.assign(_bucket=labels)
    curve = df.groupby("_bucket")["slippage_bps"].mean().to_numpy()
    mids = (bins[:-1] + bins[1:]) / 2
    return mids, curve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare slippage curves of historical and simulated trades.",
    )
    parser.add_argument("historical", help="CSV file with historical trades")
    parser.add_argument("simulated", help="CSV file with simulated trades")
    parser.add_argument(
        "--quantiles", type=int, default=10, help="Number of order size quantiles"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
        help="Maximum allowed deviation in bps between curves",
    )
    parser.add_argument(
        "--plot",
        default="slippage_curve.png",
        help="Where to save the resulting plot",
    )
    args = parser.parse_args()

    hist = _load_trades(args.historical)
    sim = _load_trades(args.simulated)
    h_x, h_y = _slippage_curve(hist, args.quantiles)
    s_x, s_y = _slippage_curve(sim, args.quantiles)

    if len(h_y) != len(s_y):
        print("Quantile mismatch between datasets", file=sys.stderr)
        sys.exit(1)

    diff = np.abs(h_y - s_y)
    ok = np.all(diff <= args.tolerance)

    for i, (x, h, s, d) in enumerate(zip(h_x, h_y, s_y, diff)):
        print(f"q{i}: size={x:.2f} hist={h:.2f} sim={s:.2f} diff={d:.2f}")

    plt.plot(h_x, h_y, label="historical")
    plt.plot(s_x, s_y, label="simulated")
    plt.xlabel("order size")
    plt.ylabel("slippage (bps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.plot)

    if not ok:
        print(
            f"slippage curves deviate by more than {args.tolerance} bps", file=sys.stderr
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
