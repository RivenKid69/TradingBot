# sim/calibrate_slippage.py
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional

import pandas as pd  # требуется pandas в окружении

from sim.slippage import SlippageConfig


def fit_k_closed_form(df: pd.DataFrame) -> float:
    """
    Простая подгонка коэффициента k в модели:
        observed_slip_bps - half_spread_bps ≈ k * vol_factor * sqrt(size/liquidity)
    => y ≈ k * x  => k = (x·y) / (x·x)
    Требуемые столбцы:
      - observed_slip_bps
      - half_spread_bps
      - vol_factor
      - size
      - liquidity
    """
    d = df.copy()
    d = d[(d["size"] > 0) & (d["liquidity"] > 0)]
    if d.empty:
        return 0.8
    x = d["vol_factor"] * (d["size"] / d["liquidity"]).pow(0.5)
    y = d["observed_slip_bps"] - d["half_spread_bps"]
    num = float((x * y).sum())
    den = float((x * x).sum())
    if den <= 0.0 or not math.isfinite(den):
        return 0.8
    k = num / den
    if not math.isfinite(k):
        k = 0.8
    return float(max(0.0, k))


def main():
    p = argparse.ArgumentParser(description="Calibrate slippage coefficient k from historical trades.")
    p.add_argument("--trades", required=True, help="CSV/Parquet with columns: price, side, qty, quote_price, observed_slip_bps, spread_bps, vol_factor, liquidity")
    p.add_argument("--out", required=True, help="Output JSON path to write {'k': ...}")
    p.add_argument("--format", choices=["csv", "parquet"], default=None, help="Optional explicit format (csv/parquet)")
    p.add_argument("--min-half-spread-bps", type=float, default=0.0)
    p.add_argument("--default-spread-bps", type=float, default=2.0)
    args = p.parse_args()

    # load
    fmt = args.format
    path = args.trades
    if fmt is None:
        fmt = "parquet" if path.lower().endswith(".parquet") else "csv"
    if fmt == "csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    # prepare
    cfg = SlippageConfig(k=0.8, min_half_spread_bps=float(args.min_half_spread_bps), default_spread_bps=float(args.default_spread_bps))
    df = df.copy()
    if "half_spread_bps" not in df.columns:
        # вычислим половину спрэда
        df["half_spread_bps"] = df["spread_bps"].fillna(cfg.default_spread_bps) * 0.5
        df["half_spread_bps"] = df["half_spread_bps"].clip(lower=cfg.min_half_spread_bps)

    # fit
    k = fit_k_closed_form(df)

    # write
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"k": float(k)}, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote k={k:.6f} to {args.out}")


if __name__ == "__main__":
    main()
