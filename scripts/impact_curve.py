#!/usr/bin/env python3
"""Collect trades and compute impact vs participation curves.

The script downloads recent trades and minute bars from Binance public API,
derives participation and slippage for each trade and fits the coefficient ``k``
in :func:`estimate_slippage_bps`.  It saves per‑trade metrics, calibrated
parameters and a plot comparing empirical and model curves.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from binance_public import BinancePublicClient
from service_calibrate_slippage import fit_k_closed_form
from slippage import SlippageConfig, estimate_slippage_bps, model_curve


@dataclass
class TradeMetrics:
    size: float
    liquidity: float
    participation: float
    observed_slip_bps: float
    spread_bps: float
    half_spread_bps: float
    vol_factor: float


def fetch_recent_trades(
    client: BinancePublicClient, symbol: str, market: str, limit: int
) -> pd.DataFrame:
    data = client.get_agg_trades(market=market, symbol=symbol, limit=limit)
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["T"], unit="ms", utc=True)
    df["price"] = df["p"].astype(float)
    df["size"] = df["q"].astype(float)
    return df


def attach_minute_stats(
    client: BinancePublicClient, df: pd.DataFrame, symbol: str, market: str
) -> pd.DataFrame:
    if df.empty:
        return df
    start = int(df["T"].min())
    end = int(df["T"].max())
    kl = client.get_klines(
        market=market, symbol=symbol, interval="1m", start_ms=start, end_ms=end
    )
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "qav",
        "trades",
        "tbbv",
        "tbqv",
        "ignore",
    ]
    kdf = pd.DataFrame(kl, columns=cols)
    kdf["minute"] = pd.to_datetime(kdf["open_time"], unit="ms", utc=True)
    kdf["open"] = kdf["open"].astype(float)
    kdf["close"] = kdf["close"].astype(float)
    kdf["high"] = kdf["high"].astype(float)
    kdf["low"] = kdf["low"].astype(float)
    kdf["volume"] = kdf["volume"].astype(float)
    kdf["spread_bps"] = (kdf["high"] - kdf["low"]) / kdf["close"] * 1e4

    df = df.copy()
    df["minute"] = df["timestamp"].dt.floor("1min")
    df = df.merge(kdf[["minute", "open", "close", "volume", "spread_bps"]], on="minute", how="left")
    df["mid"] = (df["open"] + df["close"]) / 2
    df["observed_slip_bps"] = (df["price"] - df["mid"]) / df["mid"] * 1e4
    df["liquidity"] = df["volume"].replace(0, np.nan)
    df["participation"] = df["size"] / df["liquidity"]
    df["half_spread_bps"] = df["spread_bps"] / 2
    df["vol_factor"] = 1.0
    return df.dropna(subset=["liquidity"])


def compute_empirical_curve(df: pd.DataFrame, quantiles: int) -> Tuple[np.ndarray, np.ndarray]:
    labels, bins = pd.qcut(
        df["participation"], quantiles, labels=False, retbins=True, duplicates="drop"
    )
    df = df.assign(_bucket=labels)
    curve = df.groupby("_bucket")["observed_slip_bps"].mean().to_numpy()
    mids = (bins[:-1] + bins[1:]) / 2
    return mids, curve


def main() -> None:
    p = argparse.ArgumentParser(
        description="Collect trades and build impact vs participation curve",
    )
    p.add_argument("symbol")
    p.add_argument("--market", default="futures", choices=["spot", "futures"])
    p.add_argument("--trades", type=int, default=1000, help="number of trades")
    p.add_argument("--quantiles", type=int, default=20, help="number of buckets")
    p.add_argument("--out-prefix", default="impact")
    args = p.parse_args()

    client = BinancePublicClient()
    trades = fetch_recent_trades(client, args.symbol, args.market, args.trades)
    trades = attach_minute_stats(client, trades, args.symbol, args.market)
    if trades.empty:
        raise SystemExit("no trades fetched")

    trades.to_csv(f"{args.out_prefix}_trades.csv", index=False)

    mids, curve = compute_empirical_curve(trades, args.quantiles)

    cfg = SlippageConfig(
        k=fit_k_closed_form(trades),
        default_spread_bps=float(trades["spread_bps"].median()),
        min_half_spread_bps=float(trades["half_spread_bps"].quantile(0.1)),
    )
    params = {
        "k": cfg.k,
        "default_spread_bps": cfg.default_spread_bps,
        "min_half_spread_bps": cfg.min_half_spread_bps,
    }
    with open(f"{args.out_prefix}_slippage.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)

    model_y = model_curve(
        mids / trades["liquidity"].median(),
        cfg=cfg,
        spread_bps=cfg.default_spread_bps,
    )

    plt.plot(mids, curve, label="empirical")
    plt.plot(mids, model_y, label="model")
    plt.xlabel("participation")
    plt.ylabel("slippage (bps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_curve.png")


if __name__ == "__main__":
    main()
