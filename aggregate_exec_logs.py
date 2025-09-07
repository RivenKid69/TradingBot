# scripts/aggregate_exec_logs.py
from __future__ import annotations

import argparse
import glob
import math
import os
from typing import List, Tuple

import pandas as pd


def _read_any(path: str) -> pd.DataFrame:
    """Reads CSV/Parquet or glob into a single DataFrame."""
    if "*" in path or "?" in path or ("[" in path and "]" in path):
        parts: List[str] = glob.glob(path)
        dfs = [ _read_any(p) for p in parts ]
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize trades to unified schema:
    ts, run_id, symbol, side, order_type, price, qty, fee, fee_asset, pnl, exec_status, liquidity, client_order_id, order_id, meta_json
    Supports legacy schema: ts, price, volume, side, agent_flag, order_id
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts","run_id","symbol","side","order_type","price","qty","fee","fee_asset","pnl","exec_status","liquidity","client_order_id","order_id","meta_json"])

    cols = set(df.columns)

    # Unified already
    if {"ts","run_id","symbol","side","order_type","price","quantity"}.issubset(cols):
        df = df.copy()
        df = df.rename(columns={"quantity":"qty"})
        for c in ["fee","pnl"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # ensure required cols exist
        for c in ["fee","fee_asset","pnl","exec_status","liquidity","client_order_id","order_id","meta_json"]:
            if c not in df.columns:
                df[c] = None
        return df[["ts","run_id","symbol","side","order_type","price","qty","fee","fee_asset","pnl","exec_status","liquidity","client_order_id","order_id","meta_json"]]

    # Legacy -> map
    if {"ts","price","volume","side"}.issubset(cols):
        out = pd.DataFrame()
        out["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
        out["run_id"] = ""
        out["symbol"] = "UNKNOWN"
        out["side"] = df["side"].astype(str).str.upper()
        out["order_type"] = "MARKET"
        out["price"] = pd.to_numeric(df["price"], errors="coerce")
        out["qty"] = pd.to_numeric(df["volume"], errors="coerce")
        out["fee"] = 0.0
        out["fee_asset"] = None
        out["pnl"] = None
        out["exec_status"] = "FILLED"
        out["liquidity"] = "UNKNOWN"
        out["client_order_id"] = None
        out["order_id"] = df["order_id"] if "order_id" in df.columns else None
        out["meta_json"] = "{}"
        return out

    # Unknown schema -> attempt minimal
    df = df.copy()
    if "ts" not in df.columns:
        df["ts"] = pd.NA
    if "price" not in df.columns:
        df["price"] = pd.NA
    if "qty" not in df.columns:
        if "quantity" in df.columns:
            df["qty"] = pd.to_numeric(df["quantity"], errors="coerce")
        elif "volume" in df.columns:
            df["qty"] = pd.to_numeric(df["volume"], errors="coerce")
        else:
            df["qty"] = pd.NA
    df["run_id"] = ""
    df["symbol"] = "UNKNOWN"
    df["side"] = df.get("side", "BUY")
    df["order_type"] = df.get("order_type", "MARKET")
    for c in ["fee","fee_asset","pnl","exec_status","liquidity","client_order_id","order_id","meta_json"]:
        if c not in df.columns:
            df[c] = None
    return df[["ts","run_id","symbol","side","order_type","price","qty","fee","fee_asset","pnl","exec_status","liquidity","client_order_id","order_id","meta_json"]]


def _bucket_ts_ms(ts_ms: pd.Series, *, bar_seconds: int) -> pd.Series:
    """Floors ms timestamp to bar_seconds buckets."""
    step = int(bar_seconds) * 1000
    return (pd.to_numeric(ts_ms, errors="coerce").astype("Int64") // step) * step


def aggregate(trades_path: str, reports_path: str, out_bars: str, out_days: str, bar_seconds: int = 60) -> Tuple[str, str]:
    """
    Aggregates trade logs into per-bar and per-day summaries.
    - trades_path: path or glob to log_trades_*.csv (unified) or legacy trades.csv
    - reports_path: optional path/glob to equity reports (csv/parquet) — if present, we will attach equity at bar ends
    - out_bars, out_days: output CSV paths
    Returns (out_bars, out_days).
    """
    trades_raw = _read_any(trades_path)
    trades = _normalize_trades(trades_raw)

    if trades.empty:
        # write empty frames to keep pipeline consistent
        pd.DataFrame().to_csv(out_bars, index=False)
        pd.DataFrame().to_csv(out_days, index=False)
        return out_bars, out_days

    # Ensure numeric types
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["qty"] = pd.to_numeric(trades["qty"], errors="coerce")
    trades["ts"] = pd.to_numeric(trades["ts"], errors="coerce").astype("Int64")
    trades["side_sign"] = trades["side"].astype(str).map(lambda s: 1 if s.upper()=="BUY" else -1)

    # Per-bar aggregation
    trades["ts_bucket"] = _bucket_ts_ms(trades["ts"], bar_seconds=bar_seconds)
    g = trades.groupby(["symbol","ts_bucket"], as_index=False)

    def _agg(df: pd.DataFrame) -> pd.Series:
        qty_abs = df["qty"].abs().sum()
        notional = (df["price"] * df["qty"].abs()).sum()
        vwap = float(notional / qty_abs) if qty_abs and math.isfinite(notional) else float("nan")
        buy_qty = df.loc[df["side_sign"]>0, "qty"].abs().sum()
        sell_qty = df.loc[df["side_sign"]<0, "qty"].abs().sum()
        n_trades = int(len(df))
        fee_sum = float(pd.to_numeric(df["fee"], errors="coerce").fillna(0.0).sum()) if "fee" in df.columns else 0.0
        return pd.Series({
            "volume": float(qty_abs),
            "buy_qty": float(buy_qty),
            "sell_qty": float(sell_qty),
            "trades": n_trades,
            "vwap": float(vwap),
            "fee_total": fee_sum,
        })

    bars = g.apply(_agg)
    bars = bars.rename(columns={"ts_bucket":"ts"})
    bars["ts"] = bars["ts"].astype("Int64")

    # Per-day aggregation (UTC days by ms timestamp)
    day_ms = 24*60*60*1000
    trades["day"] = (trades["ts"].astype("Int64") // day_ms) * day_ms
    gd = trades.groupby(["symbol","day"], as_index=False)
    days = gd.apply(lambda df: pd.Series({
        "volume": float(df["qty"].abs().sum()),
        "trades": int(len(df)),
        "buy_qty": float(df.loc[df["side_sign"]>0, "qty"].abs().sum()),
        "sell_qty": float(df.loc[df["side_sign"]<0, "qty"].abs().sum()),
        "fee_total": float(pd.to_numeric(df["fee"], errors="coerce").fillna(0.0).sum()) if "fee" in df.columns else 0.0,
        "vwap": float(((df["price"] * df["qty"].abs()).sum() / df["qty"].abs().sum())) if df["qty"].abs().sum() else float("nan"),
    }))
    days = days.rename(columns={"day":"ts"})
    days["ts"] = days["ts"].astype("Int64")

    os.makedirs(os.path.dirname(out_bars) or ".", exist_ok=True)
    bars.to_csv(out_bars, index=False)
    os.makedirs(os.path.dirname(out_days) or ".", exist_ok=True)
    days.to_csv(out_days, index=False)
    return out_bars, out_days


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate execution logs into per-bar and per-day summaries.")
    p.add_argument("--trades", required=True, help="Path or glob to unified Exec logs (log_trades_*.csv). Legacy trades.csv is supported but deprecated.")
    p.add_argument("--reports", default="", help="Optional path or glob to equity reports (csv/parquet)")
    p.add_argument("--out-bars", default="logs/agg_bars.csv", help="Output CSV path for per-bar aggregation")
    p.add_argument("--out-days", default="logs/agg_days.csv", help="Output CSV path for per-day aggregation")
    p.add_argument("--bar-seconds", type=int, default=60, help="Bar length in seconds (default: 60)")
    args = p.parse_args()

    aggregate(args.trades, args.reports, args.out_bars, args.out_days, bar_seconds=int(args.bar_seconds))


if __name__ == "__main__":
    main()
