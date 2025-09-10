"""Aggregate execution logs into per-bar and per-day summaries.

Example
-------
```
python aggregate_exec_logs.py \
    --trades 'logs/log_trades_*.csv' \
    --reports 'logs/report_equity_*.csv' \
    --out-bars logs/agg_bars.csv \
    --out-days logs/agg_days.csv \
    --equity-png logs/equity.png \
    --metrics-md logs/metrics.md
```
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Tuple

import pandas as pd

from services.metrics import (
    calculate_metrics,
    plot_equity_curve,
    read_any,
)


def _normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize trades to unified schema:
    ts, run_id, symbol, side, order_type, price, quantity, fee, fee_asset, pnl, exec_status, liquidity, client_order_id, order_id, meta_json
    Supports legacy schema: ts, price, volume, side, agent_flag, order_id
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts","run_id","symbol","side","order_type","price","quantity","fee","fee_asset","pnl","exec_status","liquidity","client_order_id","order_id","meta_json"])

    cols = set(df.columns)

    # Unified already
    if {"ts","run_id","symbol","side","order_type","price","quantity"}.issubset(cols):
        df = df.copy()
        for c in ["fee","pnl"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # ensure required cols exist
        for c in ["fee","fee_asset","pnl","exec_status","liquidity","client_order_id","order_id","meta_json","execution_profile"]:
            if c not in df.columns:
                df[c] = None
        return df[["ts","run_id","symbol","side","order_type","price","quantity","fee","fee_asset","pnl","exec_status","liquidity","client_order_id","order_id","execution_profile","meta_json"]]

    # Legacy -> map
    if {"ts","price","volume","side"}.issubset(cols):
        out = pd.DataFrame()
        out["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
        out["run_id"] = ""
        out["symbol"] = "UNKNOWN"
        out["side"] = df["side"].astype(str).str.upper()
        out["order_type"] = "MARKET"
        out["price"] = pd.to_numeric(df["price"], errors="coerce")
        out["quantity"] = pd.to_numeric(df["volume"], errors="coerce")
        out["fee"] = 0.0
        out["fee_asset"] = None
        out["pnl"] = None
        out["exec_status"] = "FILLED"
        out["liquidity"] = "UNKNOWN"
        out["client_order_id"] = None
        out["order_id"] = df["order_id"] if "order_id" in df.columns else None
        out["meta_json"] = "{}"
        out["execution_profile"] = None
        return out

    # Unknown schema -> attempt minimal
    df = df.copy()
    if "ts" not in df.columns:
        df["ts"] = pd.NA
    if "price" not in df.columns:
        df["price"] = pd.NA
    if "quantity" not in df.columns:
        if "volume" in df.columns:
            df["quantity"] = pd.to_numeric(df["volume"], errors="coerce")
        else:
            df["quantity"] = pd.NA
    df["run_id"] = ""
    df["symbol"] = "UNKNOWN"
    df["side"] = df.get("side", "BUY")
    df["order_type"] = df.get("order_type", "MARKET")
    for c in ["fee","fee_asset","pnl","exec_status","liquidity","client_order_id","order_id","meta_json","execution_profile"]:
        if c not in df.columns:
            df[c] = None
    return df[["ts","run_id","symbol","side","order_type","price","quantity","fee","fee_asset","pnl","exec_status","liquidity","client_order_id","order_id","execution_profile","meta_json"]]


def _normalize_reports(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize equity reports to at least ts_ms and equity columns."""
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["ts_ms", "symbol", "equity", "fee_total", "funding_cashflow", "execution_profile"]
        )

    r = df.copy()
    cols = set(r.columns)

    if "ts_ms" not in cols:
        for candidate in ["ts", "timestamp", "time"]:
            if candidate in r.columns:
                r = r.rename(columns={candidate: "ts_ms"})
                break

    if "symbol" not in r.columns:
        r["symbol"] = "UNKNOWN"

    for c in ["equity", "fee_total", "funding_cashflow", "bid", "ask", "mtm_price", "execution_profile"]:
        if c not in r.columns:
            r[c] = pd.NA

    r["ts_ms"] = pd.to_numeric(r["ts_ms"], errors="coerce").astype("Int64")
    r["equity"] = pd.to_numeric(r["equity"], errors="coerce")
    r["fee_total"] = pd.to_numeric(r["fee_total"], errors="coerce")
    r["funding_cashflow"] = pd.to_numeric(r["funding_cashflow"], errors="coerce")
    r["bid"] = pd.to_numeric(r["bid"], errors="coerce")
    r["ask"] = pd.to_numeric(r["ask"], errors="coerce")
    r["mtm_price"] = pd.to_numeric(r["mtm_price"], errors="coerce")

    return r[["ts_ms", "symbol", "equity", "fee_total", "funding_cashflow", "bid", "ask", "mtm_price", "execution_profile"]]


def _bucket_ts_ms(ts_ms: pd.Series, *, bar_seconds: int) -> pd.Series:
    """Floors ms timestamp to bar_seconds buckets."""
    step = int(bar_seconds) * 1000
    return (pd.to_numeric(ts_ms, errors="coerce").astype("Int64") // step) * step


def recompute_pnl(trades: pd.DataFrame, reports: pd.DataFrame) -> pd.Series:
    """Recompute total PnL for each report row.

    Trades must contain ``ts``, ``price``, ``quantity`` and ``side`` columns and
    are expected to be in milliseconds. Reports should include ``ts_ms`` along
    with ``bid``, ``ask`` and optional ``mtm_price`` used for mark to market.

    Returns a :class:`pandas.Series` aligned with ``reports`` containing the
    recomputed PnL (realized + unrealized) at each report timestamp.
    """

    if trades is None or trades.empty or reports is None or reports.empty:
        return pd.Series(dtype=float)

    t = trades.sort_values("ts").copy()
    r = reports.sort_values("ts_ms").copy()

    pos = 0.0
    avg = None
    realized = 0.0
    i = 0
    out: list[float] = []
    t_ts = t["ts"].values.tolist()
    t_side = t["side"].astype(str).str.upper().values.tolist()
    t_price = t["price"].astype(float).values.tolist()
    t_qty = t["quantity"].astype(float).abs().values.tolist()

    for _, rep in r.iterrows():
        ts = float(rep["ts_ms"]) if pd.notna(rep["ts_ms"]) else float("inf")
        while i < len(t_ts) and t_ts[i] <= ts:
            price = t_price[i]
            qty = t_qty[i]
            side = t_side[i]
            if side == "BUY":
                if pos < 0.0:
                    close_qty = min(qty, -pos)
                    if avg is not None:
                        realized += (avg - price) * close_qty
                    pos += close_qty
                    qty -= close_qty
                    if qty > 0.0:
                        pos += qty
                        avg = price
                    elif pos == 0.0:
                        avg = None
                else:
                    new_pos = pos + qty
                    avg = (avg * pos + price * qty) / new_pos if pos > 0.0 and avg is not None else price
                    pos = new_pos
            else:  # SELL
                if pos > 0.0:
                    close_qty = min(qty, pos)
                    if avg is not None:
                        realized += (price - avg) * close_qty
                    pos -= close_qty
                    qty -= close_qty
                    if qty > 0.0:
                        pos -= qty
                        avg = price
                    elif pos == 0.0:
                        avg = None
                else:
                    new_pos = pos - qty
                    avg = (avg * (-pos) + price * qty) / (-new_pos) if pos < 0.0 and avg is not None else price
                    pos = new_pos
            i += 1

        bid = rep.get("bid")
        ask = rep.get("ask")
        mark_p = rep.get("mtm_price")
        if pd.isna(mark_p):
            mark_p = None
        if mark_p is None:
            if pos > 0.0 and pd.notna(bid):
                mark_p = float(bid)
            elif pos < 0.0 and pd.notna(ask):
                mark_p = float(ask)
            elif pd.notna(bid) and pd.notna(ask):
                mark_p = float((float(bid) + float(ask)) / 2.0)

        unrealized = 0.0
        if mark_p is not None and avg is not None and pos != 0.0:
            if pos > 0.0:
                unrealized = (float(mark_p) - avg) * pos
            else:
                unrealized = (avg - float(mark_p)) * (-pos)
        out.append(realized + unrealized)

    return pd.Series(out, index=r.index)


def aggregate(
    trades_path: str,
    reports_path: str,
    out_bars: str,
    out_days: str,
    *,
    bar_seconds: int = 60,
    equity_png: str = "",
    metrics_md: str = "",
) -> Tuple[str, str]:
    """
    Aggregates trade logs into per-bar and per-day summaries.
    - trades_path: path or glob to log_trades_*.csv (unified) or legacy trades.csv
    - reports_path: optional path/glob to equity reports (csv/parquet) — if present, we will attach equity at bar ends
    - out_bars, out_days: output CSV paths
    Returns (out_bars, out_days).
    """
    trades_raw = read_any(trades_path)
    trades = _normalize_trades(trades_raw)

    if trades.empty:
        # write empty frames to keep pipeline consistent
        pd.DataFrame().to_csv(out_bars, index=False)
        pd.DataFrame().to_csv(out_days, index=False)
        return out_bars, out_days

    # Ensure numeric types
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce")
    trades["ts"] = pd.to_numeric(trades["ts"], errors="coerce").astype("Int64")
    trades["side_sign"] = trades["side"].astype(str).map(lambda s: 1 if s.upper() == "BUY" else -1)

    # Per-bar aggregation
    trades["ts_bucket"] = _bucket_ts_ms(trades["ts"], bar_seconds=bar_seconds)
    g = trades.groupby(["symbol","ts_bucket"], as_index=False)

    def _agg(df: pd.DataFrame) -> pd.Series:
        qty_abs = df["quantity"].abs().sum()
        notional = (df["price"] * df["quantity"].abs()).sum()
        vwap = float(notional / qty_abs) if qty_abs and math.isfinite(notional) else float("nan")
        buy_qty = df.loc[df["side_sign"]>0, "quantity"].abs().sum()
        sell_qty = df.loc[df["side_sign"]<0, "quantity"].abs().sum()
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
    bars = bars.rename(columns={"ts_bucket": "ts"})
    bars["ts"] = bars["ts"].astype("Int64")

    # Per-day aggregation (UTC days by ms timestamp)
    day_ms = 24*60*60*1000
    trades["day"] = (trades["ts"].astype("Int64") // day_ms) * day_ms
    gd = trades.groupby(["symbol","day"], as_index=False)
    days = gd.apply(lambda df: pd.Series({
        "volume": float(df["quantity"].abs().sum()),
        "trades": int(len(df)),
        "buy_qty": float(df.loc[df["side_sign"]>0, "quantity"].abs().sum()),
        "sell_qty": float(df.loc[df["side_sign"]<0, "quantity"].abs().sum()),
        "fee_total": float(pd.to_numeric(df["fee"], errors="coerce").fillna(0.0).sum()) if "fee" in df.columns else 0.0,
        "vwap": float(((df["price"] * df["quantity"].abs()).sum() / df["quantity"].abs().sum())) if df["quantity"].abs().sum() else float("nan"),
    }))
    days = days.rename(columns={"day":"ts"})
    days["ts"] = days["ts"].astype("Int64")

    reports = _normalize_reports(read_any(reports_path)) if reports_path else _normalize_reports(pd.DataFrame())
    if not reports.empty:
        rep_sorted = reports.sort_values(["symbol", "ts_ms"])
        bars = pd.merge_asof(
            bars.sort_values(["symbol", "ts"]),
            rep_sorted,
            left_on="ts",
            right_on="ts_ms",
            by="symbol",
            direction="backward",
        ).drop(columns=["ts_ms"])
        days = pd.merge_asof(
            days.sort_values(["symbol", "ts"]),
            rep_sorted,
            left_on="ts",
            right_on="ts_ms",
            by="symbol",
            direction="backward",
        ).drop(columns=["ts_ms"])
        if equity_png:
            try:
                plot_equity_curve(reports, equity_png)
            except Exception:
                pass
    else:
        if equity_png:
            os.makedirs(os.path.dirname(equity_png) or ".", exist_ok=True)

    if metrics_md:
        trades_for_metrics = trades.rename(columns={"quantity": "qty"})
        metrics = calculate_metrics(trades_for_metrics, reports)
        os.makedirs(os.path.dirname(metrics_md) or ".", exist_ok=True)
        with open(metrics_md, "w", encoding="utf-8") as f:
            f.write("# Performance Metrics\n\n")
            f.write("## Equity\n")
            for k, v in metrics["equity"].items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n## Trades\n")
            for k, v in metrics["trades"].items():
                f.write(f"- **{k}**: {v}\n")

    os.makedirs(os.path.dirname(out_bars) or ".", exist_ok=True)
    bars.to_csv(out_bars, index=False)
    os.makedirs(os.path.dirname(out_days) or ".", exist_ok=True)
    days.to_csv(out_days, index=False)
    return out_bars, out_days


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate execution logs into per-bar and per-day summaries.")
    p.add_argument("--trades", required=True, help="Path or glob to unified Exec logs (log_trades_*.csv). Legacy trades.csv is supported but deprecated.")
    p.add_argument(
        "--reports",
        default="",
        help="Optional path or glob to equity reports (report_equity_*.csv)",
    )
    p.add_argument("--out-bars", default="logs/agg_bars.csv", help="Output CSV path for per-bar aggregation")
    p.add_argument("--out-days", default="logs/agg_days.csv", help="Output CSV path for per-day aggregation")
    p.add_argument("--bar-seconds", type=int, default=60, help="Bar length in seconds (default: 60)")
    p.add_argument("--equity-png", default="", help="Optional path to save equity curve PNG")
    p.add_argument("--metrics-md", default="", help="Optional path to save metrics summary in Markdown")
    args = p.parse_args()

    aggregate(
        args.trades,
        args.reports,
        args.out_bars,
        args.out_days,
        bar_seconds=int(args.bar_seconds),
        equity_png=args.equity_png,
        metrics_md=args.metrics_md,
    )


if __name__ == "__main__":
    main()
