# scripts/ingest_funding_mark.py
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from binance_public import BinancePublicClient
from utils_time import parse_time_to_ms


def _fetch_all_funding(client: BinancePublicClient, *, symbol: str, start_ms: int, end_ms: int, limit: int = 1000, sleep_ms: int = 350) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cur = int(start_ms)
    while cur < end_ms:
        batch = client.get_funding(symbol=symbol, start_ms=cur, end_ms=end_ms, limit=limit)
        if not batch:
            cur += 8 * 60 * 60 * 1000  # funding каждые 8 часов — шагнём на интервал
            time.sleep(sleep_ms / 1000.0)
            continue
        out.extend(batch)
        last_ts = int(batch[-1]["fundingTime"])
        cur = max(cur + 1, last_ts + 1)
        time.sleep(sleep_ms / 1000.0)
    return out


def _funding_to_df(raw: List[Dict[str, Any]], symbol: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["ts_ms", "symbol", "funding_rate"])
    d = pd.DataFrame(raw)
    d["ts_ms"] = d["fundingTime"].astype("int64")
    d["symbol"] = str(symbol).upper()
    d["funding_rate"] = pd.to_numeric(d["fundingRate"], errors="coerce")
    out = d[["ts_ms", "symbol", "funding_rate"]].sort_values(["symbol", "ts_ms"]).reset_index(drop=True)
    return out


def _fetch_all_mark(client: BinancePublicClient, *, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1500, sleep_ms: int = 350) -> List[List[Any]]:
    out: List[List[Any]] = []
    cur = int(start_ms)
    while cur < end_ms:
        batch = client.get_mark_klines(symbol=symbol, interval=interval, start_ms=cur, end_ms=end_ms, limit=limit)
        if not batch:
            cur += 60_000
            time.sleep(sleep_ms / 1000.0)
            continue
        out.extend(batch)
        last_close = int(batch[-1][6])  # close_time
        cur = max(cur + 1, last_close + 1)
        time.sleep(sleep_ms / 1000.0)
    return out


def _mark_to_df(raw: List[List[Any]], symbol: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["ts_ms", "symbol", "mark_open", "mark_high", "mark_low", "mark_close"])
    cols = [
        "open_time", "open", "high", "low", "close", "ignore",
        "close_time", "ignore2", "ignore3", "ignore4", "ignore5", "ignore6"
    ]
    df = pd.DataFrame(raw, columns=cols[:len(raw[0])])
    df["ts_ms"] = df["open_time"].astype("int64")
    df["symbol"] = str(symbol).upper()
    out = df[["ts_ms", "symbol", "open", "high", "low", "close"]].copy()
    out = out.rename(columns={"open": "mark_open", "high": "mark_high", "low": "mark_low", "close": "mark_close"})
    for c in ["mark_open", "mark_high", "mark_low", "mark_close"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values(["symbol", "ts_ms"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="Ingest Binance Futures funding and mark-price klines (public).")
    p.add_argument("--symbol", required=True, help="Символ фьючерса, например BTCUSDT")
    p.add_argument("--start", required=True, help="Начало периода (YYYY-MM-DD или unix ms)")
    p.add_argument("--end", required=True, help="Конец периода (YYYY-MM-DD или unix ms)")
    p.add_argument("--mark-interval", default="1m", help="Интервал mark-price klines")
    p.add_argument("--out-dir", default="data/futures", help="Куда писать parquet")
    p.add_argument("--limit", type=int, default=1500, help="Лимит на запрос")
    p.add_argument("--sleep-ms", type=int, default=350, help="Пауза между запросами")
    args = p.parse_args()

    start_ms = parse_time_to_ms(args.start)
    end_ms = parse_time_to_ms(args.end)
    os.makedirs(args.out_dir, exist_ok=True)

    client = BinancePublicClient()

    # funding
    f_raw = _fetch_all_funding(client, symbol=args.symbol, start_ms=start_ms, end_ms=end_ms, limit=min(1000, args.limit), sleep_ms=args.sleep_ms)
    f_df = _funding_to_df(f_raw, args.symbol)
    f_out = os.path.join(args.out_dir, f"{args.symbol}_funding.parquet")
    f_df.to_parquet(f_out, index=False)
    print(f"Wrote {len(f_df)} rows: {f_out}")

    # mark price klines
    m_raw = _fetch_all_mark(client, symbol=args.symbol, interval=args.mark_interval, start_ms=start_ms, end_ms=end_ms, limit=args.limit, sleep_ms=args.sleep_ms)
    m_df = _mark_to_df(m_raw, args.symbol)
    m_out = os.path.join(args.out_dir, f"{args.symbol}_mark_{args.mark_interval}.parquet")
    m_df.to_parquet(m_out, index=False)
    print(f"Wrote {len(m_df)} rows: {m_out}")


if __name__ == "__main__":
    main()
