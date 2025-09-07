# scripts/ingest_klines.py
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from binance_public import BinancePublicClient
from utils_time import parse_time_to_ms


KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def _fetch_all_klines(client: BinancePublicClient, *, market: str, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1500, sleep_ms: int = 350) -> List[List[Any]]:
    """
    Идём по временной оси слева направо, пока не дойдём до end_ms.
    """
    out: List[List[Any]] = []
    cur = int(start_ms)
    while cur < end_ms:
        batch = client.get_klines(market=market, symbol=symbol, interval=interval, start_ms=cur, end_ms=end_ms, limit=limit)
        if not batch:
            # двигаем окно на один интервал (эвристика), чтобы избежать вечного цикла
            cur += 60_000
            time.sleep(sleep_ms / 1000.0)
            continue
        out.extend(batch)
        last_close = int(batch[-1][6])  # close_time
        # Binance включает свечи по endTime, поэтому шагаем на +1 ms от последнего close_time
        cur = max(cur + 1, last_close + 1)
        time.sleep(sleep_ms / 1000.0)
    return out


def _to_df(raw: List[List[Any]], symbol: str) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame(columns=["ts_ms", "symbol", "open", "high", "low", "close", "volume", "number_of_trades", "taker_buy_base", "taker_buy_quote"])
    df = pd.DataFrame(raw, columns=KLINE_COLS)
    df["ts_ms"] = df["open_time"].astype("int64")
    df["symbol"] = str(symbol).upper()
    out = df[["ts_ms", "symbol", "open", "high", "low", "close", "volume", "number_of_trades", "taker_buy_base", "taker_buy_quote"]].copy()
    # типы
    for c in ["open", "high", "low", "close", "volume", "taker_buy_base", "taker_buy_quote"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["number_of_trades"] = pd.to_numeric(out["number_of_trades"], errors="coerce").astype("Int64")
    return out.sort_values(["symbol", "ts_ms"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="Ingest Binance klines (public, no keys).")
    p.add_argument("--market", choices=["spot", "futures"], default="spot", help="Рынок: spot или futures (USDT-M)")
    p.add_argument("--symbols", required=True, help="Символы через запятую, например BTCUSDT,ETHUSDT")
    p.add_argument("--interval", default="1m", help="Интервал kline: 1m/3m/5m/15m/1h/4h/1d и т.п.")
    p.add_argument("--start", required=True, help="Начало периода (YYYY-MM-DD, ISO или unix ms)")
    p.add_argument("--end", required=True, help="Конец периода (YYYY-MM-DD, ISO или unix ms)")
    p.add_argument("--out-dir", default="data/klines", help="Куда писать parquet по символам")
    p.add_argument("--limit", type=int, default=1500, help="Лимит на запрос API")
    p.add_argument("--sleep-ms", type=int, default=350, help="Пауза между запросами (ms)")
    args = p.parse_args()

    start_ms = parse_time_to_ms(args.start)
    end_ms = parse_time_to_ms(args.end)
    if end_ms <= start_ms:
        raise SystemExit("end <= start")

    os.makedirs(args.out_dir, exist_ok=True)

    client = BinancePublicClient()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    for sym in symbols:
        raw = _fetch_all_klines(client, market=args.market, symbol=sym, interval=args.interval, start_ms=start_ms, end_ms=end_ms, limit=args.limit, sleep_ms=args.sleep_ms)
        df = _to_df(raw, sym)
        out_path = os.path.join(args.out_dir, f"{sym}_{args.interval}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"Wrote {len(df)} rows: {out_path}")


if __name__ == "__main__":
    main()
