# -*- coding: utf-8 -*-
"""
incremental_klines.py
---------------------------------------------------------------
Incrementally append the last CLOSED 1h Binance candles per symbol.
• Always request limit=3 to be robust against delays/outages.
• Append only the second-to-last (closed) candle.
• Deduplicate by 'open_time' (milliseconds from Binance) and write it as-is.
Output CSV per symbol at data/candles/{SYMBOL}.csv with Binance 12 fields + 'symbol'.

CLI:
  python incremental_klines.py --symbols BTCUSDT,ETHUSDT [--close-lag-ms 2000]
  python incremental_klines.py                       # load symbols from data/universe/symbols.json
"""
from __future__ import annotations

import os
import csv
import time
import argparse
import json
from typing import List, Optional
import requests
import clock
from utils_time import is_bar_closed

BASE = "https://api.binance.com/api/v3/klines"
OUT_DIR = os.path.join("data", "candles")
os.makedirs(OUT_DIR, exist_ok=True)

HEADER = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_asset_volume","number_of_trades",
    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore","symbol"
]

SYMBOL_MAX_ERRORS = 3
SYMBOL_RETRY_BACKOFF = 1.0


def _read_last_ts(path: str) -> Optional[int]:
    """Read the last numeric open_time from an existing CSV (skip header/blank lines).
    Returns the last open_time (as int) or None if file missing/empty.
    """
    if not os.path.exists(path):
        return None
    last_ts: Optional[int] = None
    try:
        with open(path, "r", newline="") as f:
            r = csv.reader(f)
            for row in r:
                if not row:
                    continue
                try:
                    ts = int(row[0])
                except Exception:
                    continue  # skip header or malformed
                else:
                    last_ts = ts
    except Exception:
        return None
    return last_ts


def _get_with_retry(url: str, *, params: dict, retries: int = 3, backoff: float = 0.5):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)


def fetch_last_closed(symbol: str) -> Optional[list]:
    """Fetch last 3 klines for 1h interval and return the second-to-last (closed) one."""
    params = {"symbol": symbol.upper(), "interval": "1h", "limit": 3}
    r = _get_with_retry(BASE, params=params)
    data = r.json()
    if not isinstance(data, list) or len(data) < 2:
        return None
    # Binance kline structure:
    # [ open_time, open, high, low, close, volume, close_time,
    #   quote_asset_volume, number_of_trades,
    #   taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore ]
    closed = data[-2]  # second to last is guaranteed CLOSED
    return closed


def append_closed(symbol: str, close_lag_ms: int) -> bool:
    """Append the latest CLOSED candle for symbol if it advances the timeline.

    Parameters
    ----------
    symbol : str
        Trading pair symbol, e.g. ``BTCUSDT``.
    close_lag_ms : int
        Allowed lag in milliseconds when verifying bar close time.

    Returns
    -------
    bool
        ``True`` if the bar was appended, ``False`` otherwise.
    """
    path = os.path.join(OUT_DIR, f"{symbol.upper()}.csv")
    row = fetch_last_closed(symbol)
    if row is None:
        return False

    close_ts = int(row[0]) + 3_600_000  # open time + 1h in ms
    now_ms = clock.now_ms()
    if not is_bar_closed(close_ts, now_ms, close_lag_ms):
        print(
            f"[WARN] {symbol}: bar not closed yet (close_ts={close_ts} now={now_ms} lag={close_lag_ms})"
        )
        return False

    open_time = int(row[0])
    out = list(map(str, row + [symbol.upper()]))  # add symbol as the 13th column

    # Dedup: only append if strictly newer than last present open_time
    last_ts = _read_last_ts(path)
    if last_ts is not None and open_time <= last_ts:
        return False

    # Append (create with header if new file)
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerow(out)
    else:
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(out)
    return True


def run_many(symbols: List[str], close_lag_ms: int) -> int:
    appended = 0
    skipped: List[str] = []
    for sym in symbols:
        errors = 0
        while True:
            try:
                if append_closed(sym, close_lag_ms):
                    appended += 1
                break
            except Exception as e:
                errors += 1
                print(
                    f"[WARN] {sym}: {e} (attempt {errors}/{SYMBOL_MAX_ERRORS})"
                )
                if errors >= SYMBOL_MAX_ERRORS:
                    print(
                        f"[WARN] {sym}: skipping after {SYMBOL_MAX_ERRORS} errors"
                    )
                    skipped.append(sym)
                    break
                time.sleep(SYMBOL_RETRY_BACKOFF * errors)
    if skipped:
        skipped_names = ", ".join(sorted(set(skipped)))
        print(f"[WARN] Skipped symbols: {skipped_names}")
    print(f"\u2713 Appended {appended} closed bars.")
    return appended


def _parse_symbols(s: str) -> List[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally append last CLOSED 1h candles from Binance."
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols or leave empty to use data/universe/symbols.json",
    )
    parser.add_argument(
        "--close-lag-ms",
        type=int,
        default=2000,
        help="Allowed lag in ms when verifying if the fetched bar is closed",
    )
    args = parser.parse_args()
    if args.symbols:
        symbols = _parse_symbols(args.symbols)
    else:
        json_path = os.path.join("data", "universe", "symbols.json")
        with open(json_path, "r") as f:
            symbols = [s.strip().upper() for s in json.load(f)]
    run_many(symbols, args.close_lag_ms)


if __name__ == "__main__":
    main()
