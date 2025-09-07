# scripts/make_prices_from_klines.py
from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd


def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    p = argparse.ArgumentParser(description="Make normalized prices table from klines parquet.")
    p.add_argument("--in-klines", required=True, help="Входной parquet клайнов (любой интервал)")
    p.add_argument("--symbol", required=True, help="Символ, например BTCUSDT")
    p.add_argument("--price-col", choices=["close", "hl2", "ohlc4"], default="close", help="Какую цену считать ценой для меток")
    p.add_argument("--out", default="data/prices.parquet", help="Куда сохранить prices.parquet")
    args = p.parse_args()

    d = _read_any(args.in_klines)
    if d.empty:
        raise SystemExit("пустой вход")

    d = d.copy()
    d = d[d["symbol"].str.upper() == args.symbol.upper()].copy()
    if d.empty:
        raise SystemExit("символ не найден в клайнах")

    if args.price_col == "close":
        d["price"] = pd.to_numeric(d["close"], errors="coerce")
    elif args.price_col == "hl2":
        d["price"] = (pd.to_numeric(d["high"], errors="coerce") + pd.to_numeric(d["low"], errors="coerce")) / 2.0
    elif args.price_col == "ohlc4":
        d["price"] = (
            pd.to_numeric(d["open"], errors="coerce")
            + pd.to_numeric(d["high"], errors="coerce")
            + pd.to_numeric(d["low"], errors="coerce")
            + pd.to_numeric(d["close"], errors="coerce")
        ) / 4.0

    out = d[["ts_ms", "symbol", "price"]].dropna().sort_values(["symbol", "ts_ms"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.out.lower().endswith(".parquet"):
        out.to_parquet(args.out, index=False)
    else:
        out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
