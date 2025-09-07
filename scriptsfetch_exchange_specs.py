# scripts/fetch_exchange_specs.py
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import requests


def _endpoint(market: str) -> str:
    m = str(market).lower().strip()
    if m == "spot":
        return "https://api.binance.com/api/v3/exchangeInfo"
    # по умолчанию возьмём USDT-маржинальные фьючи
    return "https://fapi.binance.com/fapi/v1/exchangeInfo"


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    p = argparse.ArgumentParser(description="Fetch Binance exchangeInfo and save minimal specs JSON (tickSize/stepSize/minNotional) per symbol.")
    p.add_argument("--market", choices=["spot", "futures"], default="futures", help="Какой рынок опрашивать")
    p.add_argument("--symbols", default="", help="Список символов через запятую; пусто = все")
    p.add_argument("--out", default="data/exchange_specs.json", help="Куда сохранить JSON")
    args = p.parse_args()

    url = _endpoint(args.market)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    by_symbol: Dict[str, Dict] = {}
    requested: List[str] = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()] if args.symbols else []

    for s in data.get("symbols", []):
        sym = str(s.get("symbol", "")).upper()
        if requested and sym not in requested:
            continue
        tick_size = 0.0
        step_size = 0.0
        min_notional = 0.0
        for f in s.get("filters", []):
            typ = str(f.get("filterType", ""))
            if typ == "PRICE_FILTER":
                tick_size = float(f.get("tickSize", 0.0))
            elif typ == "LOT_SIZE":
                step_size = float(f.get("stepSize", 0.0))
            elif typ in ("MIN_NOTIONAL", "NOTIONAL"):
                # на фьючах ключ называется NOTIONAL
                min_notional = float(f.get("minNotional", f.get("notional", 0.0)))
        by_symbol[sym] = {
            "tickSize": tick_size,
            "stepSize": step_size,
            "minNotional": min_notional,
        }

    _ensure_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(by_symbol, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(by_symbol)} symbols to {args.out}")


if __name__ == "__main__":
    main()
