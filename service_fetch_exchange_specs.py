from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Sequence

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


def run(
    market: str = "futures",
    symbols: Sequence[str] | str | None = None,
    out: str = "data/exchange_specs.json",
) -> Dict[str, Dict[str, float]]:
    """Fetch Binance exchangeInfo and store minimal specs JSON."""

    url = _endpoint(market)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(symbols, str):
        requested = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        requested = [s.strip().upper() for s in (symbols or []) if s.strip()]

    by_symbol: Dict[str, Dict[str, float]] = {}
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

    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_dataset": f"binance_exchangeInfo_{market}",
        "version": 1,
    }
    payload = {"metadata": meta, "specs": by_symbol}

    _ensure_dir(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(by_symbol)} symbols to {out}")
    return by_symbol


__all__ = ["run"]
