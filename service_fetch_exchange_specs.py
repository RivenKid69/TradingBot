from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Dict, Sequence

import requests
from binance_public import BinancePublicClient


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
    volume_threshold: float = 0.0,
    volume_out: str | None = None,
    days: int = 30,
) -> Dict[str, Dict[str, float]]:
    """Fetch Binance exchangeInfo and store minimal specs JSON.

    Additionally computes average daily quote volume over the last ``days``
    for each symbol and optionally filters out symbols whose average falls
    below ``volume_threshold``.  The computed averages can be stored in
    ``volume_out`` for transparency.
    """

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

    # --- compute average quote volume per symbol ---
    client = BinancePublicClient()
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(days) * 86_400_000
    avg_quote_vol: Dict[str, float] = {}
    for sym in list(by_symbol.keys()):
        try:
            kl = client.get_klines(
                market=market,
                symbol=sym,
                interval="1d",
                start_ms=start_ms,
                end_ms=end_ms,
                limit=days,
            )
            vols = [float(k[7]) for k in kl]
            avg_quote_vol[sym] = sum(vols) / len(vols) if vols else 0.0
        except Exception:
            avg_quote_vol[sym] = 0.0

    if volume_threshold > 0.0:
        before = len(by_symbol)
        by_symbol = {
            sym: spec
            for sym, spec in by_symbol.items()
            if avg_quote_vol.get(sym, 0.0) >= volume_threshold
        }
        dropped = before - len(by_symbol)
        print(
            f"Dropped {dropped} symbols below volume threshold {volume_threshold}"
        )

    if volume_out:
        _ensure_dir(volume_out)
        with open(volume_out, "w", encoding="utf-8") as f:
            json.dump(avg_quote_vol, f, ensure_ascii=False, indent=2)

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
