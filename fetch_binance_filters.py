# scripts/fetch_binance_filters.py
from __future__ import annotations

import json
import os
import sys
from typing import Dict, Any, List

import urllib.request
import urllib.error
import urllib.parse


def _fetch_exchange_info(symbols: List[str]|None=None) -> Dict[str, Any]:
    base = "https://api.binance.com/api/v3/exchangeInfo"
    if symbols:
        sym_param = "[" + ",".join(symbols) + "]"
        url = base + "?symbols=" + urllib.parse.quote(sym_param)
    else:
        url = base
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data


def _normalize_filters(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    symbols = raw.get("symbols") or []
    for s in symbols:
        sym = s.get("symbol")
        filts = s.get("filters") or []
        d: Dict[str, Any] = {}
        for f in filts:
            ftype = f.get("filterType")
            if not ftype:
                continue
            d[ftype] = {k: v for k, v in f.items() if k != "filterType"}
        out[sym] = d
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/fetch_binance_filters.py <output_json> [SYMBOL1 SYMBOL2 ...]")
        sys.exit(2)
    out_path = sys.argv[1]
    symbols = sys.argv[2:] if len(sys.argv) > 2 else None
    try:
        data = _fetch_exchange_info(symbols)
    except Exception as e:
        print(f"ERROR: failed to fetch exchangeInfo: {e}", file=sys.stderr)
        sys.exit(1)
    normalized = _normalize_filters(data)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote {len(normalized)} symbols to {out_path}")


if __name__ == "__main__":
    main()
