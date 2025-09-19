# scripts/fetch_binance_filters.py
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

from binance_public import BinancePublicClient


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/fetch_binance_filters.py <output_json> [SYMBOL1 SYMBOL2 ...]")
        sys.exit(2)
    out_path = sys.argv[1]
    symbols = sys.argv[2:] if len(sys.argv) > 2 else None
    client = BinancePublicClient()
    try:
        normalized = client.get_exchange_filters(symbols=symbols)
    except Exception as e:
        print(f"ERROR: failed to fetch exchangeInfo: {e}", file=sys.stderr)
        sys.exit(1)

    precision_keys = {
        "baseAssetPrecision",
        "quoteAssetPrecision",
        "baseCommissionPrecision",
        "quoteCommissionPrecision",
        "quotePrecision",
    }
    for sym_data in normalized.values():
        if not isinstance(sym_data, dict):
            continue
        for key in precision_keys:
            if key not in sym_data:
                continue
            try:
                sym_data[key] = int(sym_data[key])
            except (TypeError, ValueError):
                continue

    meta = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_dataset": "binance_exchange_filters",
        "version": 1,
    }
    payload = {"metadata": meta, "filters": normalized}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote {len(normalized)} symbols to {out_path}")


if __name__ == "__main__":
    main()
