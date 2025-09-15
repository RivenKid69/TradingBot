from __future__ import annotations

import json
import os
from typing import List

import requests


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d:
        os.makedirs(d, exist_ok=True)


def run(out: str = "data/universe/symbols.json") -> List[str]:
    """Fetch Binance spot symbols trading against USDT and store them.

    Parameters
    ----------
    out:
        Destination JSON file. The parent directory is created if needed.
    Returns
    -------
    List[str]
        Sorted list of symbols that were saved to ``out``.
    """

    resp = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=20)
    resp.raise_for_status()
    data = resp.json()

    symbols = [
        s["symbol"].upper()
        for s in data.get("symbols", [])
        if s.get("status") == "TRADING"
        and s.get("quoteAsset") == "USDT"
        and "SPOT" in s.get("permissions", [])
    ]
    symbols.sort()

    _ensure_dir(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(symbols, f, ensure_ascii=False, indent=2)
    return symbols


__all__ = ["run"]
