from __future__ import annotations

import json
import os
import time
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


_DEFAULT_TTL_SECONDS = 24 * 60 * 60


def _is_stale(path: str, ttl: int) -> bool:
    """Return ``True`` if ``path`` is missing or older than ``ttl`` seconds."""
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return True
    return (time.time() - mtime) > ttl


def get_symbols(
    ttl: int = _DEFAULT_TTL_SECONDS, out: str = "data/universe/symbols.json"
) -> List[str]:
    """Return cached Binance symbols list, refreshing if ``ttl`` expired."""

    if _is_stale(out, ttl):
        run(out)

    with open(out, "r", encoding="utf-8") as f:
        return json.load(f)


__all__ = ["run", "get_symbols"]

# Perform a freshness check when the module is imported so that consumers
# get an up-to-date universe without needing to call ``get_symbols`` first.
try:  # pragma: no cover - network may be unavailable during tests
    get_symbols()
except Exception:
    # The refresh is best effort; failures are surfaced on explicit call.
    pass
