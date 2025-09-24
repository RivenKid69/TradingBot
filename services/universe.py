from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import requests


_REQUEST_THROTTLE_SECONDS = 0.2
_REQUEST_MAX_ATTEMPTS = 3
_REQUEST_BACKOFF_BASE = 0.5
_last_request_ts: float = 0.0


def _throttled_get(
    url: str,
    *,
    params: Optional[Dict[str, str]] = None,
    timeout: int = 20,
    max_attempts: int = _REQUEST_MAX_ATTEMPTS,
    backoff_base: float = _REQUEST_BACKOFF_BASE,
) -> requests.Response:
    """Perform ``requests.get`` with basic throttling and backoff."""

    global _last_request_ts

    params = params or {}
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        now = time.time()
        wait = _REQUEST_THROTTLE_SECONDS - (now - _last_request_ts)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if attempt == max_attempts:
                raise
            time.sleep(backoff_base * attempt)
        else:
            _last_request_ts = time.time()
            return resp
    # ``raise`` above ensures this is unreachable, keep for mypy friendliness
    assert last_error is not None
    raise last_error


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.fspath(path)) or "."
    os.makedirs(directory, exist_ok=True)


def run(
    out: str = "data/universe/symbols.json", liquidity_threshold: float = 0.0
) -> List[str]:
    """Fetch Binance spot symbols trading against USDT and store them.

    Parameters
    ----------
    out:
        Destination JSON file. The parent directory is created if needed.
    liquidity_threshold:
        Minimum 24h quote volume (USDT) required to keep a symbol.
    Returns
    -------
    List[str]
        Sorted list of symbols that were saved to ``out``.
    """

    resp = _throttled_get("https://api.binance.com/api/v3/exchangeInfo", timeout=20)
    data = resp.json()

    volumes: Dict[str, float] = {}
    if liquidity_threshold > 0:
        resp = _throttled_get(
            "https://api.binance.com/api/v3/ticker/24hr", timeout=20
        )
        volumes = {
            t["symbol"].upper(): float(t.get("quoteVolume", 0.0))
            for t in resp.json()
        }

    symbols = [
        s["symbol"].upper()
        for s in data.get("symbols", [])
        if s.get("status") == "TRADING"
        and s.get("quoteAsset") == "USDT"
        and "SPOT" in s.get("permissions", [])
        and (
            liquidity_threshold <= 0
            or volumes.get(s["symbol"].upper(), 0.0) >= liquidity_threshold
        )
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
    ttl: int = _DEFAULT_TTL_SECONDS,
    out: str = "data/universe/symbols.json",
    liquidity_threshold: float = 0.0,
    force: bool = False,
) -> List[str]:
    """Return cached Binance symbols list, refreshing if needed."""

    if force or _is_stale(out, ttl):
        run(out, liquidity_threshold=liquidity_threshold)

    with open(out, "r", encoding="utf-8") as f:
        return json.load(f)


__all__ = ["run", "get_symbols"]


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Binance spot symbols trading against USDT."
    )
    parser.add_argument(
        "--liquidity-threshold",
        type=float,
        default=0.0,
        help="Minimum 24h quote volume (USDT) to include a symbol",
    )
    parser.add_argument(
        "--output",
        default="data/universe/symbols.json",
        help="Destination JSON file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh even if cache is fresh",
    )
    args = parser.parse_args()
    get_symbols(
        out=args.output,
        liquidity_threshold=args.liquidity_threshold,
        force=args.force,
    )


if __name__ == "__main__":  # pragma: no cover - CLI is tested via integration
    _main()
else:  # Perform a freshness check when imported
    try:  # pragma: no cover - network may be unavailable during tests
        get_symbols()
    except Exception:
        # The refresh is best effort; failures are surfaced on explicit call.
        pass
