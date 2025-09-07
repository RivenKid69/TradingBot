# data/binance_public.py
from __future__ import annotations

import time
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _http_get(url: str, params: Dict[str, Any], *, timeout: int = 20) -> Dict[str, Any] | List[Any]:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    full = f"{url}?{query}" if query else url
    req = urllib.request.Request(full, method="GET")
    req.add_header("User-Agent", "BinancePublicClient/1.0 (+https://github.com/yourrepo)")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        # в редких случаях API отдаёт текст — вернём как есть
        return {"raw": data.decode("utf-8", errors="ignore")}


def _retrying_get(url: str, params: Dict[str, Any], *, timeout: int = 20, retries: int = 5, backoff_base: float = 0.5) -> Dict[str, Any] | List[Any]:
    for i in range(max(1, retries)):
        try:
            return _http_get(url, params, timeout=timeout)
        except Exception as e:
            if i == retries - 1:
                raise
            sleep_s = backoff_base * (2 ** i) + (0.1 * i)
            time.sleep(sleep_s)
    # недостижимо
    return {}


@dataclass
class PublicEndpoints:
    spot_base: str = "https://api.binance.com"
    futures_base: str = "https://fapi.binance.com"


class BinancePublicClient:
    """
    Минимальный публичный клиент Binance (без ключей).
      - get_klines() для spot и futures
      - get_mark_klines() для mark-price (futures)
      - get_funding() для funding rate (futures)
    Пагинация — по startTime/endTime + limit.
    Времена — миллисекунды Unix.
    """
    def __init__(self, endpoints: Optional[PublicEndpoints] = None, timeout: int = 20) -> None:
        self.e = endpoints or PublicEndpoints()
        self.timeout = int(timeout)

    # -------- KLINES --------

    def get_klines(self, *, market: str, symbol: str, interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None, limit: int = 1500) -> List[List[Any]]:
        """
        Возвращает «сырые» klines: список списков, как в Binance API.
        market: "spot" | "futures"
        interval: "1m" | "5m" | "15m" | "1h" | ...
        """
        if market not in ("spot", "futures"):
            raise ValueError("market должен быть 'spot' или 'futures'")
        base = self.e.spot_base if market == "spot" else self.e.futures_base
        path = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"
        url = f"{base}{path}"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": int(limit),
        }
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        data = _retrying_get(url, params, timeout=self.timeout)
        if isinstance(data, list):
            return data  # type: ignore
        raise RuntimeError(f"Unexpected klines response: {data}")

    # -------- MARK PRICE KLINES (futures only) --------

    def get_mark_klines(self, *, symbol: str, interval: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None, limit: int = 1500) -> List[List[Any]]:
        base = self.e.futures_base
        path = "/fapi/v1/markPriceKlines"
        url = f"{base}{path}"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": int(limit),
        }
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        data = _retrying_get(url, params, timeout=self.timeout)
        if isinstance(data, list):
            return data  # type: ignore
        raise RuntimeError(f"Unexpected markPriceKlines response: {data}")

    # -------- FUNDING (futures only) --------

    def get_funding(self, *, symbol: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        base = self.e.futures_base
        path = "/fapi/v1/fundingRate"
        url = f"{base}{path}"
        params = {
            "symbol": symbol.upper(),
            "limit": int(limit),
        }
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        data = _retrying_get(url, params, timeout=self.timeout)
        if isinstance(data, list):
            return data  # type: ignore
        raise RuntimeError(f"Unexpected funding response: {data}")
