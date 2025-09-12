# data/binance_public.py
from __future__ import annotations

import time
import json
import math
import urllib.parse
import urllib.request
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from utils import SignalRateLimiter


logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        endpoints: Optional[PublicEndpoints] = None,
        timeout: int = 20,
        rate_limit: float | None = None,
        backoff_base: float = 2.0,
        max_backoff: float = 60.0,
    ) -> None:
        self.e = endpoints or PublicEndpoints()
        self.timeout = int(timeout)
        self._rate_limiter = (
            SignalRateLimiter(rate_limit, backoff_base, max_backoff)
            if rate_limit and rate_limit > 0
            else None
        )
        self._rl_total = 0
        self._rl_delayed = 0
        self._rl_rejected = 0

    def _throttle(self) -> None:
        """Apply rate limiter with blocking backoff."""
        if self._rate_limiter is None:
            return
        self._rl_total += 1
        while True:
            allowed, status = self._rate_limiter.can_send()
            if allowed:
                return
            if status == "rejected":
                self._rl_rejected += 1
            else:
                self._rl_delayed += 1
            wait = max(self._rate_limiter._cooldown_until - time.time(), 0.0)
            if wait > 0:
                time.sleep(wait)

    def __del__(self) -> None:  # pragma: no cover - best effort logging
        if getattr(self, "_rl_total", 0):
            logger.info(
                "BinancePublicClient rate limiting: delayed=%0.2f%% (%d/%d), rejected=%0.2f%% (%d/%d)",
                self._rl_delayed / self._rl_total * 100.0,
                self._rl_delayed,
                self._rl_total,
                self._rl_rejected / self._rl_total * 100.0,
                self._rl_rejected,
                self._rl_total,
            )

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
        self._throttle()
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
        self._throttle()
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
        self._throttle()
        data = _retrying_get(url, params, timeout=self.timeout)
        if isinstance(data, list):
            return data  # type: ignore
        raise RuntimeError(f"Unexpected funding response: {data}")

    # -------- EXCHANGE FILTERS --------

    def get_exchange_filters(self, *, market: str = "spot", symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает фильтры торговли для указанных символов Binance.
        Поддерживаются типы фильтров: PRICE_FILTER, LOT_SIZE, MIN_NOTIONAL,
        PERCENT_PRICE_BY_SIDE / PERCENT_PRICE.
        """
        if market not in ("spot", "futures"):
            raise ValueError("market must be 'spot' or 'futures'")
        base = self.e.spot_base if market == "spot" else self.e.futures_base
        path = "/api/v3/exchangeInfo" if market == "spot" else "/fapi/v1/exchangeInfo"
        url = f"{base}{path}"
        params: Dict[str, Any] = {}
        if symbols:
            params["symbols"] = json.dumps([s.upper() for s in symbols])
        self._throttle()
        data = _retrying_get(url, params, timeout=self.timeout)
        out: Dict[str, Dict[str, Any]] = {}
        if isinstance(data, dict):
            for s in data.get("symbols", []):
                sym = s.get("symbol")
                filts = s.get("filters", [])
                d: Dict[str, Any] = {}
                for f in filts:
                    ftype = f.get("filterType")
                    if ftype in {"PRICE_FILTER", "LOT_SIZE", "MIN_NOTIONAL", "PERCENT_PRICE_BY_SIDE", "PERCENT_PRICE"}:
                        d[ftype] = {k: v for k, v in f.items() if k != "filterType"}
                if sym and d:
                    out[sym] = d
        return out
