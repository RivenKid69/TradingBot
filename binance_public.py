# data/binance_public.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core_config import RetryConfig
from services.rest_budget import RestBudgetSession


DEFAULT_RETRY_CFG = RetryConfig(max_attempts=5, backoff_base_s=0.5, max_backoff_s=60.0)


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
        retry_cfg: RetryConfig | None = None,
        session: RestBudgetSession | None = None,
    ) -> None:
        self.e = endpoints or PublicEndpoints()
        self.timeout = int(timeout)
        cfg_retry = retry_cfg or DEFAULT_RETRY_CFG
        self._owns_session = session is None
        if session is None:
            session_cfg: Dict[str, Any] = {"timeout": self.timeout, "retry": cfg_retry}
            self.session = RestBudgetSession(session_cfg)
        else:
            self.session = session

    def close(self) -> None:
        """Release owned REST session resources."""

        if getattr(self, "_owns_session", False):
            try:
                self.session.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # -------- SERVER TIME --------

    def get_server_time(self) -> Tuple[int, float]:
        """Fetch Binance server time and measure round-trip time."""
        base = self.e.spot_base
        path = "/api/v3/time"
        url = f"{base}{path}"
        params: Dict[str, Any] = {}
        t0 = time.time()
        data = self.session.get(url, params=params, timeout=self.timeout)
        t1 = time.time()
        if isinstance(data, dict) and "serverTime" in data:
            return int(data["serverTime"]), (t1 - t0) * 1000.0
        raise RuntimeError(f"Unexpected server time response: {data}")

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
        data = self.session.get(
            url, params=params, timeout=self.timeout, budget="klines"
        )
        if isinstance(data, list):
            return data  # type: ignore
        raise RuntimeError(f"Unexpected klines response: {data}")

    # -------- AGGREGATED TRADES --------

    def get_agg_trades(
        self,
        *,
        market: str,
        symbol: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Fetch aggregated trades from Binance public API."""
        if market not in ("spot", "futures"):
            raise ValueError("market must be 'spot' or 'futures'")
        base = self.e.spot_base if market == "spot" else self.e.futures_base
        path = "/api/v3/aggTrades" if market == "spot" else "/fapi/v1/aggTrades"
        url = f"{base}{path}"
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "limit": int(limit),
        }
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        data = self.session.get(
            url, params=params, timeout=self.timeout, budget="aggTrades"
        )
        if isinstance(data, list):
            return data  # type: ignore[return-value]
        raise RuntimeError(f"Unexpected aggTrades response: {data}")

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
        data = self.session.get(
            url, params=params, timeout=self.timeout, budget="markPriceKlines"
        )
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
        data = self.session.get(
            url, params=params, timeout=self.timeout, budget="fundingRate"
        )
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
        data = self.session.get(
            url, params=params, timeout=self.timeout, budget="exchangeInfo"
        )
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
