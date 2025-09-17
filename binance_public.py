# data/binance_public.py
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple, Union

from core_config import RetryConfig
from services.rest_budget import RestBudgetSession


DEFAULT_RETRY_CFG = RetryConfig(max_attempts=5, backoff_base_s=0.5, max_backoff_s=60.0)
_BOOK_TICKER_TTL_S = 1.0
_LAST_PRICE_TTL_S = 1.0


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
        self._book_ticker_cache: Dict[
            str,
            Tuple[float, Tuple[Optional[Union[Decimal, float]], Optional[Union[Decimal, float]]]],
        ] = {}
        self._last_price_cache: Dict[str, Tuple[float, Optional[Union[Decimal, float]]]] = {}

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

    @staticmethod
    def _to_number(value: Any) -> Optional[Union[Decimal, float]]:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            try:
                return float(value)
            except Exception:
                return None

    def get_book_ticker(
        self, symbols: List[str] | str
    ) -> Union[
        Tuple[Optional[Union[Decimal, float]], Optional[Union[Decimal, float]]],
        Dict[str, Tuple[Optional[Union[Decimal, float]], Optional[Union[Decimal, float]]]],
    ]:
        """Return latest best bid/ask quotes for ``symbols``.

        The Binance REST ``bookTicker`` endpoint is queried with lightweight
        caching to avoid repeated requests within the same bar.  Results are
        returned as either a tuple ``(bid, ask)`` for a single symbol or a
        mapping ``symbol -> (bid, ask)`` for multiple symbols.  Values are
        ``Decimal`` where possible with a float fallback when conversion
        fails.
        """

        if isinstance(symbols, str):
            symbol_list = [symbols.upper()]
            single = True
        else:
            symbol_list = [s.upper() for s in symbols]
            single = len(symbol_list) == 1
        if not symbol_list:
            raise ValueError("symbols must be non-empty")

        now = time.monotonic()
        results: Dict[str, Tuple[Optional[Union[Decimal, float]], Optional[Union[Decimal, float]]]] = {}
        missing: List[str] = []
        for sym in symbol_list:
            cache_entry = self._book_ticker_cache.get(sym)
            if cache_entry is None or cache_entry[0] <= now:
                missing.append(sym)
            else:
                results[sym] = cache_entry[1]

        if missing:
            url = f"{self.e.spot_base}/api/v3/ticker/bookTicker"
            params: Dict[str, Any]
            if len(missing) == 1:
                params = {"symbol": missing[0]}
            else:
                params = {"symbols": json.dumps(missing)}
            data = self.session.get(
                url,
                params=params,
                timeout=self.timeout,
                budget="bookTicker",
            )
            parsed: Dict[
                str,
                Tuple[Optional[Union[Decimal, float]], Optional[Union[Decimal, float]]],
            ] = {}

            def _handle_entry(entry: Any) -> None:
                if not isinstance(entry, dict):
                    return
                sym = str(entry.get("symbol", "")).upper()
                if not sym:
                    return
                bid = self._to_number(entry.get("bidPrice"))
                ask = self._to_number(entry.get("askPrice"))
                parsed[sym] = (bid, ask)

            if isinstance(data, list):
                for item in data:
                    _handle_entry(item)
            elif isinstance(data, dict):
                _handle_entry(data)
            else:
                raise RuntimeError(f"Unexpected bookTicker response: {data}")

            if not parsed:
                raise RuntimeError(f"Unexpected bookTicker response: {data}")

            expires_at = now + _BOOK_TICKER_TTL_S
            for sym, quote in parsed.items():
                self._book_ticker_cache[sym] = (expires_at, quote)
                if sym in symbol_list:
                    results[sym] = quote

        if single:
            sym = symbol_list[0]
            if sym not in results:
                raise RuntimeError(
                    f"Missing bookTicker data for {sym}. Cached keys: {list(results)}"
                )
            return results[sym]
        return results

    def get_last_price(
        self, symbol: str, ttl_s: float = _LAST_PRICE_TTL_S
    ) -> Optional[Union[Decimal, float]]:
        """Return the latest trade price for ``symbol`` using ticker/price."""

        sym = str(symbol).upper()
        if not sym:
            raise ValueError("symbol must be non-empty")

        now = time.monotonic()
        cache = self._last_price_cache.get(sym)
        ttl = float(ttl_s)
        if cache is not None and cache[0] > now:
            return cache[1]

        url = f"{self.e.spot_base}/api/v3/ticker/price"
        params = {"symbol": sym}
        data = self.session.get(
            url,
            params=params,
            timeout=self.timeout,
            budget="tickerPrice",
        )
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected tickerPrice response: {data}")
        price = self._to_number(data.get("price"))
        if price is None:
            raise RuntimeError(f"Unexpected tickerPrice response: {data}")
        expires_at = now + ttl if ttl > 0 else now
        self._last_price_cache[sym] = (expires_at, price)
        return price

    def get_spread_bps(
        self,
        symbols: List[str] | str,
        *,
        market: str = "spot",
        prefer_book_ticker: bool = True,
    ) -> Union[Optional[float], Dict[str, Optional[float]]]:
        """Return spread estimates in basis points for ``symbols``.

        The method prefers real-time best bid/ask quotes obtained via
        :meth:`get_book_ticker`. When unavailable it falls back to the
        24-hour statistics ``highPrice``/``lowPrice`` using the approximation
        ``(high - low) / mid`` where ``mid`` defaults to the latest price.
        """

        if isinstance(symbols, str):
            symbol_list = [symbols.upper()]
            single = True
        else:
            symbol_list = [s.upper() for s in symbols]
            single = len(symbol_list) == 1
        if not symbol_list:
            raise ValueError("symbols must be non-empty")

        spreads: Dict[str, Optional[float]] = {sym: None for sym in symbol_list}
        missing = set(symbol_list)

        if prefer_book_ticker:
            try:
                raw_quotes = self.get_book_ticker(
                    symbol_list if not single else symbol_list[0]
                )
                if single:
                    quote_map = {symbol_list[0]: raw_quotes}
                else:
                    quote_map = raw_quotes
                for sym, quote in quote_map.items():
                    if not isinstance(quote, tuple) or len(quote) != 2:
                        continue
                    spread_val = self._spread_from_quote(*quote)
                    if spread_val is not None:
                        spreads[sym] = spread_val
                        missing.discard(sym)
            except Exception:
                pass

        if missing:
            stats = self._fetch_24h_stats(market=market, symbols=list(missing))
            for sym, (high, low, last) in stats.items():
                spread_val = self._spread_from_range(high, low, last)
                if spread_val is not None:
                    spreads[sym] = spread_val
                    missing.discard(sym)

        if single:
            return spreads.get(symbol_list[0])
        return spreads

    # -------- Helpers --------

    def _fetch_24h_stats(
        self, *, market: str, symbols: List[str]
    ) -> Dict[str, Tuple[
        Optional[Union[Decimal, float]],
        Optional[Union[Decimal, float]],
        Optional[Union[Decimal, float]],
    ]]:
        if not symbols:
            return {}
        if market not in ("spot", "futures"):
            raise ValueError("market must be 'spot' or 'futures'")
        base = self.e.spot_base if market == "spot" else self.e.futures_base
        path = "/api/v3/ticker/24hr" if market == "spot" else "/fapi/v1/ticker/24hr"
        url = f"{base}{path}"
        params: Dict[str, Any]
        if len(symbols) == 1:
            params = {"symbol": symbols[0].upper()}
        else:
            params = {"symbols": json.dumps([s.upper() for s in symbols])}
        data = self.session.get(
            url,
            params=params,
            timeout=self.timeout,
            budget="ticker24hr",
        )
        results: Dict[
            str,
            Tuple[
                Optional[Union[Decimal, float]],
                Optional[Union[Decimal, float]],
                Optional[Union[Decimal, float]],
            ],
        ] = {}

        def _handle(entry: Any) -> None:
            if not isinstance(entry, dict):
                return
            sym = str(entry.get("symbol", "")).upper()
            if not sym:
                return
            high = self._to_number(entry.get("highPrice"))
            low = self._to_number(entry.get("lowPrice"))
            last = self._to_number(entry.get("lastPrice"))
            results[sym] = (high, low, last)

        if isinstance(data, list):
            for item in data:
                _handle(item)
        elif isinstance(data, dict):
            _handle(data)
        else:
            raise RuntimeError(f"Unexpected ticker24hr response: {data}")
        return results

    @staticmethod
    def _spread_from_quote(
        bid: Optional[Union[Decimal, float]],
        ask: Optional[Union[Decimal, float]],
    ) -> Optional[float]:
        try:
            bid_val = float(bid) if bid is not None else float("nan")
            ask_val = float(ask) if ask is not None else float("nan")
        except Exception:
            return None
        if not math.isfinite(bid_val) or not math.isfinite(ask_val):
            return None
        if bid_val <= 0 or ask_val <= 0:
            return None
        mid = (bid_val + ask_val) * 0.5
        if mid <= 0:
            return None
        spread = (ask_val - bid_val) / mid * 10000.0
        return spread if spread > 0 else None

    @staticmethod
    def _spread_from_range(
        high: Optional[Union[Decimal, float]],
        low: Optional[Union[Decimal, float]],
        last: Optional[Union[Decimal, float]],
    ) -> Optional[float]:
        try:
            high_val = float(high) if high is not None else float("nan")
            low_val = float(low) if low is not None else float("nan")
            last_val = float(last) if last is not None else float("nan")
        except Exception:
            return None
        if not math.isfinite(high_val) or not math.isfinite(low_val):
            return None
        if high_val <= 0 or low_val <= 0 or high_val <= low_val:
            return None
        mid = last_val if math.isfinite(last_val) and last_val > 0 else (high_val + low_val) * 0.5
        if mid <= 0:
            return None
        spread = (high_val - low_val) / mid * 10000.0
        return spread if spread > 0 else None
