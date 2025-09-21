# data/binance_public.py
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from core_config import RetryConfig
from services.rest_budget import RestBudgetSession


DEFAULT_RETRY_CFG = RetryConfig(max_attempts=5, backoff_base_s=0.5, max_backoff_s=60.0)
_BOOK_TICKER_TTL_S = 1.0
_LAST_PRICE_TTL_S = 1.0

#: Budget keys used by :class:`BinancePublicClient` along with the documented
#: Binance REST endpoints and their indicative request weights.  The weight
#: values are based on https://binance-docs.github.io/apidocs/spot/en/ and are
#: meant to mirror how :class:`services.rest_budget.RestBudgetSession` should
#: account for the relative cost of each request.  Keeping the mapping close to
#: the client makes it easy to audit and maintain budgets when Binance updates
#: endpoint weights.
BINANCE_REST_BUDGETS: Dict[str, str] = {
    "serverTime": "GET /api/v3/time (weight 1)",
    "klines": "GET /api/v3/klines & GET /fapi/v1/klines (weight depends on limit)",
    "aggTrades": "GET /api/v3/aggTrades & GET /fapi/v1/aggTrades (weight 1-2)",
    "markPriceKlines": "GET /fapi/v1/markPriceKlines (weight depends on limit)",
    "fundingRate": "GET /fapi/v1/fundingRate (weight 1)",
    "exchangeInfo": "GET /api/v3/exchangeInfo & GET /fapi/v1/exchangeInfo (weight 10)",
    "bookTicker": "GET /api/v3/ticker/bookTicker (weight 1 single / 2 multi)",
    "tickerPrice": "GET /api/v3/ticker/price (weight 1 single / 2 multi)",
    "ticker24hr": "GET /api/v3/ticker/24hr & GET /fapi/v1/ticker/24hr (weight 1-40)",
}


def _kline_tokens(limit: int) -> float:
    """Return an approximate request weight for ``limit`` kline candles."""

    try:
        limit_val = int(limit)
    except (TypeError, ValueError):
        limit_val = 0
    limit_val = max(limit_val, 1)
    if limit_val <= 100:
        return 1.0
    if limit_val <= 500:
        return 2.0
    if limit_val <= 1000:
        return 5.0
    return 10.0


def _agg_trades_tokens(limit: int) -> float:
    """Return representative weight for aggregated trade queries."""

    try:
        limit_val = int(limit)
    except (TypeError, ValueError):
        limit_val = 0
    return 1.0 if limit_val <= 500 else 2.0


def _book_ticker_tokens(count: int) -> float:
    """Weight helper for :meth:`get_book_ticker`."""

    return 1.0 if count <= 1 else 2.0


def _ticker_price_tokens(count: int) -> float:
    """Weight helper for :meth:`get_last_price`."""

    return 1.0 if count <= 1 else 2.0


def _ticker_24hr_tokens(symbols: List[str] | None) -> float:
    """Return approximate weight for 24hr statistics requests."""

    if not symbols:
        # Requesting the entire book of tickers carries the highest weight.
        return 40.0
    count = len(symbols)
    if count <= 1:
        return 1.0
    # Binance charges up to 40 weight units; scale moderately by symbol count.
    return float(min(40.0, max(2.0, float(count))))


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

    See :data:`BINANCE_REST_BUDGETS` for a reference of budget names used by
    the client and the corresponding Binance REST endpoints/weights.
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
        self._last_response_metadata: Dict[str, Any] | None = None
        self._last_weight_headers: Dict[str, Any] | None = None

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

    def _update_last_response_metadata(self) -> None:
        metadata: Mapping[str, Any] | None = None
        getter = getattr(self.session, "get_last_response_metadata", None)
        if callable(getter):
            try:
                candidate = getter()
            except TypeError:
                candidate = None
            if isinstance(candidate, Mapping):
                metadata = dict(candidate)
        elif isinstance(getter, Mapping):
            metadata = dict(getter)

        self._last_response_metadata = metadata
        if metadata is None:
            self._last_weight_headers = None
            return

        weights = metadata.get("binance_weights") or metadata.get("weight_headers")
        if isinstance(weights, Mapping):
            self._last_weight_headers = {str(k): weights[k] for k in weights}
        else:
            self._last_weight_headers = None

    def _session_get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        budget: str,
        tokens: float,
        endpoint: str | None = None,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "params": params,
            "timeout": self.timeout,
            "budget": budget,
            "tokens": float(tokens),
        }
        if endpoint is not None:
            kwargs["endpoint"] = endpoint
        data = self.session.get(url, **kwargs)
        self._update_last_response_metadata()
        return data

    @property
    def last_response_metadata(self) -> Optional[Dict[str, Any]]:
        """Return metadata captured from the most recent REST response."""

        if self._last_response_metadata is None:
            return None
        return dict(self._last_response_metadata)

    @property
    def last_weight_headers(self) -> Optional[Dict[str, Any]]:
        """Return parsed Binance weight headers from the latest response."""

        if self._last_weight_headers is None:
            return None
        return dict(self._last_weight_headers)

    def get_server_time(self) -> Tuple[int, float]:
        """Fetch Binance server time and measure round-trip time.

        Budget ``serverTime`` tracks ``GET /api/v3/time`` which carries a
        constant weight of ``1`` per request.
        """
        base = self.e.spot_base
        path = "/api/v3/time"
        url = f"{base}{path}"
        params: Dict[str, Any] = {}
        t0 = time.time()
        data = self._session_get(url, params=params, budget="serverTime", tokens=1.0)
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
        Budget ``klines`` corresponds to ``GET /api/v3/klines`` (spot) and
        ``GET /fapi/v1/klines`` (futures) where the request weight grows with
        ``limit``.
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
        tokens = _kline_tokens(limit)
        data = self._session_get(url, params=params, budget="klines", tokens=tokens)
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
        """Fetch aggregated trades from Binance public API.

        Budget ``aggTrades`` reflects ``GET /api/v3/aggTrades`` and the futures
        equivalent where heavier ``limit`` values consume two weight units.
        """
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
        tokens = _agg_trades_tokens(limit)
        data = self._session_get(
            url, params=params, budget="aggTrades", tokens=tokens
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
        # Budget ``markPriceKlines`` mirrors futures mark price klines with the
        # same weight profile as standard klines.
        tokens = _kline_tokens(limit)
        data = self._session_get(
            url, params=params, budget="markPriceKlines", tokens=tokens
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
        # Budget ``fundingRate`` maps to ``GET /fapi/v1/fundingRate`` which has a
        # flat weight of ``1`` per request.
        data = self._session_get(url, params=params, budget="fundingRate", tokens=1.0)
        if isinstance(data, list):
            return data  # type: ignore
        raise RuntimeError(f"Unexpected funding response: {data}")

    # -------- EXCHANGE FILTERS --------

    def get_exchange_filters(self, *, market: str = "spot", symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Возвращает фильтры торговли для указанных символов Binance.
        Поддерживаются типы фильтров: PRICE_FILTER, LOT_SIZE, MIN_NOTIONAL,
        PERCENT_PRICE_BY_SIDE / PERCENT_PRICE.
        Budget ``exchangeInfo`` represents ``GET /api/v3/exchangeInfo`` and
        ``GET /fapi/v1/exchangeInfo`` which both have an advertised weight of
        ``10``.
        """
        if market not in ("spot", "futures"):
            raise ValueError("market must be 'spot' or 'futures'")
        base = self.e.spot_base if market == "spot" else self.e.futures_base
        path = "/api/v3/exchangeInfo" if market == "spot" else "/fapi/v1/exchangeInfo"
        url = f"{base}{path}"
        params: Dict[str, Any] = {}
        if symbols:
            params["symbols"] = json.dumps([s.upper() for s in symbols])
        data = self._session_get(
            url, params=params, budget="exchangeInfo", tokens=10.0
        )
        out: Dict[str, Dict[str, Any]] = {}
        if isinstance(data, dict):
            recognized_filters = {
                "PRICE_FILTER",
                "LOT_SIZE",
                "MIN_NOTIONAL",
                "PERCENT_PRICE_BY_SIDE",
                "PERCENT_PRICE",
                "COMMISSION_STEP",
            }
            for s in data.get("symbols", []):
                sym = s.get("symbol")
                filts = s.get("filters", [])
                d: Dict[str, Any] = {}
                for f in filts:
                    ftype = f.get("filterType")
                    if ftype in recognized_filters:
                        d[ftype] = {k: v for k, v in f.items() if k != "filterType"}
                for key, value in s.items():
                    if isinstance(key, str) and key.lower().endswith("precision"):
                        d[key] = value
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

        Budget ``bookTicker`` maps to ``GET /api/v3/ticker/bookTicker`` with a
        weight of ``1`` for single-symbol requests and ``2`` when fetching
        multiple symbols.
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
            request_tokens = _book_ticker_tokens(len(missing))
            data = self._session_get(
                url,
                params=params,
                budget="bookTicker",
                tokens=request_tokens,
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
        # Budget ``tickerPrice`` corresponds to ``GET /api/v3/ticker/price``
        # with a constant single-symbol weight of ``1``.
        data = self._session_get(
            url,
            params=params,
            budget="tickerPrice",
            tokens=_ticker_price_tokens(1),
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
        normalized = [s.upper() for s in symbols]
        if len(normalized) == 1:
            params = {"symbol": normalized[0]}
        else:
            params = {"symbols": json.dumps(normalized)}
        # Budget ``ticker24hr`` spans spot/futures 24-hour statistics and can
        # cost up to ``40`` weight units when requesting many symbols.
        tokens = _ticker_24hr_tokens(normalized if normalized else None)
        data = self._session_get(
            url,
            params=params,
            budget="ticker24hr",
            tokens=tokens,
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
