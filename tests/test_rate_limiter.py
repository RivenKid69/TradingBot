import sys
import pathlib

import sys
import json
import pathlib
import asyncio
import math



# Ensure stdlib logging is used instead of local logging.py
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_orig_sys_path = list(sys.path)
sys.path = [p for p in sys.path if p not in ("", str(REPO_ROOT))]
import logging as std_logging  # type: ignore
sys.modules["logging"] = std_logging
sys.path = _orig_sys_path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import logging

from unittest.mock import MagicMock
from decimal import Decimal
import pytest

from utils import SignalRateLimiter, TokenBucket

# provide dummy websockets module for importing binance_ws without dependency
class _DummyWS:
    pass
sys.modules.setdefault("websockets", _DummyWS())

import binance_public
import binance_ws
from services.event_bus import EventBus


# --- TokenBucket tests ---

def test_token_bucket_basic():
    tb = TokenBucket(rps=2.0, burst=4.0, tokens=4.0, last_ts=0.0)
    assert tb.consume(tokens=1, now=0.0)
    assert tb.tokens == pytest.approx(3.0)
    assert tb.consume(tokens=4, now=0.5)
    assert tb.tokens == pytest.approx(0.0)
    assert not tb.consume(tokens=1, now=0.5)


# --- SignalRateLimiter tests ---

def test_rate_limit_per_second():
    rl = SignalRateLimiter(max_per_sec=2)
    now = 0.0
    assert rl.can_send(now) == (True, "ok")
    assert rl.can_send(now) == (True, "ok")
    allowed, status = rl.can_send(now)
    assert not allowed and status == "rejected"
    allowed, _ = rl.can_send(now + 1.0)
    assert allowed


def test_exponential_backoff():
    rl = SignalRateLimiter(max_per_sec=100, backoff_base=2.0, max_backoff=0.05)
    now = 0.0
    for _ in range(100):
        assert rl.can_send(now)[0]
    # first rejection -> backoff 0.01
    allowed, status = rl.can_send(now)
    assert not allowed and status == "rejected"
    assert rl._current_backoff == pytest.approx(0.01)
    # second rejection -> 0.02
    allowed, status = rl.can_send(now + 0.011)
    assert not allowed and status == "rejected"
    assert rl._current_backoff == pytest.approx(0.02)
    # third rejection -> 0.04
    allowed, status = rl.can_send(now + 0.032)
    assert not allowed and status == "rejected"
    assert rl._current_backoff == pytest.approx(0.04)
    # fourth rejection capped at max_backoff 0.05
    allowed, status = rl.can_send(now + 0.073)
    assert not allowed and status == "rejected"
    assert rl._current_backoff == pytest.approx(0.05)


def test_rate_limiter_sub_unit_rate_backoff():
    rl = SignalRateLimiter(max_per_sec=0.5, backoff_base=2.0, max_backoff=60.0)
    now = 0.0

    allowed, status = rl.can_send(now)
    assert allowed and status == "ok"

    allowed, status = rl.can_send(now)
    assert not allowed and status == "rejected"
    assert rl._current_backoff == pytest.approx(2.0)
    assert rl._cooldown_until == pytest.approx(now + 2.0)

    allowed, status = rl.can_send(now + 1.0)
    assert not allowed and status == "delayed"
    assert rl._current_backoff == pytest.approx(2.0)

    resume_time = now + 2.0 + 1e-6
    allowed, status = rl.can_send(resume_time)
    assert allowed and status == "ok"
    assert rl._current_backoff == pytest.approx(0.0)

    allowed, status = rl.can_send(resume_time)
    assert not allowed and status == "rejected"
    assert rl._current_backoff == pytest.approx(2.0)


# --- BinancePublicClient REST integration ---

@pytest.mark.parametrize(
    "method_name, kwargs, response, expected_url, expected_budget, expected_tokens, expected_result",
    [
        (
            "get_exchange_filters",
            {"market": "spot", "symbols": ["btcusdt"]},
            {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.1"}
                        ],
                    }
                ]
            },
            "https://api.binance.com/api/v3/exchangeInfo",
            "exchangeInfo",
            10.0,
            {"BTCUSDT": {"PRICE_FILTER": {"tickSize": "0.1"}}},
        ),
        (
            "get_klines",
            {"market": "spot", "symbol": "btcusdt", "interval": "1m"},
            [[1, 2, 3]],
            "https://api.binance.com/api/v3/klines",
            "klines",
            10.0,
            [[1, 2, 3]],
        ),
        (
            "get_mark_klines",
            {"symbol": "btcusdt", "interval": "1m"},
            [[1, 2, 3]],
            "https://fapi.binance.com/fapi/v1/markPriceKlines",
            "markPriceKlines",
            10.0,
            [[1, 2, 3]],
        ),
        (
            "get_funding",
            {"symbol": "btcusdt"},
            [{"symbol": "BTCUSDT"}],
            "https://fapi.binance.com/fapi/v1/fundingRate",
            "fundingRate",
            1.0,
            [{"symbol": "BTCUSDT"}],
        ),
        (
            "get_last_price",
            {"symbol": "btcusdt"},
            {"symbol": "BTCUSDT", "price": "123.45"},
            "https://api.binance.com/api/v3/ticker/price",
            "tickerPrice",
            1.0,
            Decimal("123.45"),
        ),
    ],
)
def test_binance_public_uses_rest_session_budget(
    method_name: str,
    kwargs: dict[str, object],
    response: object,
    expected_url: str,
    expected_budget: str,
    expected_tokens: float,
    expected_result: object,
):
    session = MagicMock()
    session.get.return_value = response
    client = binance_public.BinancePublicClient(session=session)

    method = getattr(client, method_name)
    result = method(**kwargs)

    assert result == expected_result
    assert session.get.call_count == 1
    args, call_kwargs = session.get.call_args
    assert args == (expected_url,)
    assert call_kwargs["budget"] == expected_budget
    assert call_kwargs["tokens"] == pytest.approx(expected_tokens)
    assert call_kwargs["timeout"] == client.timeout
    params = call_kwargs["params"]
    assert isinstance(params, dict)
    if method_name == "get_exchange_filters":
        assert json.loads(params["symbols"]) == ["BTCUSDT"]
    else:
        assert params["symbol"] == "BTCUSDT"


def test_get_book_ticker_uses_budget_and_cache():
    session = MagicMock()
    session.get.return_value = {
        "symbol": "BTCUSDT",
        "bidPrice": "100.0",
        "askPrice": "101.0",
    }
    client = binance_public.BinancePublicClient(session=session)

    bid, ask = client.get_book_ticker("btcusdt")
    assert isinstance(bid, (Decimal, float))
    assert isinstance(ask, (Decimal, float))
    assert bid == Decimal("100.0")
    assert ask == Decimal("101.0")
    assert session.get.call_count == 1
    args, kwargs = session.get.call_args
    assert args == ("https://api.binance.com/api/v3/ticker/bookTicker",)
    assert kwargs["budget"] == "bookTicker"
    assert kwargs["tokens"] == pytest.approx(1.0)
    assert kwargs["params"]["symbol"] == "BTCUSDT"

    # Second call should hit the cache and avoid a REST request
    session.get.return_value = {
        "symbol": "BTCUSDT",
        "bidPrice": "999.0",
        "askPrice": "999.1",
    }
    bid2, ask2 = client.get_book_ticker("btcusdt")
    assert session.get.call_count == 1
    assert bid2 == bid and ask2 == ask


def test_get_book_ticker_multiple_symbols():
    session = MagicMock()
    session.get.return_value = [
        {"symbol": "BTCUSDT", "bidPrice": "100", "askPrice": "101"},
        {"symbol": "ETHUSDT", "bidPrice": "10", "askPrice": "11"},
    ]
    client = binance_public.BinancePublicClient(session=session)

    quotes = client.get_book_ticker(["btcusdt", "ethusdt"])
    assert set(quotes.keys()) == {"BTCUSDT", "ETHUSDT"}
    assert quotes["BTCUSDT"] == (Decimal("100"), Decimal("101"))
    assert quotes["ETHUSDT"] == (Decimal("10"), Decimal("11"))
    args, kwargs = session.get.call_args
    params = kwargs["params"]
    assert json.loads(params["symbols"]) == ["BTCUSDT", "ETHUSDT"]
    assert kwargs["tokens"] == pytest.approx(2.0)

    # Cached result should satisfy subsequent single-symbol calls without REST
    session.get.reset_mock()
    single = client.get_book_ticker("ethusdt")
    assert session.get.call_count == 0
    assert single == quotes["ETHUSDT"]


def test_get_last_price_uses_cache():
    session = MagicMock()
    session.get.return_value = {"symbol": "BTCUSDT", "price": "101.5"}
    client = binance_public.BinancePublicClient(session=session)

    price = client.get_last_price("btcusdt")
    assert isinstance(price, (Decimal, float))
    assert price == Decimal("101.5")
    assert session.get.call_count == 1
    args, kwargs = session.get.call_args
    assert args == ("https://api.binance.com/api/v3/ticker/price",)
    assert kwargs["budget"] == "tickerPrice"
    assert kwargs["tokens"] == pytest.approx(1.0)
    assert kwargs["params"]["symbol"] == "BTCUSDT"

    session.get.return_value = {"symbol": "BTCUSDT", "price": "999"}
    price2 = client.get_last_price("btcusdt")
    assert session.get.call_count == 1
    assert price2 == price


def test_get_spread_bps_prefers_book_ticker():
    session = MagicMock()
    session.get.return_value = {
        "symbol": "BTCUSDT",
        "bidPrice": "100.0",
        "askPrice": "101.0",
    }
    client = binance_public.BinancePublicClient(session=session)

    spread = client.get_spread_bps("btcusdt")

    expected = (101.0 - 100.0) / ((101.0 + 100.0) * 0.5) * 10000.0
    assert spread is not None and math.isclose(spread, expected)
    assert session.get.call_count == 1
    args, kwargs = session.get.call_args
    assert args == ("https://api.binance.com/api/v3/ticker/bookTicker",)
    assert kwargs["tokens"] == pytest.approx(1.0)


def test_last_weight_headers_exposed():
    session = MagicMock()
    session.get.return_value = {"serverTime": 0}
    session.get_last_response_metadata.return_value = {
        "binance_weights": {"x-mbx-used-weight-1m": 123.0}
    }
    client = binance_public.BinancePublicClient(session=session)

    client.get_server_time()

    assert client.last_weight_headers == {"x-mbx-used-weight-1m": 123.0}


def test_get_spread_bps_falls_back_to_range():
    session = MagicMock()
    # Only the 24hr endpoint will be called because book ticker is skipped.
    session.get.return_value = {
        "symbol": "BTCUSDT",
        "highPrice": "102.0",
        "lowPrice": "98.0",
        "lastPrice": "100.0",
    }
    client = binance_public.BinancePublicClient(session=session)

    spread = client.get_spread_bps("btcusdt", prefer_book_ticker=False)

    expected = (102.0 - 98.0) / 100.0 * 10000.0
    assert spread is not None and math.isclose(spread, expected)
    args, kwargs = session.get.call_args
    assert args == ("https://api.binance.com/api/v3/ticker/24hr",)
    assert kwargs["budget"] == "ticker24hr"


# --- BinanceWS limiter inclusion and counters ---

def test_binance_ws_rate_limit_counters(monkeypatch):
    bus = EventBus(queue_size=10, drop_policy="newest")
    ws = binance_ws.BinanceWS(symbols=["BTCUSDT"], bus=bus, rate_limit=1)
    rl_mock = MagicMock()
    rl_mock.can_send.side_effect = [
        (False, "rejected"), (True, "ok"),
        (False, "delayed"),
    ]
    rl_mock._cooldown_until = 0.0
    ws._rate_limiter = rl_mock
    assert ws._rate_limiter is not None

    async def dummy_sleep(_):
        return None

    monkeypatch.setattr(binance_ws.asyncio, "sleep", dummy_sleep)
    async def run():
        allowed = await ws._check_rate_limit()
        assert allowed
        allowed = await ws._check_rate_limit()
        assert not allowed

    asyncio.run(run())
    assert ws._rl_total == 2
    assert ws._rl_delayed == 1
    assert ws._rl_dropped == 1
