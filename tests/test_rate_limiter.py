import sys
import pathlib

import sys
import json
import pathlib
import asyncio



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


# --- BinancePublicClient REST integration ---

@pytest.mark.parametrize(
    "method_name, kwargs, response, expected_url, expected_budget, expected_result",
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
            {"BTCUSDT": {"PRICE_FILTER": {"tickSize": "0.1"}}},
        ),
        (
            "get_klines",
            {"market": "spot", "symbol": "btcusdt", "interval": "1m"},
            [[1, 2, 3]],
            "https://api.binance.com/api/v3/klines",
            "klines",
            [[1, 2, 3]],
        ),
        (
            "get_mark_klines",
            {"symbol": "btcusdt", "interval": "1m"},
            [[1, 2, 3]],
            "https://fapi.binance.com/fapi/v1/markPriceKlines",
            "markPriceKlines",
            [[1, 2, 3]],
        ),
        (
            "get_funding",
            {"symbol": "btcusdt"},
            [{"symbol": "BTCUSDT"}],
            "https://fapi.binance.com/fapi/v1/fundingRate",
            "fundingRate",
            [{"symbol": "BTCUSDT"}],
        ),
    ],
)
def test_binance_public_uses_rest_session_budget(
    method_name: str,
    kwargs: dict[str, object],
    response: object,
    expected_url: str,
    expected_budget: str,
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
    assert call_kwargs["timeout"] == client.timeout
    params = call_kwargs["params"]
    assert isinstance(params, dict)
    if method_name == "get_exchange_filters":
        assert json.loads(params["symbols"]) == ["BTCUSDT"]
    else:
        assert params["symbol"] == "BTCUSDT"


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
