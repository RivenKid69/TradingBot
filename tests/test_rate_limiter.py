import sys
import pathlib

import sys
import pathlib
import gc
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

from utils import SignalRateLimiter

# provide dummy websockets module for importing binance_ws without dependency
class _DummyWS:
    pass
sys.modules.setdefault("websockets", _DummyWS())

import binance_public
import binance_ws


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


# --- BinancePublicClient logging and limiter inclusion ---

def test_binance_public_logging_and_counts(monkeypatch, caplog):
    client = binance_public.BinancePublicClient(rate_limit=1)
    rl_mock = MagicMock()
    rl_mock.can_send.side_effect = [
        (False, "delayed"), (True, "ok"),
        (False, "rejected"), (True, "ok"),
    ]
    rl_mock._cooldown_until = 0.0
    client._rate_limiter = rl_mock
    monkeypatch.setattr(binance_public.time, "sleep", lambda _: None)

    caplog.set_level(logging.INFO, logger=binance_public.__name__)
    client._throttle()  # delayed
    client._throttle()  # rejected

    assert client._rl_total == 2
    assert client._rl_delayed == 1
    assert client._rl_rejected == 1

    del client
    gc.collect()

    rec = next(r for r in caplog.records if "BinancePublicClient rate limiting" in r.message)
    assert "delayed=50.00% (1/2)" in rec.message
    assert "rejected=50.00% (1/2)" in rec.message


def test_binance_public_limiter_enabled_and_disabled():
    with_limiter = binance_public.BinancePublicClient(rate_limit=1)
    no_limiter = binance_public.BinancePublicClient(rate_limit=None)
    try:
        assert with_limiter._rate_limiter is not None
        assert no_limiter._rate_limiter is None
    finally:
        # avoid logging on destruction
        with_limiter._rl_total = 0
        no_limiter._rl_total = 0


# --- BinanceWS limiter inclusion and counters ---

def test_binance_ws_rate_limit_counters(monkeypatch):
    async def on_bar(_):
        pass

    ws = binance_ws.BinanceWS(symbols=["BTCUSDT"], on_bar=on_bar, rate_limit=1)
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
