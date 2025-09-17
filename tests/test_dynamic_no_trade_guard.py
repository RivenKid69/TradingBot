from decimal import Decimal

from core_models import Bar
from dynamic_no_trade_guard import DynamicNoTradeGuard
from no_trade_config import DynamicGuardConfig


def _make_bar(ts: int, price: float, *, high: float | None = None, low: float | None = None) -> Bar:
    base = Decimal(str(price))
    hi = Decimal(str(high if high is not None else price))
    lo = Decimal(str(low if low is not None else price))
    return Bar(ts=ts, symbol="BTCUSDT", open=base, high=hi, low=lo, close=base)


def test_dynamic_guard_triggers_and_cooldown():
    cfg = DynamicGuardConfig(
        enable=True,
        sigma_window=3,
        atr_window=3,
        spread_abs_bps=50.0,
        hysteresis=0.1,
        cooldown_bars=2,
        log_reason=True,
    )
    guard = DynamicNoTradeGuard(cfg)
    guard.prewarm("BTCUSDT", [_make_bar(i, 100.0 + i) for i in range(3)])

    guard.update("BTCUSDT", _make_bar(3, 103.0), spread=60.0)
    blocked, reason, snapshot = guard.should_block("BTCUSDT")
    assert blocked is True
    assert reason is not None and "spread_abs" in reason
    assert snapshot["cooldown"] == 2

    guard.update("BTCUSDT", _make_bar(4, 103.0), spread=45.0)
    blocked, reason, snapshot = guard.should_block("BTCUSDT")
    assert blocked is True
    assert snapshot["cooldown"] == 1
    assert reason is not None and "cooldown" in reason

    guard.update("BTCUSDT", _make_bar(5, 103.0), spread=45.0)
    blocked, reason, snapshot = guard.should_block("BTCUSDT")
    assert blocked is True
    assert snapshot["cooldown"] == 0

    guard.update("BTCUSDT", _make_bar(6, 103.0), spread=40.0)
    blocked, reason, snapshot = guard.should_block("BTCUSDT")
    assert blocked is False
    assert reason is None
    assert snapshot["trigger_reasons"] == []


def test_dynamic_guard_uses_atr_when_spread_missing():
    cfg = DynamicGuardConfig(
        enable=True,
        sigma_window=3,
        atr_window=3,
        spread_abs_bps=500.0,
    )
    guard = DynamicNoTradeGuard(cfg)
    history = [_make_bar(i, 100.0, high=105.0, low=95.0) for i in range(3)]
    guard.prewarm("ETHUSDT", history)

    guard.update("ETHUSDT", _make_bar(3, 100.0, high=105.0, low=95.0), spread=None)
    blocked, reason, snapshot = guard.should_block("ETHUSDT")
    assert blocked is True
    assert reason is not None
    assert snapshot["spread"] >= 500.0
    assert "spread_abs" in snapshot["trigger_reasons"]


def test_dynamic_guard_prewarm_does_not_block():
    cfg = DynamicGuardConfig(
        enable=True,
        sigma_window=4,
        atr_window=4,
        spread_abs_bps=80.0,
    )
    guard = DynamicNoTradeGuard(cfg)
    guard.prewarm("SOLUSDT", [_make_bar(i, 20.0 + i) for i in range(5)])

    blocked, reason, snapshot = guard.should_block("SOLUSDT")
    assert blocked is False
    assert reason is None
    assert snapshot["blocked"] is False
