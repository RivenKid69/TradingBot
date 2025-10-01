from __future__ import annotations

import math
from collections import deque
from decimal import Decimal

from core_models import Bar
from dynamic_no_trade_guard import DynamicNoTradeGuard, _SymbolState
from no_trade_config import DynamicGuardConfig


def _make_bar(
    ts: int,
    close: float,
    *,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
) -> Bar:
    close_d = Decimal(str(close))
    open_d = Decimal(str(open_ if open_ is not None else close))
    high_d = Decimal(str(high if high is not None else close))
    low_d = Decimal(str(low if low is not None else close))
    return Bar(ts=ts, symbol="SYMBOL", open=open_d, high=high_d, low=low_d, close=close_d)


def test_update_handles_missing_values_and_guard_not_ready() -> None:
    cfg = DynamicGuardConfig(
        enable=True,
        sigma_window=3,
        sigma_min_periods=3,
        spread_pctile_window=3,
        spread_pctile_min_periods=3,
    )
    cfg.volatility.upper_multiplier = 2.0
    cfg.spread.upper_pctile = 0.9
    guard = DynamicNoTradeGuard(cfg)

    state = _SymbolState(
        returns=deque([float("nan"), 0.01, float("nan")], maxlen=guard._sigma_window),
        spread=deque([float("nan"), float("nan"), float("nan")], maxlen=guard._spread_window),
        last_close=None,
    )

    guard._update_from_bar(state, _make_bar(1, 100.0), spread=float("nan"), evaluate=True)

    assert len(state.returns) == guard._sigma_window
    assert len(state.spread) == guard._spread_window
    assert math.isnan(state.returns[-1])
    assert math.isnan(state.spread[-1])
    assert state.blocked is False
    assert state.reason is None
    assert state.last_trigger == ()

    snapshot = dict(state.last_snapshot)
    assert snapshot["ready"] is False
    assert snapshot["sigma_ready"] is False
    assert snapshot["spread_ready"] is False


def test_volatility_trigger_cooldown_and_release() -> None:
    cfg = DynamicGuardConfig(
        enable=True,
        sigma_window=4,
        sigma_min_periods=3,
        cooldown_bars=2,
    )
    cfg.volatility.upper_multiplier = 2.0
    cfg.volatility.lower_multiplier = 1.0
    guard = DynamicNoTradeGuard(cfg)

    state = _SymbolState(
        returns=deque([0.01, -0.015, 0.02], maxlen=guard._sigma_window),
        spread=deque([], maxlen=guard._spread_window),
        last_close=100.0,
    )

    guard._update_from_bar(state, _make_bar(1, 110.0), spread=None, evaluate=True)

    assert state.blocked is True
    assert state.cooldown == 2
    assert state.reason == "vol_extreme"
    assert state.last_trigger == ("vol_extreme",)
    snapshot = dict(state.last_snapshot)
    assert snapshot["blocked"] is True
    assert snapshot["trigger_reasons"] == ["vol_extreme"]

    guard._update_from_bar(state, _make_bar(2, 110.55), spread=None, evaluate=True)
    assert state.blocked is True
    assert state.cooldown == 1
    assert state.reason == "vol_extreme_cooldown"

    guard._update_from_bar(state, _make_bar(3, 111.10275), spread=None, evaluate=True)
    assert state.blocked is True
    assert state.cooldown == 0
    assert state.reason.endswith("_cooldown")

    guard._update_from_bar(state, _make_bar(4, 111.65826375), spread=None, evaluate=True)
    assert state.blocked is False
    assert state.cooldown == 0
    assert state.reason is None
    assert state.last_trigger == ()
    snapshot = dict(state.last_snapshot)
    assert snapshot["blocked"] is False
    assert snapshot["trigger_reasons"] == []


def test_spread_percentile_trigger_and_release() -> None:
    cfg = DynamicGuardConfig(
        enable=True,
        sigma_window=2,
        spread_pctile_window=4,
        spread_pctile_min_periods=3,
        cooldown_bars=1,
    )
    cfg.spread.upper_pctile = 0.75
    cfg.spread.lower_pctile = 0.5
    guard = DynamicNoTradeGuard(cfg)

    state = _SymbolState(
        returns=deque([], maxlen=guard._sigma_window),
        spread=deque([5.0, 6.0, 7.0], maxlen=guard._spread_window),
        last_close=100.0,
    )

    guard._update_from_bar(state, _make_bar(1, 101.0), spread=12.0, evaluate=True)

    assert state.blocked is True
    assert state.cooldown == 1
    assert state.reason == "spread_wide"
    assert state.last_trigger == ("spread_wide",)

    guard._update_from_bar(state, _make_bar(2, 101.5), spread=4.0, evaluate=True)
    assert state.blocked is True
    assert state.cooldown == 0
    assert state.reason == "spread_wide_cooldown"

    guard._update_from_bar(state, _make_bar(3, 102.0), spread=5.0, evaluate=True)
    assert state.blocked is False
    assert state.reason is None
    assert state.last_trigger == ()


def test_spread_absolute_trigger_with_fallback_and_release() -> None:
    cfg = DynamicGuardConfig(
        enable=True,
        sigma_window=2,
        spread_pctile_window=4,
        spread_pctile_min_periods=3,
    )
    cfg.spread.abs_bps = 40.0
    guard = DynamicNoTradeGuard(cfg)
    guard._spread_abs_lower = 20.0

    state = _SymbolState(
        returns=deque([], maxlen=guard._sigma_window),
        spread=deque([10.0, 12.0, 15.0], maxlen=guard._spread_window),
        last_close=100.0,
    )

    guard._update_from_bar(
        state,
        _make_bar(1, 100.0, high=105.0, low=95.0),
        spread=float("nan"),
        evaluate=True,
    )

    assert state.blocked is True
    assert state.reason == "spread_abs"
    assert state.last_trigger == ("spread_abs",)
    snapshot = dict(state.last_snapshot)
    assert snapshot["spread"] and snapshot["spread"] > 40.0

    guard._update_from_bar(state, _make_bar(2, 100.5), spread=10.0, evaluate=True)
    assert state.blocked is False
    assert state.reason is None
    assert state.last_trigger == ()
    snapshot = dict(state.last_snapshot)
    assert snapshot["spread"] == 10.0
    assert len(state.spread) == guard._spread_window
