import importlib.util
import pathlib
import sys

import pytest

base = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("execution_sim", base / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec)
sys.modules["execution_sim"] = exec_mod
spec.loader.exec_module(exec_mod)

ActionProto = exec_mod.ActionProto
ActionType = exec_mod.ActionType
ExecutionSimulator = exec_mod.ExecutionSimulator
estimate_slippage_bps = exec_mod.estimate_slippage_bps
apply_slippage_price = exec_mod.apply_slippage_price


def test_market_open_next_h1_slippage():
    sim = ExecutionSimulator(execution_profile="MKT_OPEN_NEXT_H1")
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)
    current_ts = 1_800_000
    pending_report = sim.run_step(
        ts=current_ts,
        ref_price=100.0,
        bid=None,
        ask=None,
        vol_factor=1.0,
        liquidity=1.0,
        bar_open=99.0,
        bar_high=101.0,
        bar_low=98.0,
        bar_close=100.0,
        bar_timeframe_ms=3_600_000,
        actions=[(ActionType.MARKET, proto)],
    )
    assert pending_report.trades == []
    assert pending_report.status == "PENDING_NEXT_BAR"
    next_open_ts = 3_600_000
    sim.set_market_snapshot(
        bid=None,
        ask=None,
        ts_ms=next_open_ts,
        bar_open=100.0,
        bar_high=102.0,
        bar_low=99.0,
        bar_close=101.0,
    )
    filled_report = sim.pop_ready(now_ts=next_open_ts, ref_price=100.0)
    assert len(filled_report.trades) == 1
    trade = filled_report.trades[0]
    assert trade.ts == next_open_ts
    expected_bps = estimate_slippage_bps(
        spread_bps=filled_report.spread_bps or 0.0,
        size=trade.qty,
        liquidity=1.0,
        vol_factor=1.0,
        cfg=sim.slippage_cfg,
    )
    expected_price = apply_slippage_price(
        side="BUY", quote_price=100.0, slippage_bps=expected_bps
    )
    assert trade.price == pytest.approx(expected_price)


def test_market_open_next_h1_missing_next_bar():
    sim = ExecutionSimulator(execution_profile="MKT_OPEN_NEXT_H1")
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)
    current_ts = 1_800_000
    report = sim.run_step(
        ts=current_ts,
        ref_price=100.0,
        bid=None,
        ask=None,
        vol_factor=1.0,
        liquidity=1.0,
        bar_open=99.0,
        bar_high=101.0,
        bar_low=98.0,
        bar_close=100.0,
        bar_timeframe_ms=3_600_000,
        actions=[(ActionType.MARKET, proto)],
    )
    assert report.trades == []
    cancel_report = sim.pop_ready(now_ts=3_600_000, ref_price=100.0)
    assert cancel_report.trades == []
    assert cancel_report.status == "CANCELED_NEXT_BAR"
    assert cancel_report.cancelled_ids
    first_id = cancel_report.cancelled_ids[0]
    assert cancel_report.cancelled_reasons[first_id] == "NO_BAR_DATA"


def test_market_open_next_h1_last_signal_wins():
    sim = ExecutionSimulator(execution_profile="MKT_OPEN_NEXT_H1")
    buy_proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)
    sell_proto = ActionProto(action_type=ActionType.MARKET, volume_frac=-2.0)
    ts = 1_800_000
    first_report = sim.run_step(
        ts=ts,
        ref_price=100.0,
        bid=None,
        ask=None,
        vol_factor=1.0,
        liquidity=1.0,
        bar_open=99.0,
        bar_high=101.0,
        bar_low=98.0,
        bar_close=100.0,
        bar_timeframe_ms=3_600_000,
        actions=[(ActionType.MARKET, buy_proto)],
    )
    assert first_report.status == "PENDING_NEXT_BAR"
    first_id = first_report.new_order_ids[0]
    second_report = sim.run_step(
        ts=ts + 60_000,
        ref_price=100.0,
        bid=None,
        ask=None,
        vol_factor=1.0,
        liquidity=1.0,
        bar_open=99.5,
        bar_high=101.5,
        bar_low=98.5,
        bar_close=100.5,
        bar_timeframe_ms=3_600_000,
        actions=[(ActionType.MARKET, sell_proto)],
    )
    assert first_id in second_report.cancelled_ids
    assert second_report.cancelled_reasons[first_id] == "SUPERSEDED"
    assert second_report.status == "PENDING_NEXT_BAR"
    next_open_ts = 3_600_000
    sim.set_market_snapshot(
        bid=None,
        ask=None,
        ts_ms=next_open_ts,
        bar_open=100.0,
        bar_high=103.0,
        bar_low=99.0,
        bar_close=102.0,
    )
    final_report = sim.pop_ready(now_ts=next_open_ts, ref_price=100.0)
    assert len(final_report.trades) == 1
    trade = final_report.trades[0]
    assert trade.side == "SELL"
    assert trade.qty == pytest.approx(2.0)
