"""Behavioural tests for next-bar-open execution handling."""

from __future__ import annotations

import types

import pytest

from execution_sim import ActionProto, ActionType, ExecutionSimulator


def _make_next_open_simulator() -> ExecutionSimulator:
    """Create a simulator with deterministic next-bar-open configuration."""

    sim = ExecutionSimulator(
        symbol="TESTUSDT",
        filters_path=None,
        latency_config={"base_ms": 0, "jitter_ms": 0, "timeout_ms": 0, "retries": 0},
        slippage_config={"default_spread_bps": 0.0, "k": 0.0, "min_half_spread_bps": 0.0},
        risk_config={"enabled": False},
    )

    # Force the simulator into next-bar-open mode without stochastic effects.
    sim.execution_entry_mode = "next_bar_open"
    sim.latency = None
    sim.risk = None
    sim.slippage_cfg = None

    def _zero_fee(
        self: ExecutionSimulator,
        *,
        side: str,
        price: float,
        qty: float,
        liquidity: str,
    ) -> float:
        return 0.0

    sim._compute_trade_fee = types.MethodType(_zero_fee, sim)
    sim._fees_apply_to_cumulative = types.MethodType(lambda self, fee: None, sim)

    # Provide stable environment hints consumed during order flushing.
    sim._last_spread_bps = 0.0
    sim._last_vol_factor = 1.0
    sim._last_liquidity = 10.0
    sim._strict_open_fill = False
    sim._clip_next_bar = True
    sim._bar_cap_base_enabled = True
    sim._used_base_in_bar = {}
    sim._reset_bar_capacity_if_needed = types.MethodType(lambda self, ts_ms: 4.0, sim)

    return sim


def test_next_bar_open_flow() -> None:
    sim = _make_next_open_simulator()
    sim.set_next_open_price(110.0)

    # Seed initial snapshot so the simulator knows when the next bar should open.
    sim._update_next_open_context(
        ts_ms=0,
        bar_open=100.0,
        bar_high=101.0,
        bar_low=99.0,
        bar_close=100.5,
        timeframe_ms=60_000,
    )

    sim.set_ref_price(110.0)

    first_order = ActionProto(action_type=ActionType.MARKET, volume_frac=6.0)
    cid_first = sim.submit(first_order, now_ts=10_000)

    assert sim._next_open_metrics == {"submitted": 1, "filled": 0, "cancelled": 0}

    sim._update_next_open_context(
        ts_ms=60_000,
        bar_open=110.0,
        bar_high=112.0,
        bar_low=108.0,
        bar_close=111.0,
        timeframe_ms=60_000,
    )

    assert sim._next_open_metrics == {"submitted": 1, "filled": 1, "cancelled": 0}
    assert not sim._pending_next_open
    assert len(sim._next_open_ready_trades) == 1

    trade = sim._next_open_ready_trades[0]
    assert trade.qty == pytest.approx(4.0)
    assert trade.fill_ratio == pytest.approx(4.0 / 6.0)
    assert trade.capacity_reason == "BAR_CAPACITY_BASE"
    assert trade.exec_status == "PARTIAL"

    report_first = sim._pop_ready_next_open(
        now_ts=60_000,
        ref_price=110.0,
        bid=109.5,
        ask=110.5,
        bar_open=110.0,
        bar_high=112.0,
        bar_low=108.0,
        bar_close=111.0,
    )

    assert report_first.status == "FILLED_NEXT_BAR"
    assert report_first.fee_total == pytest.approx(0.0)
    assert report_first.risk_events == []
    assert not sim._pending_next_open
    assert [t.qty for t in report_first.trades] == [pytest.approx(4.0)]
    assert report_first.exec_status == "PARTIAL"
    assert report_first.capacity_reason == "BAR_CAPACITY_BASE"

    sim.set_ref_price(100.0)
    second_order = ActionProto(action_type=ActionType.MARKET, volume_frac=2.0)
    cid_second = sim.submit(second_order, now_ts=70_000)

    assert sim._next_open_metrics == {"submitted": 1, "filled": 0, "cancelled": 0}

    sim._next_h1_open_price = None
    sim._update_next_open_context(
        ts_ms=120_000,
        bar_open=None,
        bar_high=None,
        bar_low=None,
        bar_close=None,
        timeframe_ms=60_000,
    )

    metrics_after_missing_open = sim._next_open_metrics
    assert metrics_after_missing_open["submitted"] == 1
    assert metrics_after_missing_open["filled"] == 0
    assert metrics_after_missing_open["cancelled"] >= 1
    assert cid_second in sim._next_open_cancelled
    assert sim._next_open_cancelled_reasons[cid_second] == "EXPIRED_NO_BAR_OPEN"

    report_second = sim._pop_ready_next_open(
        now_ts=120_000,
        ref_price=100.0,
        bid=None,
        ask=None,
        bar_open=None,
        bar_high=None,
        bar_low=None,
        bar_close=None,
    )

    assert report_second.cancelled_ids == [cid_second]
    assert report_second.status == "EXPIRED_NEXT_BAR"
    assert report_second.reason == {"code": "MISSING_BAR_OPEN"}
    assert not sim._pending_next_open

    sim.set_ref_price(100.0)
    third_order = ActionProto(action_type=ActionType.MARKET, volume_frac=1.5)
    cid_third = sim.submit(third_order, now_ts=130_000)

    assert sim._pending_next_open
    assert sim._next_open_metrics == {"submitted": 1, "filled": 0, "cancelled": 0}

    report_third = sim._pop_ready_next_open(
        now_ts=180_000,
        ref_price=100.0,
        bid=None,
        ask=None,
        bar_open=None,
        bar_high=None,
        bar_low=None,
        bar_close=None,
    )

    assert cid_third in report_third.cancelled_ids
    assert report_third.cancelled_reasons[cid_third] == "NO_BAR_DATA"
    assert report_third.status == "CANCELED_NEXT_BAR"
    assert report_third.reason == {"code": "MISSING_NEXT_BAR"}
    assert not sim._pending_next_open
    final_metrics = sim._next_open_metrics
    assert final_metrics["submitted"] == 1
    assert final_metrics["filled"] == 0
    assert final_metrics["cancelled"] >= 1
