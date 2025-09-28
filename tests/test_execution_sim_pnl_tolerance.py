import logging

import numpy as np
import pytest

lob_state = pytest.importorskip("lob_state_cython")
coreworkspace_mod = pytest.importorskip("coreworkspace")
fast_lob_mod = pytest.importorskip("fast_lob")

SimulationWorkspace = coreworkspace_mod.SimulationWorkspace
CythonLOB = fast_lob_mod.CythonLOB
CyMicrostructureGenerator = lob_state.CyMicrostructureGenerator
EnvState = lob_state.EnvState
run_full_step_logic_cython = lob_state.run_full_step_logic_cython

from execution_sim import (
    _PNL_RECONCILE_REL_TOL,
    _check_pnl_reconciliation,
)


def test_pnl_reconciliation_large_notional(caplog: pytest.LogCaptureFixture) -> None:
    expected = 1e12
    allowed_diff = expected * _PNL_RECONCILE_REL_TOL * 0.5

    with caplog.at_level(logging.WARNING):
        _check_pnl_reconciliation(expected, expected + allowed_diff)

    assert not caplog.records

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _check_pnl_reconciliation(expected, expected + allowed_diff * 4)

    assert any(
        "PnL reconciliation drift exceeds tolerance" in record.getMessage()
        for record in caplog.records
    )


def _seed_book(lob: CythonLOB, price_scale: float, *, bids=(), asks=(), timestamp: int = 0) -> None:
    for price, volume in bids:
        lob.add_limit_order(True, int(round(price * price_scale)), volume, timestamp, False)
    for price, volume in asks:
        lob.add_limit_order(False, int(round(price * price_scale)), volume, timestamp, False)


def _base_state(price_scale: int) -> EnvState:
    state = EnvState()
    state.cash = 1_000.0
    state.units = 0.0
    state.net_worth = 1_000.0
    state.prev_net_worth = 1_000.0
    state.peak_value = 1_000.0
    state._position_value = 0.0
    state.realized_pnl_cum = 0.0
    state.step_idx = 0
    state.is_bankrupt = False
    state.next_order_id = 1
    state.taker_fee = 0.001
    state.maker_fee = 0.0
    state.profit_close_bonus = 0.0
    state.loss_close_penalty = 0.0
    state.bankruptcy_threshold = 0.0
    state.bankruptcy_penalty = 0.0
    state.max_drawdown = 1.0
    state.use_atr_stop = False
    state.use_trailing_stop = False
    state.terminate_on_sl_tp = False
    state.use_potential_shaping = False
    state.use_dynamic_risk = False
    state.use_legacy_log_reward = False
    state.gamma = 0.0
    state.last_potential = 0.0
    state.potential_shaping_coef = 0.0
    state.risk_aversion_variance = 0.0
    state.risk_aversion_drawdown = 0.0
    state.trade_frequency_penalty = 0.0
    state.turnover_penalty_coef = 0.0
    state.atr_multiplier = 1.0
    state.trailing_atr_mult = 1.0
    state.tp_atr_mult = 1.0
    state.last_pos = 0.0
    state.risk_off_level = -1.0
    state.risk_on_level = 1.0
    state.max_position_risk_off = 1.0
    state.max_position_risk_on = 1.0
    state.market_impact_k = 0.0
    state.fear_greed_value = 0.0
    state.price_scale = price_scale
    state.last_agent_fill_ratio = 0.0
    state.last_event_importance = 0.0
    state.time_since_event = 0.0
    state.last_event_step = -1
    state.token_index = 0
    state.last_realized_spread = 0.0
    state.last_executed_notional = 0.0
    state.last_bar_atr = 0.0
    return state


def test_simulator_flip_keeps_cash_and_equity_consistent() -> None:
    price_scale = 100
    workspace = SimulationWorkspace(8)
    generator = CyMicrostructureGenerator()
    state = _base_state(price_scale)

    lob = CythonLOB()
    lob.set_fee_model(state.maker_fee, state.taker_fee, 0.0)
    _seed_book(lob, price_scale, bids=[(99.0, 5.0)], asks=[(100.0, 5.0)])

    action_buy = np.array([0.1, 0.0], dtype=np.float64)
    workspace.clear_step()
    run_full_step_logic_cython(
        workspace,
        lob,
        generator,
        100.0,
        100.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0,
        0.0,
        action_buy,
        state,
    )

    # Second step: sell enough to flip to a short position at a higher price.
    lob = CythonLOB()
    lob.set_fee_model(state.maker_fee, state.taker_fee, 0.0)
    _seed_book(lob, price_scale, bids=[(110.0, 3.0)], asks=[(111.0, 3.0)])

    state.step_idx += 1
    workspace.clear_step()

    prev_net_worth = state.prev_net_worth
    cash_before = state.cash
    units_before = state.units
    entry_price = state._entry_price
    realized_before = state.realized_pnl_cum

    current_price = 110.0
    trade_volume = units_before - (-1.0)
    delta_ratio = -trade_volume * current_price / prev_net_worth
    current_pos_ratio = (units_before * current_price) / (prev_net_worth + 1e-8)
    target_pos_ratio = current_pos_ratio + delta_ratio
    action_sell = np.array([target_pos_ratio, 0.0], dtype=np.float64)

    _, _, info = run_full_step_logic_cython(
        workspace,
        lob,
        generator,
        current_price,
        current_price,
        0.0,
        0.0,
        1.0,
        1.0,
        0,
        0.0,
        action_sell,
        state,
    )

    trade_cash = trade_volume * current_price
    fee_paid = trade_cash * state.taker_fee
    cash_expected = cash_before + trade_cash - fee_paid
    realized_expected = realized_before + units_before * (current_price - entry_price)

    final_mid = (lob.get_best_bid() + lob.get_best_ask()) / (2.0 * price_scale)
    final_units = units_before - trade_volume
    net_worth_expected = cash_expected + final_units * final_mid
    step_pnl_expected = net_worth_expected - prev_net_worth

    assert state.cash == pytest.approx(cash_expected, rel=1e-12)
    assert state.units == pytest.approx(final_units, rel=1e-12)
    assert state.realized_pnl_cum == pytest.approx(realized_expected, rel=1e-12)
    assert state.net_worth == pytest.approx(net_worth_expected, rel=1e-9)
    assert info["step_pnl"] == pytest.approx(step_pnl_expected, rel=1e-9)
