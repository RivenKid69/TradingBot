from decimal import Decimal

import pytest

from core_contracts import PolicyCtx
from core_models import Side
from strategies.base import SignalPosition
from strategies.momentum import MomentumStrategy


def _ctx(ts: int) -> PolicyCtx:
    return PolicyCtx(ts=ts, symbol="BTCUSDT")


def test_momentum_policy_transitions_and_orders():
    policy = MomentumStrategy()
    policy.setup(
        {
            "lookback": 2,
            "order_qty": 1.0,
            "enter_threshold": 2.0,
            "exit_threshold": 1.0,
        }
    )

    # warm-up window
    assert policy.decide({"ref_price": 100.0}, _ctx(1)) == []

    # crossing +enter_threshold -> LONG
    orders = policy.decide({"ref_price": 104.0}, _ctx(2))
    assert len(orders) == 1
    assert orders[0].side == Side.BUY
    assert orders[0].quantity == Decimal("1")
    assert policy.get_signal_state("BTCUSDT") is SignalPosition.LONG

    # signal falls below exit_threshold -> close LONG
    orders = policy.decide({"ref_price": 103.0}, _ctx(3))
    assert len(orders) == 1
    assert orders[0].side == Side.SELL
    assert policy.get_signal_state("BTCUSDT") is SignalPosition.FLAT

    # large negative signal -> open SHORT
    orders = policy.decide({"ref_price": 96.0}, _ctx(4))
    assert len(orders) == 1
    assert orders[0].side == Side.SELL
    assert policy.get_signal_state("BTCUSDT") is SignalPosition.SHORT

    # reversal SHORT -> LONG emits two BUY orders
    orders = policy.decide({"ref_price": 110.0}, _ctx(5))
    assert len(orders) == 2
    assert all(order.side == Side.BUY for order in orders)
    assert policy.get_signal_state("BTCUSDT") is SignalPosition.LONG


def test_momentum_policy_threshold_fallback_and_validation():
    policy = MomentumStrategy()

    policy.setup({"threshold": 1.5})
    assert policy.enter_threshold == 1.5
    assert policy.exit_threshold == 1.5
    assert policy.threshold == 1.5

    with pytest.raises(ValueError):
        policy.setup({"enter_threshold": 0.5, "exit_threshold": 1.0})
