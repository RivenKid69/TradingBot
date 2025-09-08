from decimal import Decimal

from core_contracts import PolicyCtx
from core_models import Side
from strategies.momentum import MomentumStrategy


def _ctx(ts: int) -> PolicyCtx:
    return PolicyCtx(ts=ts, symbol="BTCUSDT")


def test_momentum_policy_produces_orders():
    policy = MomentumStrategy()
    policy.setup({"lookback": 3, "order_qty": 1.0})

    # first two observations - insufficient window
    assert policy.decide({"ref_price": 100.0}, _ctx(1)) == []
    assert policy.decide({"ref_price": 101.0}, _ctx(2)) == []

    # third observation triggers BUY
    orders = policy.decide({"ref_price": 103.0}, _ctx(3))
    assert len(orders) == 1
    o = orders[0]
    assert o.side == Side.BUY
    assert o.quantity == Decimal("1")

    # price drops below average -> SELL
    orders = policy.decide({"ref_price": 90.0}, _ctx(4))
    assert len(orders) == 1
    o = orders[0]
    assert o.side == Side.SELL
    assert o.quantity == Decimal("1")
