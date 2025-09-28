from decimal import Decimal

import pytest

from core_config import SpotCostConfig, SpotImpactConfig
from core_models import Bar, Order, OrderType, Side

from impl_bar_executor import BarExecutor, PortfolioState, decide_spot_trade


def make_bar(ts: int, price: float) -> Bar:
    return Bar(
        ts=ts,
        symbol="BTCUSDT",
        open=Decimal(str(price)),
        high=Decimal(str(price * 1.01)),
        low=Decimal(str(price * 0.99)),
        close=Decimal(str(price)),
        volume_base=Decimal("1"),
        volume_quote=Decimal(str(price)),
    )


def test_decide_spot_trade_costs_and_net():
    state = PortfolioState(symbol="BTCUSDT", weight=0.1, equity_usd=1000.0)
    cfg = SpotCostConfig(
        taker_fee_bps=5.0,
        half_spread_bps=10.0,
        impact=SpotImpactConfig(sqrt_coeff=15.0, linear_coeff=2.0),
    )
    signal = {"target_weight": 0.3, "edge_bps": 50.0}
    metrics = decide_spot_trade(signal, state, cfg, adv_quote=50000.0, safety_margin_bps=3.0)

    # Expected delta = 0.2 -> turnover_usd = 200.0, participation = 0.004
    assert metrics.turnover_usd == pytest.approx(200.0)
    base_cost = 15.0  # 5 + 10
    impact = 15.0 * (0.004 ** 0.5) + 2.0 * 0.004
    assert metrics.cost_bps == pytest.approx(base_cost + impact)
    assert metrics.edge_bps == 50.0
    assert metrics.net_bps == pytest.approx(50.0 - (base_cost + impact) - 3.0)
    assert metrics.act_now is True


def test_bar_executor_target_weight_single_instruction():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        min_rebalance_step=0.0,
        cost_config=SpotCostConfig(),
        default_equity_usd=1000.0,
    )
    order = Order(
        ts=1,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": make_bar(1, 10000.0),
            "payload": {"target_weight": 0.5, "edge_bps": 20.0},
        },
    )
    report = executor.execute(order)
    instructions = report.meta["instructions"]
    assert len(instructions) == 1
    instr = instructions[0]
    assert instr["target_weight"] == 0.5
    assert instr["delta_weight"] == 0.5
    assert instr["slice_index"] == 0
    positions = executor.get_open_positions()
    pos = positions["BTCUSDT"]
    assert pos.meta["weight"] == 0.5


def test_bar_executor_delta_weight_twap_and_participation():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(),
        default_equity_usd=1000.0,
    )
    bar = make_bar(2, 5000.0)
    order = Order(
        ts=2,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": bar,
            "adv_quote": 1000.0,
            "payload": {
                "delta_weight": 0.4,
                "edge_bps": 30.0,
                "twap": {"parts": 2, "interval_s": 60},
                "max_participation": 0.05,
            },
        },
    )
    report = executor.execute(order)
    instructions = report.meta["instructions"]
    # Max participation with adv 1000 => max slice notional 50, total notional = 400
    # Requires at least 8 slices (400 / 50)
    assert len(instructions) == 8
    assert instructions[0]["slice_index"] == 0
    assert instructions[-1]["slice_index"] == 7
    assert instructions[-1]["target_weight"] == 0.4


def test_bar_executor_respects_min_rebalance_step():
    executor = BarExecutor(
        run_id="test",
        min_rebalance_step=0.2,
        cost_config=SpotCostConfig(),
        default_equity_usd=1000.0,
    )
    order = Order(
        ts=3,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": make_bar(3, 100.0),
            "payload": {"target_weight": 0.1, "edge_bps": 10.0},
        },
    )
    report = executor.execute(order)
    assert report.meta.get("min_step_enforced") is True
    assert report.meta["instructions"] == []
    positions = executor.get_open_positions()
    assert positions["BTCUSDT"].meta["weight"] == 0.0

