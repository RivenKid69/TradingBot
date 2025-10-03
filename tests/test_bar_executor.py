import logging
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from api.spot_signals import (
    SpotSignalEconomics,
    SpotSignalEnvelope,
    SpotSignalTargetWeightPayload,
)

from core_config import SpotCostConfig, SpotImpactConfig, SpotTurnoverCaps, SpotTurnoverLimit
from core_models import Bar, Order, OrderType, Side
from impl_bar_executor import BarExecutor, PortfolioState, decide_spot_trade
from service_signal_runner import _Worker


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
    assert metrics.impact == pytest.approx(impact)
    assert metrics.impact_mode == "model"


def test_decide_spot_trade_safety_margin_blocks_trade():
    state = PortfolioState(symbol="BTCUSDT", weight=0.2, equity_usd=500.0)
    cfg = SpotCostConfig(taker_fee_bps=1.0, half_spread_bps=1.0)

    # Incoming payload that would reduce exposure, but with low edge.
    signal = {"delta_weight": -0.1, "edge_bps": 3.5}
    metrics = decide_spot_trade(signal, state, cfg, adv_quote=None, safety_margin_bps=5.0)

    assert metrics.turnover_usd == pytest.approx(50.0)
    # Base cost of 2 bps plus safety margin 5 bps exceeds the edge => net negative
    assert metrics.cost_bps == pytest.approx(2.0)
    assert metrics.net_bps == pytest.approx(3.5 - 2.0 - 5.0)
    assert metrics.act_now is False
    assert metrics.impact == pytest.approx(0.0)
    assert metrics.impact_mode == "none"


def test_decide_spot_trade_prefers_signal_turnover_usd():
    state = PortfolioState(symbol="BTCUSDT", weight=0.0, equity_usd=0.0)
    cfg = SpotCostConfig()
    signal = {"target_weight": 0.5, "edge_bps": 10.0, "turnover_usd": 123.0}

    metrics = decide_spot_trade(signal, state, cfg, adv_quote=None, safety_margin_bps=0.0)

    assert metrics.turnover_usd == pytest.approx(123.0)
    assert metrics.act_now is True


def test_decide_spot_trade_honors_false_act_now_flag():
    state = PortfolioState(symbol="BTCUSDT", weight=0.0, equity_usd=1000.0)
    cfg = SpotCostConfig()
    signal = {
        "target_weight": 0.2,
        "edge_bps": 50.0,
        "act_now": "false",
    }

    metrics = decide_spot_trade(signal, state, cfg, adv_quote=None, safety_margin_bps=0.0)

    assert metrics.net_bps > 0.0
    assert metrics.turnover_usd == pytest.approx(200.0)
    assert metrics.act_now is False


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
    assert report.meta["execution_mode"] == "bar"
    assert report.meta["execution"]["execution_mode"] == "bar"
    instructions = report.meta["instructions"]
    assert len(instructions) == 1
    instr = instructions[0]
    assert instr["target_weight"] == 0.5
    assert instr["delta_weight"] == 0.5
    assert instr["slice_index"] == 0
    positions = executor.get_open_positions()
    pos = positions["BTCUSDT"]
    assert pos.meta["weight"] == 0.5
    assert pos.qty == Decimal("0.05")


def test_bar_executor_initial_weights_respected_on_first_execute():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        min_rebalance_step=0.0,
        cost_config=SpotCostConfig(),
        default_equity_usd=1000.0,
        initial_weights={"BTCUSDT": 0.25},
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

    assert instructions, "expected rebalance instructions on first execute"
    instr = instructions[0]
    assert instr["target_weight"] == pytest.approx(0.5)
    assert instr["delta_weight"] == pytest.approx(0.25)
    decision = report.meta["decision"]
    assert decision["delta_weight"] == pytest.approx(0.25)


def test_bar_executor_reduces_weight_after_price_increase():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        min_rebalance_step=0.0,
        cost_config=SpotCostConfig(),
        default_equity_usd=1000.0,
    )

    open_order = Order(
        ts=1,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": make_bar(1, 10000.0),
            "payload": {"target_weight": 0.5, "edge_bps": 50.0},
        },
    )
    first_report = executor.execute(open_order)
    assert first_report.meta["instructions"]

    rebalance_order = Order(
        ts=2,
        symbol="BTCUSDT",
        side=Side.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": make_bar(2, 12000.0),
            "payload": {"target_weight": 0.5, "edge_bps": 50.0},
        },
    )

    second_report = executor.execute(rebalance_order)
    instructions = second_report.meta["instructions"]
    assert instructions, "expected a sell instruction to maintain target weight"
    assert instructions[0]["delta_weight"] == pytest.approx(-0.1, rel=1e-6)


def test_bar_executor_preserves_sell_side_in_exec_report():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        min_rebalance_step=0.0,
        cost_config=SpotCostConfig(),
        default_equity_usd=1000.0,
        initial_weights={"BTCUSDT": 0.5},
    )
    order = Order(
        ts=2,
        symbol="BTCUSDT",
        side="SELL",
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": make_bar(2, 10000.0),
            "payload": {"target_weight": 0.0, "edge_bps": 20.0},
        },
    )

    report = executor.execute(order)

    assert report.side is Side.SELL
    instructions = report.meta["instructions"]
    assert instructions and instructions[0]["delta_weight"] == pytest.approx(-0.5)

    positions = executor.get_open_positions()
    pos = positions["BTCUSDT"]
    assert pos.meta["weight"] == pytest.approx(0.0)


def test_bar_executor_handles_symbol_case_variants():
    executor = BarExecutor(
        run_id="test",
        default_equity_usd=500.0,
        initial_weights={"ethusdt": 0.2},
        cost_config=SpotCostConfig(),
    )

    first_bar = Bar(
        ts=100,
        symbol="ETHUSDT",
        open=Decimal("1000"),
        high=Decimal("1005"),
        low=Decimal("995"),
        close=Decimal("1002"),
        volume_base=Decimal("1"),
        volume_quote=Decimal("1002"),
    )
    increase_order = Order(
        ts=100,
        symbol="ETHUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": first_bar,
            "payload": {"target_weight": 0.4, "edge_bps": 25.0},
        },
    )

    first_report = executor.execute(increase_order)
    assert first_report.meta["decision"]["target_weight"] == pytest.approx(0.4)
    assert first_report.meta["decision"]["delta_weight"] == pytest.approx(0.2)

    second_bar = Bar(
        ts=200,
        symbol="ETHUSDT",
        open=Decimal("1100"),
        high=Decimal("1105"),
        low=Decimal("1095"),
        close=Decimal("1102"),
        volume_base=Decimal("1"),
        volume_quote=Decimal("1102"),
    )
    decrease_order = Order(
        ts=200,
        symbol="ethusdt",
        side=Side.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": second_bar,
            "payload": {"target_weight": 0.1, "edge_bps": 25.0},
        },
    )

    second_report = executor.execute(decrease_order)
    assert second_report.meta["decision"]["target_weight"] == pytest.approx(0.1)
    price_ratio = float(second_bar.close) / float(first_bar.close)
    expected_delta = 0.1 - (0.4 * price_ratio)
    assert second_report.meta["decision"]["delta_weight"] == pytest.approx(
        expected_delta
    )

    positions_upper = executor.get_open_positions(symbols=["ETHUSDT"])
    assert "ETHUSDT" in positions_upper
    assert positions_upper["ETHUSDT"].meta["weight"] == pytest.approx(0.1)

    positions_lower = executor.get_open_positions(symbols=["ethusdt"])
    assert "ETHUSDT" in positions_lower
    assert positions_lower["ETHUSDT"].meta["weight"] == pytest.approx(0.1)


def test_bar_executor_includes_decision_costs():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(taker_fee_bps=2.0, half_spread_bps=3.0),
        safety_margin_bps=1.5,
        default_equity_usd=2000.0,
    )

    order = Order(
        ts=10,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": make_bar(10, 20000.0),
            "adv_quote": 1_000_000.0,
            "payload": {"target_weight": 0.25, "edge_bps": 20.0},
        },
    )

    report = executor.execute(order)
    decision = report.meta["decision"]
    assert decision["cost_bps"] == pytest.approx(5.0)
    assert decision["net_bps"] == pytest.approx(20.0 - 5.0 - 1.5)
    assert decision["impact_mode"] == "model"


def test_bar_executor_adv_fallback_from_economics():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(
            taker_fee_bps=2.0,
            half_spread_bps=3.0,
            impact=SpotImpactConfig(sqrt_coeff=15.0, linear_coeff=2.0),
        ),
        safety_margin_bps=0.0,
        default_equity_usd=2000.0,
    )

    order = Order(
        ts=101,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": make_bar(101, 20000.0),
            "payload": {
                "target_weight": 0.25,
                "edge_bps": 20.0,
                "economics": {"adv_quote": 1_000_000.0},
            },
        },
    )

    report = executor.execute(order)

    decision = report.meta["decision"]
    assert decision["adv_quote"] == pytest.approx(1_000_000.0)
    assert decision["impact_mode"] == "model"
    assert decision["impact"] > 0.0
    assert report.meta["adv_quote"] == pytest.approx(1_000_000.0)


def test_bar_executor_missing_adv_sets_impact_mode_none():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(taker_fee_bps=2.0, half_spread_bps=3.0),
        safety_margin_bps=0.0,
        default_equity_usd=2000.0,
    )

    order = Order(
        ts=11,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": make_bar(11, 20000.0),
            "payload": {"target_weight": 0.25, "edge_bps": 20.0},
        },
    )

    report = executor.execute(order)
    decision = report.meta["decision"]
    assert decision["impact"] == pytest.approx(0.0)
    assert decision["impact_mode"] == "none"
    assert decision["turnover_usd"] == pytest.approx(500.0)


def test_bar_executor_clears_turnover_when_skipping_execution():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(taker_fee_bps=12.5, half_spread_bps=12.5),
        default_equity_usd=1000.0,
        safety_margin_bps=0.0,
    )

    order = Order(
        ts=12,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": make_bar(12, 15000.0),
            "payload": {"target_weight": 0.2, "edge_bps": 5.0},
        },
    )

    report = executor.execute(order)

    decision = report.meta["decision"]
    assert decision["act_now"] is False
    assert decision["turnover_usd"] == pytest.approx(0.0)
    execution_meta = report.meta["execution"]
    assert execution_meta["turnover_usd"] == pytest.approx(0.0)
    assert report.meta["executed_turnover_usd"] == pytest.approx(0.0)
    assert report.meta["instructions"] == []

    snapshot = executor.monitoring_snapshot()
    assert snapshot["act_now"] is False
    assert snapshot["turnover_usd"] == pytest.approx(0.0)
    assert isinstance(order.meta, dict)
    bar_execution = order.meta.get("_bar_execution")
    assert bar_execution is not None
    assert bar_execution["turnover_usd"] == pytest.approx(0.0)


def test_bar_executor_turnover_cap_allows_quantized_trade():
    caps = SpotTurnoverCaps(per_symbol=SpotTurnoverLimit(usd=70.0))
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(),
        default_equity_usd=1000.0,
        turnover_caps=caps,
        symbol_specs={"BTCUSDT": {"step_size": Decimal("0.6")}},
    )

    order = Order(
        ts=20,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": make_bar(20, 100.0),
            "payload": {"target_weight": 0.1, "edge_bps": 50.0},
        },
    )

    report = executor.execute(order)

    instructions = report.meta["instructions"]
    assert len(instructions) == 1
    instr = instructions[0]
    assert instr["notional_usd"] == pytest.approx(60.0)
    assert instr["target_weight"] == pytest.approx(0.06)

    decision = report.meta["decision"]
    assert decision["turnover_usd"] == pytest.approx(60.0)
    assert decision["target_weight"] == pytest.approx(0.06)
    assert decision["delta_weight"] == pytest.approx(0.06)

    assert report.meta["executed_turnover_usd"] == pytest.approx(60.0)
    assert report.meta["cap_usd"] == pytest.approx(70.0)
    assert report.meta.get("turnover_cap_enforced") is None
    assert report.meta["requested_target_weight"] == pytest.approx(0.1)

    positions = executor.get_open_positions()
    assert positions["BTCUSDT"].meta["weight"] == pytest.approx(0.06)


def test_bar_executor_payload_turnover_override_caps_schedule():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(),
        default_equity_usd=1000.0,
    )

    override_turnover = 120.0

    order = Order(
        ts=21,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": make_bar(21, 100.0),
            "payload": {
                "target_weight": 0.5,
                "edge_bps": 50.0,
                "turnover_usd": override_turnover,
            },
        },
    )

    report = executor.execute(order)

    instructions = report.meta["instructions"]
    assert len(instructions) == 1
    instr = instructions[0]
    assert instr["notional_usd"] == pytest.approx(override_turnover)
    assert instr["target_weight"] == pytest.approx(0.12)
    assert instr["delta_weight"] == pytest.approx(0.12)

    decision = report.meta["decision"]
    assert decision["turnover_usd"] == pytest.approx(override_turnover)
    assert report.meta["executed_turnover_usd"] == pytest.approx(override_turnover)
    assert report.meta["requested_target_weight"] == pytest.approx(0.5)

    positions = executor.get_open_positions()
    assert positions["BTCUSDT"].meta["weight"] == pytest.approx(0.12)


def test_bar_executor_handles_envelope_meta():
    class EnvelopePayload(SpotSignalTargetWeightPayload):
        def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
            data = super().model_dump(*args, **kwargs)
            data["edge_bps"] = float(self.economics.edge_bps)
            return data

    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(),
        default_equity_usd=100.0,
    )
    bar = make_bar(42, 10_000.0)
    economics = SpotSignalEconomics(
        edge_bps=75.0,
        cost_bps=5.0,
        net_bps=65.0,
        turnover_usd=0.0,
        act_now=True,
        impact=0.0,
        impact_mode="model",
    )
    payload = EnvelopePayload(economics=economics, target_weight=0.4)
    expires_at = bar.ts + 60_000
    envelope = SpotSignalEnvelope(
        symbol="BTCUSDT",
        bar_close_ms=bar.ts,
        expires_at_ms=expires_at,
        payload=payload,
    )
    adv_quote = 50_000.0
    equity_override = 2_500.0
    object.__setattr__(envelope, "adv_quote", adv_quote)
    object.__setattr__(envelope, "equity_usd", equity_override)
    object.__setattr__(envelope, "bar", bar)

    order = Order(
        ts=bar.ts,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta=envelope,
    )

    report = executor.execute(order)

    assert isinstance(order.meta, dict)
    assert set(order.meta.keys()) >= {"payload", "_bar_execution"}
    payload_meta = order.meta["payload"]
    assert payload_meta["target_weight"] == pytest.approx(payload.target_weight)
    economics_meta = payload_meta.get("economics")
    assert economics_meta["edge_bps"] == pytest.approx(economics.edge_bps)

    instructions = report.meta["instructions"]
    assert len(instructions) == 1
    instruction = instructions[0]
    assert instruction["target_weight"] == pytest.approx(payload.target_weight)
    decision = report.meta["decision"]
    assert decision["edge_bps"] == pytest.approx(economics.edge_bps)
    assert decision["turnover_usd"] == pytest.approx(
        payload.target_weight * equity_override
    )
    assert decision["impact_mode"] == "model"
    assert report.meta["adv_quote"] == pytest.approx(adv_quote)
    assert report.meta["reference_price"] == pytest.approx(float(bar.close))
    assert report.price == bar.close


def test_bar_executor_falls_back_to_nested_economics() -> None:
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(taker_fee_bps=2.5, half_spread_bps=2.5),
        default_equity_usd=1000.0,
    )
    bar = make_bar(100, 20_000.0)
    economics = SpotSignalEconomics(
        edge_bps=55.0,
        cost_bps=5.0,
        net_bps=50.0,
        turnover_usd=321.0,
        act_now=True,
        impact=0.0,
        impact_mode="model",
    )
    payload = SpotSignalTargetWeightPayload(
        economics=economics,
        target_weight=0.4,
    )
    order = Order(
        ts=bar.ts,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": bar,
            "payload": payload.model_dump(),
        },
    )

    report = executor.execute(order)

    decision = report.meta["decision"]
    assert decision["edge_bps"] == pytest.approx(economics.edge_bps)
    assert decision["net_bps"] == pytest.approx(
        economics.edge_bps - economics.cost_bps
    )
    assert decision["act_now"] is True
    assert decision["turnover_usd"] == pytest.approx(economics.turnover_usd)
    expected_target = economics.turnover_usd / executor.default_equity_usd
    assert decision["target_weight"] == pytest.approx(expected_target)
    assert decision["delta_weight"] == pytest.approx(expected_target)


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
    assert instructions[-1]["target_weight"] == pytest.approx(0.4)


def test_bar_executor_uses_default_max_participation():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(),
        default_equity_usd=1_000_000.0,
        max_participation=0.05,
    )
    bar = make_bar(20, 10_000.0)
    order = Order(
        ts=20,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": bar,
            "adv_quote": 1_000_000.0,
            "payload": {"delta_weight": 0.2, "edge_bps": 25.0},
        },
    )

    report = executor.execute(order)
    instructions = report.meta["instructions"]
    assert len(instructions) == 4
    assert instructions[-1]["target_weight"] == pytest.approx(0.2)


def test_bar_executor_turnover_cap_blocks_trade():
    cost_cfg = SpotCostConfig(
        turnover_caps=SpotTurnoverCaps(
            per_symbol=SpotTurnoverLimit(usd=100.0)
        )
    )
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=cost_cfg,
        default_equity_usd=1000.0,
    )
    order = Order(
        ts=30,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": make_bar(30, 10_000.0),
            "payload": {"target_weight": 0.2, "edge_bps": 20.0},
        },
    )

    report = executor.execute(order)
    assert report.meta["instructions"] == []
    assert report.meta.get("turnover_cap_enforced") is True
    assert report.meta["decision"]["act_now"] is False
    assert report.meta["cap_usd"] == pytest.approx(100.0)


def test_bar_executor_invalid_target_weight_skips_trade():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(),
        default_equity_usd=1_000.0,
    )

    bar = make_bar(55, 10_000.0)
    order = Order(
        ts=bar.ts,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": bar,
            "payload": {"target_weight": "bad", "edge_bps": 50.0},
        },
    )

    report = executor.execute(order)

    decision = report.meta["decision"]
    assert decision["turnover_usd"] == pytest.approx(0.0)
    assert decision["impact"] == pytest.approx(0.0)
    assert decision["act_now"] is False
    assert report.meta["instructions"] == []


def test_bar_executor_rounds_quantities_to_step_size():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(),
        default_equity_usd=1_000.0,
        symbol_specs={"BTCUSDT": {"step_size": 0.2}},
    )
    bar = make_bar(50, 200.0)
    order = Order(
        ts=50,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": bar,
            "payload": {"target_weight": 0.05, "edge_bps": 25.0},
        },
    )

    report = executor.execute(order)
    instructions = report.meta["instructions"]
    assert len(instructions) == 1
    instr = instructions[0]
    assert instr["quantity"] == pytest.approx(0.2)
    assert instr["notional_usd"] == pytest.approx(40.0)
    assert instr["delta_weight"] == pytest.approx(0.04)
    assert instr["target_weight"] == pytest.approx(0.04)
    assert report.meta["requested_target_weight"] == pytest.approx(0.05)
    assert report.meta["target_weight"] == pytest.approx(0.04)
    assert report.meta["decision"]["turnover_usd"] == pytest.approx(40.0)


def test_bar_executor_zero_delta_after_rounding_sets_act_now_false():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(),
        default_equity_usd=1_000.0,
        symbol_specs={"BTCUSDT": {"step_size": 0.5}},
    )
    bar = make_bar(55, 100.0)
    order = Order(
        ts=55,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": bar,
            "payload": {"target_weight": 0.01, "edge_bps": 100.0},
        },
    )

    report = executor.execute(order)

    assert report.meta["instructions"] == []
    decision = report.meta["decision"]
    assert decision["act_now"] is False
    assert decision["turnover_usd"] == pytest.approx(0.0)
    snapshot = executor.monitoring_snapshot()
    assert snapshot["act_now"] is False


def test_bar_executor_rejects_below_min_notional_after_rounding():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(),
        default_equity_usd=1_000.0,
        symbol_specs={"BTCUSDT": {"step_size": 0.2, "min_notional": 50.0}},
    )
    bar = make_bar(60, 200.0)
    order = Order(
        ts=60,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": bar,
            "payload": {"target_weight": 0.05, "edge_bps": 25.0},
        },
    )

    report = executor.execute(order)
    assert report.meta["instructions"] == []
    assert report.meta.get("reason") == "below_min_notional"
    assert report.meta["decision"]["act_now"] is False
    assert report.meta["decision"]["turnover_usd"] == pytest.approx(0.0)
    assert report.meta["target_weight"] == pytest.approx(0.0)
    assert report.meta["requested_target_weight"] == pytest.approx(0.05)


def test_bar_executor_rejects_twap_slices_below_min_notional():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(),
        default_equity_usd=1_000.0,
        symbol_specs={"BTCUSDT": {"min_notional": 50.0}},
    )
    bar = make_bar(65, 200.0)
    order = Order(
        ts=65,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": bar,
            "payload": {
                "delta_weight": 0.12,
                "edge_bps": 25.0,
                "twap": {"parts": 3},
            },
        },
    )

    report = executor.execute(order)

    assert report.meta["instructions"] == []
    assert report.meta.get("reason") == "below_min_notional"
    decision = report.meta["decision"]
    assert decision["act_now"] is False
    assert decision["turnover_usd"] == pytest.approx(0.0)
    assert report.meta["target_weight"] == pytest.approx(0.0)
    assert report.meta["requested_delta_weight"] == pytest.approx(0.12)


@pytest.mark.parametrize("price", [0.0, -100.0])
def test_bar_executor_skips_when_price_invalid(price: float):
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(),
        default_equity_usd=1000.0,
    )
    bar = make_bar(123, price)
    order = Order(
        ts=123,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": bar,
            "payload": {"target_weight": 0.5, "edge_bps": 50.0},
        },
    )

    report = executor.execute(order)

    assert report.meta["instructions"] == []
    assert report.meta.get("reason") == "no_price"
    decision = report.meta["decision"]
    assert decision["act_now"] is False
    assert decision.get("reason") == "no_price"
    assert decision["turnover_usd"] == 0.0

    snapshot = executor.monitoring_snapshot()
    assert snapshot.get("act_now") is False
    assert snapshot.get("reason") == "no_price"


def test_bar_executor_turnover_cap_tracks_portfolio_usage():
    cost_cfg = SpotCostConfig(
        turnover_caps=SpotTurnoverCaps(
            portfolio=SpotTurnoverLimit(usd=150.0)
        )
    )
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=cost_cfg,
        default_equity_usd=1000.0,
    )
    bar = make_bar(40, 5000.0)

    first_order = Order(
        ts=40,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": bar,
            "payload": {"target_weight": 0.05, "edge_bps": 15.0},
        },
    )
    first_report = executor.execute(first_order)
    assert first_report.meta["instructions"]
    assert first_report.meta.get("turnover_cap_enforced") is None
    assert first_report.meta["portfolio_turnover_cap_usd"] == pytest.approx(150.0)
    assert first_report.meta["cap_usd"] == pytest.approx(150.0)

    second_order = Order(
        ts=41,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": bar,
            "payload": {"target_weight": 0.25, "edge_bps": 25.0},
        },
    )
    second_report = executor.execute(second_order)
    assert second_report.meta["instructions"] == []
    assert second_report.meta.get("turnover_cap_enforced") is True
    assert second_report.meta["cap_usd"] == pytest.approx(100.0)
    assert second_report.meta["decision"]["turnover_usd"] == pytest.approx(0.0)

    snapshot = executor.monitoring_snapshot()
    assert snapshot["turnover_usd"] == pytest.approx(0.0)
    assert snapshot.get("turnover_cap_enforced") is True
    assert snapshot["cap_usd"] == pytest.approx(100.0)

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
    assert report.meta["decision"]["turnover_usd"] == pytest.approx(0.0)
    assert report.meta["target_weight"] == pytest.approx(0.0)
    assert report.meta["delta_weight"] == pytest.approx(0.0)
    assert report.meta["decision"]["target_weight"] == pytest.approx(0.0)
    assert report.meta["decision"]["delta_weight"] == pytest.approx(0.0)
    positions = executor.get_open_positions()
    assert positions["BTCUSDT"].meta["weight"] == 0.0

    snapshot = executor.monitoring_snapshot()
    assert snapshot["turnover_usd"] == pytest.approx(0.0)
    assert snapshot.get("min_step_enforced") is True
    assert snapshot["target_weight"] == pytest.approx(0.0)
    assert snapshot["delta_weight"] == pytest.approx(0.0)


def test_bar_executor_skip_reason_preserves_weights():
    executor = BarExecutor(
        run_id="test",
        cost_config=SpotCostConfig(),
        default_equity_usd=1000.0,
        initial_weights={"BTCUSDT": 0.2},
    )
    order = Order(
        ts=7,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {"target_weight": 0.5, "edge_bps": 50.0},
        },
    )

    report = executor.execute(order)

    assert report.meta.get("reason") == "no_price"
    assert report.meta["instructions"] == []
    assert report.meta["target_weight"] == pytest.approx(0.2)
    assert report.meta["delta_weight"] == pytest.approx(0.0)
    assert report.meta["decision"]["target_weight"] == pytest.approx(0.2)
    assert report.meta["decision"]["delta_weight"] == pytest.approx(0.0)

    snapshot = executor.monitoring_snapshot()
    assert snapshot["target_weight"] == pytest.approx(0.2)
    assert snapshot["delta_weight"] == pytest.approx(0.0)
    assert snapshot.get("reason") == "no_price"


def test_bar_executor_skips_when_edge_insufficient():
    executor = BarExecutor(
        run_id="test",
        cost_config=SpotCostConfig(taker_fee_bps=10.0, half_spread_bps=5.0),
        safety_margin_bps=10.0,
        default_equity_usd=1000.0,
    )

    order = Order(
        ts=4,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": make_bar(4, 15000.0),
            "payload": {"target_weight": 0.3, "edge_bps": 10.0},
        },
    )

    report = executor.execute(order)
    assert report.meta["instructions"] == []
    assert report.meta["decision"]["act_now"] is False
    assert executor.get_open_positions()["BTCUSDT"].meta["weight"] == 0.0


def _make_worker(max_total_weight: float | None, *, existing_weights: dict[str, float] | None = None) -> _Worker:
    class DummyFeaturePipe:
        def __init__(self) -> None:
            self.spread_ttl_ms = 0
            self._spread_ttl_ms = 0
            self.signal_quality: dict[str, Any] = {}
            self.metrics = SimpleNamespace(reset_symbol=lambda *_: None)

    class DummyPolicy:
        def consume_signal_transitions(self):  # pragma: no cover - simple stub
            return []

    fp = DummyFeaturePipe()
    policy = DummyPolicy()
    logger = logging.getLogger("test_worker_normalization")
    executor = SimpleNamespace()
    worker = _Worker(
        fp,
        policy,
        logger,
        executor,
        None,
        enforce_closed_bars=True,
        close_lag_ms=0,
        ws_dedup_enabled=False,
        ws_dedup_log_skips=False,
        ws_dedup_timeframe_ms=0,
        throttle_cfg=None,
        no_trade_cfg=None,
        pipeline_cfg=None,
        signal_quality_cfg=None,
        zero_signal_alert=0,
        state_enabled=False,
        rest_candidates=None,
        monitoring=None,
        monitoring_agg=None,
        worker_id="worker-test",
        status_callback=None,
        execution_mode="bar",
        portfolio_equity=1_000.0,
        max_total_weight=max_total_weight,
    )
    worker._weights = dict(existing_weights or {})
    return worker


def test_worker_normalizes_weights_above_cap():
    worker = _make_worker(1.0, existing_weights={"ETHUSDT": 0.4})
    order1_turnover = 700.0
    order1 = Order(
        ts=1,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {
                "target_weight": 0.7,
                "edge_bps": 10.0,
                "economics": {"turnover_usd": order1_turnover},
            },
            "economics": {"turnover_usd": order1_turnover},
            "decision": {
                "turnover_usd": order1_turnover,
                "economics": {"turnover_usd": order1_turnover},
            },
        },
    )
    order2_turnover = 500.0
    order2 = Order(
        ts=1,
        symbol="LTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {
                "target_weight": 0.7,
                "edge_bps": 12.0,
                "economics": {"turnover_usd": order2_turnover},
            },
            "economics": {"turnover_usd": order2_turnover},
            "decision": {
                "turnover_usd": order2_turnover,
                "economics": {"turnover_usd": order2_turnover},
            },
        },
    )
    normalized_orders, applied = worker._normalize_weight_targets([order1, order2])
    assert applied is True
    payloads = [o.meta["payload"] for o in normalized_orders]
    assert sum(p["target_weight"] for p in payloads) == pytest.approx(0.6)
    for payload, order in zip(payloads, normalized_orders):
        assert payload["normalized"] is True
        assert order.meta["normalized"] is True
        assert payload["delta_weight"] == pytest.approx(payload["target_weight"])
        assert payload["target_weight"] == pytest.approx(0.3)
        assert payload.get("normalization")
        assert payload["normalization"]["factor"] == pytest.approx(0.6 / 1.4)
    pending = worker._pending_weight
    assert len(pending) == 2
    expected_factor = payloads[0]["normalization"]["factor"]
    for order_obj, payload, original_turnover in (
        (normalized_orders[0], payloads[0], order1_turnover),
        (normalized_orders[1], payloads[1], order2_turnover),
    ):
        assert pending[id(order_obj)]["target_weight"] == pytest.approx(
            payload["target_weight"]
        )
        assert pending[id(order_obj)]["delta_weight"] == pytest.approx(
            payload["delta_weight"]
        )
        economics = payload["economics"]
        assert economics["turnover_usd"] == pytest.approx(original_turnover * expected_factor)
        decision_meta = order_obj.meta["decision"]
        assert decision_meta["turnover_usd"] == pytest.approx(
            original_turnover * expected_factor
        )
        assert decision_meta["economics"]["turnover_usd"] == pytest.approx(
            original_turnover * expected_factor
        )
        assert order_obj.meta["economics"]["turnover_usd"] == pytest.approx(
            original_turnover * expected_factor
        )


def test_worker_normalizes_weights_accumulates_same_symbol_orders():
    worker = _make_worker(
        0.5,
        existing_weights={"BTCUSDT": 0.1, "ETHUSDT": 0.1},
    )
    order1_turnover = 200.0
    order1 = Order(
        ts=1,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {
                "target_weight": 0.6,
                "edge_bps": 10.0,
                "economics": {"turnover_usd": order1_turnover},
            },
            "economics": {"turnover_usd": order1_turnover},
            "decision": {
                "turnover_usd": order1_turnover,
                "economics": {"turnover_usd": order1_turnover},
            },
        },
    )
    order2_turnover = 300.0
    order2 = Order(
        ts=2,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {
                "target_weight": 0.8,
                "edge_bps": 12.0,
                "economics": {"turnover_usd": order2_turnover},
            },
            "economics": {"turnover_usd": order2_turnover},
            "decision": {
                "turnover_usd": order2_turnover,
                "economics": {"turnover_usd": order2_turnover},
            },
        },
    )

    normalized_orders, applied = worker._normalize_weight_targets([order1, order2])

    assert applied is True
    payloads = [order.meta["payload"] for order in normalized_orders]
    factor = payloads[0]["normalization"]["factor"]
    expected_factor = pytest.approx(0.3 / 0.7)
    assert factor == expected_factor
    assert payloads[1]["normalization"]["factor"] == pytest.approx(factor)
    assert payloads[0]["normalization"]["delta_positive_total"] == pytest.approx(0.7)
    assert payloads[0]["normalization"]["delta_negative_total"] == pytest.approx(0.0)
    assert payloads[0]["normalization"]["requested_total"] == pytest.approx(0.8)
    assert payloads[0]["normalization"]["delta_total"] == pytest.approx(0.7)
    assert payloads[0]["normalization"]["available_delta"] == pytest.approx(0.3)

    expected_target1 = 0.1 + (0.6 - 0.1) * factor
    available_total = payloads[0]["normalization"]["available_total"]
    expected_target2 = min(available_total, 0.8)
    assert payloads[0]["target_weight"] == pytest.approx(expected_target1)
    assert payloads[1]["target_weight"] == pytest.approx(expected_target2)
    assert payloads[0]["delta_weight"] == pytest.approx(expected_target1 - 0.1)
    assert payloads[1]["delta_weight"] == pytest.approx(
        expected_target2 - expected_target1
    )

    for order_obj, payload, original_turnover in (
        (normalized_orders[0], payloads[0], order1_turnover),
        (normalized_orders[1], payloads[1], order2_turnover),
    ):
        assert payload["normalized"] is True
        assert order_obj.meta["normalized"] is True
        assert order_obj.meta["normalization"]["factor"] == pytest.approx(factor)
        economics = payload["economics"]
        assert economics["turnover_usd"] == pytest.approx(original_turnover * factor)
        decision_meta = order_obj.meta["decision"]
        assert decision_meta["turnover_usd"] == pytest.approx(original_turnover * factor)
        assert decision_meta["economics"]["turnover_usd"] == pytest.approx(
            original_turnover * factor
        )
        assert order_obj.meta["economics"]["turnover_usd"] == pytest.approx(
            original_turnover * factor
        )


def test_worker_normalizes_weights_stacks_symbol_orders_to_cap():
    worker = _make_worker(0.3)

    order1 = Order(
        ts=1,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {"target_weight": 0.4, "edge_bps": 15.0},
        },
    )
    order2 = Order(
        ts=2,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {"target_weight": 0.5, "edge_bps": 20.0},
        },
    )

    requested1 = order1.meta["payload"]["target_weight"]
    requested2 = order2.meta["payload"]["target_weight"]

    normalized_orders, applied = worker._normalize_weight_targets([order1, order2])

    assert applied is True
    payloads = [order.meta["payload"] for order in normalized_orders]
    factor = payloads[0]["normalization"]["factor"]
    assert factor == pytest.approx(0.6)

    expected_first = requested1 * factor
    available_total = payloads[0]["normalization"]["available_total"]
    expected_second = min(available_total, requested2)

    assert payloads[0]["target_weight"] == pytest.approx(expected_first)
    assert payloads[1]["target_weight"] == pytest.approx(expected_second)

    initial_weight = worker._weights.get("BTCUSDT", 0.0)
    total_delta = sum(payload["delta_weight"] for payload in payloads)
    final_target = payloads[1]["target_weight"]

    assert final_target <= payloads[0]["normalization"]["available_total"]
    assert final_target - initial_weight == pytest.approx(total_delta)


def test_worker_normalizes_weights_handles_false_string_flag():
    worker = _make_worker(1.0, existing_weights={"ETHUSDT": 0.4})
    order1_turnover = 700.0
    order1 = Order(
        ts=1,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {
                "target_weight": 0.7,
                "edge_bps": 10.0,
                "economics": {"turnover_usd": order1_turnover},
                "normalized": "false",
            },
            "normalized": "false",
        },
    )
    order2_turnover = 500.0
    order2 = Order(
        ts=1,
        symbol="LTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {
                "target_weight": 0.7,
                "edge_bps": 12.0,
                "economics": {"turnover_usd": order2_turnover},
            }
        },
    )

    normalized_orders, applied = worker._normalize_weight_targets([order1, order2])

    assert applied is True
    payloads = [o.meta["payload"] for o in normalized_orders]
    assert sum(p["target_weight"] for p in payloads) == pytest.approx(0.6)
    for payload, order in zip(payloads, normalized_orders):
        assert payload["normalized"] is True
        assert order.meta["normalized"] is True
        assert payload["target_weight"] == pytest.approx(0.3)

    pending = worker._pending_weight
    assert len(pending) == 2
    for order_obj in normalized_orders:
        assert pending[id(order_obj)]["normalized"] is True


def test_worker_normalizes_weights_with_existing_weight_keeps_direction():
    worker = _make_worker(
        1.0, existing_weights={"BTCUSDT": 0.5, "ETHUSDT": 0.3}
    )
    order_up = Order(
        ts=1,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {
                "target_weight": 0.7,
                "edge_bps": 10.0,
                "economics": {"turnover_usd": 700.0},
            }
        },
    )
    order_new = Order(
        ts=1,
        symbol="LTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {
                "target_weight": 0.7,
                "edge_bps": 12.0,
                "economics": {"turnover_usd": 500.0},
            }
        },
    )

    normalized_orders, applied = worker._normalize_weight_targets([order_up, order_new])
    assert applied is True

    payloads: Dict[str, Dict[str, Any]] = {
        o.symbol: o.meta["payload"] for o in normalized_orders
    }
    btc_payload = payloads["BTCUSDT"]
    btc_current = worker._weights["BTCUSDT"]
    factor = btc_payload["normalization"]["factor"]
    assert factor == pytest.approx(
        (btc_payload["normalization"]["available_delta"]) / (
            btc_payload["normalization"]["delta_total"]
        )
    )
    assert btc_payload["target_weight"] == pytest.approx(
        btc_current + (0.7 - btc_current) * factor
    )
    assert btc_payload["target_weight"] >= btc_current
    assert btc_payload["target_weight"] <= 0.7
    assert btc_payload["delta_weight"] == pytest.approx(
        btc_payload["target_weight"] - btc_current
    )
    assert btc_payload["delta_weight"] >= 0.0
    ltc_payload = payloads["LTCUSDT"]
    assert ltc_payload["target_weight"] == pytest.approx(0.7 * factor)
    assert ltc_payload["delta_weight"] == pytest.approx(ltc_payload["target_weight"])
    total_weight = sum(p["target_weight"] for p in payloads.values())
    assert total_weight == pytest.approx(
        btc_payload["normalization"]["available_total"]
    )


def test_worker_normalizes_weights_with_current_exposure_respects_cap():
    worker = _make_worker(
        0.8,
        existing_weights={"BTCUSDT": 0.2, "ETHUSDT": 0.25, "SOLUSDT": 0.1},
    )
    current_weights = {sym: worker._weights[sym] for sym in ("BTCUSDT", "ETHUSDT")}
    btc_target = 0.5
    order_btc = Order(
        ts=1,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={"payload": {"target_weight": btc_target, "edge_bps": 20.0}},
    )
    eth_target = 0.45
    order_eth = Order(
        ts=1,
        symbol="ETHUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={"payload": {"target_weight": eth_target, "edge_bps": 18.0}},
    )

    normalized_orders, applied = worker._normalize_weight_targets([order_btc, order_eth])
    assert applied is True
    assert len(normalized_orders) == 2

    payloads = {order.symbol: order.meta["payload"] for order in normalized_orders}
    normalization = payloads["BTCUSDT"]["normalization"]

    expected_current_total = sum(current_weights.values())
    assert normalization["current_total"] == pytest.approx(expected_current_total)
    expected_delta_total = sum(
        max(0.0, target - current_weights[symbol])
        for symbol, target in (("BTCUSDT", btc_target), ("ETHUSDT", eth_target))
    )
    assert normalization["delta_total"] == pytest.approx(expected_delta_total)
    assert normalization["available_total"] == pytest.approx(
        worker._max_total_weight - worker._weights["SOLUSDT"]
    )
    assert normalization["available_delta"] == pytest.approx(
        normalization["available_total"] - normalization["current_total"]
    )
    expected_factor = normalization["factor"]
    assert expected_factor == pytest.approx(
        normalization["available_delta"] / normalization["delta_total"]
    )

    total_weight = sum(payload["target_weight"] for payload in payloads.values())
    assert total_weight == pytest.approx(normalization["available_total"])
    assert total_weight + worker._weights["SOLUSDT"] == pytest.approx(
        worker._max_total_weight
    )

    for order in (order_btc, order_eth):
        payload = payloads[order.symbol]
        current = current_weights[order.symbol]
        assert payload["target_weight"] >= current
        original_target = btc_target if order.symbol == "BTCUSDT" else eth_target
        expected_target = current + (original_target - current) * expected_factor
        assert payload["target_weight"] == pytest.approx(expected_target)
        assert payload["delta_weight"] == pytest.approx(
            payload["target_weight"] - current
        )
        assert payload["delta_weight"] >= 0.0


def test_worker_normalizes_weights_mixed_directions_under_cap():
    worker = _make_worker(
        0.7,
        existing_weights={"BTCUSDT": 0.4, "ETHUSDT": 0.2, "SOLUSDT": 0.05},
    )

    buy_turnover = 800.0
    btc_order = Order(
        ts=1,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {
                "target_weight": 0.6,
                "edge_bps": 15.0,
                "economics": {"turnover_usd": buy_turnover},
            },
            "economics": {"turnover_usd": buy_turnover},
            "decision": {
                "turnover_usd": buy_turnover,
                "economics": {"turnover_usd": buy_turnover},
            },
        },
    )

    sell_turnover = 500.0
    eth_order = Order(
        ts=1,
        symbol="ETHUSDT",
        side=Side.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {
                "target_weight": 0.05,
                "edge_bps": 12.0,
                "economics": {"turnover_usd": sell_turnover},
            },
            "economics": {"turnover_usd": sell_turnover},
            "decision": {
                "turnover_usd": sell_turnover,
                "economics": {"turnover_usd": sell_turnover},
            },
        },
    )

    ada_turnover = 400.0
    ada_order = Order(
        ts=1,
        symbol="ADAUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {
                "target_weight": 0.3,
                "edge_bps": 18.0,
                "economics": {"turnover_usd": ada_turnover},
            },
            "economics": {"turnover_usd": ada_turnover},
            "decision": {
                "turnover_usd": ada_turnover,
                "economics": {"turnover_usd": ada_turnover},
            },
        },
    )

    orders = [btc_order, eth_order, ada_order]
    normalized_orders, applied = worker._normalize_weight_targets(orders)

    assert applied is True

    payloads = {order.symbol: order.meta["payload"] for order in normalized_orders}
    normalization = payloads["BTCUSDT"]["normalization"]

    assert normalization["factor"] == pytest.approx(0.4)
    assert normalization["available_total"] == pytest.approx(0.65)
    assert normalization["available_delta"] == pytest.approx(0.2)
    assert normalization["delta_total"] == pytest.approx(0.35)
    assert normalization["delta_positive_total"] == pytest.approx(0.5)
    assert normalization["delta_negative_total"] == pytest.approx(-0.15)
    assert normalization["desired_total"] == pytest.approx(0.95)

    btc_payload = payloads["BTCUSDT"]
    assert btc_payload["target_weight"] == pytest.approx(0.48)
    assert btc_payload["delta_weight"] == pytest.approx(0.08)
    assert btc_payload["economics"]["turnover_usd"] == pytest.approx(
        buy_turnover * normalization["factor"]
    )

    ada_payload = payloads["ADAUSDT"]
    assert ada_payload["target_weight"] == pytest.approx(0.12)
    assert ada_payload["delta_weight"] == pytest.approx(0.12)
    assert ada_payload["economics"]["turnover_usd"] == pytest.approx(
        ada_turnover * normalization["factor"]
    )

    eth_payload = payloads["ETHUSDT"]
    assert eth_payload["target_weight"] == pytest.approx(0.05)
    assert eth_payload["delta_weight"] == pytest.approx(-0.15)
    assert eth_payload["economics"]["turnover_usd"] == pytest.approx(sell_turnover)

    total_target = sum(p["target_weight"] for p in payloads.values())
    assert total_target == pytest.approx(normalization["available_total"])
    pending = worker._pending_weight
    assert pending[id(eth_order)]["delta_weight"] == pytest.approx(-0.15)

def test_bar_executor_propagates_normalized_flag():
    executor = BarExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(),
        default_equity_usd=1000.0,
    )
    order = Order(
        ts=5,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "bar": make_bar(5, 10000.0),
            "normalized": True,
            "payload": {
                "target_weight": 0.8,
                "edge_bps": 25.0,
                "normalized": True,
                "normalization": {"factor": 0.75},
            },
        },
    )
    report = executor.execute(order)
    assert report.meta["normalized"] is True
    assert report.meta["decision"]["normalized"] is True
    assert report.meta["decision"]["normalization"] == {"factor": 0.75}
    instructions = report.meta["instructions"]
    assert len(instructions) == 1
    assert instructions[0]["target_weight"] == pytest.approx(0.8)
    snapshot = executor.monitoring_snapshot()
    assert snapshot["normalized"] is True
    assert snapshot["decision"]["normalized"] is True


def test_bar_executor_aborts_when_weight_would_exceed_one():
    class OvershootExecutor(BarExecutor):
        def _resolve_target_weight(self, state: PortfolioState, payload: Any):  # type: ignore[override]
            return 1.2, "target", 0.4

    executor = OvershootExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(),
        default_equity_usd=1_000.0,
        initial_weights={"BTCUSDT": 0.8},
    )
    bar = make_bar(200, 20_000.0)
    order = Order(
        ts=200,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={"bar": bar, "payload": {"edge_bps": 50.0}},
    )

    report = executor.execute(order)

    instructions = report.meta["instructions"]
    assert instructions, "expected rebalance instructions when clamping above 100%"
    instr = instructions[0]
    assert instr["target_weight"] == pytest.approx(1.0)
    assert instr["delta_weight"] == pytest.approx(0.2)
    assert report.meta.get("reason") is None
    assert report.meta["target_weight"] == pytest.approx(1.0)
    decision = report.meta["decision"]
    assert decision["act_now"] is True
    assert decision["delta_weight"] == pytest.approx(0.2)


def test_bar_executor_aborts_when_weight_would_drop_below_zero():
    class UndershootExecutor(BarExecutor):
        def _resolve_target_weight(self, state: PortfolioState, payload: Any):  # type: ignore[override]
            return -0.2, "target", -0.4

    executor = UndershootExecutor(
        run_id="test",
        bar_price="close",
        cost_config=SpotCostConfig(),
        default_equity_usd=1_000.0,
        initial_weights={"BTCUSDT": 0.2},
    )
    bar = make_bar(210, 19_000.0)
    order = Order(
        ts=210,
        symbol="BTCUSDT",
        side=Side.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={"bar": bar, "payload": {"edge_bps": 50.0}},
    )

    report = executor.execute(order)

    instructions = report.meta["instructions"]
    assert instructions, "expected rebalance instructions when clamping below 0%"
    instr = instructions[0]
    assert instr["target_weight"] == pytest.approx(0.0)
    assert instr["delta_weight"] == pytest.approx(-0.2)
    assert report.meta.get("reason") is None
    assert report.meta["target_weight"] == pytest.approx(0.0)
    decision = report.meta["decision"]
    assert decision["act_now"] is True
    assert decision["delta_weight"] == pytest.approx(-0.2)

