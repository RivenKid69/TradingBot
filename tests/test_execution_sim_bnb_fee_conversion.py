import importlib.util
import math
import pathlib
import sys

import pytest


base = pathlib.Path(__file__).resolve().parents[1]

spec_exec = importlib.util.spec_from_file_location("execution_sim", base / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec_exec)
sys.modules["execution_sim"] = exec_mod
spec_exec.loader.exec_module(exec_mod)

ActionProto = exec_mod.ActionProto
ActionType = exec_mod.ActionType
ExecutionSimulator = exec_mod.ExecutionSimulator


class DummyQuantizer:
    def quantize_qty(self, symbol, qty):
        return float(qty)

    def quantize_price(self, symbol, price):
        return float(price)

    def clamp_notional(self, symbol, ref_price, qty):
        return float(qty)

    def check_percent_price_by_side(self, symbol, side, price, ref_price):
        return True


@pytest.mark.parametrize("volume_frac", [1.0, -1.0])
def test_bnb_fee_conversion_updates_equity(volume_frac):
    conversion_rate = 200.0
    taker_bps = 12.0
    fees_config = {
        "maker_bps": taker_bps,
        "taker_bps": taker_bps,
        "use_bnb_discount": True,
        "maker_discount_mult": 0.75,
        "taker_discount_mult": 0.75,
        "settlement": {"mode": "bnb", "currency": "BNB"},
        "bnb_conversion_rate": conversion_rate,
    }

    sim = ExecutionSimulator(filters_path=None, fees_config=fees_config)
    sim.set_quantizer(DummyQuantizer())

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=volume_frac)

    baseline = sim.run_step(
        ts=500_000,
        ref_price=100.0,
        bid=99.5,
        ask=100.5,
        liquidity=1.0,
        actions=[],
    )
    fees_before = sim.fees_cum
    funding_before = sim.funding_cum

    report = sim.run_step(
        ts=1_000_000,
        ref_price=100.0,
        bid=99.5,
        ask=100.5,
        liquidity=1.0,
        actions=[(ActionType.MARKET, proto)],
    )

    assert report.trades, "Expected at least one trade to be executed"
    trade = report.trades[0]

    discount_mult = 0.75
    expected_fee_quote = trade.price * trade.qty * (taker_bps * discount_mult) / 1e4
    expected_fee_bnb = expected_fee_quote / conversion_rate

    fees_delta = sim.fees_cum - fees_before
    assert math.isclose(fees_delta, expected_fee_quote, rel_tol=1e-9)
    assert math.isclose(report.fee_total, expected_fee_bnb, rel_tol=1e-9)
    assert math.isclose(trade.fee, expected_fee_bnb, rel_tol=1e-9)
    assert math.isclose(report.fee_total * conversion_rate, expected_fee_quote, rel_tol=1e-9)

    eq_delta = report.equity - baseline.equity
    realized_delta = report.realized_pnl - baseline.realized_pnl
    unrealized_delta = report.unrealized_pnl - baseline.unrealized_pnl
    funding_delta = sim.funding_cum - funding_before

    assert math.isclose(
        eq_delta - realized_delta - unrealized_delta - funding_delta,
        -expected_fee_quote,
        rel_tol=1e-9,
    )


def test_bnb_fee_conversion_extreme_conversion_precision():
    conversion_rate = 1e-8
    taker_bps = 12.0
    fees_config = {
        "maker_bps": taker_bps,
        "taker_bps": taker_bps,
        "use_bnb_discount": True,
        "maker_discount_mult": 0.75,
        "taker_discount_mult": 0.75,
        "settlement": {"mode": "bnb", "currency": "BNB"},
        "bnb_conversion_rate": conversion_rate,
    }

    sim = ExecutionSimulator(filters_path=None, fees_config=fees_config)
    sim.set_quantizer(DummyQuantizer())

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)

    sim.run_step(
        ts=1,
        ref_price=150.0,
        bid=149.5,
        ask=150.5,
        liquidity=1.0,
        actions=[],
    )
    fees_before = sim.fees_cum

    report = sim.run_step(
        ts=2,
        ref_price=150.0,
        bid=149.5,
        ask=150.5,
        liquidity=1.0,
        actions=[(ActionType.MARKET, proto)],
    )

    assert report.trades, "Expected trade execution"
    trade = report.trades[0]
    discount_mult = 0.75
    expected_fee_quote = trade.price * trade.qty * (taker_bps * discount_mult) / 1e4
    fees_delta = sim.fees_cum - fees_before
    assert math.isclose(fees_delta, expected_fee_quote, rel_tol=1e-12)

