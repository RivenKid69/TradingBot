import importlib.util
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


def test_zero_fee_discount_multiplier_maintains_zero_fees():
    fees_config = {
        "maker_bps": 12.0,
        "taker_bps": 12.0,
        "maker_discount_mult": 0.0,
        "taker_discount_mult": 0.0,
    }

    sim = ExecutionSimulator(filters_path=None, fees_config=fees_config)
    sim.set_quantizer(DummyQuantizer())

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)

    sim.run_step(
        ts=500_000,
        ref_price=100.0,
        bid=99.5,
        ask=100.5,
        liquidity=1.0,
        actions=[],
    )
    fees_before = sim.fees_cum

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

    assert report.fee_total == pytest.approx(0.0)
    assert trade.fee == pytest.approx(0.0)
    assert sim.fees_cum == pytest.approx(fees_before)


def test_negative_fee_reduces_cumulative_fee(monkeypatch):
    sim = ExecutionSimulator(filters_path=None)
    sim.set_quantizer(DummyQuantizer())

    def fake_compute_trade_fee(self, *, side, price, qty, liquidity):
        del side, price, qty, liquidity
        self._fees_last_quote_equivalent = -2.5
        return -1.0

    monkeypatch.setattr(
        ExecutionSimulator, "_compute_trade_fee", fake_compute_trade_fee
    )

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)

    sim.run_step(
        ts=1,
        ref_price=100.0,
        bid=99.5,
        ask=100.5,
        liquidity=1.0,
        actions=[],
    )
    fees_before = sim.fees_cum

    report = sim.run_step(
        ts=2,
        ref_price=100.0,
        bid=99.5,
        ask=100.5,
        liquidity=1.0,
        actions=[(ActionType.MARKET, proto)],
    )

    assert report.trades, "Expected at least one trade to be executed"
    trade = report.trades[0]

    assert trade.fee == pytest.approx(-1.0)
    assert report.fee_total == pytest.approx(-1.0)
    assert sim.fees_cum == pytest.approx(fees_before - 2.5)


def test_negative_fee_alt_currency_reduces_fees_and_updates_equity(monkeypatch):
    conversion_rate = 200.0
    pending_rebate = -0.02
    settlement_rebate = -0.01
    expected_total_quote = (pending_rebate + settlement_rebate) * conversion_rate

    sim = ExecutionSimulator(filters_path=None)
    sim.set_quantizer(DummyQuantizer())
    sim.fees_conversion_rates["BNB"] = conversion_rate

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)

    baseline = sim.run_step(
        ts=1,
        ref_price=100.0,
        bid=99.5,
        ask=100.5,
        liquidity=1.0,
        actions=[],
    )

    sim._fees_pending_settlements[("BNB", "USDT")] = pending_rebate

    fees_before = sim.fees_cum
    funding_before = sim.funding_cum
    equity_before = baseline.equity

    def fake_compute_trade_fee(self, *, side, price, qty, liquidity):
        del side, price, qty, liquidity
        self._fees_store_last_info(
            symbol="BTCUSDT",
            quote_currency="USDT",
            fee_currency="BNB",
            settlement_amount=settlement_rebate,
            conversion_rate=conversion_rate,
            requires_conversion=True,
        )
        return settlement_rebate

    monkeypatch.setattr(
        ExecutionSimulator,
        "_compute_trade_fee",
        fake_compute_trade_fee,
    )

    report = sim.run_step(
        ts=2,
        ref_price=100.0,
        bid=99.5,
        ask=100.5,
        liquidity=1.0,
        actions=[(ActionType.MARKET, proto)],
    )

    assert report.trades, "Expected at least one trade to be executed"
    trade = report.trades[0]

    assert trade.fee == pytest.approx(settlement_rebate)
    assert report.fee_total == pytest.approx(settlement_rebate)

    fees_delta = sim.fees_cum - fees_before
    assert fees_delta == pytest.approx(expected_total_quote)

    equity_delta = report.equity - equity_before
    realized_delta = report.realized_pnl - baseline.realized_pnl
    unrealized_delta = report.unrealized_pnl - baseline.unrealized_pnl
    funding_delta = sim.funding_cum - funding_before

    assert equity_delta - realized_delta - unrealized_delta - funding_delta == pytest.approx(
        -expected_total_quote
    )
