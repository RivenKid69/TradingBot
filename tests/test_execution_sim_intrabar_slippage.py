import importlib.util
import pathlib
import sys

import pytest


base = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(base))
spec = importlib.util.spec_from_file_location("execution_sim", base / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec)
sys.modules["execution_sim"] = exec_mod
spec.loader.exec_module(exec_mod)

ActionProto = exec_mod.ActionProto
ActionType = exec_mod.ActionType
ExecutionSimulator = exec_mod.ExecutionSimulator


def test_intrabar_fallback_slip_uses_estimate(monkeypatch):
    fallback_bps = 12.5

    def fake_apply_slippage_price(*, side, quote_price, slippage_bps):
        base = float(quote_price)
        slip = float(slippage_bps) / 1e4
        if str(side).upper() == "BUY":
            return base * (1.0 + slip)
        return base * (1.0 - slip)

    def fake_estimate_slippage_bps(**_kwargs):
        return fallback_bps

    blend_calls: list[float] = []

    def no_blend(self, *, taker_bps, maker_bps=None, maker_share=None):
        blend_calls.append(float(taker_bps))
        return None

    monkeypatch.setattr(exec_mod, "apply_slippage_price", fake_apply_slippage_price)
    monkeypatch.setattr(exec_mod, "estimate_slippage_bps", fake_estimate_slippage_bps)
    monkeypatch.setattr(ExecutionSimulator, "_blend_expected_spread", no_blend)

    latency_cfg = {"base_ms": 0, "jitter_ms": 0, "spike_p": 0.0, "timeout_ms": 1000, "retries": 0}
    slippage_cfg = {"default_spread_bps": 0.0, "k": 0.0, "min_half_spread_bps": 0.0}

    sim = ExecutionSimulator(
        latency_config=latency_cfg,
        slippage_config=slippage_cfg,
    )

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)

    report = sim.run_step(
        ts=0,
        ref_price=100.0,
        bid=99.5,
        ask=100.0,
        vol_factor=1.0,
        liquidity=10.0,
        trade_price=100.0,
        trade_qty=0.0,
        actions=[(ActionType.MARKET, proto)],
    )

    assert report.trades, "Expected at least one trade to be generated"
    trade = report.trades[0]
    expected_price = fake_apply_slippage_price(
        side="BUY", quote_price=100.0, slippage_bps=fallback_bps
    )

    assert trade.slippage_bps == pytest.approx(fallback_bps)
    assert trade.price == pytest.approx(expected_price)
    assert any(pytest.approx(fallback_bps) == call for call in blend_calls)


def test_maker_limit_intrabar_order_executes_immediately():
    latency_cfg = {
        "base_ms": 0,
        "jitter_ms": 0,
        "spike_p": 0.0,
        "timeout_ms": 1000,
        "retries": 0,
    }
    execution_cfg = {"intrabar_price_model": "low"}

    sim = ExecutionSimulator(
        filters_path=None,
        latency_config=latency_cfg,
        execution_config=execution_cfg,
    )

    sim.run_step(
        ts=0,
        ref_price=101.0,
        bid=100.0,
        ask=101.0,
        bar_open=101.0,
        bar_high=105.0,
        bar_low=99.0,
        bar_close=102.0,
        liquidity=10.0,
        trade_price=101.0,
        trade_qty=1.0,
        actions=[],
    )

    limit_proto = ActionProto(
        action_type=ActionType.LIMIT,
        volume_frac=1.0,
        abs_price=99.5,
    )

    sim.submit(limit_proto, now_ts=0)
    report = sim.pop_ready(now_ts=0, ref_price=101.0)

    assert report.trades, "Maker limit order should fill intrabar"
    trade = report.trades[0]
    assert trade.liquidity == "maker"
    assert trade.price == pytest.approx(99.5)
