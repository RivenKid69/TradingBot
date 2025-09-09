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
    sim.set_next_open_price(100.0)
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)
    mid_ts = 3_600_000 + 1_800_000  # 1h30m
    rep = sim.run_step(
        ts=mid_ts,
        ref_price=100.0,
        bid=None,
        ask=None,
        vol_factor=1.0,
        liquidity=1.0,
        actions=[(ActionType.MARKET, proto)],
    )
    assert len(rep.trades) == 1
    trade = rep.trades[0]
    assert trade.ts == 7_200_000  # next hour open
    expected_bps = estimate_slippage_bps(
        spread_bps=2.0,
        size=1.0,
        liquidity=1.0,
        vol_factor=1.0,
        cfg=sim.slippage_cfg,
    )
    expected_price = apply_slippage_price(
        side="BUY", quote_price=100.0, slippage_bps=expected_bps
    )
    assert trade.price == pytest.approx(expected_price)
