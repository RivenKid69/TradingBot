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


def test_vwap_execution_price():
    sim = ExecutionSimulator(
        execution_config={"algo": "VWAP"},
        slippage_config={"default_spread_bps": 0.0, "k": 0.0, "min_half_spread_bps": 0.0},
    )
    ticks = [
        (0, 100.0, 1.0),
        (1_000, 101.0, 2.0),
        (2_000, 102.0, 3.0),
    ]
    for ts, price, vol in ticks:
        sim.run_step(
            ts=ts,
            ref_price=price,
            bid=None,
            ask=None,
            vol_factor=1.0,
            liquidity=vol,
            trade_price=price,
            trade_qty=vol,
            actions=None,
        )
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)
    rep = sim.run_step(
        ts=3_599_000,
        ref_price=102.0,
        bid=None,
        ask=None,
        vol_factor=1.0,
        liquidity=1.0,
        trade_price=102.0,
        trade_qty=0.0,
        actions=[(ActionType.MARKET, proto)],
    )
    assert len(rep.trades) == 1
    trade = rep.trades[0]
    expected_vwap = (100 * 1 + 101 * 2 + 102 * 3) / (1 + 2 + 3)
    assert trade.price == pytest.approx(expected_vwap)
