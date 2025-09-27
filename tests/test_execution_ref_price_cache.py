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


def test_market_order_uses_cached_ref_price_when_missing_snapshot():
    sim = ExecutionSimulator(filters_path=None, latency_steps=0)
    sim.set_symbol("TESTUSDT")

    sim.run_step(ts=1_000, ref_price=100.0, actions=[])
    assert sim._last_ref_price == pytest.approx(100.0)

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)
    report = sim.run_step(
        ts=2_000,
        ref_price=None,
        actions=[(ActionType.MARKET, proto)],
    )

    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.price == pytest.approx(100.0)
    assert sim._last_ref_price == pytest.approx(100.0)
