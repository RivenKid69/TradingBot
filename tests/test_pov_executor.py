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

POVExecutor = exec_mod.POVExecutor
ExecutionSimulator = exec_mod.ExecutionSimulator


def test_pov_participation_propagation():
    sim = ExecutionSimulator(
        execution_config={
            "algo": "POV",
            "pov": {
                "participation": 0.2,
                "child_interval_s": 1,
                "min_child_notional": 0.0,
            },
        },
        slippage_config={
            "default_spread_bps": 0.0,
            "k": 0.0,
            "min_half_spread_bps": 0.0,
        },
    )
    executor = sim._executor
    assert isinstance(executor, POVExecutor)
    assert executor.participation == pytest.approx(0.2)


def test_pov_plan_uses_participation():
    execu = POVExecutor(participation=0.2, child_interval_s=1, min_child_notional=0.0)
    plan = execu.plan_market(
        now_ts_ms=0,
        side="BUY",
        target_qty=300,
        snapshot={"liquidity": 1000.0, "ref_price": 100.0},
    )
    assert len(plan) == 2
    assert plan[0].qty == pytest.approx(200.0)
    assert plan[1].qty == pytest.approx(100.0)
