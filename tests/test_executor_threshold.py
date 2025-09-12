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


def _make_sim(algo: str):
    cfg = {
        "notional_threshold": 100.0,
        "large_order_algo": algo,
        "twap": {"parts": 3, "child_interval_s": 1},
        "pov": {
            "participation": 0.5,
            "child_interval_s": 1,
            "min_child_notional": 1.0,
        },
    }
    lat = {"base_ms": 0, "jitter_ms": 0, "spike_p": 0.0, "timeout_ms": 1000, "retries": 0}
    slip = {"default_spread_bps": 0.0, "k": 0.0, "min_half_spread_bps": 0.0}
    return ExecutionSimulator(execution_config=cfg, slippage_config=slip, latency_config=lat)


@pytest.mark.parametrize("algo,expected", [("TWAP", 3), ("POV", 3)])
def test_threshold_executor_selection(algo: str, expected: int):
    # below threshold -> taker
    sim = _make_sim(algo)
    sim._executor = None
    sim.set_market_snapshot(bid=None, ask=None, liquidity=10.0, ts_ms=0)
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=9.0)
    sim.submit(proto, now_ts=0)
    rep = sim.pop_ready(now_ts=0, ref_price=10.0)
    assert len(rep.trades) == 1

    # above threshold -> large order algo
    sim = _make_sim(algo)
    sim._executor = None
    sim.set_market_snapshot(bid=None, ask=None, liquidity=10.0, ts_ms=0)
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=11.0)
    sim.submit(proto, now_ts=0)
    rep = sim.pop_ready(now_ts=0, ref_price=10.0)
    assert len(rep.trades) == expected
