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
TWAPExecutor = exec_mod.TWAPExecutor
POVExecutor = exec_mod.POVExecutor


latency_cfg = {"base_ms": 0, "jitter_ms": 0, "spike_p": 0.0, "timeout_ms": 1000, "retries": 0}
slippage_cfg = {"default_spread_bps": 0.0, "k": 0.0, "min_half_spread_bps": 0.0}


def test_twap_determinism():
    execu = TWAPExecutor(parts=3, child_interval_s=1)
    snap = {"liquidity": 1.0, "ref_price": 100.0}
    plan1 = execu.plan_market(now_ts_ms=0, side="BUY", target_qty=9, snapshot=snap)
    plan2 = execu.plan_market(now_ts_ms=0, side="BUY", target_qty=9, snapshot=snap)
    assert plan1 == plan2

    sim1 = ExecutionSimulator(
        execution_config={"algo": "TWAP", "twap": {"parts": 3, "child_interval_s": 1}},
        slippage_config=slippage_cfg,
        latency_config=latency_cfg,
    )
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=9.0)
    rep1 = sim1.run_step(
        ts=0,
        ref_price=10.0,
        bid=None,
        ask=None,
        vol_factor=1.0,
        liquidity=1.0,
        trade_price=10.0,
        trade_qty=0.0,
        actions=[(ActionType.MARKET, proto)],
    )

    sim2 = ExecutionSimulator(
        execution_config={"algo": "TWAP", "twap": {"parts": 3, "child_interval_s": 1}},
        slippage_config=slippage_cfg,
        latency_config=latency_cfg,
    )
    rep2 = sim2.run_step(
        ts=0,
        ref_price=10.0,
        bid=None,
        ask=None,
        vol_factor=1.0,
        liquidity=1.0,
        trade_price=10.0,
        trade_qty=0.0,
        actions=[(ActionType.MARKET, proto)],
    )

    trades1 = [(t.ts, t.qty, t.price) for t in rep1.trades]
    trades2 = [(t.ts, t.qty, t.price) for t in rep2.trades]
    assert trades1 == trades2


def test_pov_determinism():
    execu = POVExecutor(participation=0.5, child_interval_s=1, min_child_notional=1.0)
    snap = {"liquidity": 10.0, "ref_price": 100.0}
    plan1 = execu.plan_market(now_ts_ms=0, side="BUY", target_qty=11, snapshot=snap)
    plan2 = execu.plan_market(now_ts_ms=0, side="BUY", target_qty=11, snapshot=snap)
    assert plan1 == plan2

    sim1 = ExecutionSimulator(
        execution_config={"algo": "POV", "pov": {"participation": 0.5, "child_interval_s": 1, "min_child_notional": 1.0}},
        slippage_config=slippage_cfg,
        latency_config=latency_cfg,
    )
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=11.0)
    rep1 = sim1.run_step(
        ts=0,
        ref_price=10.0,
        bid=None,
        ask=None,
        vol_factor=1.0,
        liquidity=10.0,
        trade_price=10.0,
        trade_qty=0.0,
        actions=[(ActionType.MARKET, proto)],
    )

    sim2 = ExecutionSimulator(
        execution_config={"algo": "POV", "pov": {"participation": 0.5, "child_interval_s": 1, "min_child_notional": 1.0}},
        slippage_config=slippage_cfg,
        latency_config=latency_cfg,
    )
    rep2 = sim2.run_step(
        ts=0,
        ref_price=10.0,
        bid=None,
        ask=None,
        vol_factor=1.0,
        liquidity=10.0,
        trade_price=10.0,
        trade_qty=0.0,
        actions=[(ActionType.MARKET, proto)],
    )

    trades1 = [(t.ts, t.qty, t.price) for t in rep1.trades]
    trades2 = [(t.ts, t.qty, t.price) for t in rep2.trades]
    assert trades1 == trades2
