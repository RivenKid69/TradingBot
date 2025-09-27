import importlib.util
import pathlib
import random
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
VWAPExecutor = exec_mod.VWAPExecutor
DataDegradationConfig = exec_mod.DataDegradationConfig


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


def test_vwap_profile_planning():
    execu = VWAPExecutor()
    snap = {
        "bar_timeframe_ms": 1_000,
        "bar_start_ts": 0,
        "intrabar_volume_profile": [
            {"ts": 100, "volume": 1.0},
            {"ts": 400, "volume": 3.0},
            {"ts": 900, "volume": 1.0},
        ],
    }
    plan = execu.plan_market(now_ts_ms=100, side="BUY", target_qty=50.0, snapshot=snap)
    assert len(plan) == 3
    offsets = [child.ts_offset_ms for child in plan]
    assert offsets == [0, 300, 800]
    qtys = [child.qty for child in plan]
    assert qtys[0] == pytest.approx(10.0)
    assert qtys[1] == pytest.approx(30.0)
    assert qtys[2] == pytest.approx(10.0)
    for child in plan:
        assert child.liquidity_hint == pytest.approx(child.qty)


def test_vwap_fallback_plan():
    execu = VWAPExecutor(fallback_parts=4)
    snap = {
        "bar_timeframe_ms": 400,
        "bar_start_ts": 0,
        "bar_end_ts": 400,
        "intrabar_volume_profile": [],
    }
    plan = execu.plan_market(now_ts_ms=100, side="BUY", target_qty=40.0, snapshot=snap)
    assert len(plan) == 4
    assert plan[0].ts_offset_ms == 0
    assert plan[-1].ts_offset_ms == 300
    qty_total = sum(child.qty for child in plan)
    assert qty_total == pytest.approx(40.0)


def test_data_degradation_seed_zero_is_distinct_from_none():
    dd_zero = DataDegradationConfig.default()
    dd_zero.seed = 0
    sim_zero = ExecutionSimulator(seed=123, data_degradation=dd_zero)
    zero_val = sim_zero._rng_dd.random()

    dd_none = DataDegradationConfig.default()
    dd_none.seed = None
    sim_none = ExecutionSimulator(seed=123, data_degradation=dd_none)
    none_val = sim_none._rng_dd.random()

    assert zero_val == random.Random(0).random()
    assert none_val == random.Random(123).random()


@pytest.mark.parametrize("mode", ["python", "hash", "xor", "mix", "stable"])
def test_intrabar_seed_modes_are_stable(mode):
    common_kwargs = {"seed": 98765, "execution_config": {"intrabar_seed_mode": mode}}
    sim_a = ExecutionSimulator(**common_kwargs)
    sim_b = ExecutionSimulator(**common_kwargs)

    inputs = [
        {"bar_ts": 0, "side": "buy", "order_seq": 1},
        {"bar_ts": 123456789, "side": "sell", "order_seq": 42},
        {"bar_ts": None, "side": "BUY", "order_seq": 7},
    ]

    seeds_a = [sim_a._intrabar_rng_seed(**params) for params in inputs]
    seeds_b = [sim_b._intrabar_rng_seed(**params) for params in inputs]

    assert seeds_a == seeds_b
