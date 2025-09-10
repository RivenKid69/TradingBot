import importlib.util
import pathlib
import sys
import datetime

BASE = pathlib.Path(__file__).resolve().parents[1]

spec_exec = importlib.util.spec_from_file_location("execution_sim", BASE / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec_exec)
sys.modules["execution_sim"] = exec_mod
spec_exec.loader.exec_module(exec_mod)

ExecutionSimulator = exec_mod.ExecutionSimulator


def test_liquidity_and_spread_seasonality_multiplier():
    liq_mult = [1.0] * 168
    spr_mult = [1.0] * 168
    hour_idx = 10
    liq_mult[hour_idx] = 2.0
    spr_mult[hour_idx] = 3.0
    sim = ExecutionSimulator(
        liquidity_seasonality=liq_mult,
        spread_seasonality=spr_mult,
    )
    base_dt = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    ts_ms = int(base_dt.timestamp() * 1000 + hour_idx * 3_600_000)
    sim.set_market_snapshot(bid=100.0, ask=101.0, liquidity=5.0, spread_bps=1.0, ts_ms=ts_ms)
    assert sim._last_liquidity == 10.0
    assert sim._last_spread_bps == 3.0
