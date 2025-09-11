import importlib.util
import pathlib
import sys
import json
import datetime

BASE = pathlib.Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# Load ExecutionSimulator dynamically
spec_exec = importlib.util.spec_from_file_location("execution_sim", BASE / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec_exec)
sys.modules["execution_sim"] = exec_mod
spec_exec.loader.exec_module(exec_mod)
ExecutionSimulator = exec_mod.ExecutionSimulator

# Load latency implementation
def _load_latency_impl():
    spec_lat = importlib.util.spec_from_file_location("impl_latency", BASE / "impl_latency.py")
    lat_mod = importlib.util.module_from_spec(spec_lat)
    sys.modules["impl_latency"] = lat_mod
    spec_lat.loader.exec_module(lat_mod)
    return lat_mod

lat_mod = _load_latency_impl()
LatencyCfg = lat_mod.LatencyCfg
LatencyImpl = lat_mod.LatencyImpl


def test_simulation_applies_liquidity_and_latency_seasonality(tmp_path):
    hour = 10
    # liquidity multipliers: double liquidity at target hour
    liq_mult = [1.0] * 168
    liq_mult[hour] = 2.0

    # latency multipliers: double latency at same hour
    lat_mult = [1.0] * 168
    lat_mult[hour] = 2.0
    lat_path = tmp_path / "latency.json"
    lat_path.write_text(json.dumps({"latency": lat_mult}))

    sim = ExecutionSimulator(liquidity_seasonality=liq_mult)

    cfg = LatencyCfg(
        base_ms=100,
        jitter_ms=0,
        spike_p=0.0,
        spike_mult=1.0,
        timeout_ms=1000,
        seed=0,
        seasonality_path=str(lat_path),
    )
    lat_impl = LatencyImpl(cfg)
    lat_impl.attach_to(sim)

    base = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    ts_ms = int(base.timestamp() * 1000 + hour * 3_600_000)

    sim.set_market_snapshot(
        bid=100.0,
        ask=101.0,
        liquidity=5.0,
        spread_bps=1.0,
        ts_ms=ts_ms,
    )

    assert sim._last_liquidity == 10.0

    sample = sim.latency.sample(ts_ms)
    assert sample["total_ms"] == 200
