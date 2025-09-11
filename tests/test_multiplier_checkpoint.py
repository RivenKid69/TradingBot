import json
import pathlib
import sys
import importlib.util

import numpy as np

BASE = pathlib.Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

# Load execution_sim module
spec_exec = importlib.util.spec_from_file_location("execution_sim", BASE / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec_exec)
sys.modules["execution_sim"] = exec_mod
spec_exec.loader.exec_module(exec_mod)
ExecutionSimulator = exec_mod.ExecutionSimulator

# Load impl_latency module
spec_lat = importlib.util.spec_from_file_location("impl_latency", BASE / "impl_latency.py")
lat_mod = importlib.util.module_from_spec(spec_lat)
sys.modules["impl_latency"] = lat_mod
spec_lat.loader.exec_module(lat_mod)
LatencyImpl = lat_mod.LatencyImpl


def test_execution_simulator_round_trip():
    liq = [1.0 + i * 0.001 for i in range(168)]
    spread = [0.5 + i * 0.0005 for i in range(168)]
    sim = ExecutionSimulator(liquidity_seasonality=liq, spread_seasonality=spread)
    dump = sim.dump_seasonality_multipliers()
    data = json.loads(json.dumps(dump))
    sim2 = ExecutionSimulator()
    sim2.load_seasonality_multipliers(data)
    assert sim2.dump_seasonality_multipliers() == dump


def test_latency_impl_round_trip():
    cfg = {
        "base_ms": 100,
        "jitter_ms": 0,
        "spike_p": 0.0,
        "timeout_ms": 1000,
    }
    impl = LatencyImpl.from_dict(cfg)
    mult = [1.0 + i * 0.01 for i in range(168)]
    impl.load_multipliers(mult)
    dump = impl.dump_multipliers()
    data = json.loads(json.dumps(dump))
    impl2 = LatencyImpl.from_dict(cfg)
    impl2.load_multipliers(data)
    assert np.allclose(impl2.dump_multipliers(), dump)
