import importlib.util
import pathlib
import sys
import json
import datetime

BASE = pathlib.Path(__file__).resolve().parents[1]

# Load latency module
spec_lat = importlib.util.spec_from_file_location("latency", BASE / "latency.py")
lat_module = importlib.util.module_from_spec(spec_lat)
sys.modules["latency"] = lat_module
spec_lat.loader.exec_module(lat_module)

# Load impl_latency module
spec_impl = importlib.util.spec_from_file_location("impl_latency", BASE / "impl_latency.py")
impl_module = importlib.util.module_from_spec(spec_impl)
sys.modules["impl_latency"] = impl_module
spec_impl.loader.exec_module(impl_module)

LatencyImpl = impl_module.LatencyImpl


def test_latency_seasonality(tmp_path):
    multipliers = [1.0] * 168
    hour_high = 5
    hour_low = 10
    multipliers[hour_high] = 2.0
    multipliers[hour_low] = 0.5
    path = tmp_path / "latency.json"
    path.write_text(json.dumps({"latency": multipliers}))

    cfg = {
        "base_ms": 100,
        "jitter_ms": 0,
        "spike_p": 0.0,
        "timeout_ms": 1000,
        "seasonality_path": str(path),
    }
    impl = LatencyImpl.from_dict(cfg)

    class Dummy:
        pass

    sim = Dummy()
    impl.attach_to(sim)
    lat = sim.latency

    base_dt = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    ts_high = int(base_dt.timestamp() * 1000 + hour_high * 3_600_000)
    ts_low = int(base_dt.timestamp() * 1000 + hour_low * 3_600_000)

    d_high = lat.sample(ts_high)
    d_low = lat.sample(ts_low)

    assert d_high["total_ms"] == 200
    assert d_low["total_ms"] == 50
