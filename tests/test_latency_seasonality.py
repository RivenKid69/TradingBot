import importlib.util
import pathlib
import sys
import json
import datetime
import pytest

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


def test_latency_seasonality_disabled(tmp_path):
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
        "use_seasonality": False,
    }
    impl = LatencyImpl.from_dict(cfg)

    class Dummy:
        pass

    sim = Dummy()
    impl.attach_to(sim)
    lat = sim.latency

    d_high = lat.sample()
    d_low = lat.sample()
    assert d_high["total_ms"] == 100
    assert d_low["total_ms"] == 100


def test_seasonal_latency_statistics_regression(tmp_path):
    multipliers = [1.0] * 168
    multipliers[0] = 2.0
    multipliers[1] = 0.5
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

    ts0 = int(base_dt.timestamp() * 1000)
    for _ in range(5):
        lat.sample(ts0)
    stats0 = lat.stats()
    assert stats0["p50_ms"] == pytest.approx(200)
    assert stats0["p95_ms"] == pytest.approx(200)

    lat.reset_stats()
    ts1 = int(base_dt.timestamp() * 1000 + 3_600_000)
    for _ in range(5):
        lat.sample(ts1)
    stats1 = lat.stats()
    assert stats1["p50_ms"] == pytest.approx(50)
    assert stats1["p95_ms"] == pytest.approx(50)


def test_latency_seed_deterministic(tmp_path):
    multipliers = [1.0] * 168
    multipliers[0] = 2.0
    path = tmp_path / "latency.json"
    path.write_text(json.dumps({"latency": multipliers}))

    cfg = {
        "base_ms": 100,
        "jitter_ms": 50,
        "spike_p": 0.0,
        "timeout_ms": 1000,
        "seed": 123,
        "seasonality_path": str(path),
    }
    impl1 = LatencyImpl.from_dict(cfg)
    impl2 = LatencyImpl.from_dict(cfg)

    class Dummy:
        pass

    sim1 = Dummy()
    impl1.attach_to(sim1)
    lat1 = sim1.latency

    sim2 = Dummy()
    impl2.attach_to(sim2)
    lat2 = sim2.latency

    base_dt = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    ts0 = int(base_dt.timestamp() * 1000)
    ts1 = ts0 + 3_600_000

    seq1 = [
        lat1.sample(ts0)["total_ms"],
        lat1.sample(ts1)["total_ms"],
        lat1.sample(ts0)["total_ms"],
    ]
    seq2 = [
        lat2.sample(ts0)["total_ms"],
        lat2.sample(ts1)["total_ms"],
        lat2.sample(ts0)["total_ms"],
    ]

    assert seq1 == seq2
