import importlib.util
import pathlib
import sys
import threading
import time
import datetime

BASE = pathlib.Path(__file__).resolve().parents[1]

# Load latency module
spec_lat = importlib.util.spec_from_file_location("latency", BASE / "latency.py")
lat_module = importlib.util.module_from_spec(spec_lat)
sys.modules["latency"] = lat_module
spec_lat.loader.exec_module(lat_module)

LatencyModel = lat_module.LatencyModel
SeasonalLatencyModel = lat_module.SeasonalLatencyModel


def test_seasonal_latency_thread_safe_concurrent():
    multipliers = [1.0] * 168
    hour_high = 5
    hour_low = 10
    multipliers[hour_high] = 2.0
    multipliers[hour_low] = 0.5

    model = LatencyModel(base_ms=100, jitter_ms=0, spike_p=0.0, timeout_ms=1000)
    lat = SeasonalLatencyModel(model, multipliers)

    base_dt = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    ts_high = int(base_dt.timestamp() * 1000 + hour_high * 3_600_000)
    ts_low = int(base_dt.timestamp() * 1000 + hour_low * 3_600_000)

    orig_sample = model.sample
    def slow_sample():
        time.sleep(0.001)
        return orig_sample()
    model.sample = slow_sample

    high_results: list[int] = []
    low_results: list[int] = []

    def worker(ts: int, dest: list[int]):
        for _ in range(100):
            dest.append(lat.sample(ts)["total_ms"])

    threads = [
        threading.Thread(target=worker, args=(ts_high, high_results)),
        threading.Thread(target=worker, args=(ts_low, low_results)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r == 200 for r in high_results)
    assert all(r == 50 for r in low_results)
    assert len(model.lat_samples) == len(high_results) + len(low_results)
    assert model.base_ms == 100
    assert model.jitter_ms == 0
    assert model.timeout_ms == 1000
