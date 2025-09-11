import importlib.util
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parents[1]

# Load latency module
spec_lat = importlib.util.spec_from_file_location("latency", BASE / "latency.py")
lat_module = importlib.util.module_from_spec(spec_lat)
sys.modules["latency"] = lat_module
spec_lat.loader.exec_module(lat_module)

LatencyModel = lat_module.LatencyModel
SeasonalLatencyModel = lat_module.SeasonalLatencyModel


def test_latency_rng_sequence_unaffected_by_seasonality():
    multipliers = [1.0] * 168
    cfg = {
        "base_ms": 100,
        "jitter_ms": 50,
        "spike_p": 0.2,
        "spike_mult": 2.0,
        "timeout_ms": 1000,
        "seed": 12345,
    }

    plain = LatencyModel(**cfg)
    seasonal_model = LatencyModel(**cfg)
    seasonal = SeasonalLatencyModel(seasonal_model, multipliers)

    seq_plain = [plain.sample() for _ in range(5)]
    seq_seasonal = [seasonal.sample(0) for _ in range(5)]

    assert seq_plain == seq_seasonal
    assert plain._rng.getstate() == seasonal_model._rng.getstate()
