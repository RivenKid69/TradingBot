import importlib.util
import pathlib
import sys
import logging

def _load_lat_module():
    BASE = pathlib.Path(__file__).resolve().parents[1]
    sys.path.append(str(BASE))
    spec_lat = importlib.util.spec_from_file_location("latency", BASE / "latency.py")
    lat_module = importlib.util.module_from_spec(spec_lat)
    sys.modules["latency"] = lat_module
    spec_lat.loader.exec_module(lat_module)
    return lat_module


def test_scaled_latency_capped_at_timeout(caplog):
    lat_module = _load_lat_module()
    LatencyModel = lat_module.LatencyModel
    SeasonalLatencyModel = lat_module.SeasonalLatencyModel

    multipliers = [10.0] * 168
    model = LatencyModel(base_ms=300, jitter_ms=0, spike_p=0.0, timeout_ms=1000)
    lat = SeasonalLatencyModel(model, multipliers)

    with caplog.at_level(logging.WARNING):
        res = lat.sample(0)

    assert res["total_ms"] == 1000
    assert any("exceeds timeout_ms" in r.message for r in caplog.records)
