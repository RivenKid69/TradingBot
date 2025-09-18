import importlib.util
import pathlib
import sys
import numpy as np
import pytest

BASE = pathlib.Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

# Load latency module
spec_lat = importlib.util.spec_from_file_location("latency", BASE / "latency.py")
lat = importlib.util.module_from_spec(spec_lat)
sys.modules["latency"] = lat
spec_lat.loader.exec_module(lat)

# Load impl_latency module
spec_impl = importlib.util.spec_from_file_location("impl_latency", BASE / "impl_latency.py")
impl = importlib.util.module_from_spec(spec_impl)
sys.modules["impl_latency"] = impl
spec_impl.loader.exec_module(impl)

validate_multipliers = lat.validate_multipliers
LatencyImpl = impl.LatencyImpl


def test_validate_multipliers_checks():
    cap = lat.SEASONALITY_MULT_MAX
    with pytest.raises(ValueError, match="length 168"):
        validate_multipliers([1.0] * 167)
    arr = [1.0] * 168
    arr[0] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        validate_multipliers(arr)
    arr = [1.0] * 168
    arr[1] = -0.5
    with pytest.raises(ValueError, match="positive"):
        validate_multipliers(arr)
    arr = [1.0] * 168
    arr[2] = cap + 1.0
    with pytest.raises(ValueError, match="exceeds cap"):
        validate_multipliers(arr)


def test_latency_impl_load_multipliers_validation():
    cfg = {
        "base_ms": 100,
        "jitter_ms": 0,
        "spike_p": 0.0,
        "timeout_ms": 1000,
    }
    impl_instance = LatencyImpl.from_dict(cfg)
    with pytest.raises(ValueError, match="length 168"):
        impl_instance.load_multipliers([1.0] * 167)


def test_latency_impl_load_multipliers_accepts_daily_for_hourly():
    cfg = {
        "base_ms": 50,
        "jitter_ms": 10,
        "spike_p": 0.0,
        "timeout_ms": 500,
    }
    impl_instance = LatencyImpl.from_dict(cfg)
    impl_instance.load_multipliers([1.0] * 7)
    dump = impl_instance.dump_multipliers()
    assert len(dump) == 168
    assert np.allclose(dump, 1.0)


def test_latency_impl_load_multipliers_hourly_to_daily():
    cfg = {
        "base_ms": 50,
        "jitter_ms": 10,
        "spike_p": 0.0,
        "timeout_ms": 500,
        "seasonality_day_only": True,
    }
    impl_instance = LatencyImpl.from_dict(cfg)
    impl_instance.load_multipliers([1.0] * 168)
    dump = impl_instance.dump_multipliers()
    assert len(dump) == 7
    assert np.allclose(dump, 1.0)


def test_latency_impl_load_multipliers_mapping():
    cfg = {
        "base_ms": 50,
        "jitter_ms": 10,
        "spike_p": 0.0,
        "timeout_ms": 500,
    }
    impl_instance = LatencyImpl.from_dict(cfg)
    payload = {str(i): 1.0 for i in range(168)}
    impl_instance.load_multipliers(payload)
    dump = impl_instance.dump_multipliers()
    assert len(dump) == 168
    assert np.allclose(dump, 1.0)
