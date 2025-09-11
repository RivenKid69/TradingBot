import importlib.util
import pathlib
import sys
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
