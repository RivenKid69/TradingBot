from datetime import datetime, timezone, timedelta
import numpy as np
import pathlib, sys, importlib.util
import logging

import pytest

# Pre-load stdlib logging before adding repo path to avoid local module shadowing
logging.getLogger()

BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from utils.time import hour_of_week
from utils_time import hour_of_week as hour_of_week_dt

# Dynamically load modules that re-export hour_of_week
spec_lat = importlib.util.spec_from_file_location("latency", BASE / "latency.py")
lat_module = importlib.util.module_from_spec(spec_lat)
sys.modules["latency"] = lat_module
spec_lat.loader.exec_module(lat_module)
LatencyModel = lat_module.LatencyModel

spec_impl = importlib.util.spec_from_file_location("impl_latency", BASE / "impl_latency.py")
impl_module = importlib.util.module_from_spec(spec_impl)
sys.modules["impl_latency"] = impl_module
spec_impl.loader.exec_module(impl_module)
hour_of_week_latency = impl_module.hour_of_week
_LatencyWithSeasonality = impl_module._LatencyWithSeasonality

spec_exec = importlib.util.spec_from_file_location("execution_sim", BASE / "execution_sim.py")
exec_module = importlib.util.module_from_spec(spec_exec)
sys.modules["execution_sim"] = exec_module
spec_exec.loader.exec_module(exec_module)
hour_of_week_exec = exec_module.hour_of_week
ExecutionSimulator = exec_module.ExecutionSimulator


@pytest.mark.parametrize("func", [hour_of_week, hour_of_week_dt])
def test_hour_of_week_known_timestamps(func):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts0 = int(base.timestamp() * 1000)
    ts1 = int((base + timedelta(hours=37)).timestamp() * 1000)  # Tuesday 13:00
    ts_last = int((base + timedelta(days=6, hours=23)).timestamp() * 1000)  # Sunday 23:00

    assert func(ts0) == 0
    assert func(ts1) == 37
    assert func(ts_last) == 167

    arr = np.array([ts0, ts1, ts_last])
    np.testing.assert_array_equal(func(arr), np.array([0, 37, 167]))


@pytest.mark.parametrize("func", [hour_of_week, hour_of_week_dt])
def test_hour_of_week_week_boundary(func):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts_last = int((base + timedelta(days=6, hours=23)).timestamp() * 1000)
    ts_next = ts_last + 3_600_000

    assert func(ts_last) == 167
    assert func(ts_next) == 0
    arr = np.array([ts_last, ts_next])
    np.testing.assert_array_equal(func(arr), np.array([167, 0]))


def test_cross_module_consistency():
    """Ensure hour-of-week helpers agree across modules."""
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts_samples = np.array(
        [
            int(base.timestamp() * 1000),
            int((base + timedelta(hours=55)).timestamp() * 1000),
            int((base + timedelta(days=6, hours=23)).timestamp() * 1000),
        ]
    )

    expected = hour_of_week(ts_samples)
    out_dt = hour_of_week_dt(ts_samples)
    out_latency = hour_of_week_latency(ts_samples)
    out_exec = hour_of_week_exec(ts_samples)
    np.testing.assert_array_equal(out_dt, expected)
    np.testing.assert_array_equal(out_latency, expected)
    np.testing.assert_array_equal(out_exec, expected)


def test_timestamp_hour_index_alignment_across_components():
    """Identical timestamps should map to the same hour index for latency,
    liquidity and spread seasonality handlers."""
    hour_idx = 42
    liq_mult = [1.0] * 168
    spr_mult = [1.0] * 168
    lat_mult = [1.0] * 168
    liq_mult[hour_idx] = 2.0
    spr_mult[hour_idx] = 3.0
    lat_mult[hour_idx] = 4.0

    sim = ExecutionSimulator(
        liquidity_seasonality=liq_mult,
        spread_seasonality=spr_mult,
    )
    lat_model = LatencyModel(base_ms=100, jitter_ms=0, spike_p=0.0, spike_mult=1.0, timeout_ms=1000)
    lat = _LatencyWithSeasonality(lat_model, lat_mult)

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts_ms = int((base + timedelta(hours=hour_idx)).timestamp() * 1000)

    # Ensure all helpers agree on the hour index
    idx_expected = hour_of_week(ts_ms)
    assert hour_of_week_latency(ts_ms) == idx_expected
    assert hour_of_week_exec(ts_ms) == idx_expected

    lat.sample(ts_ms)
    sim.set_market_snapshot(
        bid=100.0,
        ask=101.0,
        liquidity=5.0,
        spread_bps=1.0,
        ts_ms=ts_ms,
    )

    assert lat._mult_sum[idx_expected] == pytest.approx(lat_mult[idx_expected])
    assert sim._last_liquidity == pytest.approx(5.0 * liq_mult[idx_expected])
    assert sim._last_spread_bps == pytest.approx(1.0 * spr_mult[idx_expected])

