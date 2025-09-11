from datetime import datetime, timezone, timedelta
import numpy as np
import pathlib, sys

import pytest

BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from utils.time import hour_of_week
from utils_time import hour_of_week as hour_of_week_dt
from impl_latency import hour_of_week as hour_of_week_latency


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
    """Ensure all hour-of-week helpers agree on index mapping."""
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
    np.testing.assert_array_equal(out_dt, expected)
    np.testing.assert_array_equal(out_latency, expected)
    np.testing.assert_array_equal(out_dt, out_latency)
