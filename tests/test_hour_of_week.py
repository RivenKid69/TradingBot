from datetime import datetime, timezone, timedelta
import numpy as np
import importlib.util
import pathlib
import sys

BASE = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("utils_time", BASE / "utils_time.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["utils_time"] = mod
spec.loader.exec_module(mod)

hour_of_week = mod.hour_of_week


def test_hour_of_week_known_timestamps():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts0 = int(base.timestamp() * 1000)
    ts1 = int((base + timedelta(hours=37)).timestamp() * 1000)  # Tuesday 13:00
    ts_last = int((base + timedelta(days=6, hours=23)).timestamp() * 1000)  # Sunday 23:00

    assert hour_of_week(ts0) == 0
    assert hour_of_week(ts1) == 37
    assert hour_of_week(ts_last) == 167

    arr = np.array([ts0, ts1, ts_last])
    np.testing.assert_array_equal(hour_of_week(arr), np.array([0, 37, 167]))
