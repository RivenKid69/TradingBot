import os
import sys
import numpy as np
import pytest

sys.path.append(os.getcwd())

from no_trade import _parse_daily_windows_min, _in_funding_buffer, _in_custom_window


def test_parse_daily_windows_min_valid_and_invalid():
    windows = ["00:00-01:00", "22:00-24:00", "bad", "23:00-22:00"]
    assert _parse_daily_windows_min(windows) == [(0, 60), (1320, 1440)]


def test_in_funding_buffer_with_midnight_and_day_marks():
    ts_minutes = np.array([
        0,          # exactly at midnight
        5,          # within buffer after midnight
        475,        # 7:55, 5 min before 8h mark
        495,        # 8:15, outside 10 min buffer
        951,        # 15:51, 9 min before 16h mark
        971,        # 16:11, outside buffer
        1435,       # 23:55, far from midnight mark
    ], dtype=np.int64)
    ts_ms = ts_minutes * 60_000
    mask = _in_funding_buffer(ts_ms, 10)
    expected = np.array([True, True, True, False, True, False, True])
    np.testing.assert_array_equal(mask, expected)


def test_in_custom_window_respects_ranges():
    ts_minutes = np.array([0, 4, 5, 10, 12], dtype=np.int64)
    ts_ms = ts_minutes * 60_000
    windows = [
        {"start_ts_ms": 0, "end_ts_ms": 5 * 60_000},
        {"start_ts_ms": 10 * 60_000, "end_ts_ms": 11 * 60_000},
    ]
    mask = _in_custom_window(ts_ms, windows)
    expected = np.array([True, True, True, True, False])
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.parametrize(
    "windows,match",
    [
        ([{"start_ts_ms": "bad", "end_ts_ms": 20}], "integer"),
        ([{"start_ts_ms": 10, "end_ts_ms": 5}], "must be <"),
        ([{"start_ts_ms": 10, "end_ts_ms": 10}], "must be <"),
    ],
)
def test_in_custom_window_invalid_windows_raise(windows, match):
    ts_ms = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError, match=match):
        _in_custom_window(ts_ms, windows)
