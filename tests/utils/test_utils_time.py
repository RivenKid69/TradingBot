import pytest

from utils_time import (
    _normalize_bar_bounds,
    bar_start_ms,
    bar_close_ms,
    is_bar_closed,
    next_bar_open_ms,
)


@pytest.mark.parametrize(
    "ts_ms,timeframe_ms",
    [
        (1_650_000_000_000, 60_000),
        (1_650_000_030_001, 60_000),
        (1_650_000_090_000, 90_000),
        (1_650_000_210_500, 120_000),
    ],
)
def test_normalize_bar_bounds_and_helpers(ts_ms, timeframe_ms):
    start, close = _normalize_bar_bounds(ts_ms, timeframe_ms)
    expected_start = (ts_ms // timeframe_ms) * timeframe_ms
    expected_close = expected_start + timeframe_ms
    assert (start, close) == (expected_start, expected_close)
    assert bar_start_ms(ts_ms, timeframe_ms) == start
    assert bar_close_ms(ts_ms, timeframe_ms) == close
    assert start <= ts_ms < close
    assert next_bar_open_ms(close, timeframe_ms) == close


@pytest.mark.parametrize(
    "bad_ts,bad_timeframe,error_message",
    [
        ("abc", 60_000, "ts_ms"),
        (1_650_000_000_000, "1m", "timeframe_ms"),
    ],
)
def test_normalize_bar_bounds_rejects_invalid_inputs(bad_ts, bad_timeframe, error_message):
    with pytest.raises(ValueError) as exc:
        _normalize_bar_bounds(bad_ts, bad_timeframe)
    assert error_message in str(exc.value)


@pytest.mark.parametrize("bad_timeframe", [0, -1, -60_000])
def test_bar_helpers_reject_non_positive_timeframes(bad_timeframe):
    with pytest.raises(ValueError):
        _normalize_bar_bounds(1_650_000_000_000, bad_timeframe)
    with pytest.raises(ValueError):
        bar_start_ms(1_650_000_000_000, bad_timeframe)
    with pytest.raises(ValueError):
        bar_close_ms(1_650_000_000_000, bad_timeframe)
    with pytest.raises(ValueError):
        next_bar_open_ms(1_650_000_000_000, bad_timeframe)


def test_is_bar_closed_honors_lag_offset():
    close_ts = 1_650_000_120_000
    now_ts = close_ts + 500
    assert not is_bar_closed(close_ts, now_ts, lag_ms=1_000)
    assert is_bar_closed(close_ts, now_ts + 500, lag_ms=1_000)
    assert is_bar_closed(close_ts, now_ts, lag_ms=0)


def test_next_bar_open_follows_close_timestamp():
    ts_ms = 1_650_000_060_001
    timeframe_ms = 60_000
    close_ts = bar_close_ms(ts_ms, timeframe_ms)
    assert next_bar_open_ms(ts_ms, timeframe_ms) == close_ts
    assert next_bar_open_ms(close_ts, timeframe_ms) == close_ts
