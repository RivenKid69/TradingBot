import pytest

from utils_time import (
    bar_close_ms,
    bar_start_ms,
    floor_to_timeframe,
    next_bar_open_ms,
)


REFERENCE_TS = 1_650_000_000_123


@pytest.mark.parametrize(
    "timeframe_ms, offsets",
    [
        (60_000, [0, 1, 59_999, 120_000]),
        (90_000, [0, 15, 45_000, 135_000]),
        (3_600_000, [0, 10_000, 1_800_000, 7_200_000]),
    ],
)
def test_bar_boundaries_are_consistent(timeframe_ms: int, offsets):
    aligned_base = (REFERENCE_TS // timeframe_ms) * timeframe_ms
    for delta in offsets:
        ts = aligned_base + delta
        start = bar_start_ms(ts, timeframe_ms)
        close = bar_close_ms(ts, timeframe_ms)
        assert close - start == timeframe_ms
        assert start == floor_to_timeframe(ts, timeframe_ms)
        assert start <= ts < close
        if ts > start:
            assert next_bar_open_ms(ts, timeframe_ms) == close
        assert next_bar_open_ms(close - 1, timeframe_ms) == close
        assert next_bar_open_ms(close, timeframe_ms) == close


@pytest.mark.parametrize("bad_timeframe", [0, -60_000])
def test_bar_helpers_reject_non_positive_timeframes(bad_timeframe: int) -> None:
    with pytest.raises(ValueError):
        bar_start_ms(REFERENCE_TS, bad_timeframe)
    with pytest.raises(ValueError):
        bar_close_ms(REFERENCE_TS, bad_timeframe)
    with pytest.raises(ValueError):
        next_bar_open_ms(REFERENCE_TS, bad_timeframe)


def test_bar_helpers_require_numeric_timestamp() -> None:
    with pytest.raises(ValueError):
        bar_start_ms("not-a-timestamp", 60_000)
    with pytest.raises(ValueError):
        bar_close_ms("not-a-timestamp", 60_000)
