from __future__ import annotations

from typing import Any, Mapping

from execution_algos import _BarWindowAware


class _TestAware(_BarWindowAware):
    """Expose the protected resolution helper for testing."""

    def resolve(self, now_ts_ms: int, snapshot: Mapping[str, Any]):
        return self._resolve_bar_window(now_ts_ms, snapshot)


def test_resolve_bar_window_timeframe_only_aligns_now():
    aware = _TestAware()

    timeframe, start, end = aware.resolve(1_701_234, {"bar_timeframe_ms": 60_000})

    assert timeframe == 60_000
    assert start == (1_701_234 // 60_000) * 60_000
    assert end == start + 60_000
    assert aware._last_bar_timeframe_ms == timeframe
    assert aware._last_bar_start_ts == start
    assert aware._last_bar_end_ts == end


def test_resolve_bar_window_start_and_end_combinations():
    aware = _TestAware()

    timeframe, start, end = aware.resolve(
        1_700_000,
        {
            "bar_timeframe_ms": 60_000,
            "bar_start_ts": 1_620_000,
        },
    )
    assert (timeframe, start, end) == (60_000, 1_620_000, 1_680_000)
    assert aware._last_bar_end_ts == 1_680_000

    timeframe, start, end = aware.resolve(
        1_740_000,
        {
            "bar_timeframe_ms": 60_000,
            "bar_end_ts": 1_740_000,
        },
    )
    assert (timeframe, start, end) == (60_000, 1_620_000, 1_740_000)
    assert aware._last_bar_start_ts == 1_620_000
    assert aware._last_bar_end_ts == 1_740_000

    timeframe, start, end = aware.resolve(1_800_000, {"timeframe_ms": 60_000})
    assert (timeframe, start, end) == (60_000, 1_620_000, 1_740_000)

    timeframe, start, end = aware.resolve(
        1_860_000,
        {
            "bar_start_ts": 1_860_000,
            "intrabar_end_ts": 1_920_000,
        },
    )
    assert (timeframe, start, end) == (60_000, 1_860_000, 1_920_000)
    assert aware._last_bar_timeframe_ms == 60_000
    assert aware._last_bar_start_ts == 1_860_000
    assert aware._last_bar_end_ts == 1_920_000


def test_resolve_bar_window_invalid_inputs_reset_cache():
    aware = _TestAware()

    timeframe, start, end = aware.resolve(
        2_000_000,
        {
            "bar_timeframe_ms": "",
            "bar_start_ts": "bad",
            "bar_end_ts": None,
        },
    )
    assert (timeframe, start, end) == (None, None, None)
    assert aware._last_bar_timeframe_ms is None
    assert aware._last_bar_start_ts is None
    assert aware._last_bar_end_ts is None

    aware.resolve(
        2_060_000,
        {
            "bar_timeframe_ms": 60_000,
            "bar_start_ts": 2_000_000,
            "bar_end_ts": 2_060_000,
        },
    )

    timeframe, start, end = aware.resolve(
        2_120_000,
        {
            "bar_timeframe_ms": 0,
        },
    )
    assert (timeframe, start, end) == (60_000, 2_000_000, 2_060_000)
