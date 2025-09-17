from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence, Union
import numpy as np

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
HOURS_IN_WEEK = 168
# 1970-01-01 00:00 UTC was a Thursday, which is hour 72 of the week
_EPOCH_HOW = 72  # Hour-of-week for Unix epoch (0 = Monday 00:00 UTC)

def hour_of_week(ts_ms: Union[int, Sequence[int], np.ndarray]) -> Union[int, np.ndarray]:
    """Return hour-of-week index where ``0`` is Monday 00:00 UTC.

    Parameters
    ----------
    ts_ms:
        UTC timestamp(s) in milliseconds.
    """
    arr = np.asarray(ts_ms, dtype=np.int64)

    def _calc(ts: int) -> int:
        # Convert milliseconds since epoch to an hour-of-week index.  We add the
        # epoch offset (1970‑01‑01 00:00 UTC was Thursday, hour 72 of the week)
        # and wrap by 168 so that Monday 00:00 UTC maps to ``0``.
        #
        # This formula is used across the codebase via this shared helper, so it
        # assumes ``ts`` is a UTC timestamp.
        idx = int((ts // HOUR_MS + _EPOCH_HOW) % HOURS_IN_WEEK)
        assert 0 <= idx < HOURS_IN_WEEK
        return idx

    if arr.shape == ():
        return _calc(int(arr))

    vec = np.vectorize(_calc, otypes=[int])
    return vec(arr)


def _normalize_reset_hour(reset_hour: Union[int, float]) -> int:
    """Return ``reset_hour`` normalized to the ``0``-``23`` range."""

    hour = int(reset_hour)
    return hour % 24


def _daily_reset_start_ms(ts_ms: int, reset_hour: int) -> int:
    """Timestamp (ms) corresponding to the start of the reset period for ``ts_ms``."""

    hour = _normalize_reset_hour(reset_hour)
    offset_ms = hour * HOUR_MS
    day_index = (int(ts_ms) - offset_ms) // DAY_MS
    return day_index * DAY_MS + offset_ms


def daily_reset_key(ts_ms: int, reset_hour: int) -> str:
    """Return a stable key identifying the trading "day" for ``ts_ms``.

    Parameters
    ----------
    ts_ms:
        UTC timestamp in **milliseconds**.
    reset_hour:
        Hour (UTC) when a new "trading day" starts. Values outside the
        ``0``-``23`` range are wrapped around to the valid range.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> ts = int(datetime(2024, 5, 17, 0, tzinfo=timezone.utc).timestamp() * 1000)
    >>> daily_reset_key(ts, 0)
    '2024-05-17T00'
    >>> daily_reset_key(ts + 3_600_000, 5)  # 2024-05-17 01:00 UTC is before 05:00 reset
    '2024-05-16T05'
    >>> daily_reset_key(ts + 10 * 3_600_000, 5)  # 2024-05-17 10:00 UTC is after reset
    '2024-05-17T05'
    """

    start_ms = _daily_reset_start_ms(ts_ms, reset_hour)
    dt = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H")


def next_daily_reset_ms(ts_ms: int, reset_hour: int) -> int:
    """Return the UTC timestamp (ms) for the next daily reset after ``ts_ms``.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> ts = int(datetime(2024, 5, 17, 0, tzinfo=timezone.utc).timestamp() * 1000)
    >>> next_daily_reset_ms(ts, 0) == ts + DAY_MS
    True
    >>> first_reset = int(datetime(2024, 5, 17, 5, tzinfo=timezone.utc).timestamp() * 1000)
    >>> next_daily_reset_ms(ts + 2 * HOUR_MS, 5) == first_reset
    True
    """

    return _daily_reset_start_ms(ts_ms, reset_hour) + DAY_MS
