from __future__ import annotations

from typing import Sequence, Union
import numpy as np

HOUR_MS = 3_600_000
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
