from __future__ import annotations

from typing import Sequence, Union
import numpy as np

HOUR_MS = 3_600_000
HOURS_IN_WEEK = 168
_EPOCH_HOW = 72  # Hour-of-week for Unix epoch (Thursday 00:00 UTC)

def hour_of_week(ts_ms: Union[int, Sequence[int], np.ndarray]) -> Union[int, np.ndarray]:
    """Return hour-of-week (0-167) for UTC timestamps in milliseconds."""
    arr = np.asarray(ts_ms, dtype=np.int64)

    def _calc(ts: int) -> int:
        return int((ts // HOUR_MS + _EPOCH_HOW) % HOURS_IN_WEEK)

    if arr.shape == ():
        return _calc(int(arr))

    vec = np.vectorize(_calc, otypes=[int])
    return vec(arr)
