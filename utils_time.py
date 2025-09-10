# data/utils_time.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, Sequence, Union
import os
import json
import numpy as np


HOUR_MS = 3_600_000
HOURS_IN_WEEK = 168
EPOCH_HOW_OFFSET = 72


def hour_of_week(ts_ms: Union[int, Sequence[int], np.ndarray]) -> Union[int, np.ndarray]:
    """Return hour-of-week (0-167) for timestamps in milliseconds."""
    arr = np.asarray(ts_ms, dtype=np.int64)
    how = ((arr // HOUR_MS) + EPOCH_HOW_OFFSET) % HOURS_IN_WEEK
    if arr.shape == ():
        return int(how)
    return how.astype(int)


def load_hourly_seasonality(path: str, *keys: str) -> np.ndarray | None:
    """Load hourly multipliers array from JSON file.

    Parameters
    ----------
    path : str
        Path to JSON file.
    keys : str
        Candidate keys within JSON mapping to extract array from.

    Returns
    -------
    numpy.ndarray | None
        Array of length 168 if successful, otherwise ``None``.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in keys:
                if k in data:
                    data = data[k]
                    break
        arr = np.asarray(data, dtype=float)
        if arr.shape[0] == HOURS_IN_WEEK:
            return arr
    except Exception:
        return None
    return None


def parse_time_to_ms(s: str) -> int:
    """
    Поддерживает:
      - Unix миллисекунды (строка из цифр длиной >= 10)
      - ISO 8601 / 'YYYY-MM-DD HH:MM:SS' / 'YYYY-MM-DD'
      - Специальные ключи: 'now', 'today'
    Возвращает Unix ms (int).
    """
    zs = str(s).strip()
    if zs.lower() in ("now",):
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    if zs.lower() in ("today",):
        dt = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return int(dt.timestamp() * 1000)
    if zs.isdigit():
        v = int(zs)
        # если это секунды — домножим
        if v < 10_000_000_000:
            v *= 1000
        return v
    # попробуем несколько форматов
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(zs, fmt)
            dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            pass
    # ISO
    try:
        dt = datetime.fromisoformat(zs)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        pass
    raise ValueError(f"Не удалось распарсить время: {s}")
