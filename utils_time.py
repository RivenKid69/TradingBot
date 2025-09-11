"""Assorted time utilities.

The hour-of-week index assumes ``0 = Monday 00:00 UTC``.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, Sequence, Union
import os
import json
import hashlib
import importlib.util
import sysconfig
from pathlib import Path
import numpy as np

_logging_spec = importlib.util.spec_from_file_location(
    "py_logging", Path(sysconfig.get_path("stdlib")) / "logging/__init__.py"
)
logging = importlib.util.module_from_spec(_logging_spec)
_logging_spec.loader.exec_module(logging)


HOUR_MS = 3_600_000
HOURS_IN_WEEK = 168


def hour_of_week(ts_ms: Union[int, Sequence[int], np.ndarray]) -> Union[int, np.ndarray]:
    """Return hour-of-week index where ``0`` is Monday 00:00 UTC.

    The calculation uses :func:`datetime.utcfromtimestamp` to avoid any
    dependence on the local timezone.
    """
    arr = np.asarray(ts_ms, dtype=np.int64)

    def _calc(ts: int) -> int:
        dt = datetime.utcfromtimestamp(int(ts) / 1000)
        return dt.weekday() * 24 + dt.hour

    if arr.shape == ():
        return _calc(int(arr))

    vec = np.vectorize(_calc, otypes=[int])
    return vec(arr)


def load_hourly_seasonality(path: str, *keys: str, expected_hash: str | None = None) -> np.ndarray | None:
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
        with open(path, "rb") as f:
            raw = f.read()
        digest = hashlib.sha256(raw).hexdigest()
        logger = logging.getLogger(__name__)
        logger.info("Loaded seasonality multipliers from %s (sha256=%s)", path, digest)
        if expected_hash and digest.lower() != expected_hash.lower():
            logger.warning(
                "Seasonality hash mismatch for %s: expected %s got %s",
                path,
                expected_hash,
                digest,
            )
        data = json.loads(raw.decode("utf-8"))
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
