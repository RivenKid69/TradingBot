"""Assorted time utilities.

The hour-of-week index assumes ``0 = Monday 00:00 UTC``.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Sequence, Callable
import os
import json
import hashlib
import importlib.util
import sysconfig
import threading
import time
from pathlib import Path
import numpy as np
import clock
from utils.time import hour_of_week, HOUR_MS, HOURS_IN_WEEK

__all__ = ["floor_to_timeframe", "is_bar_closed", "now_ms", "next_bar_open_ms"]

_logging_spec = importlib.util.spec_from_file_location(
    "py_logging", Path(sysconfig.get_path("stdlib")) / "logging/__init__.py"
)
logging = importlib.util.module_from_spec(_logging_spec)
_logging_spec.loader.exec_module(logging)
seasonality_logger = logging.getLogger("seasonality").getChild(__name__)

# Clamp limits applied to liquidity and latency seasonality multipliers.
SEASONALITY_MULT_MIN = 0.1
SEASONALITY_MULT_MAX = 10.0


def floor_to_timeframe(ts_ms: int, timeframe_ms: int) -> int:
    """Floor ``ts_ms`` down to the nearest multiple of ``timeframe_ms``."""

    return (ts_ms // timeframe_ms) * timeframe_ms


def is_bar_closed(ts_close_ms: int, now_utc_ms: int, lag_ms: int = 0) -> bool:
    """Return ``True`` if current time exceeds ``ts_close_ms`` plus ``lag_ms``."""

    return now_utc_ms >= ts_close_ms + lag_ms


def now_ms() -> int:
    """Return current corrected time accounting for clock skew."""

    return int(clock.system_utc_ms() + clock.clock_skew())


def next_bar_open_ms(close_ms: int, timeframe_ms: int) -> int:
    """Return the open timestamp of the bar following ``close_ms``."""

    return floor_to_timeframe(close_ms, timeframe_ms) + timeframe_ms


def interpolate_daily_multipliers(days: Sequence[float]) -> np.ndarray:
    """Expand 7-element day-of-week multipliers to 168 hours.

    Linear interpolation is applied between adjacent days to provide a smooth
    transition for each hour of the week. The first value is appended to the
    end to ensure wrap-around continuity.
    """

    arr = np.asarray(list(days), dtype=float)
    if arr.size != 7:
        raise ValueError("days must have length 7")
    hours = np.arange(0, HOURS_IN_WEEK + 24, 24)
    vals = np.concatenate([arr, arr[:1]])
    return np.interp(np.arange(HOURS_IN_WEEK), hours, vals)


def daily_from_hourly(hours: Sequence[float]) -> np.ndarray:
    """Collapse 168-hour multipliers into 7 day-of-week averages."""

    arr = np.asarray(list(hours), dtype=float)
    if arr.size != HOURS_IN_WEEK:
        raise ValueError("hours must have length 168")
    return arr.reshape(7, 24).mean(axis=1)


def load_hourly_seasonality(
    path: str,
    *keys: str,
    symbol: str | None = None,
    expected_hash: str | None = None,
) -> np.ndarray | None:
    """Load hourly or daily multipliers array from JSON file.

    Parameters
    ----------
    path : str
        Path to JSON file.
    keys : str
        Candidate keys within JSON mapping to extract array from.
    symbol : str | None
        Optional instrument symbol if the JSON file contains mappings per symbol.

    Returns
    -------
    numpy.ndarray | None
        Array of length 168 or 7 if successful, otherwise ``None``.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            raw = f.read()
        digest = hashlib.sha256(raw).hexdigest()
        seasonality_logger.info(
            "Loaded seasonality multipliers from %s (sha256=%s)", path, digest
        )
        if expected_hash and digest.lower() != expected_hash.lower():
            seasonality_logger.warning(
                "Seasonality hash mismatch for %s: expected %s got %s",
                path,
                expected_hash,
                digest,
            )
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            return None
        # Allow new structure {"SYMBOL": {"latency": [...]}}
        if symbol and symbol in data:
            data = data[symbol]
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
                break
        if not isinstance(data, list):
            return None
        arr = np.asarray(data, dtype=float)
        if arr.shape[0] in (HOURS_IN_WEEK, 7):
            if np.any(arr <= 0):
                raise ValueError("Seasonality array values must be > 0")
            if any(k in {"liquidity", "latency"} for k in keys):
                arr = np.clip(arr, SEASONALITY_MULT_MIN, SEASONALITY_MULT_MAX)
            return arr
    except ValueError:
        raise
    except Exception:
        return None
    return None


def load_seasonality(path: str) -> Dict[str, np.ndarray]:
    """Load all available seasonality arrays from ``path``.

    The JSON file is expected to contain arrays of length :data:`HOURS_IN_WEEK`
    (168) or 7 (one per weekday). It may either expose the arrays at the top
    level, or nest them under an instrument symbol. Only keys with list values
    of an accepted length are returned.

    Parameters
    ----------
    path:
        Path to a JSON file. ``FileNotFoundError`` is raised if the path does
        not exist.

    Returns
    -------
    Dict[str, numpy.ndarray]
        Mapping of keys such as ``"liquidity"``, ``"latency"`` or
        ``"spread"`` to numpy arrays.

    Raises
    ------
    ValueError
        If the file cannot be parsed or does not contain any valid arrays.
    """

    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)

    try:
        with open(path, "rb") as f:
            raw = f.read()
        digest = hashlib.sha256(raw).hexdigest()
        seasonality_logger.info(
            "Loaded seasonality multipliers from %s (sha256=%s)", path, digest
        )
        data = json.loads(raw.decode("utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - unexpected parse error
        raise ValueError(f"Invalid seasonality file {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("Seasonality JSON must be an object")

    def _extract(obj: Dict[str, object]) -> Dict[str, np.ndarray]:
        res: Dict[str, np.ndarray] = {}
        for key in ("liquidity", "latency", "spread"):
            if key in obj:
                arr = np.asarray(obj[key], dtype=float)
                if arr.shape[0] not in (HOURS_IN_WEEK, 7):
                    raise ValueError(
                        "Seasonality array '%s' must have length 168 or 7" % key
                    )
                if np.any(arr <= 0):
                    raise ValueError(
                        "Seasonality array '%s' must contain positive values" % key
                    )
                if key in {"liquidity", "latency"}:
                    arr = np.clip(arr, SEASONALITY_MULT_MIN, SEASONALITY_MULT_MAX)
                res[key] = arr
        return res

    arrays = _extract(data)
    if arrays:
        return arrays

    # Handle structure where arrays are nested under a symbol key.
    candidates = []
    for val in data.values():
        if isinstance(val, dict):
            arrs = _extract(val)
            if arrs:
                candidates.append(arrs)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError("No seasonality arrays found")
    raise ValueError("Multiple seasonality mappings found; specify symbol")


def watch_seasonality_file(
    path: str,
    callback: Callable[[Dict[str, np.ndarray]], None],
    *,
    poll_interval: float = 60.0,
) -> threading.Thread:
    """Watch ``path`` for changes and invoke ``callback`` when updated.

    The callback receives the mapping returned by :func:`load_seasonality`.
    Reloading is attempted whenever the file modification time increases.
    The watcher runs in a daemon thread and logs any file read errors,
    retaining the last successfully loaded multipliers for operator
    awareness.
    """

    def _loop() -> None:
        last_mtime: float | None = None
        last_data: Dict[str, np.ndarray] | None = None
        while True:
            try:
                mtime = os.path.getmtime(path)
                if last_mtime is None or mtime > last_mtime:
                    try:
                        data = load_seasonality(path)
                    except Exception:
                        seasonality_logger.exception(
                            "Failed to reload seasonality multipliers from %s", path
                        )
                        if last_data is not None:
                            seasonality_logger.warning(
                                "Retaining last known good multipliers from %s", path
                            )
                            try:
                                callback(last_data)
                            except Exception:
                                seasonality_logger.exception(
                                    "Failed to reapply previous multipliers from %s",
                                    path,
                                )
                    else:
                        callback(data)
                        last_data = data
                        last_mtime = mtime
            except Exception:
                seasonality_logger.exception(
                    "Error while watching seasonality file %s", path
                )
            time.sleep(float(poll_interval))

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def _hour_index(ts_ms: int, length: int) -> int:
    """Return hour-of-week index clamped to ``length``.

    The helper mirrors :func:`utils.time.hour_of_week` but additionally
    wraps the result by ``length`` to support arrays shorter than a full
    week. ``length`` defaults to :data:`HOURS_IN_WEEK`.
    """

    # Reuse the shared hour_of_week helper which computes
    # ``(ts_ms // HOUR_MS + 72) % 168`` so that ``0`` maps to Monday 00:00 UTC.
    # ``ts_ms`` must therefore be a UTC timestamp.
    hour = hour_of_week(int(ts_ms))
    if length:
        length = int(length)
        if length == 7:
            hour = (hour // 24) % 7
        else:
            hour %= length
    return int(hour)


def get_hourly_multiplier(
    ts_ms: int, multipliers: Sequence[float], *, interpolate: bool = False
) -> float:
    """Return multiplier for ``ts_ms`` from ``multipliers``.

    If ``interpolate`` is ``False`` (default) the multiplier of the nearest
    hour is returned.  When ``True``, the result is linearly interpolated
    between the current hour and the next using the minute offset within the
    hour.  Missing or short arrays gracefully default to ``1.0``.
    """

    if multipliers is None:
        return 1.0
    try:
        length = len(multipliers)
    except Exception:
        return 1.0
    if length == 0:
        return 1.0
    idx = _hour_index(ts_ms, length)
    try:
        base = float(multipliers[idx])
        if not interpolate:
            return base
        nxt = float(multipliers[(idx + 1) % length])
        frac = (ts_ms % HOUR_MS) / float(HOUR_MS)
        return base + (nxt - base) * frac
    except Exception:
        return 1.0


def get_liquidity_multiplier(
    ts_ms: int, liquidity: Sequence[float], *, interpolate: bool = False
) -> float:
    """Convenience wrapper around :func:`get_hourly_multiplier` for liquidity."""

    return get_hourly_multiplier(ts_ms, liquidity, interpolate=interpolate)


def get_latency_multiplier(
    ts_ms: int, latency: Sequence[float], *, interpolate: bool = False
) -> float:
    """Convenience wrapper around :func:`get_hourly_multiplier` for latency."""

    return get_hourly_multiplier(ts_ms, latency, interpolate=interpolate)


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
        dt = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
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
