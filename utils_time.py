"""Assorted time utilities.

The hour-of-week index assumes ``0 = Monday 00:00 UTC``.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional, Sequence, Union, Dict
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

# Re-export shared time utilities to avoid duplicate implementations.
from utils.time import hour_of_week, HOUR_MS, HOURS_IN_WEEK

# Clamp limits applied to liquidity and latency seasonality multipliers.
SEASONALITY_MULT_MIN = 0.1
SEASONALITY_MULT_MAX = 10.0


def load_hourly_seasonality(
    path: str,
    *keys: str,
    symbol: str | None = None,
    expected_hash: str | None = None,
) -> np.ndarray | None:
    """Load hourly multipliers array from JSON file.

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
            # Allow new structure {"SYMBOL": {"latency": [...]}}
            if symbol and symbol in data:
                data = data[symbol]
            for k in keys:
                if isinstance(data, dict) and k in data:
                    data = data[k]
                    break
        arr = np.asarray(data, dtype=float)
        if arr.shape[0] == HOURS_IN_WEEK:
            if any(k in {"liquidity", "latency"} for k in keys):
                arr = np.clip(arr, SEASONALITY_MULT_MIN, SEASONALITY_MULT_MAX)
            return arr
    except Exception:
        return None
    return None


def load_seasonality(path: str) -> Dict[str, np.ndarray]:
    """Load all available seasonality arrays from ``path``.

    The JSON file is expected to contain arrays of length :data:`HOURS_IN_WEEK`
    (168). It may either expose the arrays at the top level, or nest them under
    an instrument symbol. Only keys with list values of the correct length are
    returned.

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
        logging.getLogger(__name__).info(
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
        for key in ("liquidity", "latency", "spread", "multipliers"):
            if key in obj:
                arr = np.asarray(obj[key], dtype=float)
                if arr.shape[0] != HOURS_IN_WEEK:
                    raise ValueError(
                        f"Seasonality array '{key}' must have length {HOURS_IN_WEEK}"
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


def _hour_index(ts_ms: int, length: int) -> int:
    """Return hour-of-week index clamped to ``length``.

    The helper mirrors :func:`utils.time.hour_of_week` but additionally
    wraps the result by ``length`` to support arrays shorter than a full
    week. ``length`` defaults to :data:`HOURS_IN_WEEK`.
    """

    hour = hour_of_week(int(ts_ms))
    if length:
        hour %= int(length)
    return int(hour)


def get_hourly_multiplier(ts_ms: int, multipliers: Sequence[float]) -> float:
    """Return hourly multiplier for ``ts_ms`` from ``multipliers``.

    Missing or short arrays gracefully default to ``1.0``.
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
        return float(multipliers[idx])
    except Exception:
        return 1.0


def get_liquidity_multiplier(ts_ms: int, liquidity: Sequence[float]) -> float:
    """Convenience wrapper around :func:`get_hourly_multiplier` for liquidity."""

    return get_hourly_multiplier(ts_ms, liquidity)


def get_latency_multiplier(ts_ms: int, latency: Sequence[float]) -> float:
    """Convenience wrapper around :func:`get_hourly_multiplier` for latency."""

    return get_hourly_multiplier(ts_ms, latency)


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
