# -*- coding: utf-8 -*-
"""Utility helpers for working with timeframes and timestamps."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Union

# Supported timeframe identifiers
_VALID_TF = {
    "1s", "5s", "10s", "15s", "30s",
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d",
}


def ensure_timeframe(tf: str) -> str:
    """Validate and normalise timeframe string."""
    tf = str(tf).lower()
    if tf not in _VALID_TF:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return tf


def timeframe_to_ms(tf: str) -> int:
    """Convert a timeframe like '1m' or '1h' to milliseconds."""
    tf = ensure_timeframe(tf)
    unit = tf[-1]
    value = int(tf[:-1])
    mult = {"s": 1000, "m": 60 * 1000, "h": 60 * 60 * 1000, "d": 24 * 60 * 60 * 1000}[unit]
    return value * mult


def to_ms(dt: Union[int, float, str, datetime]) -> int:
    """Convert various datetime representations to Unix milliseconds."""
    if isinstance(dt, int):
        return dt
    if isinstance(dt, float):
        return int(dt)
    if isinstance(dt, str):
        try:
            return int(datetime.fromisoformat(dt.replace("Z", "+00:00")).timestamp() * 1000)
        except Exception as e:  # pragma: no cover - defensive
            raise ValueError(f"Cannot parse datetime string: {dt}") from e
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    raise TypeError(f"Unsupported datetime type: {type(dt)}")


def binance_tf(tf: str) -> str:
    """Convert project timeframe to Binance kline interval string."""
    tf = ensure_timeframe(tf)
    return tf.lower()

