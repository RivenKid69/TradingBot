# -*- coding: utf-8 -*-
import os
from typing import Any

def _norm(s: Any) -> str:
    return str(s).strip().lower()

def get_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return _norm(v) in ("1", "true", "yes", "y", "on")

def get_int(name: str, default: int = 0) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def get_float(name: str, default: float = 0.0) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

def get_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return str(v) if v is not None else default


def seasonality_enabled(default: bool = True) -> bool:
    """Return whether seasonality features are enabled.

    Controlled via the ``ENABLE_SEASONALITY`` environment variable which
    accepts typical truthy values ("1", "true", "yes", "on"). Defaults to
    ``True`` when the variable is unset.
    """
    return get_bool("ENABLE_SEASONALITY", default)
