# legacy_bridge.py
"""
Мост совместимости: из «устаревшего бокса» [pos_frac, order_flag]
в строгий ActionProto.

pos_frac: float в диапазоне [-1.0, 1.0]
order_flag: int {0=HOLD, 1=MARKET, 2=LIMIT}
"""

from __future__ import annotations
from typing import Sequence
import warnings

from action_proto import ActionProto, ActionType


def _coerce_pos(x) -> float:
    try:
        v = float(x)
    except Exception:
        raise ValueError(f"pos_frac must be float-like, got {type(x).__name__}")
    # мягкий клип, чтобы не взрывать старые пайплайны
    if v < -1.0 or v > 1.0:
        warnings.warn(f"pos_frac={v} is outside [-1,1], clipping.", RuntimeWarning, stacklevel=2)
        v = max(-1.0, min(1.0, v))
    return v


def _coerce_flag(x) -> int:
    try:
        f = int(x)
    except Exception:
        raise ValueError(f"order_flag must be int-like, got {type(x).__name__}")
    if f not in (0, 1, 2):
        raise ValueError(f"Bad legacy flag value: {f} (expected 0=HOLD,1=MARKET,2=LIMIT)")
    return f


def from_legacy_box(arr: Sequence) -> ActionProto:
    """
    Преобразует массив формата [pos_frac, order_flag] в ActionProto.
    Возвращает строго типизированный ActionProto.

    >>> from_legacy_box([0.5, 1]).action_type == ActionType.MARKET
    True
    """
    if not isinstance(arr, (list, tuple)) or len(arr) != 2:
        raise ValueError("legacy box must be a 2-element list/tuple: [pos_frac, order_flag]")

    pos = _coerce_pos(arr[0])
    flag = _coerce_flag(arr[1])

    if flag == 0:
        warnings.warn("Using legacy HOLD action.", DeprecationWarning, stacklevel=2)
        return ActionProto(action_type=ActionType.HOLD, volume_frac=0.0)

    if flag == 1:
        warnings.warn("Using legacy MARKET action.", DeprecationWarning, stacklevel=2)
        return ActionProto(action_type=ActionType.MARKET, volume_frac=pos)

    # flag == 2
    warnings.warn("Using legacy LIMIT action.", DeprecationWarning, stacklevel=2)
    return ActionProto(action_type=ActionType.LIMIT, volume_frac=pos)
