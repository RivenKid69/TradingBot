# -*- coding: utf-8 -*-
"""
action_proto.py
Структурированное представление торгового действия, совместимое с legacy-боксом [pos_frac, order_flag].

Определения:
- ActionType: IntEnum со значениями, совместимыми со старым кодом (0=HOLD,1=MARKET,2=LIMIT,3=CANCEL_ALL).
- ActionProto: dataclass с полями action_type, volume_frac, price_offset_ticks, tif, client_tag.
  volume_frac в диапазоне [-1.0, 1.0]; знак определяет сторону: >0 BUY, <0 SELL; 0 допускается для HOLD/CANCEL_ALL.
  price_offset_ticks — целочисленный сдвиг цены от референсной (mid/last) в тиках для LIMIT; 0 означает по текущей.
  tif — строка TimeInForce из core_models.TimeInForce.value ('GTC'|'IOC'|'FOK').

Методы:
- from_legacy_box(arr): принимает [pos_frac, order_flag] и строит ActionProto.
- to_dict(): сериализация в словарь с передачей числового кода action_type в поле 'type'.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Any, Mapping, Sequence

# Сохраняем числовые коды для обратной совместимости с risk_guard и медиацией.
class ActionType(IntEnum):
    HOLD = 0
    MARKET = 1
    LIMIT = 2
    CANCEL_ALL = 3


@dataclass(frozen=True)
class ActionProto:
    action_type: ActionType
    volume_frac: float
    price_offset_ticks: int = 0
    ttl_steps: int = 0
    abs_price: float | None = None
    tif: str = "GTC"           # 'GTC'|'IOC'|'FOK'
    client_tag: str | None = None

    @staticmethod
    def _coerce_pos(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            raise ValueError(f"pos_frac должен быть числом, получено {x!r}")
        if not (-1.0 <= v <= 1.0):
            raise ValueError(f"pos_frac вне диапазона [-1.0,1.0]: {v}")
        return v

    @staticmethod
    def _coerce_flag(x: Any) -> int:
        try:
            iv = int(x)
        except Exception:
            raise ValueError(f"order_flag должен быть целым, получено {x!r}")
        if iv not in (0, 1, 2, 3):
            raise ValueError(f"order_flag вне допустимых значений {0,1,2,3}: {iv}")
        return iv

    @classmethod
    def from_legacy_box(cls, arr: Sequence[Any]) -> "ActionProto":
        if not isinstance(arr, (list, tuple)) or len(arr) < 2:
            raise ValueError("Ожидался массив вида [pos_frac, order_flag]")
        pos = cls._coerce_pos(arr[0])
        flag = cls._coerce_flag(arr[1])
        if flag == 0:
            return cls(action_type=ActionType.HOLD, volume_frac=0.0)
        if flag == 1:
            return cls(action_type=ActionType.MARKET, volume_frac=pos)
        if flag == 2:
            return cls(action_type=ActionType.LIMIT, volume_frac=pos)
        # flag == 3
        return cls(action_type=ActionType.CANCEL_ALL, volume_frac=0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        # совместимость со старым кодом: числовое поле 'type' и плоские значения
        d["type"] = int(self.action_type)
        return d
