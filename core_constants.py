# -*- coding: utf-8 -*-
"""
core_constants.py
Единый источник правды для констант и енамов на Python-уровне.
Согласован с core_constants.pxd и core_constants.h.

Договорённости:
- PRICE_SCALE: количество тиков на единицу цены (100 означает шаг 0.01 при цене в единицах).
- MarketRegime: коды режимов рынка согласованы с C++/Cython.
- Дополнительные константы (лимиты и коды сторон/типов) используются только на Python-уровне.
"""

from __future__ import annotations
from enum import IntEnum

# ВНИМАНИЕ: это значение должно совпадать с core_constants.pxd и core_constants.h
PRICE_SCALE: int = 100  # шаг цены = 1/PRICE_SCALE

class MarketRegime(IntEnum):
    NORMAL = 0
    CHOPPY_FLAT = 1
    STRONG_TREND = 2
    ILLIQUID = 3

# Лимиты для симуляции (Python-only)
DEFAULT_MAX_TRADES_PER_STEP: int = 10000
DEFAULT_MAX_GENERATED_EVENTS_PER_TYPE: int = 5000

# Коды типов ордеров
ORDER_TYPE_MARKET: int = 1
ORDER_TYPE_LIMIT: int = 2

# Коды сторон сделки
SIDE_BUY: int = 1
SIDE_SELL: int = -1

# Флаги роли агента в сделке
AGENT_MAKER: int = 1
AGENT_TAKER: int = 0
