# -*- coding: utf-8 -*-
"""
core_constants.py
Единый источник правды для констант и енамов на Python-уровне.
Согласован с core_constants.pxd и core_constants.h.

Договорённости:
- PRICE_SCALE: количество тиков на единицу цены (100 означает шаг 0.01 при цене в единицах).
- MarketRegime: коды режимов рынка согласованы с C++/Cython.
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