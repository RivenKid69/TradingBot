# core_constants.pxd
# Единый источник правды для Cython. Согласован с core_constants.py и core_constants.h.

cdef enum MarketRegime:
    NORMAL = 0
    CHOPPY_FLAT = 1
    STRONG_TREND = 2
    ILLIQUID = 3

# ВНИМАНИЕ: значение должно совпадать с core_constants.py и core_constants.h
DEF PRICE_SCALE = 100
