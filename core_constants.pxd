# core_constants.pxd
# Единый источник правды для Cython. Согласован с core_constants.py и core_constants.h.

cdef extern from "core_constants.h":
    cdef enum MarketRegime:
        NORMAL
        CHOPPY_FLAT
        STRONG_TREND
        ILLIQUID

# ВНИМАНИЕ: значение должно совпадать с core_constants.py и core_constants.h
DEF PRICE_SCALE = 100
