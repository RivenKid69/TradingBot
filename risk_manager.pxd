# cython: language_level=3
cpdef enum ClosedReason:
    NONE
    ATR_SL_LONG
    ATR_SL_SHORT
    TRAILING_SL_LONG
    TRAILING_SL_SHORT
    STATIC_TP_LONG
    STATIC_TP_SHORT
    BANKRUPTCY
    MAX_DRAWDOWN

