# cython: language_level=3
from lob_state_cython cimport EnvState

cdef enum ClosedReason:
    NONE
    ATR_SL_LONG
    ATR_SL_SHORT
    TRAILING_SL_LONG
    TRAILING_SL_SHORT
    STATIC_TP_LONG
    STATIC_TP_SHORT
    BANKRUPTCY
    MAX_DRAWDOWN

cdef double compute_max_position_frac(EnvState* state) nogil
cdef ClosedReason check_static_atr_stop(EnvState* state) nogil
cdef ClosedReason check_trailing_stop(EnvState* state) nogil
cdef ClosedReason check_take_profit(EnvState* state) nogil
cdef void update_trailing_extrema(EnvState* state) nogil
cdef ClosedReason check_bankruptcy(EnvState* state) nogil
cdef ClosedReason check_max_drawdown(EnvState* state) nogil
cdef ClosedReason apply_close_if_needed(EnvState* state, bint readonly=*) nogil
