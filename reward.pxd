# reward.pxd

from lob_state_cython cimport EnvState

cdef enum ClosedReason:  # Importing ClosedReason enum for closed position reasons
    NONE = 0
    ATR_SL_LONG = 1
    ATR_SL_SHORT = 2
    TRAILING_SL_LONG = 3
    TRAILING_SL_SHORT = 4
    STATIC_TP_LONG = 5
    STATIC_TP_SHORT = 6
    BANKRUPTCY = 7
    MAX_DRAWDOWN = 8

cdef double log_return(EnvState* state) nogil noexcept
cdef double potential_phi(EnvState* state) nogil noexcept
cdef double potential_shaping(EnvState* state, double phi_t) nogil noexcept
cdef double trade_frequency_penalty_fn(EnvState* state, int trades_count) nogil noexcept
cdef double event_reward(EnvState* state, ClosedReason closed_reason) nogil noexcept
cdef double compute_reward(EnvState* state, ClosedReason closed_reason, int trades_count) nogil noexcept
