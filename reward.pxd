# cython: language_level=3

from lob_state_cython cimport EnvState


cdef enum ClosedReason:
    NONE = 0
    ATR_SL_LONG = 1
    ATR_SL_SHORT = 2
    TRAILING_SL_LONG = 3
    TRAILING_SL_SHORT = 4
    STATIC_TP_LONG = 5
    STATIC_TP_SHORT = 6
    BANKRUPTCY = 7
    MAX_DRAWDOWN = 8


cdef double log_return(double net_worth, double prev_net_worth) noexcept nogil
cdef double potential_phi(
    double net_worth,
    double peak_value,
    double units,
    double atr,
    double risk_aversion_variance,
    double risk_aversion_drawdown,
    double potential_shaping_coef,
) noexcept nogil
cdef double potential_shaping(double gamma, double last_potential, double phi_t) noexcept nogil
cdef double trade_frequency_penalty_fn(double penalty, int trades_count) noexcept nogil
cdef double event_reward(
    double profit_bonus,
    double loss_penalty,
    double bankruptcy_penalty,
    ClosedReason closed_reason,
) noexcept nogil
cdef double compute_reward_view(
    double net_worth,
    double prev_net_worth,
    double last_potential,
    bint use_legacy_log_reward,
    bint use_potential_shaping,
    double gamma,
    double potential_shaping_coef,
    double units,
    double atr,
    double risk_aversion_variance,
    double peak_value,
    double risk_aversion_drawdown,
    int trades_count,
    double trade_frequency_penalty,
    double last_executed_notional,
    double turnover_penalty_coef,
    double profit_close_bonus,
    double loss_close_penalty,
    double bankruptcy_penalty,
    ClosedReason closed_reason,
    double* out_potential,
) noexcept nogil

cpdef double compute_reward(EnvState state, ClosedReason closed_reason, int trades_count)

