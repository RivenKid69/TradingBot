# cython: language_level=3, language=c++

from lob_state_cython cimport EnvState
from risk_enums cimport ClosedReason


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
    double spot_cost_taker_fee_bps,
    double spot_cost_half_spread_bps,
    double spot_cost_impact_coeff,
    double spot_cost_impact_exponent,
    double spot_cost_adv_quote,
    double turnover_penalty_coef,
    double profit_close_bonus,
    double loss_close_penalty,
    double bankruptcy_penalty,
    ClosedReason closed_reason,
    double* out_potential,
) noexcept nogil

cpdef double compute_reward(EnvState state, ClosedReason closed_reason, int trades_count)

