# cython: language_level=3, language=c++, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
"""Reward shaping utilities shared between Python and Cython environments."""

from libc.math cimport fabs, log, tanh

from lob_state_cython cimport EnvState
from risk_enums cimport ClosedReason


cdef inline double _clamp(double value, double lower, double upper) noexcept nogil:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


cdef double log_return(double net_worth, double prev_net_worth) noexcept nogil:
    cdef double ratio
    if prev_net_worth <= 0.0 or net_worth <= 0.0:
        return 0.0
    ratio = net_worth / (prev_net_worth + 1e-9)
    ratio = _clamp(ratio, 0.1, 10.0)
    return log(ratio)


cdef double potential_phi(
    double net_worth,
    double peak_value,
    double units,
    double atr,
    double risk_aversion_variance,
    double risk_aversion_drawdown,
    double potential_shaping_coef,
) noexcept nogil:
    cdef double risk_penalty = 0.0
    cdef double dd_penalty = 0.0

    if net_worth > 1e-9 and atr > 0.0 and units != 0.0:
        risk_penalty = -risk_aversion_variance * fabs(units) * atr / (fabs(net_worth) + 1e-9)

    if peak_value > 1e-9:
        dd_penalty = -risk_aversion_drawdown * (peak_value - net_worth) / peak_value

    return potential_shaping_coef * tanh(risk_penalty + dd_penalty)


cdef double potential_shaping(double gamma, double last_potential, double phi_t) noexcept nogil:
    return gamma * phi_t - last_potential


cdef double trade_frequency_penalty_fn(double penalty, int trades_count) noexcept nogil:
    if penalty <= 0.0 or trades_count <= 0:
        return 0.0
    return penalty * trades_count


cdef double event_reward(
    double profit_bonus,
    double loss_penalty,
    double bankruptcy_penalty,
    ClosedReason closed_reason,
) noexcept nogil:
    if closed_reason == ClosedReason.NONE:
        return 0.0

    if closed_reason == ClosedReason.BANKRUPTCY:
        if bankruptcy_penalty > 0.0:
            return -bankruptcy_penalty
        return -loss_penalty

    if closed_reason == ClosedReason.STATIC_TP_LONG or closed_reason == ClosedReason.STATIC_TP_SHORT:
        return profit_bonus

    return -loss_penalty


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
) noexcept nogil:
    cdef double reward = net_worth - prev_net_worth
    cdef double phi_t = 0.0
    cdef double base_cost_bps = 0.0
    cdef double total_cost_bps = 0.0
    cdef double participation = 0.0
    cdef double impact_exp = 0.0

    if use_legacy_log_reward:
        reward += log_return(net_worth, prev_net_worth)
        if use_potential_shaping:
            phi_t = potential_phi(
                net_worth,
                peak_value,
                units,
                atr,
                risk_aversion_variance,
                risk_aversion_drawdown,
                potential_shaping_coef,
            )
            reward += potential_shaping(gamma, last_potential, phi_t)
    elif use_potential_shaping:
        phi_t = potential_phi(
            net_worth,
            peak_value,
            units,
            atr,
            risk_aversion_variance,
            risk_aversion_drawdown,
            potential_shaping_coef,
        )

    reward -= trade_frequency_penalty_fn(trade_frequency_penalty, trades_count)

    cdef double trade_notional = fabs(last_executed_notional)
    if trade_notional > 0.0:
        base_cost_bps = spot_cost_taker_fee_bps + spot_cost_half_spread_bps
        total_cost_bps = base_cost_bps if base_cost_bps > 0.0 else 0.0
        if spot_cost_impact_coeff > 0.0 and spot_cost_adv_quote > 0.0:
            participation = trade_notional / spot_cost_adv_quote
            if participation > 0.0:
                impact_exp = spot_cost_impact_exponent if spot_cost_impact_exponent > 0.0 else 1.0
                total_cost_bps += spot_cost_impact_coeff * participation ** impact_exp
        if total_cost_bps > 0.0:
            reward -= trade_notional * total_cost_bps * 1e-4

    if turnover_penalty_coef > 0.0 and last_executed_notional > 0.0:
        reward -= turnover_penalty_coef * last_executed_notional

    reward += event_reward(
        profit_close_bonus,
        loss_close_penalty,
        bankruptcy_penalty,
        closed_reason,
    )

    if out_potential != <double*>0:
        out_potential[0] = phi_t

    return reward


cpdef double compute_reward(EnvState state, ClosedReason closed_reason, int trades_count):
    cdef double new_potential = 0.0
    cdef double reward = compute_reward_view(
        state.net_worth,
        state.prev_net_worth,
        state.last_potential,
        state.use_legacy_log_reward,
        state.use_potential_shaping,
        state.gamma,
        state.potential_shaping_coef,
        state.units,
        state.last_bar_atr,
        state.risk_aversion_variance,
        state.peak_value,
        state.risk_aversion_drawdown,
        trades_count,
        state.trade_frequency_penalty,
        state.last_executed_notional,
        state.spot_cost_taker_fee_bps,
        state.spot_cost_half_spread_bps,
        state.spot_cost_impact_coeff,
        state.spot_cost_impact_exponent,
        state.spot_cost_adv_quote,
        state.turnover_penalty_coef,
        state.profit_close_bonus,
        state.loss_close_penalty,
        state.bankruptcy_penalty,
        closed_reason,
        &new_potential,
    )

    if state.use_potential_shaping:
        state.last_potential = new_potential
    else:
        state.last_potential = 0.0

    return reward
