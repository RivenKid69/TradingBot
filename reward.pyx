# reward.pyx

from libc.math cimport log, tanh, fabs  # use C math for performance (no Python)
# Import definitions from reward.pxd
cdef class EnvState  # (Ensure consistent with EnvState declared elsewhere)
cdef enum ClosedReason:
    NONE, ATR_SL_LONG, ATR_SL_SHORT, TRAILING_SL_LONG, TRAILING_SL_SHORT, STATIC_TP_LONG, STATIC_TP_SHORT, BANKRUPTCY, MAX_DRAWDOWN

cdef inline double _safe_div(double a, double b) nogil:
    """Safe division to avoid division by zero, returns 0.0 if denominator is 0"""
    return a / b if b != 0.0 else 0.0

cdef double log_return(EnvState* state) nogil:
    """Logarithmic return: log(net_worth / prev_net_worth) with clipping:contentReference[oaicite:0]{index=0}"""
    cdef double ratio = 0.0
    # Compute ratio and guard against non-positive values (clip to 1e-12 to avoid log(0))
    if state.prev_net_worth > 0.0:
        ratio = state.net_worth / state.prev_net_worth
    # Ensure ratio is at least 1e-12 (avoid log of zero or negative)
    if ratio < 1e-12:
        ratio = 1e-12
    return log(ratio)

cdef double potential_phi(EnvState* state) nogil:
    """Compute potential Φ(state) combining risk variance and drawdown penalties:contentReference[oaicite:1]{index=1}"""
    cdef double phi_variance = 0.0
    cdef double phi_drawdown = 0.0
    # Penalty for open risk: |units| * ATR / net_worth, scaled by risk_aversion_variance:contentReference[oaicite:2]{index=2}
    if state.net_worth > 0.0:
        # Assume state has a method or attribute to get current ATR value
        cdef double atr_value = 0.0
        # If EnvState provides current ATR via a method or attribute, use it (for example: state.get_atr() or state.atr)
        # Otherwise, atr_value remains 0 (no penalty if ATR not available)
        try:
            atr_value = state.get_atr(state.step_idx)
        except:
            try:
                atr_value = state.atr  # if current ATR stored as attribute
            except:
                atr_value = 0.0
        phi_variance = state.risk_aversion_variance * (fabs(state.units) * atr_value / state.net_worth)
    else:
        phi_variance = 0.0
    # Penalty for drawdown from peak: (1 - net_worth / peak_value), scaled by risk_aversion_drawdown:contentReference[oaicite:3]{index=3}
    if state.peak_value > 0.0:
        phi_drawdown = state.risk_aversion_drawdown * (1.0 - _safe_div(state.net_worth, state.peak_value))
    else:
        # If peak_value is 0 (unlikely, usually initial equity), treat drawdown penalty as 0
        phi_drawdown = 0.0
    # Combine and apply smoothing tanh, then scale by potential_shaping_coef:contentReference[oaicite:4]{index=4}
    cdef double raw_potential = phi_variance + phi_drawdown
    cdef double phi = state.potential_shaping_coef * tanh(raw_potential)
    return phi

cdef double potential_shaping(EnvState* state, double phi_t) nogil:
    """Compute potential-based shaping reward: γ * Φ_t - Φ_{t-1}, and update last_potential:contentReference[oaicite:5]{index=5}"""
    cdef double shaping_reward = 0.0
    if state.use_potential_shaping:
        shaping_reward = state.gamma * phi_t - state.last_potential  # γ·Φ_t – Φ_{t-1}
    else:
        shaping_reward = 0.0
    # Update last_potential in state to current Φ_t for next step
    state.last_potential = phi_t
    return shaping_reward

cdef double trade_frequency_penalty_fn(EnvState* state, int trades_count) nogil:
    """Penalty for trade frequency: trade_frequency_penalty * number of trades this step:contentReference[oaicite:6]{index=6}"""
    if trades_count <= 0:
        return 0.0
    return state.trade_frequency_penalty * trades_count

cdef double event_reward(EnvState* state, ClosedReason closed_reason) nogil:
    """Event-based bonus/penalty for closing a position:contentReference[oaicite:7]{index=7}"""
    if closed_reason == ClosedReason.NONE:
        return 0.0
    # If closed in profit (e.g. take-profit), grant bonus; if closed in loss (e.g. stop-loss), apply penalty:contentReference[oaicite:8]{index=8}
    if closed_reason == ClosedReason.STATIC_TP_LONG or closed_reason == ClosedReason.STATIC_TP_SHORT:
        # Take-profit closure (profit scenario)
        return state.profit_close_bonus
    else:
        # Any other closure (stop-loss, trailing stop, bankruptcy, drawdown) treated as loss scenario
        return -state.loss_close_penalty

cdef double compute_reward(EnvState* state, ClosedReason closed_reason, int trades_count) nogil:
    """Compute total reward for the step: log_return + γ·Φ_t – Φ_{t-1} + frequency penalty + event reward:contentReference[oaicite:9]{index=9}:contentReference[oaicite:10]{index=10}"""
    cdef double r_log = log_return(state)
    cdef double phi_t = 0.0
    cdef double shaping_component = 0.0
    if state.use_potential_shaping:
        # Compute current potential Φ_t
        phi_t = potential_phi(state)
        # Compute shaping reward = γ·Φ_t – Φ_{t-1}
        shaping_component = state.gamma * phi_t - state.last_potential
        # Update last_potential for next step
        state.last_potential = phi_t
    else:
        shaping_component = 0.0
    # Frequency penalty for trades this step
    cdef double freq_penalty = 0.0
    if trades_count > 0:
        freq_penalty = state.trade_frequency_penalty * trades_count
    # Event-based reward/penalty for closing position
    cdef double evt_reward = event_reward(state, closed_reason)
    # Total reward
    return r_log + shaping_component + (-freq_penalty) + evt_reward
