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

from libc.math cimport floor, ceil
from lob_state_cython cimport EnvState

cdef double _ticks_to_price(long long ticks, long long price_scale):
    """Convert integer ticks to monetary price."""
    return ticks / <double>price_scale

cdef double compute_max_position_frac(EnvState* state) nogil:
    """Compute maximum allowed position fraction of equity based on dynamic risk profile."""
    cdef double fg = 0.0
    cdef double frac = 0.0
    cdef double range_level = 0.0
    if state.use_dynamic_risk:
        fg = state.fear_greed_value
        # Ensure valid range
        range_level = state.risk_on_level - state.risk_off_level
        if range_level <= 0:
            # If thresholds are equal or mis-ordered, use conservative limit
            return state.max_position_risk_off
        if fg <= state.risk_off_level:
            frac = state.max_position_risk_off
        elif fg >= state.risk_on_level:
            frac = state.max_position_risk_on
        else:
            frac = state.max_position_risk_off + (fg - state.risk_off_level) / range_level * (state.max_position_risk_on - state.max_position_risk_off)
    else:
        frac = state.max_position_risk_on
    return frac

cdef ClosedReason check_static_atr_stop(EnvState* state) nogil:
    """Check if static ATR stop-loss is triggered."""
    if not state.use_atr_stop:
        return ClosedReason.NONE
    if state.units == 0:
        return ClosedReason.NONE
    if state._trailing_active:
        # If trailing stop is active, static ATR stop is no longer used
        return ClosedReason.NONE
    cdef long long last_price_ticks = 0
    # Compute current price in ticks from position value and units
    if state.units > 0:
        last_price_ticks = <long long>ceil(state._position_value * state.price_scale / state.units)
        if last_price_ticks <= state._initial_sl:
            return ClosedReason.ATR_SL_LONG
    elif state.units < 0:
        last_price_ticks = <long long>floor(state._position_value * state.price_scale / state.units)
        if last_price_ticks >= state._initial_sl:
            return ClosedReason.ATR_SL_SHORT
    return ClosedReason.NONE

cdef ClosedReason check_trailing_stop(EnvState* state) nogil:
    """Check if trailing stop-loss is triggered (after trailing active)."""
    if not state.use_trailing_stop or state.units == 0:
        return ClosedReason.NONE
    if not state._trailing_active:
        return ClosedReason.NONE
    cdef long long last_price_ticks = 0
    if state.units > 0:
        last_price_ticks = <long long>ceil(state._position_value * state.price_scale / state.units)
        if last_price_ticks <= state._initial_sl:
            return ClosedReason.TRAILING_SL_LONG
    elif state.units < 0:
        last_price_ticks = <long long>floor(state._position_value * state.price_scale / state.units)
        if last_price_ticks >= state._initial_sl:
            return ClosedReason.TRAILING_SL_SHORT
    return ClosedReason.NONE

cdef ClosedReason check_take_profit(EnvState* state) nogil:
    """Check if take-profit is triggered."""
    if state.units == 0 or state.tp_atr_mult <= 0:
        return ClosedReason.NONE
    cdef long long last_price_ticks = 0
    if state.units > 0:
        last_price_ticks = <long long>ceil(state._position_value * state.price_scale / state.units)
        if last_price_ticks >= state._initial_tp:
            return ClosedReason.STATIC_TP_LONG
    elif state.units < 0:
        last_price_ticks = <long long>floor(state._position_value * state.price_scale / state.units)
        if last_price_ticks <= state._initial_tp:
            return ClosedReason.STATIC_TP_SHORT
    return ClosedReason.NONE

cdef void update_trailing_extrema(EnvState* state) nogil:
    """Update trailing stop extrema and activate trailing stop if conditions met."""
    if not state.use_trailing_stop or state.units == 0:
        return
    cdef long long last_price_ticks = 0
    if state.units > 0:
        # Long position: update highest price reached
        last_price_ticks = <long long>ceil(state._position_value * state.price_scale / state.units)
        if last_price_ticks > state._high_extremum:
            state._high_extremum = last_price_ticks
        if not state._trailing_active:
            # Activate trailing stop if price moved sufficiently above entry
            cdef double activate_threshold = state._entry_price + state._atr_at_entry * state.trailing_atr_mult
            if last_price_ticks >= <long long>ceil(activate_threshold):
                state._trailing_active = True
        if state._trailing_active:
            # Adjust stop-loss upward to trailing level
            cdef double new_stop_level = state._high_extremum - state._atr_at_entry * state.trailing_atr_mult
            cdef long long new_stop_ticks = <long long>floor(new_stop_level)
            if new_stop_ticks > state._initial_sl:
                state._initial_sl = new_stop_ticks
    elif state.units < 0:
        # Short position: update lowest price reached
        last_price_ticks = <long long>floor(state._position_value * state.price_scale / state.units)
        if last_price_ticks < state._low_extremum:
            state._low_extremum = last_price_ticks
        if not state._trailing_active:
            cdef double activate_threshold = state._entry_price - state._atr_at_entry * state.trailing_atr_mult
            if last_price_ticks <= <long long>floor(activate_threshold):
                state._trailing_active = True
        if state._trailing_active:
            cdef double new_stop_level = state._low_extremum + state._atr_at_entry * state.trailing_atr_mult
            cdef long long new_stop_ticks = <long long>ceil(new_stop_level)
            if new_stop_ticks < state._initial_sl:
                state._initial_sl = new_stop_ticks
    return

cdef ClosedReason check_bankruptcy(EnvState* state) nogil:
    """Check if account is bankrupt."""
    if state.net_worth <= state.bankruptcy_threshold:
        # Mark bankrupt
        state.is_bankrupt = True
        return ClosedReason.BANKRUPTCY
    return ClosedReason.NONE

cdef ClosedReason check_max_drawdown(EnvState* state) nogil:
    """Check if max drawdown limit is hit."""
    # Update peak net_worth
    if state.net_worth > state.peak_value:
        state.peak_value = state.net_worth
    if state.net_worth <= state.peak_value * (1.0 - state.max_drawdown):
        return ClosedReason.MAX_DRAWDOWN
    return ClosedReason.NONE

cdef ClosedReason apply_close_if_needed(EnvState* state, bint readonly=False) nogil:
    """Apply position close if any risk or TP/SL condition is triggered. Returns the close reason code (or NONE).

    If ``readonly`` is True the original ``state`` remains unmodified.
    """
    cdef EnvState local_state
    if readonly:
        local_state = state[0]
        state = &local_state
    cdef ClosedReason reason = ClosedReason.NONE
    cdef ClosedReason res_check
    # Check critical conditions first (bankruptcy, drawdown)
    res_check = check_bankruptcy(state)
    if res_check != ClosedReason.NONE:
        reason = res_check
    else:
        res_check = check_max_drawdown(state)
        if res_check != ClosedReason.NONE:
            reason = res_check
    # If not already closed by risk limits, check SL/TP
    if reason == ClosedReason.NONE and state.units != 0:
        # Update trailing extrema each step
        if state.use_trailing_stop:
            update_trailing_extrema(state)
        # Static ATR stop (only if trailing not yet active)
        res_check = check_static_atr_stop(state)
        if res_check != ClosedReason.NONE:
            reason = res_check
        else:
            # Trailing stop check
            res_check = check_trailing_stop(state)
            if res_check != ClosedReason.NONE:
                reason = res_check
            else:
                # Take-profit check
                res_check = check_take_profit(state)
                if res_check != ClosedReason.NONE:
                    reason = res_check
    # Apply close actions if triggered
    if reason != ClosedReason.NONE:
        # Close the position and update state
        if state.units != 0:
            # Calculate P/L realization
            # (Position value already reflects current price)
            state.cash += state._position_value
            # Charge taker fee for closing trade
            state.cash -= abs(state._position_value) * state.taker_fee
        # Reset position-related fields
        state.units = 0
        state._position_value = 0.0
        state.prev_net_worth = state.net_worth
        # Update net_worth after closing (cash + no position)
        state.net_worth = state.cash
        # Handle bankruptcy separately
        if reason == ClosedReason.BANKRUPTCY:
            # Zero out state on bankruptcy
            state.cash = 0.0
            state.net_worth = 0.0
            state.is_bankrupt = True
        # Reset trailing stop tracking
        state._trailing_active = False
        state._high_extremum = 0
        state._low_extremum = 0
    return reason
