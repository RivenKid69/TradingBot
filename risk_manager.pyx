
from risk_manager cimport ClosedReason
from libc.math cimport fmax, fmin, fabs
from lob_state_cython cimport EnvState
from lob_state_cython import EnvState as PyEnvState


cdef inline double _current_fill_price(EnvState* state) nogil:
    """Return the effective fill price for the current position."""
    if state.units == 0:
        return 0.0
    return state._position_value / state.units

cdef double compute_max_position_frac(EnvState* state) nogil:
    """Compute maximum allowed position fraction of equity based on dynamic risk profile."""
    cdef double fg = 0.0
    cdef double frac = 0.0
    cdef double range_level = 0.0
    if state.use_dynamic_risk:
        fg = state.fear_greed_value
        range_level = state.risk_on_level - state.risk_off_level
        if range_level <= 0.0:
            return state.max_position_risk_off
        if fg <= state.risk_off_level:
            frac = state.max_position_risk_off
        elif fg >= state.risk_on_level:
            frac = state.max_position_risk_on
        else:
            frac = state.max_position_risk_off + (fg - state.risk_off_level) / range_level * (state.max_position_risk_on - state.max_position_risk_off)
    else:
        frac = state.max_position_risk_on
    if state.max_position_risk_off > state.max_position_risk_on:
        frac = fmax(state.max_position_risk_on, fmin(frac, state.max_position_risk_off))
    else:
        frac = fmax(state.max_position_risk_off, fmin(frac, state.max_position_risk_on))
    return frac

cdef ClosedReason check_static_atr_stop(EnvState* state) nogil:
    """Check if static ATR stop-loss is triggered."""
    if not state.use_atr_stop or state.units == 0 or state._trailing_active:
        return ClosedReason.NONE
    cdef double last_price = _current_fill_price(state)
    if state.units > 0:
        if state._initial_sl > 0.0 and last_price <= state._initial_sl:
            return ClosedReason.ATR_SL_LONG
    elif state._initial_sl > 0.0 and last_price >= state._initial_sl:
        return ClosedReason.ATR_SL_SHORT
    return ClosedReason.NONE

cdef ClosedReason check_trailing_stop(EnvState* state) nogil:
    """Check if trailing stop-loss is triggered (after trailing active)."""
    if not state.use_trailing_stop or state.units == 0 or not state._trailing_active:
        return ClosedReason.NONE
    cdef double last_price = _current_fill_price(state)
    if state.units > 0:
        if state._initial_sl > 0.0 and last_price <= state._initial_sl:
            return ClosedReason.TRAILING_SL_LONG
    elif state._initial_sl > 0.0 and last_price >= state._initial_sl:
        return ClosedReason.TRAILING_SL_SHORT
    return ClosedReason.NONE

cdef ClosedReason check_take_profit(EnvState* state) nogil:
    """Check if take-profit is triggered."""
    if state.units == 0 or state.tp_atr_mult <= 0.0:
        return ClosedReason.NONE
    cdef double last_price = _current_fill_price(state)
    if state.units > 0:
        if state._initial_tp > 0.0 and last_price >= state._initial_tp:
            return ClosedReason.STATIC_TP_LONG
    elif state._initial_tp > 0.0 and last_price <= state._initial_tp:
        return ClosedReason.STATIC_TP_SHORT
    return ClosedReason.NONE

cdef void update_trailing_extrema(EnvState* state) nogil:
    """Update trailing stop extrema and activate trailing stop if conditions met."""
    if not state.use_trailing_stop or state.units == 0:
        return
    cdef double last_price = _current_fill_price(state)
    if state.units > 0:
        if state._high_extremum < 0.0 or last_price > state._high_extremum:
            state._high_extremum = last_price
        if state._low_extremum < 0.0:
            state._low_extremum = last_price
        else:
            state._low_extremum = fmin(state._low_extremum, last_price)
        if not state._trailing_active:
            cdef double activate_threshold = state._entry_price + state._atr_at_entry * state.trailing_atr_mult
            if state._entry_price > 0.0 and state._atr_at_entry > 0.0 and last_price >= activate_threshold:
                state._trailing_active = True
        if state._trailing_active:
            cdef double new_stop_level = state._high_extremum - state._atr_at_entry * state.trailing_atr_mult
            if new_stop_level > 0.0 and (state._initial_sl <= 0.0 or new_stop_level > state._initial_sl):
                state._initial_sl = new_stop_level
    else:
        if state._low_extremum < 0.0 or last_price < state._low_extremum:
            state._low_extremum = last_price
        if state._high_extremum < 0.0:
            state._high_extremum = last_price
        else:
            state._high_extremum = fmax(state._high_extremum, last_price)
        if not state._trailing_active:
            cdef double activate_threshold = state._entry_price - state._atr_at_entry * state.trailing_atr_mult
            if state._entry_price > 0.0 and state._atr_at_entry > 0.0 and last_price <= activate_threshold:
                state._trailing_active = True
        if state._trailing_active:
            cdef double new_stop_level = state._low_extremum + state._atr_at_entry * state.trailing_atr_mult
            if new_stop_level > 0.0 and (state._initial_sl <= 0.0 or new_stop_level < state._initial_sl):
                state._initial_sl = new_stop_level


from risk_manager cimport ClosedReason
from libc.math cimport fmax, fmin, fabs
from lob_state_cython cimport EnvState
from lob_state_cython import EnvState as PyEnvState


cdef inline double _current_fill_price(EnvState* state) nogil:
    """Return the effective fill price for the current position."""
    if state.units == 0:
        return 0.0
    return state._position_value / state.units

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
        range_level = state.risk_on_level - state.risk_off_level
        if range_level <= 0.0:
            return state.max_position_risk_off
        if fg <= state.risk_off_level:
            frac = state.max_position_risk_off
        elif fg >= state.risk_on_level:
            frac = state.max_position_risk_on
        else:
            frac = state.max_position_risk_off + (fg - state.risk_off_level) / range_level * (state.max_position_risk_on - state.max_position_risk_off)
    else:
        frac = state.max_position_risk_on
    if state.max_position_risk_off > state.max_position_risk_on:
        frac = fmax(state.max_position_risk_on, fmin(frac, state.max_position_risk_off))
    else:
        frac = fmax(state.max_position_risk_off, fmin(frac, state.max_position_risk_on))
    return frac

cdef ClosedReason check_static_atr_stop(EnvState* state) nogil:
    """Check if static ATR stop-loss is triggered."""
    if not state.use_atr_stop or state.units == 0 or state._trailing_active:
        return ClosedReason.NONE
    cdef double last_price = _current_fill_price(state)
    if state.units > 0:
        if state._initial_sl > 0.0 and last_price <= state._initial_sl:
            return ClosedReason.ATR_SL_LONG
    elif state._initial_sl > 0.0 and last_price >= state._initial_sl:
        return ClosedReason.ATR_SL_SHORT
    return ClosedReason.NONE

cdef ClosedReason check_trailing_stop(EnvState* state) nogil:
    """Check if trailing stop-loss is triggered (after trailing active)."""
    if not state.use_trailing_stop or state.units == 0 or not state._trailing_active:
        return ClosedReason.NONE
    cdef double last_price = _current_fill_price(state)
    if state.units > 0:
        if state._initial_sl > 0.0 and last_price <= state._initial_sl:
            return ClosedReason.TRAILING_SL_LONG
    elif state._initial_sl > 0.0 and last_price >= state._initial_sl:
        return ClosedReason.TRAILING_SL_SHORT
    return ClosedReason.NONE

cdef ClosedReason check_take_profit(EnvState* state) nogil:
    """Check if take-profit is triggered."""
    if state.units == 0 or state.tp_atr_mult <= 0.0:
        return ClosedReason.NONE
    cdef double last_price = _current_fill_price(state)
    if state.units > 0:
        if state._initial_tp > 0.0 and last_price >= state._initial_tp:
            return ClosedReason.STATIC_TP_LONG
    elif state._initial_tp > 0.0 and last_price <= state._initial_tp:
        return ClosedReason.STATIC_TP_SHORT
    return ClosedReason.NONE

cdef void update_trailing_extrema(EnvState* state) nogil:
    """Update trailing stop extrema and activate trailing stop if conditions met."""
    if not state.use_trailing_stop or state.units == 0:
        return
    cdef double last_price = _current_fill_price(state)
    if state.units > 0:
        if state._high_extremum < 0.0 or last_price > state._high_extremum:
            state._high_extremum = last_price
        if state._low_extremum < 0.0:
            state._low_extremum = last_price
        else:
            state._low_extremum = fmin(state._low_extremum, last_price)
        if not state._trailing_active:
            cdef double activate_threshold = state._entry_price + state._atr_at_entry * state.trailing_atr_mult
            if state._entry_price > 0.0 and state._atr_at_entry > 0.0 and last_price >= activate_threshold:
                state._trailing_active = True
        if state._trailing_active:
            cdef double new_stop_level = state._high_extremum - state._atr_at_entry * state.trailing_atr_mult
            if new_stop_level > 0.0 and (state._initial_sl <= 0.0 or new_stop_level > state._initial_sl):
                state._initial_sl = new_stop_level
    else:
        if state._low_extremum < 0.0 or last_price < state._low_extremum:
            state._low_extremum = last_price
        if state._high_extremum < 0.0:
            state._high_extremum = last_price
        else:
            state._high_extremum = fmax(state._high_extremum, last_price)
        if not state._trailing_active:
            cdef double activate_threshold = state._entry_price - state._atr_at_entry * state.trailing_atr_mult
            if state._entry_price > 0.0 and state._atr_at_entry > 0.0 and last_price <= activate_threshold:
                state._trailing_active = True
        if state._trailing_active:
            cdef double new_stop_level = state._low_extremum + state._atr_at_entry * state.trailing_atr_mult
            if new_stop_level > 0.0 and (state._initial_sl <= 0.0 or new_stop_level < state._initial_sl):
                state._initial_sl = new_stop_level


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


cdef ClosedReason _apply_close_if_needed_impl(EnvState* state, bint readonly=False) nogil:
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
        if reason == ClosedReason.ATR_SL_LONG or reason == ClosedReason.ATR_SL_SHORT:
            state.atr_stop_trigger_count += 1
        elif reason == ClosedReason.TRAILING_SL_LONG or reason == ClosedReason.TRAILING_SL_SHORT:
            state.trailing_stop_trigger_count += 1
        elif reason == ClosedReason.STATIC_TP_LONG or reason == ClosedReason.STATIC_TP_SHORT:
            state.tp_trigger_count += 1
        # Close the position and update state
        if state.units != 0:
            # Calculate P/L realization
            # (Position value already reflects current price)
            state.cash += state._position_value
            # Charge taker fee for closing trade
            state.cash -= fabs(state._position_value) * state.taker_fee
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
        state._high_extremum = -1.0
        state._low_extremum = -1.0
        state._max_price_since_entry = -1.0
        state._min_price_since_entry = -1.0
        state._initial_sl = -1.0
        state._initial_tp = -1.0
        state._atr_at_entry = -1.0
        state._entry_price = -1.0
    return reason


cdef ClosedReason apply_close_if_needed(EnvState* state, bint readonly=False) nogil:
    return _apply_close_if_needed_impl(state, readonly)


def apply_close_if_needed(object state, bint readonly=False):
    """Python wrapper that accepts either EnvState or a duck-typed object."""
    cdef ClosedReason reason
    if isinstance(state, PyEnvState):
        reason = _apply_close_if_needed_impl(<EnvState*>state, readonly)
        return reason
    if readonly:
        import copy
        working = copy.deepcopy(state)
    else:
        working = state
    if hasattr(working, "net_worth") and hasattr(working, "peak_value"):
        if working.net_worth > working.peak_value:
            working.peak_value = working.net_worth
    return ClosedReason.NONE


# Export enum values for Python callers to maintain backwards compatibility.
NONE = ClosedReason.NONE
ATR_SL_LONG = ClosedReason.ATR_SL_LONG
ATR_SL_SHORT = ClosedReason.ATR_SL_SHORT
TRAILING_SL_LONG = ClosedReason.TRAILING_SL_LONG
TRAILING_SL_SHORT = ClosedReason.TRAILING_SL_SHORT
STATIC_TP_LONG = ClosedReason.STATIC_TP_LONG
STATIC_TP_SHORT = ClosedReason.STATIC_TP_SHORT
BANKRUPTCY = ClosedReason.BANKRUPTCY
MAX_DRAWDOWN = ClosedReason.MAX_DRAWDOWN
=======
cdef ClosedReason _apply_close_if_needed_impl(EnvState* state, bint readonly=False) nogil:
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
        if reason == ClosedReason.ATR_SL_LONG or reason == ClosedReason.ATR_SL_SHORT:
            state.atr_stop_trigger_count += 1
        elif reason == ClosedReason.TRAILING_SL_LONG or reason == ClosedReason.TRAILING_SL_SHORT:
            state.trailing_stop_trigger_count += 1
        elif reason == ClosedReason.STATIC_TP_LONG or reason == ClosedReason.STATIC_TP_SHORT:
            state.tp_trigger_count += 1
        # Close the position and update state
        if state.units != 0:
            # Calculate P/L realization
            # (Position value already reflects current price)
            state.cash += state._position_value
            # Charge taker fee for closing trade
            state.cash -= fabs(state._position_value) * state.taker_fee
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
        state._high_extremum = -1.0
        state._low_extremum = -1.0
        state._max_price_since_entry = -1.0
        state._min_price_since_entry = -1.0
        state._initial_sl = -1.0
        state._initial_tp = -1.0
        state._atr_at_entry = -1.0
        state._entry_price = -1.0
    return reason


cdef ClosedReason apply_close_if_needed(EnvState* state, bint readonly=False) nogil:
    return _apply_close_if_needed_impl(state, readonly)


def apply_close_if_needed(object state, bint readonly=False):
    """Python wrapper that accepts either EnvState or a duck-typed object."""
    cdef ClosedReason reason
    if isinstance(state, PyEnvState):
        reason = _apply_close_if_needed_impl(<EnvState*>state, readonly)
        return reason
    if readonly:
        import copy
        working = copy.deepcopy(state)
    else:
        working = state
    if hasattr(working, "net_worth") and hasattr(working, "peak_value"):
        if working.net_worth > working.peak_value:
            working.peak_value = working.net_worth
    return ClosedReason.NONE


# Export enum values for Python callers to maintain backwards compatibility.
NONE = ClosedReason.NONE
ATR_SL_LONG = ClosedReason.ATR_SL_LONG
ATR_SL_SHORT = ClosedReason.ATR_SL_SHORT
TRAILING_SL_LONG = ClosedReason.TRAILING_SL_LONG
TRAILING_SL_SHORT = ClosedReason.TRAILING_SL_SHORT
STATIC_TP_LONG = ClosedReason.STATIC_TP_LONG
STATIC_TP_SHORT = ClosedReason.STATIC_TP_SHORT
BANKRUPTCY = ClosedReason.BANKRUPTCY
MAX_DRAWDOWN = ClosedReason.MAX_DRAWDOWN

