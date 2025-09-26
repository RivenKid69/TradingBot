# cython: language_level=3
# distutils: language = c++
from libc.math cimport fmax, fmin, fabs

from risk_enums cimport ClosedReason
from risk_enums import ClosedReason as PyClosedReason
from lob_state_cython cimport EnvState
from lob_state_cython import EnvState as PyEnvState

# Re-export the Python enumeration for convenient access from Python code.
ClosedReasonEnum = PyClosedReason
ClosedReasonPy = PyClosedReason


cdef inline double _current_fill_price(EnvState state):
    """Return the effective fill price for the current position."""
    if state.units == 0:
        return 0.0
    return state._position_value / state.units


cdef double compute_max_position_frac(EnvState state):
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


cdef ClosedReason check_static_atr_stop(EnvState state):
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


cdef ClosedReason check_trailing_stop(EnvState state):
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


cdef ClosedReason check_take_profit(EnvState state):
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


cdef void update_trailing_extrema(EnvState state):
    """Update trailing stop extrema and activate trailing stop if conditions met."""
    if not state.use_trailing_stop or state.units == 0:
        return
    cdef double last_price = _current_fill_price(state)
    cdef double activate_threshold
    cdef double new_stop_level
    if state.units > 0:
        if state._high_extremum < 0.0 or last_price > state._high_extremum:
            state._high_extremum = last_price
        if state._low_extremum < 0.0:
            state._low_extremum = last_price
        else:
            state._low_extremum = fmin(state._low_extremum, last_price)
        if not state._trailing_active:
            activate_threshold = state._entry_price + state._atr_at_entry * state.trailing_atr_mult
            if state._entry_price > 0.0 and state._atr_at_entry > 0.0 and last_price >= activate_threshold:
                state._trailing_active = True
        if state._trailing_active:
            new_stop_level = state._high_extremum - state._atr_at_entry * state.trailing_atr_mult
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
            activate_threshold = state._entry_price - state._atr_at_entry * state.trailing_atr_mult
            if state._entry_price > 0.0 and state._atr_at_entry > 0.0 and last_price <= activate_threshold:
                state._trailing_active = True
        if state._trailing_active:
            new_stop_level = state._low_extremum + state._atr_at_entry * state.trailing_atr_mult
            if new_stop_level > 0.0 and (state._initial_sl <= 0.0 or new_stop_level < state._initial_sl):
                state._initial_sl = new_stop_level


cdef ClosedReason check_bankruptcy(EnvState state):
    """Check if account is bankrupt."""
    if state.net_worth <= state.bankruptcy_threshold:
        return ClosedReason.BANKRUPTCY
    return ClosedReason.NONE


cdef ClosedReason check_max_drawdown(EnvState state):
    """Check if max drawdown limit is hit."""
    if state.net_worth > state.peak_value:
        state.peak_value = state.net_worth
    if state.max_drawdown > 0.0 and state.net_worth <= state.peak_value * (1.0 - state.max_drawdown):
        return ClosedReason.MAX_DRAWDOWN
    return ClosedReason.NONE


cdef ClosedReason _apply_close_if_needed_impl(EnvState state, bint readonly=False):
    """Apply position close if any risk or TP/SL condition is triggered."""
    cdef double orig_cash = 0.0
    cdef double orig_position_value = 0.0
    cdef double orig_prev_net_worth = 0.0
    cdef double orig_net_worth = 0.0
    cdef double orig_high_extremum = 0.0
    cdef double orig_low_extremum = 0.0
    cdef double orig_max_price = 0.0
    cdef double orig_min_price = 0.0
    cdef double orig_initial_sl = 0.0
    cdef double orig_initial_tp = 0.0
    cdef double orig_atr_at_entry = 0.0
    cdef double orig_entry_price = 0.0
    cdef double orig_peak_value = 0.0
    cdef double orig_units = 0.0
    cdef bint orig_trailing_active = False
    cdef bint orig_is_bankrupt = False
    cdef int orig_atr_count = 0
    cdef int orig_trailing_count = 0
    cdef int orig_tp_count = 0
    if readonly:
        orig_cash = state.cash
        orig_position_value = state._position_value
        orig_prev_net_worth = state.prev_net_worth
        orig_net_worth = state.net_worth
        orig_high_extremum = state._high_extremum
        orig_low_extremum = state._low_extremum
        orig_max_price = state._max_price_since_entry
        orig_min_price = state._min_price_since_entry
        orig_initial_sl = state._initial_sl
        orig_initial_tp = state._initial_tp
        orig_atr_at_entry = state._atr_at_entry
        orig_entry_price = state._entry_price
        orig_peak_value = state.peak_value
        orig_units = state.units
        orig_trailing_active = state._trailing_active
        orig_is_bankrupt = state.is_bankrupt
        orig_atr_count = state.atr_stop_trigger_count
        orig_trailing_count = state.trailing_stop_trigger_count
        orig_tp_count = state.tp_trigger_count

    cdef ClosedReason reason = check_bankruptcy(state)
    if reason == ClosedReason.NONE:
        reason = check_max_drawdown(state)
    if reason == ClosedReason.NONE and state.units != 0:
        if state.use_trailing_stop:
            update_trailing_extrema(state)
        reason = check_static_atr_stop(state)
        if reason == ClosedReason.NONE:
            reason = check_trailing_stop(state)
            if reason == ClosedReason.NONE:
                reason = check_take_profit(state)

    if reason != ClosedReason.NONE:
        if reason == ClosedReason.ATR_SL_LONG or reason == ClosedReason.ATR_SL_SHORT:
            state.atr_stop_trigger_count += 1
        elif reason == ClosedReason.TRAILING_SL_LONG or reason == ClosedReason.TRAILING_SL_SHORT:
            state.trailing_stop_trigger_count += 1
        elif reason == ClosedReason.STATIC_TP_LONG or reason == ClosedReason.STATIC_TP_SHORT:
            state.tp_trigger_count += 1

        if state.units != 0:
            state.cash += state._position_value
            state.cash -= fabs(state._position_value) * state.taker_fee
        state.units = 0.0
        state._position_value = 0.0
        state.prev_net_worth = state.net_worth
        state.net_worth = state.cash
        if reason == ClosedReason.BANKRUPTCY:
            state.cash = 0.0
            state.net_worth = 0.0
            state.is_bankrupt = True
        state._trailing_active = False
        state._high_extremum = -1.0
        state._low_extremum = -1.0
        state._max_price_since_entry = -1.0
        state._min_price_since_entry = -1.0
        state._initial_sl = -1.0
        state._initial_tp = -1.0
        state._atr_at_entry = -1.0
        state._entry_price = -1.0

    if readonly:
        state.cash = orig_cash
        state._position_value = orig_position_value
        state.prev_net_worth = orig_prev_net_worth
        state.net_worth = orig_net_worth
        state._high_extremum = orig_high_extremum
        state._low_extremum = orig_low_extremum
        state._max_price_since_entry = orig_max_price
        state._min_price_since_entry = orig_min_price
        state._initial_sl = orig_initial_sl
        state._initial_tp = orig_initial_tp
        state._atr_at_entry = orig_atr_at_entry
        state._entry_price = orig_entry_price
        state.peak_value = orig_peak_value
        state.units = <float>orig_units
        state._trailing_active = orig_trailing_active
        state.is_bankrupt = orig_is_bankrupt
        state.atr_stop_trigger_count = orig_atr_count
        state.trailing_stop_trigger_count = orig_trailing_count
        state.tp_trigger_count = orig_tp_count

    return reason


cdef inline double _py_current_fill_price(object state):
    units = getattr(state, "units", 0.0)
    try:
        units_val = float(units)
    except (TypeError, ValueError):
        units_val = 0.0
    if units_val == 0.0:
        return 0.0
    position_value = getattr(state, "_position_value", 0.0)
    try:
        pos_val = float(position_value)
    except (TypeError, ValueError):
        pos_val = 0.0
    if units_val == 0.0:
        return 0.0
    return pos_val / units_val


def _py_check_bankruptcy(object state):
    net_worth = float(getattr(state, "net_worth", 0.0))
    threshold = float(getattr(state, "bankruptcy_threshold", 0.0))
    if net_worth <= threshold:
        return PyClosedReason.BANKRUPTCY
    return PyClosedReason.NONE


def _py_check_max_drawdown(object state):
    net_worth = float(getattr(state, "net_worth", 0.0))
    peak_value = float(getattr(state, "peak_value", 0.0))
    if net_worth > peak_value:
        setattr(state, "peak_value", net_worth)
        peak_value = net_worth
    max_dd = float(getattr(state, "max_drawdown", 0.0))
    if max_dd > 0.0 and net_worth <= peak_value * (1.0 - max_dd):
        return PyClosedReason.MAX_DRAWDOWN
    return PyClosedReason.NONE


def _py_update_trailing_extrema(object state):
    if not getattr(state, "use_trailing_stop", False):
        return
    units = float(getattr(state, "units", 0.0))
    if units == 0.0:
        return
    last_price = _py_current_fill_price(state)
    high_ext = float(getattr(state, "_high_extremum", -1.0))
    low_ext = float(getattr(state, "_low_extremum", -1.0))
    trailing_active = bool(getattr(state, "_trailing_active", False))
    entry_price = float(getattr(state, "_entry_price", -1.0))
    atr_at_entry = float(getattr(state, "_atr_at_entry", -1.0))
    trailing_mult = float(getattr(state, "trailing_atr_mult", 0.0))
    initial_sl = float(getattr(state, "_initial_sl", -1.0))
    if units > 0.0:
        if high_ext < 0.0 or last_price > high_ext:
            high_ext = last_price
        if low_ext < 0.0:
            low_ext = last_price
        else:
            low_ext = min(low_ext, last_price)
        if not trailing_active:
            activate_threshold = entry_price + atr_at_entry * trailing_mult
            if entry_price > 0.0 and atr_at_entry > 0.0 and last_price >= activate_threshold:
                trailing_active = True
        if trailing_active:
            new_stop = high_ext - atr_at_entry * trailing_mult
            if new_stop > 0.0 and (initial_sl <= 0.0 or new_stop > initial_sl):
                initial_sl = new_stop
    else:
        if low_ext < 0.0 or last_price < low_ext:
            low_ext = last_price
        if high_ext < 0.0:
            high_ext = last_price
        else:
            high_ext = max(high_ext, last_price)
        if not trailing_active:
            activate_threshold = entry_price - atr_at_entry * trailing_mult
            if entry_price > 0.0 and atr_at_entry > 0.0 and last_price <= activate_threshold:
                trailing_active = True
        if trailing_active:
            new_stop = low_ext + atr_at_entry * trailing_mult
            if new_stop > 0.0 and (initial_sl <= 0.0 or new_stop < initial_sl):
                initial_sl = new_stop
    setattr(state, "_high_extremum", high_ext)
    setattr(state, "_low_extremum", low_ext)
    setattr(state, "_trailing_active", trailing_active)
    setattr(state, "_initial_sl", initial_sl)


def _py_check_static_atr_stop(object state):
    if not getattr(state, "use_atr_stop", False):
        return PyClosedReason.NONE
    if bool(getattr(state, "_trailing_active", False)):
        return PyClosedReason.NONE
    units = float(getattr(state, "units", 0.0))
    if units == 0.0:
        return PyClosedReason.NONE
    last_price = _py_current_fill_price(state)
    initial_sl = float(getattr(state, "_initial_sl", -1.0))
    if units > 0.0:
        if initial_sl > 0.0 and last_price <= initial_sl:
            return PyClosedReason.ATR_SL_LONG
    else:
        if initial_sl > 0.0 and last_price >= initial_sl:
            return PyClosedReason.ATR_SL_SHORT
    return PyClosedReason.NONE


def _py_check_trailing_stop(object state):
    if not getattr(state, "use_trailing_stop", False):
        return PyClosedReason.NONE
    if not bool(getattr(state, "_trailing_active", False)):
        return PyClosedReason.NONE
    units = float(getattr(state, "units", 0.0))
    if units == 0.0:
        return PyClosedReason.NONE
    last_price = _py_current_fill_price(state)
    initial_sl = float(getattr(state, "_initial_sl", -1.0))
    if units > 0.0:
        if initial_sl > 0.0 and last_price <= initial_sl:
            return PyClosedReason.TRAILING_SL_LONG
    else:
        if initial_sl > 0.0 and last_price >= initial_sl:
            return PyClosedReason.TRAILING_SL_SHORT
    return PyClosedReason.NONE


def _py_check_take_profit(object state):
    units = float(getattr(state, "units", 0.0))
    if units == 0.0:
        return PyClosedReason.NONE
    tp_mult = float(getattr(state, "tp_atr_mult", 0.0))
    if tp_mult <= 0.0:
        return PyClosedReason.NONE
    last_price = _py_current_fill_price(state)
    initial_tp = float(getattr(state, "_initial_tp", -1.0))
    if units > 0.0:
        if initial_tp > 0.0 and last_price >= initial_tp:
            return PyClosedReason.STATIC_TP_LONG
    else:
        if initial_tp > 0.0 and last_price <= initial_tp:
            return PyClosedReason.STATIC_TP_SHORT
    return PyClosedReason.NONE


def _apply_close_if_needed_py(object state):
    reason = _py_check_bankruptcy(state)
    if reason == PyClosedReason.NONE:
        reason = _py_check_max_drawdown(state)
    units = float(getattr(state, "units", 0.0))
    if reason == PyClosedReason.NONE and units != 0.0:
        if getattr(state, "use_trailing_stop", False):
            _py_update_trailing_extrema(state)
        reason = _py_check_static_atr_stop(state)
        if reason == PyClosedReason.NONE:
            reason = _py_check_trailing_stop(state)
            if reason == PyClosedReason.NONE:
                reason = _py_check_take_profit(state)
    if reason != PyClosedReason.NONE:
        atr_count = int(getattr(state, "atr_stop_trigger_count", 0))
        trailing_count = int(getattr(state, "trailing_stop_trigger_count", 0))
        tp_count = int(getattr(state, "tp_trigger_count", 0))
        if reason in (PyClosedReason.ATR_SL_LONG, PyClosedReason.ATR_SL_SHORT):
            atr_count += 1
            setattr(state, "atr_stop_trigger_count", atr_count)
        elif reason in (PyClosedReason.TRAILING_SL_LONG, PyClosedReason.TRAILING_SL_SHORT):
            trailing_count += 1
            setattr(state, "trailing_stop_trigger_count", trailing_count)
        elif reason in (PyClosedReason.STATIC_TP_LONG, PyClosedReason.STATIC_TP_SHORT):
            tp_count += 1
            setattr(state, "tp_trigger_count", tp_count)
        position_value = float(getattr(state, "_position_value", 0.0))
        cash = float(getattr(state, "cash", 0.0))
        taker_fee = float(getattr(state, "taker_fee", 0.0))
        if units != 0.0:
            cash += position_value
            cash -= fabs(position_value) * taker_fee
        setattr(state, "cash", cash)
        prev_net_worth = float(getattr(state, "net_worth", 0.0))
        setattr(state, "prev_net_worth", prev_net_worth)
        setattr(state, "net_worth", cash)
        setattr(state, "units", 0.0)
        setattr(state, "_position_value", 0.0)
        if reason == PyClosedReason.BANKRUPTCY:
            setattr(state, "cash", 0.0)
            setattr(state, "net_worth", 0.0)
            setattr(state, "is_bankrupt", True)
        setattr(state, "_trailing_active", False)
        setattr(state, "_high_extremum", -1.0)
        setattr(state, "_low_extremum", -1.0)
        setattr(state, "_max_price_since_entry", -1.0)
        setattr(state, "_min_price_since_entry", -1.0)
        setattr(state, "_initial_sl", -1.0)
        setattr(state, "_initial_tp", -1.0)
        setattr(state, "_atr_at_entry", -1.0)
        setattr(state, "_entry_price", -1.0)
    return reason


def apply_close_if_needed(object state, bint readonly=False):
    """Python wrapper that accepts either EnvState or a duck-typed object."""
    cdef ClosedReason reason
    if isinstance(state, PyEnvState):
        reason = _apply_close_if_needed_impl(state, readonly)
        return PyClosedReason(reason)
    if readonly:
        import copy
        working = copy.deepcopy(state)
        return _apply_close_if_needed_py(working)
    return _apply_close_if_needed_py(state)
