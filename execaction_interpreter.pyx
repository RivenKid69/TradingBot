# cython: language_level=3
import math

from execevents import (
    build_agent_limit_add,
    build_agent_market_match,
    build_agent_cancel_specific,
)
from execevents cimport Side
import core_constants as constants

# For generating unique order IDs for agent orders (shim for environment's next_order_id)
cdef int _next_order_id = 1  # NOTE: shim for integration; replace with state.next_order_id management later

# Expose enum values as Python integers to avoid attribute lookups on the cdef enum.
SIDE_BUY = <int> Side.BUY
SIDE_SELL = <int> Side.SELL


cdef inline int _side_from_any(object value, int fallback):
    """Coerce arbitrary Python value to +/-1 side convention."""
    cdef int coerced
    if value is None:
        return fallback
    try:
        coerced = int(value)
        if coerced > 0:
            return 1
        elif coerced < 0:
            return -1
    except Exception:
        pass
    try:
        if bool(value):
            return 1
        else:
            return -1
    except Exception:
        pass
    return fallback


cdef inline tuple _normalize_tracker_descriptor(object descriptor, int fallback_side):
    """Normalize tracker return values to (order_id, side_value, price_ticks)."""
    cdef long long order_id = -1
    cdef int side_value = fallback_side if fallback_side in (1, -1) else 1
    cdef long long price_ticks = -1
    cdef object tmp

    if descriptor is None:
        return (order_id, side_value, price_ticks)

    if isinstance(descriptor, (tuple, list)):
        if len(descriptor) > 0:
            try:
                order_id = int(descriptor[0])
            except Exception:
                order_id = -1
        if len(descriptor) > 1:
            side_value = _side_from_any(descriptor[1], side_value)
        if len(descriptor) > 2:
            try:
                price_ticks = int(descriptor[2])
            except Exception:
                price_ticks = -1
        elif len(descriptor) == 2:
            try:
                price_ticks = int(descriptor[1])
            except Exception:
                price_ticks = -1
        return (order_id, side_value, price_ticks)

    if isinstance(descriptor, dict):
        try:
            order_id = int(descriptor.get("order_id", descriptor.get("id", -1)))
        except Exception:
            order_id = -1
        side_value = _side_from_any(descriptor.get("side", descriptor.get("is_buy_side")), side_value)
        tmp = descriptor.get("price", descriptor.get("price_ticks", descriptor.get("price_tick")))
        if tmp is not None:
            try:
                price_ticks = int(tmp)
            except Exception:
                price_ticks = -1
        return (order_id, side_value, price_ticks)

    try:
        order_id = int(getattr(descriptor, "order_id"))
    except Exception:
        try:
            order_id = int(getattr(descriptor, "id"))
        except Exception:
            try:
                order_id = int(descriptor)
            except Exception:
                order_id = -1

    tmp = None
    try:
        tmp = getattr(descriptor, "side")
    except AttributeError:
        try:
            tmp = getattr(descriptor, "is_buy_side")
        except AttributeError:
            tmp = None
    side_value = _side_from_any(tmp, side_value)

    try:
        tmp = getattr(descriptor, "price")
        if tmp is not None:
            price_ticks = int(tmp)
    except AttributeError:
        pass
    try:
        tmp = getattr(descriptor, "price_ticks")
        if tmp is not None:
            price_ticks = int(tmp)
    except AttributeError:
        pass

    return (order_id, side_value, price_ticks)


cdef tuple _tracker_lookup(object tracker, long long price_ticks, int desired_side_value):
    """Attempt to find an existing order near the desired price."""
    cdef object descriptor = None
    cdef tuple normalized
    cdef long long order_id
    cdef int side_value
    cdef long long price_found

    if tracker is None:
        return (False, -1, desired_side_value, -1)

    try:
        descriptor = tracker.find_closest_order(price_ticks, desired_side_value)
    except TypeError:
        try:
            descriptor = tracker.find_closest_order(price_ticks)
        except TypeError:
            try:
                descriptor = tracker.find_closest_order(price_ticks=price_ticks, side=desired_side_value)
            except Exception:
                descriptor = None
        except Exception:
            descriptor = None
    except AttributeError:
        descriptor = None
    except Exception:
        descriptor = None

    normalized = _normalize_tracker_descriptor(descriptor, desired_side_value)
    order_id = <long long> normalized[0]
    side_value = <int> normalized[1]
    price_found = <long long> normalized[2]

    if order_id < 0:
        return (False, -1, desired_side_value, -1)

    if price_found < 0 or side_value not in (1, -1):
        try:
            descriptor = tracker.get_info(order_id)
        except AttributeError:
            try:
                descriptor = tracker.get_order(order_id)
            except AttributeError:
                descriptor = None
        except Exception:
            descriptor = None
        normalized = _normalize_tracker_descriptor(
            descriptor,
            side_value if side_value in (1, -1) else desired_side_value,
        )
        if normalized[2] >= 0:
            price_found = <long long> normalized[2]
        if normalized[1] in (1, -1):
            side_value = <int> normalized[1]

    if side_value not in (1, -1):
        side_value = desired_side_value

    return (True, order_id, side_value, price_found)

def build_agent_event_set(state, tracker, params, action):
    """
    Interpret the agent's action and generate a set of agent events for this step.
    Returns a list of event tuples (to be mixed with public events).
    """
    global _next_order_id
    cdef double target_frac
    cdef double cur_units = 0.0
    cdef double net_worth = 0.0
    cdef double cash = 0.0
    cdef double price = 0.0
    cdef list events = []
    cdef double style_param = 0.0
    cdef double price_scale = 1.0

    try:
        price_scale = float(constants.PRICE_SCALE)
    except Exception:
        price_scale = 1.0

    # Extract action components
    if hasattr(action, "__len__") and len(action) > 1:
        target_frac = float(action[0])
        style_param = float(action[1])
    else:
        target_frac = float(action) if hasattr(action, "__float__") else 0.0
        style_param = 0.0  # default prefer limit

    # Get current state values
    try:
        net_worth = float(state.net_worth)
    except Exception:
        net_worth = 0.0
    try:
        cur_units = float(state.units)
    except Exception:
        cur_units = 0.0
    try:
        cash = float(state.cash)
    except Exception:
        cash = net_worth  # if units=0, net_worth ~ cash

    # Determine current price for calculations
    price = 0.0
    # Try to derive price from state (if position exists)
    if cur_units != 0.0:
        price = (net_worth - cash) / cur_units  # current mark price of asset
    # If no position or price still 0, try to get last price from state or market simulator
    if price <= 0.0:
        try:
            price = float(state.last_price)
        except Exception:
            price = 0.0
    if price <= 0.0:
        # If we still have no price reference, skip generating trading events
        return events

    # Calculate target position in value and units
    cdef double target_position_value = target_frac * net_worth
    cdef double current_position_value = cur_units * price
    cdef double diff_value = target_position_value - current_position_value
    cdef double diff_units = 0.0
    if price != 0.0:
        diff_units = diff_value / price

    cdef bint has_desired_side = False
    cdef Side desired_side = Side.BUY
    cdef double vol = 0.0
    if diff_units > 1e-9:
        desired_side = Side.BUY
        has_desired_side = True
        vol = diff_units
    elif diff_units < -1e-9:
        desired_side = Side.SELL
        has_desired_side = True
        vol = -diff_units
    else:
        has_desired_side = False
        vol = 0.0

    if not has_desired_side or vol < 1e-6:
        return events

    cdef int volume_units = <int> math.floor(vol + 1e-8)
    if volume_units <= 0:
        return events

    cdef bint use_market = style_param > 0.5

    cdef long long target_price_ticks = <long long> round(price * price_scale)
    cdef tuple lookup = _tracker_lookup(tracker, target_price_ticks, 1 if desired_side == Side.BUY else -1)
    cdef bint has_existing = bool(lookup[0])
    cdef int existing_id = <int> lookup[1] if has_existing else -1
    cdef Side existing_side = desired_side
    cdef long long existing_price = -1
    cdef int side_value = <int> desired_side
    cdef bint should_cancel = False
    if has_existing and existing_id >= 0:
        try:
            side_value = int(lookup[2])
            existing_side = <Side> side_value
        except Exception:
            existing_side = desired_side
        try:
            existing_price = <long long> lookup[3]
        except Exception:
            existing_price = -1
    should_cancel = False
    if has_existing and existing_id >= 0:
        if use_market:
            should_cancel = True
        elif existing_side != desired_side:
            should_cancel = True
        elif existing_price >= 0 and abs(existing_price - target_price_ticks) > 1:
            should_cancel = True
        elif existing_price < 0:
            should_cancel = True
        if should_cancel:
            events.append(build_agent_cancel_specific(existing_id, existing_side))

    cdef double mid
    cdef int oid
    if volume_units > 0:
        if use_market:
            events.append(build_agent_market_match(desired_side, volume_units))
        else:
            mid = 0.0
            try:
                lob_obj = getattr(state, "lob", None)
                if lob_obj is not None:
                    mid = float(lob_obj.mid_price())
            except Exception:
                mid = 0.0
            if mid <= 0.0:
                mid = price * price_scale
            oid = _next_order_id
            _next_order_id += 1
            events.append(build_agent_limit_add(mid, desired_side, volume_units, oid))
    return events
