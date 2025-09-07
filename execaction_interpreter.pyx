# cython: language_level=3
import math
from exec.events import build_agent_limit_add, build_agent_market_match, build_agent_cancel_specific
from exec.lob_book cimport CythonLOB
from exec.events cimport EventType, Side
from core.constants cimport PRICE_SCALE

# For generating unique order IDs for agent orders (shim for environment's next_order_id)
cdef int _next_order_id = 1  # NOTE: shim for integration; replace with state.next_order_id management later

def build_agent_event_set(state, tracker, params, action):
    """
    Interpret the agent's action and generate a set of agent events for this step.
    Returns a list of event tuples (to be mixed with public events).
    """
    cdef double target_frac
    cdef double cur_units = 0.0
    cdef double net_worth = 0.0
    cdef double cash = 0.0
    cdef double price = 0.0
    cdef double position_value = 0.0
    cdef list events = []

    # Extract action components
    if hasattr(action, "__len__") and len(action) > 1:
        target_frac = float(action[0])
        cdef double style_param = float(action[1])
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

    # Determine side and volume from diff_units
    cdef int side = 0
    cdef double vol = 0.0
    if diff_units > 1e-9:
        side = 1   # BUY
        vol = diff_units
    elif diff_units < -1e-9:
        side = -1  # SELL
        vol = -diff_units
    else:
        side = 0
        vol = 0.0

    # If volume is negligible or no change, no events
    if side == 0 or vol < 1e-6:
        # If agent has no new action, optionally consider canceling stale orders (hysteresis logic)
        return events

    # Convert volume to integer number of units (round down to nearest whole unit)
    cdef int volume_units = <int> math.floor(vol + 1e-8)
    if volume_units <= 0:
        return events

    # Determine order type (market or limit) based on style_param (or default)
    cdef bint use_market = False
    if style_param > 0.5:
        use_market = True

    # If agent has an existing open order from previous steps, handle cancellation if needed
    try:
        cdef int existing_id = -1
        # Check buy side orders
        if tracker is not None:
            existing_id = tracker.find_closest_order(price * PRICE_SCALE, Side.BUY)
            if existing_id == -1:
                existing_id = tracker.find_closest_order(price * PRICE_SCALE, Side.SELL)
    except AttributeError:
        existing_id = -1

    if existing_id != -1:
        # If agent is changing side or placing a new order, cancel the existing one first
        if (side == 1 and tracker is not None and tracker.has_sell_orders) or \
           (side == -1 and tracker is not None and tracker.has_buy_orders) or \
           use_market or True:
            events.append(build_agent_cancel_specific(existing_id, 1 if side == 1 else -1))

    # Now build new order event if needed
    if volume_units > 0:
        if use_market:
            # Market order
            events.append(build_agent_market_match(side, volume_units))
        else:
            # Limit order
            cdef double mid = 0.0
            try:
                cdef CythonLOB lob = state.lob
                if lob is not None:
                    mid = lob.mid_price()
            except Exception:
                mid = price * PRICE_SCALE
            cdef int oid = _next_order_id
            _next_order_id += 1
            events.append(build_agent_limit_add(mid, side, volume_units, oid))
    return events
