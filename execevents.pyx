# cython: language_level=3
import random
from enum import IntEnum

from execevents cimport EventType, Side


class EventTypeEnum(IntEnum):
    """Python-facing view of the execution event types."""
    AGENT_LIMIT_ADD = <int> EventType.AGENT_LIMIT_ADD
    AGENT_MARKET_MATCH = <int> EventType.AGENT_MARKET_MATCH
    AGENT_CANCEL_SPECIFIC = <int> EventType.AGENT_CANCEL_SPECIFIC
    PUBLIC_LIMIT_ADD = <int> EventType.PUBLIC_LIMIT_ADD
    PUBLIC_MARKET_MATCH = <int> EventType.PUBLIC_MARKET_MATCH
    PUBLIC_CANCEL_RANDOM = <int> EventType.PUBLIC_CANCEL_RANDOM


class SideEnum(IntEnum):
    """Python-facing view of the order sides."""
    BUY = <int> Side.BUY
    SELL = <int> Side.SELL


# Export frequently used enum values as Python-level constants for convenience.
SIDE_BUY: int = <int> Side.BUY
SIDE_SELL: int = <int> Side.SELL
EVENT_AGENT_LIMIT_ADD: int = <int> EventType.AGENT_LIMIT_ADD
EVENT_AGENT_MARKET_MATCH: int = <int> EventType.AGENT_MARKET_MATCH
EVENT_AGENT_CANCEL_SPECIFIC: int = <int> EventType.AGENT_CANCEL_SPECIFIC

# Backwards-compatible aliases exposed through the module globals without shadowing ctypedefs.
EventTypePy = EventTypeEnum
SidePy = SideEnum

__all__ = [
    "EventTypeEnum",
    "SideEnum",
    "EventTypePy",
    "SidePy",
    "EventType",
    "Side",
    "SIDE_BUY",
    "SIDE_SELL",
    "EVENT_AGENT_LIMIT_ADD",
    "EVENT_AGENT_MARKET_MATCH",
    "EVENT_AGENT_CANCEL_SPECIFIC",
    "build_agent_limit_add",
    "build_agent_market_match",
    "build_agent_cancel_specific",
    "apply_agent_events",
]


def _register_python_enums():
    """Expose Python enum views under the historical attribute names."""
    globals()["EventType"] = EventTypeEnum
    globals()["Side"] = SideEnum


_register_python_enums()
del _register_python_enums


cpdef tuple build_agent_limit_add(double mid_price, Side side, int qty, int next_order_id):
    """
    Build an agent limit add event. mid_price is in ticks (as float if fractional mid).
    side: 1 for buy, -1 for sell.
    qty: volume of the order.
    next_order_id: unique id to assign to this new order.
    Returns a tuple representing the MarketEvent.
    """
    # Determine offset range based on mid price and volatility (approximate with mid value)
    cdef double mid = mid_price
    cdef int mid_ticks = <int> mid  # use integer part of mid for offset scaling
    cdef int offset_range = 5  # default minimal range
    if mid_ticks > 0:
        # Use ~0.5% of mid as range (at least 1 tick)
        offset_range = <int> (0.005 * mid)
        if offset_range < 1:
            offset_range = 1
    else:
        offset_range = 1
    cdef int offset = random.randint(1, offset_range)  # at least 1 tick offset to remain passive
    cdef int price
    if side == Side.BUY:
        price = mid_ticks - offset
        if price < 1:
            price = 1  # do not allow zero or negative price
    else:
        price = mid_ticks + offset
        if price < 1:
            price = 1
    return (EVENT_AGENT_LIMIT_ADD, <int> side, price, qty, next_order_id)


cpdef tuple build_agent_market_match(Side side, int qty):
    """
    Build an agent market match event.
    side: 1 for buy (market buy), -1 for sell (market sell).
    qty: volume to match at market.
    """
    return (EVENT_AGENT_MARKET_MATCH, <int> side, 0, qty, 0)


cpdef tuple build_agent_cancel_specific(int order_id, Side side):
    """
    Build an agent cancel specific event for the given order id.
    side: side of the order to cancel (1 for buy side order, -1 for sell side order).
    """
    return (EVENT_AGENT_CANCEL_SPECIFIC, <int> side, 0, 0, order_id)


def apply_agent_events(state, tracker, microgen, lob, ws, events_list):
    """
    Mix agent events with public events and apply them to the LOB using SimulationWorkspace ws.
    """
    # Generate public microstructure events (if any)
    if microgen is not None:
        try:
            public_events = microgen.generate_public_events(state, tracker, lob)
        except Exception:
            public_events = []
    else:
        public_events = []
    # Combine agent and public events
    all_events = []
    if events_list is not None:
        all_events.extend(events_list)
    all_events.extend(public_events)
    cdef int n = len(all_events)
    if n == 0:
        return
    # Dispatch events through the LOB helper
    lob.apply_events_batch(all_events, ws)
