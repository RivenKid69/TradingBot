# cython: language_level=3
import random

cimport cython
from libc.stdlib cimport malloc, free

from execevents cimport EventType, Side, MarketEvent
from execlob_book cimport CythonLOB
from coreworkspace cimport SimulationWorkspace

def build_agent_limit_add(double mid_price, int side, int qty, int next_order_id):
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
    return (<int>EventType.AGENT_LIMIT_ADD, <int>side, price, qty, next_order_id)

def build_agent_market_match(int side, int qty):
    """
    Build an agent market match event.
    side: 1 for buy (market buy), -1 for sell (market sell).
    qty: volume to match at market.
    """
    return (<int>EventType.AGENT_MARKET_MATCH, <int>side, 0, qty, 0)

def build_agent_cancel_specific(int order_id, int side):
    """
    Build an agent cancel specific event for the given order id.
    side: side of the order to cancel (1 for buy side order, -1 for sell side order).
    """
    return (<int>EventType.AGENT_CANCEL_SPECIFIC, <int>side, 0, 0, order_id)

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
    # Allocate C array for events
    cdef MarketEvent* events = <MarketEvent*> malloc(n * cython.sizeof(MarketEvent))
    if events == NULL:
        raise MemoryError("Failed to allocate events array")
    cdef int i
    cdef object evt
    for i in range(n):
        # Each event in list is expected as a tuple (type, side, price, qty, order_id)
        evt = all_events[i]
        events[i].type = <EventType> evt[0]
        events[i].side = <Side> evt[1]
        events[i].price = <int> evt[2]
        events[i].qty = <int> evt[3]
        events[i].order_id = <int> evt[4]
    (<CythonLOB> lob).apply_events_batch_nogil(events, n, <SimulationWorkspace> ws)
    # Free allocated events array
    free(events)
