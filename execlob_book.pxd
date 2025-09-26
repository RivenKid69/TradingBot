cimport cython

from execevents cimport EventType, MarketEvent, Side
from coreworkspace cimport SimulationWorkspace


cdef class CythonLOB:
    """Cython implementation of a simple in-memory limit order book used in tests."""

    # Internal storage for bids and asks
    cdef int capacity_bids
    cdef int capacity_asks
    cdef int n_bids
    cdef int n_asks
    cdef MarketEvent* bid_orders
    cdef MarketEvent* ask_orders

    # Cached indices for best prices
    cdef int best_bid_index
    cdef int best_ask_index

    # Flag denoting whether the currently executing market order belongs to the agent
    cdef bint _pending_market_is_agent
    cdef bint _has_error
    cdef object _pending_exception

    cpdef CythonLOB clone(self)
    cdef bint _ensure_capacity(self, bint is_bid, int min_capacity) noexcept nogil
    cdef int _find_order_index_by_id(self, int order_id, bint is_bid) noexcept nogil
    cdef bint add_limit(self, int side, int price, int qty, bint is_agent, int order_id) noexcept nogil
    cdef bint cancel_order(self, int order_id) noexcept nogil
    cdef bint match_market(self, int side, int qty, SimulationWorkspace ws) noexcept nogil
    cdef bint _record_trade(
        self,
        SimulationWorkspace ws,
        int price,
        int qty,
        int side,
        bint agent_maker,
        bint agent_taker,
        int maker_order_id,
    ) noexcept nogil
    cdef void _reset_error_state(self)
    cdef void _set_error(self, object exc) noexcept
    cdef void _raise_pending_error(self)
    cpdef double mid_price(self)
    cdef bint apply_events_batch_nogil(self, MarketEvent* events, int num_events, SimulationWorkspace ws) noexcept nogil
    cpdef void apply_events_batch(self, list events, SimulationWorkspace ws)
    cpdef list iter_agent_orders(self)
