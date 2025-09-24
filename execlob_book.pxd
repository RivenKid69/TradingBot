cimport cython
from libcpp.vector cimport vector
# cimport dependencies from other modules
from execevents cimport MarketEvent, EventType, Side
from coreworkspace cimport SimulationWorkspace

cdef class CythonLOB:
    """
    Cython implementation of a Limit Order Book (LOB) for simulation.
    """
    # Internal storage for orders (bids and asks)
    cdef int capacity_bids
    cdef int capacity_asks
    cdef int n_bids
    cdef int n_asks
    cdef MarketEvent* bid_orders  # array of orders for bids (each contains id, price, qty, etc.)
    cdef MarketEvent* ask_orders  # array of orders for asks
    cdef:
        # Best bid and ask indices (for quick access, updated as needed)
        int best_bid_index
        int best_ask_index
    cdef bint _pending_market_is_agent

    cpdef CythonLOB clone(self)
    cdef void _ensure_capacity(self, bint is_bid, int min_capacity) nogil
    cdef int _find_order_index_by_id(self, int order_id, bint is_bid) nogil
    cdef void add_limit(self, int side, int price, int qty, bint is_agent, int order_id) nogil
    cdef void cancel_order(self, int order_id) nogil
    cdef void match_market(self, int side, int qty, SimulationWorkspace ws) nogil
    cdef void _record_trade(self, SimulationWorkspace ws, int price, int qty, int side,
                            bint agent_maker, bint agent_taker, int maker_order_id) nogil
    cpdef double mid_price(self)
    cdef void apply_events_batch_nogil(self, MarketEvent* events, int num_events, SimulationWorkspace ws) nogil
    cpdef void apply_events_batch(self, list events, SimulationWorkspace ws)
    cpdef list iter_agent_orders(self)
