cimport cython
from libcpp.vector cimport vector
# cimport dependencies from other modules
from exec.events cimport MarketEvent, EventType, Side
from core.workspace cimport SimulationWorkspace

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

    cpdef CythonLOB clone(self)
    cdef void _ensure_capacity(self, bint is_bid, int min_capacity) nogil
    cdef void add_limit(self, int side, int price, int qty, bint is_agent, int order_id) nogil
    cdef void cancel_order(self, int order_id) nogil
    cdef void match_market(self, int side, int qty, SimulationWorkspace ws) nogil
    cpdef double mid_price(self)
    cdef void apply_events_batch_nogil(self, MarketEvent* events, int num_events, SimulationWorkspace ws) nogil
