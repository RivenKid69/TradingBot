# Declaration of the SimulationWorkspace Cython class (for use in Cython code).
cdef class SimulationWorkspace:
    """
    Cython workspace for accumulating trade data within a simulation step.

    This class manages pre-allocated, C-contiguous buffers to store information 
    about trades and filled orders that occur in a single simulation step. 
    It is optimized for performance and can be used without the Global Interpreter Lock (GIL) 
    in time-critical sections of the simulation.
    """

    # C-contiguous arrays (1-dimensional) for trade data and filled order IDs.
    cdef double[::1] trade_prices       # Prices of trades (each step)
    cdef double[::1] trade_qtys         # Quantities of trades
    cdef char[::1] trade_sides          # Sides of trades (e.g., 1 for buy, -1 for sell)
    cdef char[::1] trade_is_agent_maker # Flags indicating if agent was maker (1) or taker (0) for each trade
    cdef long long[::1] trade_ts        # Timestamps of trades
    cdef long long[::1] filled_order_ids# IDs of orders that were fully filled in the step

    cdef int trade_count    # Number of trades recorded in the current step
    cdef int filled_count   # Number of order IDs recorded as fully filled in the current step

    # Methods for managing the workspace buffers.
    cdef void ensure_capacity(self, int min_capacity) nogil
    cdef void clear_step(self) nogil
    cdef void push_trade(self, double price, double qty, char side, char is_agent_maker, long long ts) nogil
    cdef void push_filled_order_id(self, long long order_id) nogil
