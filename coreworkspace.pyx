# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# The above directives (if present) disable certain Python checks for performance.

from libc.stdlib cimport malloc, free  # (Optional: if using C memory allocation, but here we use Python memory)
from libcpp.vector cimport vector      # (Not used, but available if needed for C++ structures)

# Import Python-level constants (ensuring no other project modules are used).
from core_constants cimport PRICE_SCALE, MarketRegime

cdef class SimulationWorkspace:
    """
    SimulationWorkspace manages preallocated buffers for trades in a simulation step.

    The workspace allocates C-contiguous arrays for trade data (prices, quantities, sides, 
    agent maker flags, timestamps) and for filled order IDs. This allows the simulation to 
    collect all events in a step without frequent memory (re)allocation, improving performance 
    and avoiding garbage collection overhead.

    All methods are designed for use under `nogil` (no Python GIL), meaning they do not 
    allocate Python objects or call Python APIs on the hot path. This makes it safe to record 
    simulation events from low-level code without holding the GIL.

    Attributes:
        trade_prices (double[::1]): Buffer of trade prices for the current step.
        trade_qtys (double[::1]): Buffer of trade quantities for the current step.
        trade_sides (char[::1]): Buffer of trade sides (1 for buy, -1 for sell).
        trade_is_agent_maker (char[::1]): Buffer of flags indicating if agent was maker (1) or taker (0) in each trade.
        trade_ts (long long[::1]): Buffer of trade timestamps.
        filled_order_ids (long long[::1]): Buffer of fully filled order IDs in the current step.
        trade_count (int): Current number of trades stored for this step.
        filled_count (int): Current number of filled order IDs stored for this step.
    """
    cdef int _capacity  # internal capacity of each buffer (number of slots allocated)

    def __cinit__(self):
        # Initialize counters to zero and capacity to 0 (will allocate in __init__)
        self.trade_count = 0
        self.filled_count = 0
        self._capacity = 0

    def __init__(self, int initial_capacity=0):
        """Initialize the SimulationWorkspace with given initial capacity (number of trades).

        Args:
            initial_capacity (int): Initial number of trade slots to allocate. If not provided 
                                     or <= 0, uses constants.DEFAULT_MAX_TRADES_PER_STEP.
        """
        cdef int capacity
        # Determine initial capacity
        if initial_capacity <= 0:
            capacity = constants.DEFAULT_MAX_TRADES_PER_STEP
        else:
            capacity = initial_capacity

        # Allocate buffers as bytearrays and cast to typed memoryviews for C-contiguous arrays.
        # Using bytearray ensures Python manages the memory and we get a writable buffer.
        cdef int bytes_double = sizeof(double)
        cdef int bytes_longlong = sizeof(long long)

        # Allocate byte buffers for each array
        cdef bytes b_prices = bytearray(capacity * bytes_double)
        cdef bytes b_qtys = bytearray(capacity * bytes_double)
        cdef bytes b_sides = bytearray(capacity * sizeof(char))
        cdef bytes b_makers = bytearray(capacity * sizeof(char))
        cdef bytes b_ts = bytearray(capacity * bytes_longlong)
        cdef bytes b_filled = bytearray(capacity * bytes_longlong)

        # Cast byte buffers to typed memoryviews (C-contiguous arrays)
        self.trade_prices = memoryview(b_prices).cast('d')   # double -> 'd'
        self.trade_qtys = memoryview(b_qtys).cast('d')
        self.trade_sides = memoryview(b_sides).cast('b')     # signed char -> 'b'
        self.trade_is_agent_maker = memoryview(b_makers).cast('b')
        self.trade_ts = memoryview(b_ts).cast('q')           # long long -> 'q'
        self.filled_order_ids = memoryview(b_filled).cast('q')

        # Set the internal capacity and verify buffers are C-contiguous.
        self._capacity = capacity
        assert self.trade_prices.strides[0] == sizeof(double)
        assert self.trade_qtys.strides[0] == sizeof(double)
        assert self.trade_sides.strides[0] == sizeof(char)
        assert self.trade_is_agent_maker.strides[0] == sizeof(char)
        assert self.trade_ts.strides[0] == sizeof(long long)
        assert self.filled_order_ids.strides[0] == sizeof(long long)
        # Note: The above asserts ensure each memoryview has contiguous stride (1 element step).
        # They rely on the fact that memoryview.strides is accessible with GIL (we are in __init__).

    cdef void ensure_capacity(self, int min_capacity) nogil:
        """Ensure the internal buffers have capacity for at least `min_capacity` elements.

        If the current buffer capacity is less than min_capacity, new buffers twice the current 
        size (or min_capacity, whichever is larger) are allocated and existing data is copied over. 
        This operation may temporarily acquire the GIL for memory allocation and copying, but is 
        designed to happen infrequently (only when buffers need to grow).
        """
        cdef int new_capacity, current_capacity
        current_capacity = self._capacity
        if min_capacity <= current_capacity:
            return  # Sufficient capacity already available.

        # Determine new capacity (grow by factor of 2, or to min_capacity if larger).
        new_capacity = current_capacity * 2
        if new_capacity < min_capacity:
            new_capacity = min_capacity

        # Perform allocation and copying with GIL, as this involves Python object management.
        with gil:
            # Allocate new byte buffers for each array with the larger capacity.
            cdef int bytes_double = sizeof(double)
            cdef int bytes_longlong = sizeof(long long)
            cdef bytes b_prices_new = bytearray(new_capacity * bytes_double)
            cdef bytes b_qtys_new = bytearray(new_capacity * bytes_double)
            cdef bytes b_sides_new = bytearray(new_capacity * sizeof(char))
            cdef bytes b_makers_new = bytearray(new_capacity * sizeof(char))
            cdef bytes b_ts_new = bytearray(new_capacity * bytes_longlong)
            cdef bytes b_filled_new = bytearray(new_capacity * bytes_longlong)

            # Create memoryviews for new buffers
            cdef double[::1] new_prices = memoryview(b_prices_new).cast('d')
            cdef double[::1] new_qtys = memoryview(b_qtys_new).cast('d')
            cdef char[::1] new_sides = memoryview(b_sides_new).cast('b')
            cdef char[::1] new_makers = memoryview(b_makers_new).cast('b')
            cdef long long[::1] new_ts = memoryview(b_ts_new).cast('q')
            cdef long long[::1] new_filled = memoryview(b_filled_new).cast('q')

            # Copy old data into new buffers (up to current counts).
            if self.trade_count > 0:
                new_prices[0:self.trade_count] = self.trade_prices[0:self.trade_count]
                new_qtys[0:self.trade_count] = self.trade_qtys[0:self.trade_count]
                new_sides[0:self.trade_count] = self.trade_sides[0:self.trade_count]
                new_makers[0:self.trade_count] = self.trade_is_agent_maker[0:self.trade_count]
                new_ts[0:self.trade_count] = self.trade_ts[0:self.trade_count]
            if self.filled_count > 0:
                new_filled[0:self.filled_count] = self.filled_order_ids[0:self.filled_count]

            # Assign new buffers to self (old buffers will be deallocated when their refs drop out of scope).
            self.trade_prices = new_prices
            self.trade_qtys = new_qtys
            self.trade_sides = new_sides
            self.trade_is_agent_maker = new_makers
            self.trade_ts = new_ts
            self.filled_order_ids = new_filled

        # Update internal capacity (no GIL needed here for plain C int).
        self._capacity = new_capacity

        # Assert that new buffers are C-contiguous (for safety in debug mode). 
        assert self.trade_prices.strides[0] == sizeof(double)
        assert self.trade_qtys.strides[0] == sizeof(double)
        assert self.trade_sides.strides[0] == sizeof(char)
        assert self.trade_is_agent_maker.strides[0] == sizeof(char)
        assert self.trade_ts.strides[0] == sizeof(long long)
        assert self.filled_order_ids.strides[0] == sizeof(long long)

    cdef void clear_step(self) nogil:
        """Reset the workspace for a new simulation step.

        This clears the counters for trades and filled orders, allowing reuse of the existing buffers 
        without resizing. The data in the buffers remains allocated (and may still hold old values), 
        but new writes will simply overwrite old data. This method should be called at the beginning 
        of each new simulation step to start fresh.
        """
        self.trade_count = 0
        self.filled_count = 0
        # Note: We do not clear the buffer contents for performance reasons. The trade_count and 
        # filled_count define the active range of data, and old data beyond these counts is ignored.

    cdef void push_trade(self, double price, double qty, char side, char is_agent_maker, long long ts) nogil:
        """Append a trade record to the workspace buffers.

        This records a new trade with the given price, quantity, side, agent maker flag, and timestamp.
        If necessary, the internal buffers are expanded to accommodate the new trade (which may involve 
        acquiring the GIL briefly).

        Args:
            price (double): The trade price.
            qty (double): The trade quantity.
            side (char): The trade side (e.g., 1 for buy, -1 for sell).
            is_agent_maker (char): Flag indicating if the agent was the maker (1) or taker (0) in this trade.
            ts (long long): The timestamp of the trade (in nanoseconds or appropriate unit).
        """
        cdef int idx = self.trade_count
        if idx >= self._capacity:
            # Need to grow the buffers to fit at least one more trade.
            self.ensure_capacity(idx + 1)
        # After ensure_capacity, it's safe to write the new trade at index idx.
        self.trade_prices[idx] = price
        self.trade_qtys[idx] = qty
        self.trade_sides[idx] = side
        self.trade_is_agent_maker[idx] = is_agent_maker
        self.trade_ts[idx] = ts
        self.trade_count += 1

    cdef void push_filled_order_id(self, long long order_id) nogil:
        """Append a filled order ID to the workspace buffer.

        This records the ID of an order that was completely filled during the step.
        If necessary, the internal buffer for filled order IDs is expanded (which may acquire the GIL).

        Args:
            order_id (long long): The identifier of the order that has been fully filled.
        """
        cdef int idx = self.filled_count
        if idx >= self._capacity:
            # Ensure there is space for the new filled order ID.
            self.ensure_capacity(idx + 1)
        self.filled_order_ids[idx] = order_id
        self.filled_count += 1
