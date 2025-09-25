# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
# The above directives (if present) disable certain Python checks for performance.

from libc.stdlib cimport malloc, free  # (Optional: if using C memory allocation, but here we use Python memory)

# Import Python-level constants (ensuring no other project modules are used).
import core_constants as constants

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
    def __cinit__(self):
        # Initialize counters to zero and capacity to 0 (will allocate in __init__)
        self.trade_count = 0
        self.filled_count = 0
        self._capacity = 0
        self._has_error = False
        self._pending_exception = None

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

        # Allocate byte buffers for each array. Using bytearray ensures the
        # underlying storage is writable and remains alive while referenced by
        # typed memoryviews.
        cdef bytearray b_prices = bytearray(capacity * bytes_double)
        cdef bytearray b_qtys = bytearray(capacity * bytes_double)
        cdef bytearray b_sides = bytearray(capacity * sizeof(char))
        cdef bytearray b_makers = bytearray(capacity * sizeof(char))
        cdef bytearray b_ts = bytearray(capacity * bytes_longlong)
        cdef bytearray b_filled = bytearray(capacity * bytes_longlong)
        cdef bytearray b_maker_ids = bytearray(capacity * bytes_longlong)
        cdef bytearray b_taker_flags = bytearray(capacity * sizeof(char))

        # Cast byte buffers to typed memoryviews (C-contiguous arrays)
        self.trade_prices = memoryview(b_prices).cast('d')   # double -> 'd'
        self.trade_qtys = memoryview(b_qtys).cast('d')
        self.trade_sides = memoryview(b_sides).cast('b')     # signed char -> 'b'
        self.trade_is_agent_maker = memoryview(b_makers).cast('b')
        self.trade_ts = memoryview(b_ts).cast('q')           # long long -> 'q'
        self.filled_order_ids = memoryview(b_filled).cast('q')
        self.maker_ids_all_arr = memoryview(b_maker_ids).cast('Q')
        self.taker_is_agent_all_arr = memoryview(b_taker_flags).cast('b')

        # Retain references to the Python buffers for the lifetime of the workspace.
        self._buf_trade_prices = b_prices
        self._buf_trade_qtys = b_qtys
        self._buf_trade_sides = b_sides
        self._buf_trade_is_agent_maker = b_makers
        self._buf_trade_ts = b_ts
        self._buf_filled_order_ids = b_filled
        self._buf_maker_ids = b_maker_ids
        self._buf_taker_flags = b_taker_flags

        # Legacy aliases for backwards compatibility with lob_state_cython
        self.prices_all_arr = self.trade_prices
        self.volumes_all_arr = self.trade_qtys
        self.is_buy_side_all_arr = self.trade_sides
        self.maker_is_agent_all_arr = self.trade_is_agent_maker
        self.timestamps_all_arr = self.trade_ts
        self.fully_executed_ids_all_arr = self.filled_order_ids

        # Set the internal capacity and verify buffers are C-contiguous.
        self._capacity = capacity
        assert self.trade_prices.strides[0] == sizeof(double)
        assert self.trade_qtys.strides[0] == sizeof(double)
        assert self.trade_sides.strides[0] == sizeof(char)
        assert self.trade_is_agent_maker.strides[0] == sizeof(char)
        assert self.trade_ts.strides[0] == sizeof(long long)
        assert self.filled_order_ids.strides[0] == sizeof(long long)
        assert self.maker_ids_all_arr.strides[0] == sizeof(unsigned long long)
        assert self.taker_is_agent_all_arr.strides[0] == sizeof(char)
        # Note: The above asserts ensure each memoryview has contiguous stride (1 element step).
        # They rely on the fact that memoryview.strides is accessible with GIL (we are in __init__).

    cdef void ensure_capacity(self, int min_capacity):
        """Ensure the internal buffers have capacity for at least ``min_capacity`` elements."""
        cdef int new_capacity, current_capacity

        current_capacity = self._capacity
        if min_capacity <= current_capacity:
            return

        new_capacity = current_capacity * 2
        if new_capacity < min_capacity:
            new_capacity = min_capacity

        self._resize_buffers(new_capacity)
        self._capacity = new_capacity

    cdef void _resize_buffers(self, int new_capacity):
        """Resize all workspace buffers to ``new_capacity`` while holding the GIL."""
        cdef int bytes_double = sizeof(double)
        cdef int bytes_longlong = sizeof(long long)

        cdef double[::1] old_prices = self.trade_prices
        cdef double[::1] old_qtys = self.trade_qtys
        cdef char[::1] old_sides = self.trade_sides
        cdef char[::1] old_makers = self.trade_is_agent_maker
        cdef long long[::1] old_ts = self.trade_ts
        cdef long long[::1] old_filled = self.filled_order_ids
        cdef unsigned long long[::1] old_maker_ids = self.maker_ids_all_arr
        cdef char[::1] old_taker_flags = self.taker_is_agent_all_arr

        cdef bytearray b_prices_new = bytearray(new_capacity * bytes_double)
        cdef bytearray b_qtys_new = bytearray(new_capacity * bytes_double)
        cdef bytearray b_sides_new = bytearray(new_capacity * sizeof(char))
        cdef bytearray b_makers_new = bytearray(new_capacity * sizeof(char))
        cdef bytearray b_ts_new = bytearray(new_capacity * bytes_longlong)
        cdef bytearray b_filled_new = bytearray(new_capacity * bytes_longlong)
        cdef bytearray b_maker_ids_new = bytearray(new_capacity * bytes_longlong)
        cdef bytearray b_taker_flags_new = bytearray(new_capacity * sizeof(char))

        cdef double[::1] new_prices = memoryview(b_prices_new).cast('d')
        cdef double[::1] new_qtys = memoryview(b_qtys_new).cast('d')
        cdef char[::1] new_sides = memoryview(b_sides_new).cast('b')
        cdef char[::1] new_makers = memoryview(b_makers_new).cast('b')
        cdef long long[::1] new_ts = memoryview(b_ts_new).cast('q')
        cdef long long[::1] new_filled = memoryview(b_filled_new).cast('q')
        cdef unsigned long long[::1] new_maker_ids = memoryview(b_maker_ids_new).cast('Q')
        cdef char[::1] new_taker_flags = memoryview(b_taker_flags_new).cast('b')

        if self.trade_count > 0:
            new_prices[0:self.trade_count] = old_prices[0:self.trade_count]
            new_qtys[0:self.trade_count] = old_qtys[0:self.trade_count]
            new_sides[0:self.trade_count] = old_sides[0:self.trade_count]
            new_makers[0:self.trade_count] = old_makers[0:self.trade_count]
            new_ts[0:self.trade_count] = old_ts[0:self.trade_count]
            new_maker_ids[0:self.trade_count] = old_maker_ids[0:self.trade_count]
            new_taker_flags[0:self.trade_count] = old_taker_flags[0:self.trade_count]
        if self.filled_count > 0:
            new_filled[0:self.filled_count] = old_filled[0:self.filled_count]

        self.trade_prices = new_prices
        self.trade_qtys = new_qtys
        self.trade_sides = new_sides
        self.trade_is_agent_maker = new_makers
        self.trade_ts = new_ts
        self.filled_order_ids = new_filled
        self.maker_ids_all_arr = new_maker_ids
        self.taker_is_agent_all_arr = new_taker_flags

        self._buf_trade_prices = b_prices_new
        self._buf_trade_qtys = b_qtys_new
        self._buf_trade_sides = b_sides_new
        self._buf_trade_is_agent_maker = b_makers_new
        self._buf_trade_ts = b_ts_new
        self._buf_filled_order_ids = b_filled_new
        self._buf_maker_ids = b_maker_ids_new
        self._buf_taker_flags = b_taker_flags_new

        self.prices_all_arr = self.trade_prices
        self.volumes_all_arr = self.trade_qtys
        self.is_buy_side_all_arr = self.trade_sides
        self.maker_is_agent_all_arr = self.trade_is_agent_maker
        self.timestamps_all_arr = self.trade_ts
        self.fully_executed_ids_all_arr = self.filled_order_ids

        assert self.trade_prices.strides[0] == sizeof(double)
        assert self.trade_qtys.strides[0] == sizeof(double)
        assert self.trade_sides.strides[0] == sizeof(char)
        assert self.trade_is_agent_maker.strides[0] == sizeof(char)
        assert self.trade_ts.strides[0] == sizeof(long long)
        assert self.filled_order_ids.strides[0] == sizeof(long long)
        assert self.maker_ids_all_arr.strides[0] == sizeof(unsigned long long)
        assert self.taker_is_agent_all_arr.strides[0] == sizeof(char)

    cdef void clear_step(self) nogil:
        """Reset the workspace for a new simulation step.

        This clears the counters for trades and filled orders, allowing reuse of the existing buffers
        without resizing. The data in the buffers remains allocated (and may still hold old values),
        but new writes will simply overwrite old data. This method should be called at the beginning
        of each new simulation step to start fresh.
        """
        self.trade_count = 0
        self.filled_count = 0
        self._has_error = False
        with gil:
            self._pending_exception = None
        # Note: We do not clear the buffer contents for performance reasons. The trade_count and 
        # filled_count define the active range of data, and old data beyond these counts is ignored.

    cdef bint push_trade(self, double price, double qty, char side, char is_agent_maker, long long ts) noexcept nogil:
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
        cdef int idx
        if self._has_error:
            return False

        idx = self.trade_count
        if idx >= self._capacity:
            # Need to grow the buffers to fit at least one more trade.
            with gil:
                if not self._has_error:
                    try:
                        self.ensure_capacity(idx + 1)
                    except Exception as exc:
                        self._has_error = True
                        self._pending_exception = exc
                        return False
                else:
                    return False
        if self._has_error:
            return False
        # After ensure_capacity, it's safe to write the new trade at index idx.
        self.trade_prices[idx] = price
        self.trade_qtys[idx] = qty
        self.trade_sides[idx] = side
        self.trade_is_agent_maker[idx] = is_agent_maker
        self.trade_ts[idx] = ts
        self.taker_is_agent_all_arr[idx] = <char>0
        self.maker_ids_all_arr[idx] = <unsigned long long>0
        self.trade_count += 1
        return True

    cdef bint push_filled_order_id(self, long long order_id) noexcept nogil:
        """Append a filled order ID to the workspace buffer.

        This records the ID of an order that was completely filled during the step.
        If necessary, the internal buffer for filled order IDs is expanded (which may acquire the GIL).

        Args:
            order_id (long long): The identifier of the order that has been fully filled.
        """
        cdef int idx
        if self._has_error:
            return False

        idx = self.filled_count
        if idx >= self._capacity:
            # Ensure there is space for the new filled order ID.
            with gil:
                if not self._has_error:
                    try:
                        self.ensure_capacity(idx + 1)
                    except Exception as exc:
                        self._has_error = True
                        self._pending_exception = exc
                        return False
                else:
                    return False
        if self._has_error:
            return False
        self.filled_order_ids[idx] = order_id
        self.filled_count += 1
        return True

    cdef bint has_error(self) noexcept nogil:
        return self._has_error

    cpdef void raise_pending_error(self):
        if self._has_error:
            exc = self._pending_exception
            self._pending_exception = None
            self._has_error = False
            if exc is None:
                raise MemoryError("SimulationWorkspace encountered an unknown error")
            raise exc
