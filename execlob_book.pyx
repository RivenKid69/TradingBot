# cython: language_level=3, boundscheck=False, wraparound=False

from libc.stdlib cimport malloc, realloc, free, rand
from libc.stddef cimport size_t
from libc.string cimport memcpy, memmove
cimport cython
from execevents cimport MarketEvent, EventType, Side
from coreworkspace cimport SimulationWorkspace

cdef class CythonLOB:

    """
    Cython implementation of a Limit Order Book (LOB) supporting basic operations.
    """

    def __cinit__(self):
        # Initialize with default capacity
        self.capacity_bids = 64
        self.capacity_asks = 64
        self.n_bids = 0
        self.n_asks = 0
        self.bid_orders = <MarketEvent*> malloc(self.capacity_bids * cython.sizeof(MarketEvent))
        self.ask_orders = <MarketEvent*> malloc(self.capacity_asks * cython.sizeof(MarketEvent))
        self.best_bid_index = -1
        self.best_ask_index = -1
        self._pending_market_is_agent = False
        self._has_error = False
        self._pending_exception = None

    def __dealloc__(self):
        # Free allocated memory
        if self.bid_orders != <MarketEvent*> 0:
            free(self.bid_orders)
        if self.ask_orders != <MarketEvent*> 0:
            free(self.ask_orders)
        self.bid_orders = <MarketEvent*> 0
        self.ask_orders = <MarketEvent*> 0

    cpdef CythonLOB clone(self):
        """
        Create a deep copy of the order book (for atomic step simulation).
        """
        cdef CythonLOB newlob = CythonLOB()
        # Ensure capacity in newlob arrays
        newlob._ensure_capacity(True, self.n_bids)
        newlob._ensure_capacity(False, self.n_asks)
        # Copy bid orders
        if self.n_bids > 0:
            memcpy(<void*> newlob.bid_orders, <void*> self.bid_orders,
                   <size_t> self.n_bids * cython.sizeof(MarketEvent))
        # Copy ask orders
        if self.n_asks > 0:
            memcpy(<void*> newlob.ask_orders, <void*> self.ask_orders,
                   <size_t> self.n_asks * cython.sizeof(MarketEvent))
        newlob.n_bids = self.n_bids
        newlob.n_asks = self.n_asks
        # Update best indices in clone
        newlob.best_bid_index = self.best_bid_index
        newlob.best_ask_index = self.best_ask_index
        newlob._has_error = False
        newlob._pending_exception = None
        return newlob

    cdef bint _ensure_capacity(self, bint is_bid, int min_capacity) noexcept nogil:
        """Ensure the internal array capacity for bids or asks is at least ``min_capacity``."""
        cdef int current
        cdef int new_cap
        cdef size_t bytes_required
        cdef MarketEvent* new_ptr

        if self._has_error:
            return False

        if is_bid:
            current = self.capacity_bids
        else:
            current = self.capacity_asks

        if min_capacity <= current:
            return True

        new_cap = current if current > 0 else 1
        while new_cap < min_capacity:
            if new_cap > (1 << 29):
                new_cap = min_capacity
                break
            new_cap = new_cap * 2

        bytes_required = <size_t> new_cap * cython.sizeof(MarketEvent)
        if is_bid:
            new_ptr = <MarketEvent*> realloc(self.bid_orders, bytes_required)
        else:
            new_ptr = <MarketEvent*> realloc(self.ask_orders, bytes_required)

        if new_ptr == <MarketEvent*> 0:
            with gil:
                if self._pending_exception is None:
                    self._pending_exception = MemoryError("Failed to grow order buffer")
                self._has_error = True
            return False

        if is_bid:
            self.bid_orders = new_ptr
            self.capacity_bids = new_cap
        else:
            self.ask_orders = new_ptr
            self.capacity_asks = new_cap

        return True

    cdef void _reset_error_state(self):
        self._has_error = False
        self._pending_exception = None

    cdef void _set_error(self, object exc) noexcept:
        if self._pending_exception is None:
            self._pending_exception = exc
        self._has_error = True

    cdef void _raise_pending_error(self):
        if self._has_error:
            exc = self._pending_exception
            self._pending_exception = None
            self._has_error = False
            if exc is None:
                raise MemoryError("CythonLOB encountered an unknown error")
            raise exc

    cdef int _find_order_index_by_id(self, int order_id, bint is_bid) noexcept nogil:
        """
        Find the index of order with given id in the specified side array.
        Returns -1 if not found.
        """
        cdef int i
        if is_bid:
            for i in range(self.n_bids):
                if self.bid_orders[i].order_id == order_id:
                    return i
        else:
            for i in range(self.n_asks):
                if self.ask_orders[i].order_id == order_id:
                    return i
        return -1

    cdef bint add_limit(self, int side, int price, int qty, bint is_agent, int order_id) noexcept nogil:
        """Add a limit order to the book, optionally matching against resting liquidity."""
        cdef int insert_idx

        if qty <= 0:
            return True
        if self._has_error:
            return False

        if side == Side.BUY:
            while qty > 0 and self.n_asks > 0:
                self.best_ask_index = 0
                if self.ask_orders[0].price <= price:
                    if self.ask_orders[0].qty <= qty:
                        qty -= self.ask_orders[0].qty
                        if self.n_asks > 1:
                            memmove(
                                <void*> self.ask_orders,
                                <void*> (self.ask_orders + 1),
                                <size_t> (self.n_asks - 1) * cython.sizeof(MarketEvent),
                            )
                        self.n_asks -= 1
                        if self.n_asks == 0:
                            self.best_ask_index = -1
                        else:
                            self.best_ask_index = 0
                    else:
                        self.ask_orders[0].qty -= qty
                        qty = 0
                else:
                    break
            if qty <= 0 or self._has_error:
                return not self._has_error
            if not self._ensure_capacity(True, self.n_bids + 1):
                return False
            insert_idx = 0
            while insert_idx < self.n_bids and self.bid_orders[insert_idx].price < price:
                insert_idx += 1
            if self.n_bids > insert_idx:
                memmove(
                    <void*> (self.bid_orders + insert_idx + 1),
                    <void*> (self.bid_orders + insert_idx),
                    <size_t> (self.n_bids - insert_idx) * cython.sizeof(MarketEvent),
                )
            self.bid_orders[insert_idx].type = EventType.PUBLIC_LIMIT_ADD
            self.bid_orders[insert_idx].side = Side.BUY
            self.bid_orders[insert_idx].price = price
            self.bid_orders[insert_idx].qty = qty
            self.bid_orders[insert_idx].order_id = order_id
            if is_agent:
                self.bid_orders[insert_idx].type = EventType.AGENT_LIMIT_ADD
            self.n_bids += 1
            self.best_bid_index = self.n_bids - 1
        else:
            while qty > 0 and self.n_bids > 0:
                self.best_bid_index = self.n_bids - 1
                if self.bid_orders[self.best_bid_index].price >= price:
                    if self.bid_orders[self.best_bid_index].qty <= qty:
                        qty -= self.bid_orders[self.best_bid_index].qty
                        self.n_bids -= 1
                        if self.n_bids == 0:
                            self.best_bid_index = -1
                        else:
                            self.best_bid_index = self.n_bids - 1
                    else:
                        self.bid_orders[self.best_bid_index].qty -= qty
                        qty = 0
                else:
                    break
            if qty <= 0 or self._has_error:
                return not self._has_error
            if not self._ensure_capacity(False, self.n_asks + 1):
                return False
            insert_idx = 0
            while insert_idx < self.n_asks and self.ask_orders[insert_idx].price < price:
                insert_idx += 1
            if self.n_asks > insert_idx:
                memmove(
                    <void*> (self.ask_orders + insert_idx + 1),
                    <void*> (self.ask_orders + insert_idx),
                    <size_t> (self.n_asks - insert_idx) * cython.sizeof(MarketEvent),
                )
            self.ask_orders[insert_idx].type = EventType.PUBLIC_LIMIT_ADD
            self.ask_orders[insert_idx].side = Side.SELL
            self.ask_orders[insert_idx].price = price
            self.ask_orders[insert_idx].qty = qty
            self.ask_orders[insert_idx].order_id = order_id
            if is_agent:
                self.ask_orders[insert_idx].type = EventType.AGENT_LIMIT_ADD
            self.n_asks += 1
            self.best_ask_index = 0

        return not self._has_error

    cdef bint cancel_order(self, int order_id) noexcept nogil:
        """Cancel (remove) an order by its ``order_id`` if it exists."""
        cdef int idx

        if self._has_error:
            return False

        idx = self._find_order_index_by_id(order_id, True)
        if idx != -1:
            if idx + 1 < self.n_bids:
                memmove(
                    <void*> (self.bid_orders + idx),
                    <void*> (self.bid_orders + idx + 1),
                    <size_t> (self.n_bids - idx - 1) * cython.sizeof(MarketEvent),
                )
            self.n_bids -= 1
            if self.n_bids == 0:
                self.best_bid_index = -1
            else:
                self.best_bid_index = self.n_bids - 1
            return True

        idx = self._find_order_index_by_id(order_id, False)
        if idx != -1:
            if idx + 1 < self.n_asks:
                memmove(
                    <void*> (self.ask_orders + idx),
                    <void*> (self.ask_orders + idx + 1),
                    <size_t> (self.n_asks - idx - 1) * cython.sizeof(MarketEvent),
                )
            self.n_asks -= 1
            if self.n_asks == 0:
                self.best_ask_index = -1
            else:
                self.best_ask_index = 0
        return True

    cdef bint _record_trade(self, SimulationWorkspace ws, int price, int qty, int side,
                            bint agent_maker, bint agent_taker, int maker_order_id) noexcept nogil:
        cdef int idx

        if self._has_error or ws.has_error():
            return False

        idx = ws.trade_count
        if not ws.push_trade(<double> price, <double> qty, <char> side,
                             <char> (1 if agent_maker else 0), 0):
            with gil:
                self._set_error(RuntimeError("SimulationWorkspace.push_trade failed"))
            return False
        if ws.has_error():
            with gil:
                self._set_error(RuntimeError("SimulationWorkspace reported an error"))
            return False

        ws.taker_is_agent_all_arr[idx] = <char> (1 if agent_taker else 0)
        if agent_maker:
            ws.maker_ids_all_arr[idx] = <unsigned long long> maker_order_id
        else:
            ws.maker_ids_all_arr[idx] = <unsigned long long> 0
        return True

    cdef bint match_market(self, int side, int qty, SimulationWorkspace ws) noexcept nogil:
        """Execute a market order of given side and quantity against the book."""
        cdef int remaining = qty
        cdef int trade_qty
        cdef int trade_price
        cdef int j
        cdef bint is_agent_market = self._pending_market_is_agent
        cdef bint maker_is_agent

        if qty <= 0:
            return True
        if self._has_error:
            return False

        if side == Side.BUY:
            while remaining > 0 and self.n_asks > 0 and not self._has_error and not ws.has_error():
                self.best_ask_index = 0
                trade_price = self.ask_orders[0].price
                maker_is_agent = self.ask_orders[0].type == EventType.AGENT_LIMIT_ADD
                if self.ask_orders[0].qty <= remaining:
                    trade_qty = self.ask_orders[0].qty
                    remaining -= trade_qty
                    if not self._record_trade(ws, trade_price, trade_qty, 1,
                                              maker_is_agent, is_agent_market, self.ask_orders[0].order_id):
                        return False
                    if maker_is_agent:
                        if not ws.push_filled_order_id(self.ask_orders[0].order_id):
                            with gil:
                                self._set_error(RuntimeError("Failed to push filled order id"))
                            return False
                        if ws.has_error():
                            return False
                    if self.n_asks > 1:
                        memmove(
                            <void*> self.ask_orders,
                            <void*> (self.ask_orders + 1),
                            <size_t> (self.n_asks - 1) * cython.sizeof(MarketEvent),
                        )
                    self.n_asks -= 1
                    if self.n_asks == 0:
                        self.best_ask_index = -1
                    else:
                        self.best_ask_index = 0
                else:
                    trade_qty = remaining
                    self.ask_orders[0].qty -= trade_qty
                    remaining = 0
                    if not self._record_trade(ws, trade_price, trade_qty, 1,
                                              maker_is_agent, is_agent_market, self.ask_orders[0].order_id):
                        return False
                    if maker_is_agent and self.ask_orders[0].qty == 0:
                        if not ws.push_filled_order_id(self.ask_orders[0].order_id):
                            with gil:
                                self._set_error(RuntimeError("Failed to push filled order id"))
                            return False
                        if ws.has_error():
                            return False
        else:
            while remaining > 0 and self.n_bids > 0 and not self._has_error and not ws.has_error():
                self.best_bid_index = self.n_bids - 1
                trade_price = self.bid_orders[self.best_bid_index].price
                maker_is_agent = self.bid_orders[self.best_bid_index].type == EventType.AGENT_LIMIT_ADD
                if self.bid_orders[self.best_bid_index].qty <= remaining:
                    trade_qty = self.bid_orders[self.best_bid_index].qty
                    remaining -= trade_qty
                    if not self._record_trade(ws, trade_price, trade_qty, -1,
                                              maker_is_agent, is_agent_market, self.bid_orders[self.best_bid_index].order_id):
                        return False
                    if maker_is_agent:
                        if not ws.push_filled_order_id(self.bid_orders[self.best_bid_index].order_id):
                            with gil:
                                self._set_error(RuntimeError("Failed to push filled order id"))
                            return False
                        if ws.has_error():
                            return False
                    self.n_bids -= 1
                    if self.n_bids == 0:
                        self.best_bid_index = -1
                    else:
                        self.best_bid_index = self.n_bids - 1
                else:
                    trade_qty = remaining
                    self.bid_orders[self.best_bid_index].qty -= trade_qty
                    remaining = 0
                    if not self._record_trade(ws, trade_price, trade_qty, -1,
                                              maker_is_agent, is_agent_market, self.bid_orders[self.best_bid_index].order_id):
                        return False
                    if maker_is_agent and self.bid_orders[self.best_bid_index].qty == 0:
                        if not ws.push_filled_order_id(self.bid_orders[self.best_bid_index].order_id):
                            with gil:
                                self._set_error(RuntimeError("Failed to push filled order id"))
                            return False
                        if ws.has_error():
                            return False

        return not self._has_error and not ws.has_error()

    cpdef double mid_price(self):
        """
        Compute the mid price of the book. If both sides present, returns (best_ask + best_bid)/2.
        If one side is empty, returns the available best price on the other side.
        Returns 0 if book is empty.
        """
        if self.n_bids == 0 and self.n_asks == 0:
            return 0.0
        elif self.n_bids == 0:
            # No bids, mid = best ask
            return <double> self.ask_orders[0].price
        elif self.n_asks == 0:
            # No asks, mid = best bid
            return <double> self.bid_orders[self.n_bids - 1].price
        else:
            self.best_bid_index = self.n_bids - 1
            self.best_ask_index = 0
            return (self.bid_orders[self.best_bid_index].price + self.ask_orders[self.best_ask_index].price) / 2.0

    cdef bint apply_events_batch_nogil(self, MarketEvent* events, int num_events, SimulationWorkspace ws) noexcept nogil:
        """Apply a batch of events (agent + public) to the order book."""
        cdef int i
        cdef int j

        if self._has_error or ws.has_error():
            return False

        self._pending_market_is_agent = False
        for i in range(num_events):
            if events[i].type == EventType.AGENT_LIMIT_ADD or events[i].type == EventType.PUBLIC_LIMIT_ADD:
                if not self.add_limit(events[i].side, events[i].price, events[i].qty,
                                      events[i].type == EventType.AGENT_LIMIT_ADD, events[i].order_id):
                    return False
                if self._has_error or ws.has_error():
                    return False

        for i in range(num_events):
            if events[i].type == EventType.AGENT_MARKET_MATCH or events[i].type == EventType.PUBLIC_MARKET_MATCH:
                self._pending_market_is_agent = events[i].type == EventType.AGENT_MARKET_MATCH
                if not self.match_market(events[i].side, events[i].qty, ws):
                    self._pending_market_is_agent = False
                    return False
                if self._has_error or ws.has_error():
                    self._pending_market_is_agent = False
                    return False
        self._pending_market_is_agent = False

        for i in range(num_events):
            if events[i].type == EventType.AGENT_CANCEL_SPECIFIC or events[i].type == EventType.PUBLIC_CANCEL_RANDOM:
                if events[i].type == EventType.AGENT_CANCEL_SPECIFIC:
                    if not self.cancel_order(events[i].order_id):
                        return False
                else:
                    if self.n_bids + self.n_asks == 0:
                        continue
                    if events[i].side == Side.BUY or events[i].side == Side.SELL:
                        if events[i].side == Side.BUY and self.n_bids > 0:
                            j = rand() % self.n_bids
                            if not self.cancel_order(self.bid_orders[j].order_id):
                                return False
                        elif events[i].side == Side.SELL and self.n_asks > 0:
                            j = rand() % self.n_asks
                            if not self.cancel_order(self.ask_orders[j].order_id):
                                return False
                    else:
                        j = rand() % (self.n_bids + self.n_asks)
                        if j < self.n_bids:
                            if not self.cancel_order(self.bid_orders[j].order_id):
                                return False
                        else:
                            if not self.cancel_order(self.ask_orders[j - self.n_bids].order_id):
                                return False
                if self._has_error:
                    return False

        return not self._has_error and not ws.has_error()


    cpdef void apply_events_batch(self, list events, SimulationWorkspace ws):
        """Apply a sequence of Python-level events to the book via the workspace."""
        cdef Py_ssize_t n = len(events)
        cdef MarketEvent* buffer
        cdef Py_ssize_t i
        cdef object evt
        cdef bint success

        ws.raise_pending_error()
        self._raise_pending_error()
        self._reset_error_state()

        if n == 0:
            return

        buffer = <MarketEvent*> malloc(n * cython.sizeof(MarketEvent))
        if buffer == <MarketEvent*> 0:
            raise MemoryError("Failed to allocate temporary event buffer")

        try:
            for i in range(n):
                evt = events[i]
                buffer[i].type = <EventType> <int> evt[0]
                buffer[i].side = <Side> <int> evt[1]
                buffer[i].price = <int> evt[2]
                buffer[i].qty = <int> evt[3]
                buffer[i].order_id = <int> evt[4]

            with nogil:
                success = self.apply_events_batch_nogil(buffer, <int> n, ws)
        finally:
            free(buffer)

        if ws.has_error():
            ws.raise_pending_error()
        if self._has_error:
            self._raise_pending_error()
        if not success:
            raise RuntimeError("CythonLOB failed to apply events batch")



    cpdef list iter_agent_orders(self):
        """Return a Python list of the current agent limit orders."""
        cdef list result = []
        cdef int i

        self._raise_pending_error()

        for i in range(self.n_bids):
            if self.bid_orders[i].type == EventType.AGENT_LIMIT_ADD:
                result.append((self.bid_orders[i].order_id, 1, self.bid_orders[i].price))
        for i in range(self.n_asks):
            if self.ask_orders[i].type == EventType.AGENT_LIMIT_ADD:
                result.append((self.ask_orders[i].order_id, -1, self.ask_orders[i].price))
        return result
