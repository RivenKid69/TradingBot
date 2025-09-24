# cython: language_level=3

from libc.stdlib cimport malloc, realloc, free, rand
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
        for i in range(self.n_bids):
            newlob.bid_orders[i] = self.bid_orders[i]
        # Copy ask orders
        for i in range(self.n_asks):
            newlob.ask_orders[i] = self.ask_orders[i]
        newlob.n_bids = self.n_bids
        newlob.n_asks = self.n_asks
        # Update best indices in clone
        newlob.best_bid_index = self.best_bid_index
        newlob.best_ask_index = self.best_ask_index
        return newlob

    cdef void _ensure_capacity(self, bint is_bid, int min_capacity) nogil:
        """
        Ensure the internal array capacity for bids or asks is at least min_capacity.
        """
        cdef int new_cap
        if is_bid:
            if min_capacity <= self.capacity_bids:
                return
            new_cap = self.capacity_bids
            while new_cap < min_capacity:
                new_cap = new_cap * 2
            self.bid_orders = <MarketEvent*> realloc(self.bid_orders, new_cap * cython.sizeof(MarketEvent))
            self.capacity_bids = new_cap
        else:
            if min_capacity <= self.capacity_asks:
                return
            new_cap = self.capacity_asks
            while new_cap < min_capacity:
                new_cap = new_cap * 2
            self.ask_orders = <MarketEvent*> realloc(self.ask_orders, new_cap * cython.sizeof(MarketEvent))
            self.capacity_asks = new_cap

    cdef int _find_order_index_by_id(self, int order_id, bint is_bid) nogil:
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

    cdef void add_limit(self, int side, int price, int qty, bint is_agent, int order_id) nogil:
        """
        Add a limit order to the book. If the order crosses the opposing side, 
        it will match available volume first, then any remainder is added.
        side: Side.BUY (1) for buy order, Side.SELL (-1) for sell order.
        price: price in ticks.
        qty: quantity of the order.
        is_agent: whether this order belongs to the agent.
        order_id: unique identifier for the order.
        """
        cdef int i, insert_idx

        if qty <= 0:
            return

        if side == Side.BUY:
            # Matching against asks (sell orders) at or below price
            while qty > 0 and self.n_asks > 0:
                # Best ask is the lowest ask price
                self.best_ask_index = 0  # best ask at index 0 (since we maintain asks sorted ascending by price)
                if self.ask_orders[0].price <= price:
                    # There is an ask at price <= buy price -> trade occurs at ask price
                    if self.ask_orders[0].qty <= qty:
                        # The ask order is fully filled
                        qty -= self.ask_orders[0].qty
                        # Record trade in SimulationWorkspace (done in match_market for uniformity, so skip here)
                        # Remove this ask order from book (shift array)
                        for i in range(1, self.n_asks):
                            self.ask_orders[i-1] = self.ask_orders[i]
                        self.n_asks -= 1
                    else:
                        # The ask order has more quantity than the buy order; partial fill
                        self.ask_orders[0].qty -= qty
                        qty = 0
                    # Continue matching until qty is 0 or no asks at price <= limit price
                else:
                    break
            if qty <= 0:
                return  # fully matched by existing asks, nothing to add
            # If remaining qty, add as new bid order
            # Ensure capacity for new bid
            self._ensure_capacity(True, self.n_bids + 1)
            # Find insertion index to keep bid_orders sorted ascending by price (lowest first)
            insert_idx = 0
            while insert_idx < self.n_bids and self.bid_orders[insert_idx].price < price:
                insert_idx += 1
            # Shift orders to make space at insert_idx
            for i in range(self.n_bids, insert_idx, -1):
                self.bid_orders[i] = self.bid_orders[i-1]
            # Insert new order
            self.bid_orders[insert_idx].type = EventType.PUBLIC_LIMIT_ADD  # default label, will be updated by context if needed
            self.bid_orders[insert_idx].side = Side.BUY
            self.bid_orders[insert_idx].price = price
            self.bid_orders[insert_idx].qty = qty
            self.bid_orders[insert_idx].order_id = order_id
            if is_agent:
                # Mark agent order by event type (for clarity) - uses AGENT_LIMIT_ADD for identification if needed
                self.bid_orders[insert_idx].type = EventType.AGENT_LIMIT_ADD
            self.n_bids += 1
            # Update best bid index (should be last index for highest price since list ascending)
            # The highest price bid is at the end of list
            self.best_bid_index = self.n_bids - 1

        else:
            # side == Side.SELL
            # Matching against bids (buy orders) at or above price
            while qty > 0 and self.n_bids > 0:
                # Best bid is the highest bid price (at end of array since ascending sort)
                self.best_bid_index = self.n_bids - 1
                if self.bid_orders[self.best_bid_index].price >= price:
                    # There is a bid at price >= sell price -> trade occurs at bid price
                    if self.bid_orders[self.best_bid_index].qty <= qty:
                        qty -= self.bid_orders[self.best_bid_index].qty
                        # Remove this bid order from book
                        self.n_bids -= 1
                        # (no need to shift because we remove last element in array due to highest price at end)
                    else:
                        self.bid_orders[self.best_bid_index].qty -= qty
                        qty = 0
                    # Continue matching until qty is 0 or no bids at price >= limit price
                else:
                    break
            if qty <= 0:
                return  # fully matched by existing bids
            # Add remaining as new ask order
            self._ensure_capacity(False, self.n_asks + 1)
            insert_idx = 0
            while insert_idx < self.n_asks and self.ask_orders[insert_idx].price < price:
                insert_idx += 1
            for i in range(self.n_asks, insert_idx, -1):
                self.ask_orders[i] = self.ask_orders[i-1]
            self.ask_orders[insert_idx].type = EventType.PUBLIC_LIMIT_ADD
            self.ask_orders[insert_idx].side = Side.SELL
            self.ask_orders[insert_idx].price = price
            self.ask_orders[insert_idx].qty = qty
            self.ask_orders[insert_idx].order_id = order_id
            if is_agent:
                self.ask_orders[insert_idx].type = EventType.AGENT_LIMIT_ADD
            self.n_asks += 1
            # Update best ask index (should always be 0 for lowest price ask since sorted ascending)
            self.best_ask_index = 0

    cdef void cancel_order(self, int order_id) nogil:
        """
        Cancel (remove) an order by its order_id if it exists.
        """
        cdef int idx
        cdef int j
        # Try find in bids
        idx = self._find_order_index_by_id(order_id, True)
        if idx != -1:
            # Remove bid order at idx
            for j in range(idx + 1, self.n_bids):
                self.bid_orders[j - 1] = self.bid_orders[j]
            self.n_bids -= 1
            return
        # Try find in asks
        idx = self._find_order_index_by_id(order_id, False)
        if idx != -1:
            for j in range(idx + 1, self.n_asks):
                self.ask_orders[j - 1] = self.ask_orders[j]
            self.n_asks -= 1
            return

    cdef void _record_trade(self, SimulationWorkspace ws, int price, int qty, int side,
                            bint agent_maker, bint agent_taker, int maker_order_id) nogil:
        cdef int idx = ws.trade_count
        ws.push_trade(<double> price, <double> qty, <char> side,
                      <char> (1 if agent_maker else 0), 0)
        ws.taker_is_agent_all_arr[idx] = <char> (1 if agent_taker else 0)
        if agent_maker:
            ws.maker_ids_all_arr[idx] = <unsigned long long> maker_order_id
        else:
            ws.maker_ids_all_arr[idx] = <unsigned long long> 0

    cdef void match_market(self, int side, int qty, SimulationWorkspace ws) nogil:
        """
        Execute a market order of given side and quantity against the book.
        Consumes orders from the opposite side up to the given qty.
        Records trades in SimulationWorkspace.
        """
        cdef int remaining = qty
        cdef int trade_qty
        cdef int trade_price
        cdef int j
        cdef bint is_agent_market = self._pending_market_is_agent
        if qty <= 0:
            return
        if side == Side.BUY:
            # Market buy: take from lowest ask prices
            while remaining > 0 and self.n_asks > 0:
                self.best_ask_index = 0
                trade_price = self.ask_orders[0].price
                if self.ask_orders[0].qty <= remaining:
                    trade_qty = self.ask_orders[0].qty
                    remaining -= trade_qty
                    # Record trade in SimulationWorkspace
                    self._record_trade(ws, trade_price, trade_qty, 1,
                                       self.ask_orders[0].type == EventType.AGENT_LIMIT_ADD,
                                       is_agent_market, self.ask_orders[0].order_id)
                    if self.ask_orders[0].type == EventType.AGENT_LIMIT_ADD:
                        ws.push_filled_order_id(self.ask_orders[0].order_id)
                    # Remove the ask order
                    for j in range(1, self.n_asks):
                        self.ask_orders[j - 1] = self.ask_orders[j]
                    self.n_asks -= 1
                else:
                    # Partial fill of the ask order
                    trade_qty = remaining
                    self.ask_orders[0].qty -= trade_qty
                    remaining = 0
                    self._record_trade(ws, trade_price, trade_qty, 1,
                                       self.ask_orders[0].type == EventType.AGENT_LIMIT_ADD,
                                       is_agent_market, self.ask_orders[0].order_id)
                    if self.ask_orders[0].type == EventType.AGENT_LIMIT_ADD and self.ask_orders[0].qty == 0:
                        ws.push_filled_order_id(self.ask_orders[0].order_id)
                    # remaining = 0 will break loop
            # Note: any remaining unmatched volume is discarded (market order not fully filled)
        else:
            # side == Side.SELL
            # Market sell: take from highest bid prices
            while remaining > 0 and self.n_bids > 0:
                self.best_bid_index = self.n_bids - 1
                trade_price = self.bid_orders[self.best_bid_index].price
                if self.bid_orders[self.best_bid_index].qty <= remaining:
                    trade_qty = self.bid_orders[self.best_bid_index].qty
                    remaining -= trade_qty
                    self._record_trade(ws, trade_price, trade_qty, -1,
                                       self.bid_orders[self.best_bid_index].type == EventType.AGENT_LIMIT_ADD,
                                       is_agent_market, self.bid_orders[self.best_bid_index].order_id)
                    if self.bid_orders[self.best_bid_index].type == EventType.AGENT_LIMIT_ADD:
                        ws.push_filled_order_id(self.bid_orders[self.best_bid_index].order_id)
                    # Remove this bid
                    self.n_bids -= 1
                else:
                    trade_qty = remaining
                    self.bid_orders[self.best_bid_index].qty -= trade_qty
                    remaining = 0
                    self._record_trade(ws, trade_price, trade_qty, -1,
                                       self.bid_orders[self.best_bid_index].type == EventType.AGENT_LIMIT_ADD,
                                       is_agent_market, self.bid_orders[self.best_bid_index].order_id)
                    if self.bid_orders[self.best_bid_index].type == EventType.AGENT_LIMIT_ADD and self.bid_orders[self.best_bid_index].qty == 0:
                        ws.push_filled_order_id(self.bid_orders[self.best_bid_index].order_id)

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

    cdef void apply_events_batch_nogil(self, MarketEvent* events, int num_events, SimulationWorkspace ws) nogil:
        """
        Apply a batch of events (agent + public) to the order book under nogil.
        Events are processed in an order: all limit adds, then all market matches, then all cancels.
        This enforces atomic step processing.
        """
        cdef int i
        cdef int j
        self._pending_market_is_agent = False
        # Process limit add events first
        for i in range(num_events):
            if events[i].type == EventType.AGENT_LIMIT_ADD or events[i].type == EventType.PUBLIC_LIMIT_ADD:
                self.add_limit(events[i].side, events[i].price, events[i].qty,
                               events[i].type == EventType.AGENT_LIMIT_ADD, events[i].order_id)
        # Process market match events
        for i in range(num_events):
            if events[i].type == EventType.AGENT_MARKET_MATCH or events[i].type == EventType.PUBLIC_MARKET_MATCH:
                self._pending_market_is_agent = events[i].type == EventType.AGENT_MARKET_MATCH
                self.match_market(events[i].side, events[i].qty, ws)
        self._pending_market_is_agent = False
        # Process cancel events
        for i in range(num_events):
            if events[i].type == EventType.AGENT_CANCEL_SPECIFIC or events[i].type == EventType.PUBLIC_CANCEL_RANDOM:
                if events[i].type == EventType.AGENT_CANCEL_SPECIFIC:
                    # Cancel specific order by id
                    self.cancel_order(events[i].order_id)
                else:
                    # PUBLIC_CANCEL_RANDOM: cancel a random order from the specified side (or either side if side=0)
                    if self.n_bids + self.n_asks == 0:
                        continue
                    if events[i].side == Side.BUY or events[i].side == Side.SELL:
                        # Cancel random order from the given side
                        if events[i].side == Side.BUY and self.n_bids > 0:
                            j = rand() % self.n_bids
                            self.cancel_order(self.bid_orders[j].order_id)
                        elif events[i].side == Side.SELL and self.n_asks > 0:
                            j = rand() % self.n_asks
                            self.cancel_order(self.ask_orders[j].order_id)
                    else:
                        # Cancel random order from either side
                        j = rand() % (self.n_bids + self.n_asks)
                        if j < self.n_bids:
                            self.cancel_order(self.bid_orders[j].order_id)
                        else:
                            self.cancel_order(self.ask_orders[j - self.n_bids].order_id)

    cpdef void apply_events_batch(self, list events, SimulationWorkspace ws):
        """Apply a sequence of Python-level events to the book via the workspace."""
        cdef Py_ssize_t n = len(events)
        cdef MarketEvent* buffer
        cdef Py_ssize_t i
        cdef object evt

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
                self.apply_events_batch_nogil(buffer, <int> n, ws)
        finally:
            free(buffer)

    cpdef list iter_agent_orders(self):
        """Return a Python list of the current agent limit orders."""
        cdef list result = []
        cdef int i
        for i in range(self.n_bids):
            if self.bid_orders[i].type == EventType.AGENT_LIMIT_ADD:
                result.append((self.bid_orders[i].order_id, 1, self.bid_orders[i].price))
        for i in range(self.n_asks):
            if self.ask_orders[i].type == EventType.AGENT_LIMIT_ADD:
                result.append((self.ask_orders[i].order_id, -1, self.ask_orders[i].price))
        return result
