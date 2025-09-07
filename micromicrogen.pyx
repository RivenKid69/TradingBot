# cython: language_level=3, boundscheck=False, wraparound=False
from libc.stdint cimport uint32_t, uint64_t
from libc.math cimport exp

# Import PRICE_SCALE constant from core.constants (Python module)
import core.constants as _const
cdef int PRICE_SCALE = _const.PRICE_SCALE

cdef class CyMicrostructureGenerator:
    """Cython class to generate public market microstructure events under nogil."""
    cdef uint64_t _state        # PCG32 RNG state
    cdef uint64_t _inc          # PCG32 RNG increment (stream id)
    cdef double momentum_factor
    cdef double mean_reversion_factor
    cdef double base_order_imbalance_ratio
    cdef double base_cancel_ratio
    cdef double adversarial_factor
    cdef int _last_side         # Last event side (1 for buy, -1 for sell, 0 if none)
    cdef int current_price      # Approximate last trade price (in ticks)
    cdef int best_bid           # Best bid price (tick)
    cdef int best_ask           # Best ask price (tick)

    def __cinit__(self):
        # Initialize default parameters and RNG state
        self._state = 0
        self._inc = 0x14057B7EF767814F  # default stream (odd constant)
        self.momentum_factor = 0.0
        self.mean_reversion_factor = 0.0
        self.base_order_imbalance_ratio = 1.0
        self.base_cancel_ratio = 0.0
        self.adversarial_factor = 0.0
        self._last_side = 0
        # Set a default initial price and spread (avoid negative prices)
        self.current_price = 100 * PRICE_SCALE
        if self.current_price < 0:
            self.current_price = 0
        self.best_bid = self.current_price - 1 if self.current_price > 0 else 0
        self.best_ask = self.current_price + 1

    cpdef void seed(self, uint64_t seed):
        """Seed the internal PCG32 random number generator with a 64-bit seed."""
        # NOTE: Using PCG32 seeding routine for reproducibility
        self._state = 0
        self._inc = (seed << 1) | 1  # ensure increment is odd
        # Advance state with initial sequence selection and state injection
        self._pcg32_random()        # discard first output, updates state
        self._state += seed        # mix in the seed as initial state
        self._pcg32_random()        # advance again to finalize state
        self._last_side = 0        # reset momentum tracking
        # Reset price and spread to default values (optional)
        self.current_price = 100 * PRICE_SCALE
        self.best_bid = self.current_price - 1 if self.current_price > 0 else 0
        self.best_ask = self.current_price + 1

    cpdef void set_regime(self,
                          double base_order_imbalance_ratio,
                          double base_cancel_ratio,
                          double momentum_factor,
                          double mean_reversion_factor,
                          double adversarial_factor):
        """Set the microstructure regime parameters for event generation."""
        self.base_order_imbalance_ratio = base_order_imbalance_ratio
        self.base_cancel_ratio = base_cancel_ratio
        self.momentum_factor = momentum_factor
        self.mean_reversion_factor = mean_reversion_factor
        self.adversarial_factor = adversarial_factor

    cpdef int generate_public_events(self, object out_events, int max_events):
        """Generate public market events and fill the out_events buffer (owner=0).
        
        Returns the number of events generated. This function performs all 
        event generation under nogil for performance. The output buffer out_events 
        should be a contiguous memoryview of MarketEvent objects.
        """
        cdef int i, events_count
        # Acquire a typed memoryview for output events (1D contiguous)
        cdef MarketEvent[::1] ev_buf = out_events  # memoryview of MarketEvent
        # Local copies of state for nogil operations
        cdef uint64_t state = self._state
        cdef uint64_t inc = self._inc
        cdef int last_side = self._last_side
        cdef int best_bid = self.best_bid
        cdef int best_ask = self.best_ask
        cdef int cur_price = self.current_price

        # Calculate base probabilities
        cdef double pb_base = 0.5
        if self.base_order_imbalance_ratio > 0.0:
            pb_base = self.base_order_imbalance_ratio / (1.0 + self.base_order_imbalance_ratio)
        elif self.base_order_imbalance_ratio == 0.0:
            pb_base = 0.0  # extreme case: no buy flow if ratio is 0

        # Compute expected events count (Poisson intensity = 1 + adversarial_factor)
        cdef double lam = 1.0 + self.adversarial_factor
        if lam < 0.0:
            lam = 0.0
        # Sample events_count ~ Poisson(lam) using inversion by multiplication
        cdef double L = exp(-lam)
        cdef double p = 1.0
        events_count = 0
        # Draw Poisson outcome
        while True:
            # Draw uniform [0,1) from PCG32
            p *= (<double>(state >> 18 ^ state) * 2.3283064365386963e-10)  # Using PCG output as uniform
            # The above uses an inline PCG step for efficiency (XSH RR output)
            # Actually, ensure to advance state properly:
            if p <= L:
                break
            events_count += 1
            # Advance PCG state manually inside loop for subsequent randoms
            state = state * 6364136223846793005ULL + (inc | 1ULL)
            if events_count > max_events:
                # Cap events at max_events to avoid overflow
                events_count = max_events
                break

        if events_count > max_events:
            events_count = max_events

        # Generate each event
        with nogil:
            # Use the latest state/inc values for RNG under nogil
            self._state = state
            self._inc = inc
            for i in range(events_count):
                # Determine event type (cancel or order) based on base_cancel_ratio
                cdef double u = self._rand_uniform()
                cdef int event_type
                if u < self.base_cancel_ratio and (best_bid > 0 or best_ask > 0):
                    event_type = PUBLIC_CANCEL_RANDOM
                else:
                    # Determine if event is a market order or a limit add
                    cdef double u_type = self._rand_uniform()
                    # Base probability of market order (more with momentum, less with mean reversion)
                    cdef double p_market = 0.5 + 0.5 * (self.momentum_factor - self.mean_reversion_factor)
                    if p_market < 0.0:
                        p_market = 0.0
                    elif p_market > 1.0:
                        p_market = 1.0
                    if u_type < p_market:
                        event_type = PUBLIC_MARKET_MATCH
                    else:
                        event_type = PUBLIC_LIMIT_ADD
                # Determine side of event (buy or sell) with momentum/mean-reversion adjustments
                cdef double pb = pb_base
                if last_side == 1:
                    # Last event was buy: momentum favors buy, reversion favors sell
                    pb = pb_base + self.momentum_factor - self.mean_reversion_factor
                elif last_side == -1:
                    # Last event was sell: momentum favors sell (reducing buy prob), reversion favors buy
                    pb = pb_base - self.momentum_factor + self.mean_reversion_factor
                if pb < 0.0:
                    pb = 0.0
                elif pb > 1.0:
                    pb = 1.0
                cdef double u_side = self._rand_uniform()
                cdef int side = 1 if u_side < pb else -1  # 1 for buy, -1 for sell

                # Prepare event data
                ev_buf[i].owner = 0
                ev_buf[i].side = side
                ev_buf[i].type = event_type
                # Default price and qty
                ev_buf[i].price = 0
                ev_buf[i].qty = 1

                if event_type == PUBLIC_LIMIT_ADD:
                    # Generate a limit order (no immediate match)
                    cdef int price = 0
                    if side == 1:
                        # Buy limit: place at or below best_ask - 1
                        cdef int max_price = best_ask - 1 if best_ask > 0 else best_bid
                        if max_price < 0:
                            max_price = 0
                        cdef int range = 5 + <int>(self.adversarial_factor * 5)
                        if range < 0:
                            range = 0
                        # random offset within [0, range]
                        cdef uint32_t rnd = self._pcg32_random_fast()
                        cdef int offset = 0
                        if range > 0:
                            offset = rnd % (range + 1)
                        price = best_bid
                        if offset > 0:
                            # Add offset but do not exceed max_price
                            cdef long long cand_price = best_bid + offset
                            price = cand_price if cand_price <= max_price else max_price
                        # Update best_bid if improved
                        if price > best_bid:
                            best_bid = price
                        # best_ask remains unchanged
                    else:
                        # Sell limit: place at or above best_bid + 1
                        cdef int min_price = best_bid + 1
                        if min_price < 0:
                            min_price = 0
                        cdef int range = 5 + <int>(self.adversarial_factor * 5)
                        if range < 0:
                            range = 0
                        cdef uint32_t rnd = self._pcg32_random_fast()
                        cdef int offset = 0
                        if range > 0:
                            offset = rnd % (range + 1)
                        price = best_ask
                        if offset > 0:
                            # Subtract offset but ensure not below min_price
                            cdef long long cand_price = best_ask - offset
                            price = cand_price if cand_price >= min_price else min_price
                        # Update best_ask if improved (lowered)
                        if price < best_ask:
                            best_ask = price
                        # best_bid remains unchanged
                    ev_buf[i].price = price if price >= 0 else 0
                    # Quantity: at least 1, add adversarial factor influence
                    cdef int base_max_qty = 5
                    cdef int add_range = <int>(self.adversarial_factor * 10)
                    if add_range < 0:
                        add_range = 0
                    cdef uint32_t rndq = self._pcg32_random_fast()
                    cdef int qty = 1
                    if base_max_qty + add_range > 1:
                        qty = 1 + (rndq % (base_max_qty + add_range))
                    if qty < 1:
                        qty = 1
                    ev_buf[i].qty = qty
                    # current_price (last trade) remains unchanged (no trade occurred)
                elif event_type == PUBLIC_MARKET_MATCH:
                    # Generate a market order (immediate match with opposite side)
                    if side == 1:
                        # Buy market order: takes the best ask
                        cdef int trade_price = best_ask
                        ev_buf[i].price = trade_price if trade_price >= 0 else 0
                        # Update current price to trade price
                        cur_price = trade_price if trade_price >= 0 else 0
                        # Remove best ask level (simulate fill)
                        best_ask = trade_price + 1  # next ask is higher (spread widens)
                        # Optionally narrow bid (simulate price up move)
                        if best_bid < trade_price:
                            best_bid += 1  # buyers chase price up by one tick
                    else:
                        # Sell market order: takes the best bid
                        cdef int trade_price = best_bid
                        ev_buf[i].price = trade_price if trade_price >= 0 else 0
                        cur_price = trade_price if trade_price >= 0 else 0
                        # Remove best bid level
                        best_bid = trade_price - 1 if trade_price > 0 else 0  # next bid is lower
                        # Optionally narrow ask (simulate price down move)
                        if best_ask > trade_price:
                            best_ask -= 1  # sellers push price down by one tick
                    # Quantity for market order
                    cdef int base_max_qty = 5
                    cdef int add_range = <int>(self.adversarial_factor * 10)
                    if add_range < 0:
                        add_range = 0
                    cdef uint32_t rndq = self._pcg32_random_fast()
                    cdef int qty = 1
                    if base_max_qty + add_range > 1:
                        qty = 1 + (rndq % (base_max_qty + add_range))
                    if qty < 1:
                        qty = 1
                    ev_buf[i].qty = qty
                else:
                    # PUBLIC_CANCEL_RANDOM event: cancel a random order on one side
                    ev_buf[i].price = 0  # price not applicable
                    ev_buf[i].qty = 0    # qty not applicable for cancel (not used)
                    if side == 1:
                        # Cancel a buy order: likely remove best bid
                        if best_bid > 0:
                            best_bid -= 1  # next lower bid becomes best
                            if best_bid < 0:
                                best_bid = 0
                    else:
                        # Cancel a sell order: remove best ask
                        best_ask += 1  # next higher ask becomes best
                    # current_price remains unchanged
                # Update momentum tracking (last_side) after each event
                last_side = side

        # Re-acquire GIL here
        # Save updated internal state and market state back to object
        self._last_side = last_side
        self.best_bid = best_bid
        self.best_ask = best_ask
        self.current_price = cur_price
        return events_count

    # Internal PCG32 functions (no GIL required)
    cdef inline uint32_t _pcg32_random(self) nogil:
        """Advance the RNG state and produce a 32-bit random output (PCG32)."""
        cdef uint64_t oldstate = self._state
        # Advance internal state (LCG step)
        self._state = oldstate * 6364136223846793005ULL + (self._inc | 1ULL)
        # Calculate output function (XSH RR)
        cdef uint32_t xorshifted = <uint32_t>(((oldstate >> 18) ^ oldstate) >> 27)
        cdef uint32_t rot = <uint32_t>(oldstate >> 59)
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))

    cdef inline uint32_t _pcg32_random_fast(self) nogil:
        """Fast PCG32 step without state update (use when state updated separately)."""
        # This assumes state has been advanced externally. Use with caution.
        cdef uint64_t oldstate = self._state
        # We do not advance state here (state should be advanced prior to call)
        cdef uint32_t xorshifted = <uint32_t>(((oldstate >> 18) ^ oldstate) >> 27)
        cdef uint32_t rot = <uint32_t>(oldstate >> 59)
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))

    cdef inline double _rand_uniform(self) nogil:
        """Generate a uniform random number in [0.0, 1.0) using PCG32."""
        cdef uint32_t r = self._pcg32_random()
        # Convert to double in [0,1)
        return r * (1.0 / 4294967296.0)

# Define event type constants (assuming unique codes for public events)
cdef int PUBLIC_LIMIT_ADD = 1
cdef int PUBLIC_MARKET_MATCH = 2
cdef int PUBLIC_CANCEL_RANDOM = 3
