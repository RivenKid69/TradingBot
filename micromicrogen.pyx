
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Low level public microstructure event generator.

This module exposes :class:`CyMicrostructureGenerator` which mirrors the
behaviour of the historical implementation relied upon by the execution
simulator.  The generator operates entirely under ``nogil`` when filling a
``MarketEvent`` memoryview, uses a PCG32 random number generator and maintains
an internal representation of the best bid/ask and last trade price so that the
produced flow reacts to prior activity.  Each limit order receives a unique
``order_id`` allowing downstream components to reconstruct book state.
"""

from libc.math cimport exp, fabs
from libc.stdint cimport uint32_t, uint64_t

from cpython.mem cimport PyMem_Free, PyMem_Malloc
from libc.stddef cimport size_t

import cython
import core_constants as _const

from execevents cimport EventType, MarketEvent, Side


cdef int PRICE_SCALE = _const.PRICE_SCALE


cdef inline uint32_t _pcg32_step(uint64_t* state, uint64_t inc) nogil:
    """Advance the PCG32 state and return a 32-bit output."""
    cdef uint64_t oldstate = state[0]
    state[0] = oldstate * 6364136223846793005ULL + (inc | 1ULL)
    cdef uint32_t xorshifted = <uint32_t>(((oldstate >> 18) ^ oldstate) >> 27)
    cdef uint32_t rot = <uint32_t>(oldstate >> 59)
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))


cdef inline double _rand_uniform(uint64_t* state, uint64_t inc) nogil:
    """Uniform variate in [0, 1)."""
    return _pcg32_step(state, inc) * (1.0 / 4294967296.0)


cdef inline uint32_t _rand_uint(uint64_t* state, uint64_t inc) nogil:
    """Return an unsigned 32-bit random integer."""
    return _pcg32_step(state, inc)


cdef inline int _sample_poisson(uint64_t* state, uint64_t inc, double lam, int cap) nogil:
    """Sample a Poisson distributed integer with intensity ``lam`` (cap enforced)."""
    if lam <= 0.0:
        return 0
    cdef double L = exp(-lam)
    cdef double p = 1.0
    cdef int k = 0
    while True:
        p *= _rand_uniform(state, inc)
        if p <= L:
            break
        k += 1
        if k >= cap:
            return cap
    if k > cap:
        return cap
    return k


cdef class CyMicrostructureGenerator:
    """Generate public order flow compatible with the execution simulator."""

    def __cinit__(self):
        self._state = 0
        self._inc = 0x14057B7EF767814F
        self._order_seq = 1
        self._reset_parameters()

    cpdef void seed(self, uint64_t seed):
        """Seed the internal PCG32 RNG and reset book state."""
        self._state = 0
        self._inc = (seed << 1) | 1
        cdef uint64_t state = self._state
        cdef uint64_t inc = self._inc
        _pcg32_step(&state, inc)
        state += seed
        _pcg32_step(&state, inc)
        self._state = state
        self._order_seq = 1
        self._last_side = 0
        self.current_price = 100 * PRICE_SCALE
        if self.current_price < PRICE_SCALE:
            self.current_price = PRICE_SCALE
        self.best_bid = self.current_price - 1
        if self.best_bid < 0:
            self.best_bid = 0
        self.best_ask = self.current_price + 1


"""Simple microstructure generator used by the Python simulation stack.

The previous revision attempted to implement a fully ``nogil`` PCG based
event generator that wrote directly into a ``MarketEvent`` memory view.  The
structure definition that code relied on no longer exists which caused Cython
to emit multiple syntax and attribute errors (missing ``owner`` field,
unmatched method signature used from Python, invalid ``cdef`` statements in a
``with gil`` block, etc.).  For the high level simulator we only need a small
Python friendly generator that returns tuples describing public events.  The
rewritten implementation below keeps the configuration surface intact while
removing the invalid low level constructs so the module can compile again.
"""

# cython: language_level=3
from libc.stdint cimport uint64_t

import random

import core_constants as constants
from execevents cimport EventType, Side


cdef class CyMicrostructureGenerator:
    """Generate lightweight public order flow for tests and Python sims."""

    def __cinit__(self):
        self._rng = random.Random()
        self._reset_defaults()

    cpdef void seed(self, uint64_t seed):
        """Seed the internal RNG (``random.Random`` wrapper)."""
        self._rng.seed(seed)
        self._last_side = 0



    cpdef void set_regime(self,
                          double base_order_imbalance_ratio,
                          double base_cancel_ratio,
                          double momentum_factor,
                          double mean_reversion_factor,
                          double adversarial_factor):



        """Configure the generator parameters used when producing events."""


        self.base_order_imbalance_ratio = base_order_imbalance_ratio
        self.base_cancel_ratio = base_cancel_ratio
        self.momentum_factor = momentum_factor
        self.mean_reversion_factor = mean_reversion_factor
        self.adversarial_factor = adversarial_factor


    cdef int generate_public_events_into(self,
                                         MarketEvent* out_events,
                                         size_t buf_len,
                                         int max_events):
        """Fill ``out_events`` with up to ``max_events`` public market events."""
        if out_events == NULL or buf_len <= 0 or max_events <= 0:
            return 0
        if max_events > buf_len:
            max_events = <int>buf_len

        cdef double lam = 1.0 + self.adversarial_factor
        if lam < 0.0:
            lam = 0.0

        cdef uint64_t state = self._state
        cdef uint64_t inc = self._inc | 1
        cdef int last_side = self._last_side
        cdef int best_bid = self.best_bid
        cdef int best_ask = self.best_ask
        cdef int current_price = self.current_price
        cdef uint32_t order_seq = self._order_seq

        cdef double cancel_ratio = self.base_cancel_ratio
        if cancel_ratio < 0.0:
            cancel_ratio = 0.0
        elif cancel_ratio > 1.0:
            cancel_ratio = 1.0

        cdef double imbalance_ratio = self.base_order_imbalance_ratio
        cdef double pb_base
        if imbalance_ratio > 0.0:
            pb_base = imbalance_ratio / (1.0 + imbalance_ratio)
        elif imbalance_ratio == 0.0:
            pb_base = 0.0
        else:
            pb_base = 0.5

        cdef double momentum = self.momentum_factor
        cdef double mean_reversion = self.mean_reversion_factor
        cdef double adversarial = self.adversarial_factor

        cdef int events_count = _sample_poisson(&state, inc, lam, max_events)

        cdef MarketEvent* events = out_events
        cdef int i
        cdef double u
        cdef double pb
        cdef double p_market
        cdef int side
        cdef int price
        cdef int qty
        cdef int range_extra
        cdef uint32_t rnd
        cdef double abs_adv = fabs(adversarial)

        with nogil:
            for i in range(events_count):
                u = _rand_uniform(&state, inc)
                if u < cancel_ratio and (best_bid > 0 or best_ask > 0):
                    events[i].type = EventType.PUBLIC_CANCEL_RANDOM
                    pb = pb_base
                    if last_side == 1:
                        pb = pb + momentum - mean_reversion
                    elif last_side == -1:
                        pb = pb - momentum + mean_reversion
                    if pb < 0.0:
                        pb = 0.0
                    elif pb > 1.0:
                        pb = 1.0
                    side = 1 if _rand_uniform(&state, inc) < pb else -1
                    events[i].side = <Side>side
                    events[i].price = 0
                    events[i].qty = 0
                    events[i].order_id = <int>(_rand_uint(&state, inc) & 0x7FFFFFFF)
                    if side == 1 and best_bid > 0:
                        best_bid -= 1
                        if best_bid < 0:
                            best_bid = 0
                    elif side == -1:
                        best_ask += 1
                    if best_ask <= best_bid:
                        best_ask = best_bid + 1
                    last_side = side
                    continue

                p_market = 0.5 + 0.5 * (momentum - mean_reversion)
                if p_market < 0.0:
                    p_market = 0.0
                elif p_market > 1.0:
                    p_market = 1.0

                pb = pb_base
                if last_side == 1:
                    pb = pb + momentum - mean_reversion
                elif last_side == -1:
                    pb = pb - momentum + mean_reversion
                if pb < 0.0:
                    pb = 0.0
                elif pb > 1.0:
                    pb = 1.0

                side = 1 if _rand_uniform(&state, inc) < pb else -1
                events[i].side = <Side>side

                if _rand_uniform(&state, inc) < p_market:
                    events[i].type = EventType.PUBLIC_MARKET_MATCH
                    if side == 1:
                        price = best_ask
                        if price <= 0:
                            price = current_price + 1
                        current_price = price
                        if best_bid < price:
                            best_bid = price
                        best_ask = price + 1
                    else:
                        price = best_bid
                        if price <= 0:
                            price = current_price - 1
                            if price < 0:
                                price = 0
                        current_price = price
                        if best_ask > price:
                            best_ask = price
                        if price > 0:
                            best_bid = price - 1
                        else:
                            best_bid = 0
                    if best_ask <= best_bid:
                        best_ask = best_bid + 1
                    range_extra = 4 + <int>(abs_adv * 10.0)
                    if range_extra < 1:
                        range_extra = 1
                    qty = 1 + <int>(_rand_uint(&state, inc) % <uint32_t>(range_extra))
                    if qty < 1:
                        qty = 1
                    events[i].price = price
                    events[i].qty = qty
                    events[i].order_id = 0
                else:
                    events[i].type = EventType.PUBLIC_LIMIT_ADD
                    range_extra = 5 + <int>(abs_adv * 5.0)
                    if range_extra < 0:
                        range_extra = 0
                    rnd = _rand_uint(&state, inc)
                    if side == 1:
                        price = best_bid
                        if range_extra > 0:
                            price += <int>(rnd % <uint32_t>(range_extra + 1))
                        if best_ask > best_bid + 1 and price >= best_ask:
                            price = best_ask - 1
                        if price < 0:
                            price = 0
                        if price > best_bid:
                            best_bid = price
                    else:
                        price = best_ask
                        if range_extra > 0:
                            price -= <int>(rnd % <uint32_t>(range_extra + 1))
                        if price <= best_bid:
                            price = best_bid + 1
                        if price < 0:
                            price = 0
                        if price < best_ask:
                            best_ask = price

                    range_extra = 5 + <int>(abs_adv * 10.0)
                    if range_extra < 1:
                        range_extra = 1
                    qty = 1 + <int>(_rand_uint(&state, inc) % <uint32_t>(range_extra))
                    if qty < 1:
                        qty = 1
                    events[i].price = price
                    events[i].qty = qty
                    events[i].order_id = <int>order_seq
                    order_seq += 1
                last_side = side

        self._state = state
        self._inc = inc
        self._last_side = last_side
        if best_bid < 0:
            best_bid = 0
        if best_ask <= best_bid:
            best_ask = best_bid + 1
        if current_price < 0:
            current_price = 0
        self.best_bid = best_bid
        self.best_ask = best_ask
        self.current_price = current_price
        self._order_seq = order_seq

        return events_count

    def generate_public_events(self, object state, object tracker, object lob, int max_events=16):
        """Python helper returning a list of tuples for compatibility layers."""
        if max_events <= 0:
            return []

        cdef long long bid_hint = 0
        cdef long long ask_hint = 0
        cdef bint have_bid = False
        cdef bint have_ask = False
        cdef double last_price = 0.0
        cdef int price_ticks = 0

        try:
            if lob is not None:
                getter = getattr(lob, "get_best_bid", None)
                if getter is not None:
                    bid_hint = <long long> getter()
                    have_bid = bid_hint > 0
                elif hasattr(lob, "best_bid"):
                    bid_hint = <long long> getattr(lob, "best_bid")
                    have_bid = bid_hint > 0

                getter = getattr(lob, "get_best_ask", None)
                if getter is not None:
                    ask_hint = <long long> getter()
                    have_ask = ask_hint > 0
                elif hasattr(lob, "best_ask"):
                    ask_hint = <long long> getattr(lob, "best_ask")
                    have_ask = ask_hint > 0
        except Exception:
            have_bid = have_ask = False

        if have_bid and have_ask and ask_hint > bid_hint:
            self.best_bid = <int> bid_hint
            self.best_ask = <int> ask_hint
            self.current_price = <int> ((bid_hint + ask_hint) // 2)
        elif have_bid and not have_ask and bid_hint > 0:
            self.best_bid = <int> bid_hint
            self.best_ask = self.best_bid + 1
            self.current_price = self.best_bid
        elif have_ask and not have_bid and ask_hint > 0:
            self.best_ask = <int> ask_hint
            self.best_bid = self.best_ask - 1 if self.best_ask > 0 else 0
            self.current_price = self.best_ask

        if state is not None:
            try:
                price_attr = getattr(state, "last_price", None)
                if price_attr is not None:
                    last_price = float(price_attr)
                    if last_price > 0:
                        price_ticks = <int> (last_price * PRICE_SCALE)
                        if price_ticks > 0:
                            self.current_price = price_ticks
                            if self.best_bid <= 0:
                                self.best_bid = price_ticks - 1 if price_ticks > 0 else 0
                            if self.best_ask <= self.best_bid:
                                self.best_ask = self.best_bid + 1
            except Exception:
                pass

        cdef size_t capacity = <size_t> max_events
        cdef MarketEvent* buffer = <MarketEvent*> PyMem_Malloc(capacity * cython.sizeof(MarketEvent))
        if buffer == NULL:
            raise MemoryError("Failed to allocate event buffer")

        cdef int produced = 0
        cdef int i
        try:
            produced = self.generate_public_events_into(buffer, capacity, max_events)
            result = []
            for i in range(produced):
                result.append((<int> buffer[i].type,
                               <int> buffer[i].side,
                               buffer[i].price,
                               buffer[i].qty,
                               buffer[i].order_id))
            return result
        finally:
            PyMem_Free(buffer)

    cdef void _reset_parameters(self):
        self.momentum_factor = 0.0
        self.mean_reversion_factor = 0.0
        self.base_order_imbalance_ratio = 1.0
        self.base_cancel_ratio = 0.05
        self.adversarial_factor = 0.0
        self._last_side = 0
        self.current_price = 100 * PRICE_SCALE
        if self.current_price < PRICE_SCALE:
            self.current_price = PRICE_SCALE
        self.best_bid = self.current_price - 1
        if self.best_bid < 0:
            self.best_bid = 0
        self.best_ask = self.current_price + 1


    cpdef list generate_public_events(self,
                                      object state,
                                      object tracker,
                                      object lob,
                                      int max_events=16):
        """Return a list of public events as tuples.

        The tuples follow the same layout as agent events generated elsewhere in
        the code base: ``(event_type, side, price_ticks, qty, order_id)``.
        ``event_type`` and ``side`` are returned as plain integers compatible
        with the ``EventType`` and ``Side`` enums defined in :mod:`execevents`.
        """

        cdef list events = []
        cdef int num_events = self._determine_event_count(max_events)
        if num_events == 0:
            return events

        cdef double mid_price = self._resolve_mid_price(state, lob)
        cdef int mid_ticks = <int> (mid_price * constants.PRICE_SCALE)
        if mid_ticks < 1:
            mid_ticks = constants.PRICE_SCALE  # fall back to one currency unit

        cdef int i
        for i in range(num_events):
            events.append(self._build_single_event(mid_ticks))

        return events

    cdef void _reset_defaults(self):
        self.momentum_factor = 0.0
        self.mean_reversion_factor = 0.0
        self.base_order_imbalance_ratio = 1.0
        self.base_cancel_ratio = 0.1
        self.adversarial_factor = 0.0
        self._last_side = 0

    cdef int _determine_event_count(self, int max_events):
        if max_events <= 0:
            return 0
        # Use a simple geometric style distribution to keep things light weight.
        cdef double intensity = 0.5 + max(0.0, self.adversarial_factor)
        cdef int count = 0
        while count < max_events and self._rng.random() < intensity:
            count += 1
            intensity *= 0.6  # diminishing probability of long bursts
        return count

    cdef double _resolve_mid_price(self, object state, object lob):
        """Best effort mid price retrieval used for pricing new orders."""
        try:
            if lob is not None:
                return float(lob.mid_price()) / constants.PRICE_SCALE
        except Exception:
            pass

        try:
            return float(getattr(state, "last_price"))
        except Exception:
            return 1.0  # final fallback prevents zero pricing

    cdef tuple _build_single_event(self, int mid_ticks):
        cdef int side = self._choose_side()
        cdef double cancel_threshold = max(0.0, min(1.0, self.base_cancel_ratio))
        cdef double draw = self._rng.random()

        if draw < cancel_threshold:
            return (<int> EventType.PUBLIC_CANCEL_RANDOM, side, 0, 0, 0)

        cdef double market_bias = 0.5 + 0.5 * (self.momentum_factor - self.mean_reversion_factor)
        market_bias = max(0.0, min(1.0, market_bias))

        if self._rng.random() < market_bias:
            price = max(1, mid_ticks + (constants.PRICE_SCALE if side == <int> Side.BUY else -constants.PRICE_SCALE))
            qty = self._sample_quantity()
            self._last_side = side
            return (<int> EventType.PUBLIC_MARKET_MATCH, side, price, qty, 0)

        price = self._sample_limit_price(mid_ticks, side)
        qty = self._sample_quantity()
        self._last_side = side
        return (<int> EventType.PUBLIC_LIMIT_ADD, side, price, qty, 0)

    cdef int _choose_side(self):
        cdef double base_prob = 0.5
        if self.base_order_imbalance_ratio > 0.0:
            base_prob = self.base_order_imbalance_ratio / (1.0 + self.base_order_imbalance_ratio)

        if self._last_side == <int> Side.BUY:
            base_prob += self.momentum_factor
            base_prob -= self.mean_reversion_factor
        elif self._last_side == <int> Side.SELL:
            base_prob -= self.momentum_factor
            base_prob += self.mean_reversion_factor

        base_prob = max(0.0, min(1.0, base_prob))
        return <int> Side.BUY if self._rng.random() < base_prob else <int> Side.SELL

    cdef int _sample_limit_price(self, int mid_ticks, int side):
        cdef int tick_offset = 1 + self._rng.randint(0, 5 + int(abs(self.adversarial_factor) * 4))
        if side == <int> Side.BUY:
            return max(1, mid_ticks - tick_offset)
        else:
            return max(1, mid_ticks + tick_offset)

    cdef int _sample_quantity(self):
        return 1 + self._rng.randint(0, 4 + int(abs(self.adversarial_factor) * 6))


