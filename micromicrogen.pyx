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
