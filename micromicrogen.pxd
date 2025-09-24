
cimport libc.stdint
from libc.stddef cimport size_t

from execevents cimport MarketEvent


cdef class CyMicrostructureGenerator:
    cdef libc.stdint.uint64_t _state
    cdef libc.stdint.uint64_t _inc
    cdef libc.stdint.uint32_t _order_seq




# Cython declarations for the microstructure event generator
cdef class CyMicrostructureGenerator:
    """Simple microstructure generator used from Python code."""
    cdef object _rng



    cdef double momentum_factor
    cdef double mean_reversion_factor
    cdef double base_order_imbalance_ratio
    cdef double base_cancel_ratio
    cdef double adversarial_factor


    cdef int _last_side
    cdef int current_price
    cdef int best_bid
    cdef int best_ask

    cpdef void seed(self, libc.stdint.uint64_t seed)
    cpdef void set_regime(self, double base_order_imbalance_ratio,
                          double base_cancel_ratio,
                          double momentum_factor,
                          double mean_reversion_factor,
                          double adversarial_factor)
    cdef int generate_public_events_into(self, MarketEvent* out_events,
                                         size_t buf_len,
                                         int max_events)
    cdef void _reset_parameters(self)



    cdef int _last_side
    cpdef void seed(self, unsigned long long seed)
    cpdef void set_regime(self, double base_order_imbalance_ratio,
                           double base_cancel_ratio,
                           double momentum_factor,
                           double mean_reversion_factor,
                           double adversarial_factor)
    cpdef list generate_public_events(self, object state, object tracker,
                                      object lob, int max_events=*)
    cdef void _reset_defaults(self)
    cdef int _determine_event_count(self, int max_events)
    cdef double _resolve_mid_price(self, object state, object lob)
    cdef tuple _build_single_event(self, int mid_ticks)
    cdef int _choose_side(self)
    cdef int _sample_limit_price(self, int mid_ticks, int side)
    cdef int _sample_quantity(self)


