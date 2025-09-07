cimport libc.stdint

# Cython declarations for the microstructure event generator
cdef class CyMicrostructureGenerator:
    """Cython microstructure events generator (public market events)."""
    cpdef void seed(self, libc.stdint.uint64_t seed)
    cpdef void set_regime(self, double base_order_imbalance_ratio,
                           double base_cancel_ratio,
                           double momentum_factor,
                           double mean_reversion_factor,
                           double adversarial_factor)
    cpdef int generate_public_events(self, object out_events, int max_events)
