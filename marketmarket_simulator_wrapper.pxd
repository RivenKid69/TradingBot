cimport libc.stdint

# Cython interface for the MarketSimulatorWrapper
cdef class MarketSimulatorWrapper:
    """Cython wrapper for the C++ MarketSimulator (safe interface)."""
    cdef public bint _random_shocks_enabled
    cdef public bint _last_shock
    cdef double _last_price
    cpdef void set_seed(self, libc.stdint.uint64_t seed)
    cpdef void enable_random_shocks(self, bint enable)
    cpdef void step(self, int step_index, double black_swan_probability, bint is_training_mode)
    cpdef bint shock_triggered(self)
    cpdef double get_last_price(self)
    cpdef double get_ma5(self)
    cpdef double get_ma20(self)
    cpdef double get_atr(self)
    cpdef double get_rsi(self)
    cpdef double get_macd(self)
    cpdef double get_macd_signal(self)
    cpdef double get_momentum(self)
    cpdef double get_cci(self)
    cpdef double get_obv(self)
    cpdef double get_bb_lower(self)
    cpdef double get_bb_upper(self)
    cpdef void force_market_regime(self, object name, int start, int duration)
