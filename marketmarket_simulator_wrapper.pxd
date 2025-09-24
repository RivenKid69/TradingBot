from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from libcpp.array cimport array

from core_constants cimport MarketRegime

cdef extern from "MarketSimulator.h" nogil:
    cdef cppclass MarketSimulator:
        MarketSimulator(
            double* price,
            double* open,
            double* high,
            double* low,
            double* volume_usd,
            size_t n_steps,
            uint64_t seed=0
        ) except +
        double step(size_t i, double black_swan_probability, bint is_training_mode)
        void set_seed(uint64_t seed)
        void set_regime_distribution(const array[double, 4]& probs)
        void enable_random_shocks(bint enable, double probability_per_step)
        void force_market_regime(MarketRegime regime, size_t start, size_t duration)
        void set_liquidity_seasonality(const array[double, 168]& multipliers)
        int shock_triggered(size_t i) const
        double get_ma5(size_t i) const
        double get_ma20(size_t i) const
        double get_atr(size_t i) const
        double get_rsi(size_t i) const
        double get_macd(size_t i) const
        double get_macd_signal(size_t i) const
        double get_momentum(size_t i) const
        double get_cci(size_t i) const
        double get_obv(size_t i) const
        double get_bb_lower(size_t i) const
        double get_bb_upper(size_t i) const


cdef class MarketSimulatorWrapper:
    """Cython wrapper for the C++ MarketSimulator (safe interface)."""
    cdef MarketSimulator* _sim
    cdef public bint _random_shocks_enabled
    cdef public bint _last_shock
    cdef double _last_price
    cdef size_t _last_step
    cdef double _flash_probability
    cdef array[double, 4] _regime_probs
    cdef array[double, 168] _liquidity_multipliers
    cdef object _price_ref
    cdef object _open_ref
    cdef object _high_ref
    cdef object _low_ref
    cdef object _volume_ref
    cdef size_t _n_steps

    cpdef void set_seed(self, uint64_t seed)
    cpdef void enable_random_shocks(self, bint enable, double probability=0.01)
    cpdef double step(self, int step_index, double black_swan_probability, bint is_training_mode)
    cpdef int shock_triggered(self, long step_idx=-1)
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
    cpdef void set_regime_distribution(self, object probabilities)
    cpdef object get_regime_distribution(self)
    cpdef void set_liquidity_seasonality(self, object multipliers)
    cpdef object get_liquidity_seasonality(self)
    cpdef void force_market_regime(self, object regime, int start=0, int duration=0)
