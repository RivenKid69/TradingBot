# cython: language=c++

from libc.stddef cimport size_t
from libc.stdint cimport uint64_t
from core_constants cimport MarketRegime

cdef extern from "<array>" namespace "std":
    cdef cppclass ArrayDouble4 "std::array<double, 4>":
        ArrayDouble4()
        ArrayDouble4(const ArrayDouble4&)
        double& operator[](size_t)
        const double& operator[](size_t) const
        double* data()
        const double* data() const

    cdef cppclass ArrayDouble168 "std::array<double, 168>":
        ArrayDouble168()
        ArrayDouble168(const ArrayDouble168&)
        double& operator[](size_t)
        const double& operator[](size_t) const
        double* data()
        const double* data() const

cdef extern from "MarketSimulator.h" nogil:
    cdef cppclass MarketSimulator:
        MarketSimulator(
            double* price,
            double* open,
            double* high,
            double* low,
            double* volume_usd,
            size_t n_steps,
            uint64_t seed
        ) except +
        double step(size_t i, double black_swan_probability, bint is_training_mode)
        void set_seed(uint64_t seed)
        void set_regime_distribution(const ArrayDouble4& probs)
        void enable_random_shocks(bint enable, double probability_per_step)
        void force_market_regime(MarketRegime regime, size_t start, size_t duration)
        void set_liquidity_seasonality(const ArrayDouble168& multipliers)
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
    cdef double _regime_probs[4]
    cdef double _liquidity_multipliers[168]
    cdef object _price_ref
    cdef object _open_ref
    cdef object _high_ref
    cdef object _low_ref
    cdef object _volume_ref
    cdef size_t _n_steps
    cdef double* _price_data
