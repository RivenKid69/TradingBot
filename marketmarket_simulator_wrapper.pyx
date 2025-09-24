# cython: language_level=3
# distutils: language = c++

import numpy as np

cimport numpy as np

from libc.stddef cimport size_t
from libc.stdint cimport uint64_t

from core_constants cimport MarketRegime

np.import_array()


cdef inline double _clamp_non_negative(double value) nogil:
    if value < 0.0:
        return 0.0
    return value


cdef class MarketSimulatorWrapper:
    """Cython wrapper for MarketSimulator, providing safe access to simulation and indicators."""

    def __cinit__(self,
                  object price_arr not None,
                  object open_arr not None,
                  object high_arr not None,
                  object low_arr not None,
                  object volume_usd_arr not None,
                  uint64_t seed=0):
        cdef np.ndarray[np.float64_t, ndim=1] price = np.ascontiguousarray(price_arr, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] open_ = np.ascontiguousarray(open_arr, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] high = np.ascontiguousarray(high_arr, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] low = np.ascontiguousarray(low_arr, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] volume = np.ascontiguousarray(volume_usd_arr, dtype=np.float64)

        if price.shape[0] != open_.shape[0] or price.shape[0] != high.shape[0] or \
           price.shape[0] != low.shape[0] or price.shape[0] != volume.shape[0]:
            raise ValueError("All OHLCV arrays must have the same length")
        if price.shape[0] == 0:
            raise ValueError("Market simulator requires at least one timestep")

        self._price_ref = price
        self._open_ref = open_
        self._high_ref = high
        self._low_ref = low
        self._volume_ref = volume
        self._n_steps = <size_t>price.shape[0]

        self._sim = new MarketSimulator(&price[0], &open_[0], &high[0], &low[0], &volume[0], self._n_steps, seed)
        if self._sim is NULL:
            raise MemoryError("Failed to allocate MarketSimulator")

        self._random_shocks_enabled = False
        self._flash_probability = 0.0
        self._last_shock = False
        self._last_price = 0.0
        self._last_step = 0

        cdef int i
        for i in range(4):
            self._regime_probs[i] = 0.0
        for i in range(168):
            self._liquidity_multipliers[i] = 1.0

        # Mirror the C++ default distribution (or configuration file if present)
        try:
            import json as _json
            import os as _os
            path = _os.getenv("MARKET_REGIMES_JSON", "configs/market_regimes.json")
            with open(path, "r") as fh:
                data = _json.load(fh)
            probs = data.get("regime_probs", [0.8, 0.05, 0.10, 0.05])
        except Exception:
            probs = [0.8, 0.05, 0.10, 0.05]
        self.set_regime_distribution(probs)

    def __dealloc__(self):
        if self._sim is not NULL:
            del self._sim
            self._sim = NULL

    cpdef void set_seed(self, uint64_t seed):
        self._sim.set_seed(seed)
        self._last_price = 0.0
        self._last_shock = False
        self._last_step = 0

    cpdef void enable_random_shocks(self, bint enable, double probability=0.01):
        if probability < 0.0:
            probability = 0.0
        self._sim.enable_random_shocks(enable, probability)
        self._random_shocks_enabled = enable
        self._flash_probability = probability

    cpdef double step(self, int step_index, double black_swan_probability, bint is_training_mode):
        if step_index < 0:
            raise ValueError("step_index must be non-negative")
        self._last_step = <size_t>step_index
        self._last_price = self._sim.step(self._last_step, black_swan_probability, is_training_mode)
        self._last_shock = self._sim.shock_triggered(self._last_step) != 0
        return self._last_price

    cpdef int shock_triggered(self, long step_idx=-1):
        cdef size_t idx
        if step_idx < 0:
            idx = self._last_step
        else:
            idx = <size_t>step_idx
        return self._sim.shock_triggered(idx)

    cpdef double get_last_price(self):
        if self._last_step < self._n_steps:
            return (<np.ndarray[np.float64_t, ndim=1]>self._price_ref)[self._last_step]
        return self._last_price

    cpdef double get_ma5(self):
        return self._sim.get_ma5(self._last_step)

    cpdef double get_ma20(self):
        return self._sim.get_ma20(self._last_step)

    cpdef double get_atr(self):
        return self._sim.get_atr(self._last_step)

    cpdef double get_rsi(self):
        return self._sim.get_rsi(self._last_step)

    cpdef double get_macd(self):
        return self._sim.get_macd(self._last_step)

    cpdef double get_macd_signal(self):
        return self._sim.get_macd_signal(self._last_step)

    cpdef double get_momentum(self):
        return self._sim.get_momentum(self._last_step)

    cpdef double get_cci(self):
        return self._sim.get_cci(self._last_step)

    cpdef double get_obv(self):
        return self._sim.get_obv(self._last_step)

    cpdef double get_bb_lower(self):
        return self._sim.get_bb_lower(self._last_step)

    cpdef double get_bb_upper(self):
        return self._sim.get_bb_upper(self._last_step)

    cpdef void set_regime_distribution(self, object probabilities):
        cdef np.ndarray[np.float64_t, ndim=1] probs = np.ascontiguousarray(probabilities, dtype=np.float64)
        if probs.size != 4:
            raise ValueError("regime distribution must contain exactly four values")
        cdef double total = 0.0
        cdef int i
        for i in range(4):
            if probs[i] < 0.0:
                raise ValueError("regime probabilities must be non-negative")
            total += probs[i]
        if total <= 0.0:
            raise ValueError("regime probabilities must sum to a positive value")
        for i in range(4):
            self._regime_probs[i] = probs[i] / total
        self._sim.set_regime_distribution(self._regime_probs)

    cpdef object get_regime_distribution(self):
        cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(4, dtype=np.float64)
        cdef int i
        for i in range(4):
            out[i] = self._regime_probs[i]
        return out

    cpdef void set_liquidity_seasonality(self, object multipliers):
        cdef np.ndarray[np.float64_t, ndim=1] mult = np.ascontiguousarray(multipliers, dtype=np.float64)
        if mult.size != 168:
            raise ValueError("liquidity seasonality must contain 168 hourly multipliers")
        cdef int i
        for i in range(168):
            self._liquidity_multipliers[i] = _clamp_non_negative(mult[i])
        self._sim.set_liquidity_seasonality(self._liquidity_multipliers)

    cpdef object get_liquidity_seasonality(self):
        cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(168, dtype=np.float64)
        cdef int i
        for i in range(168):
            out[i] = self._liquidity_multipliers[i]
        return out

    cpdef void force_market_regime(self, object regime, int start=0, int duration=0):
        cdef MarketRegime regime_code
        try:
            regime_code = <MarketRegime><int>regime
        except Exception:
            raise ValueError("regime must be convertible to MarketRegime enum")
        if start < 0 or duration < 0:
            raise ValueError("start and duration must be non-negative")
        self._sim.force_market_regime(regime_code, <size_t>start, <size_t>duration)
