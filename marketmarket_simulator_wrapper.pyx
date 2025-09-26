# cython: language_level=3
# cython: language=c++
# distutils: language = c++

import numpy as np

cimport numpy as np

from libc.stddef cimport size_t
from cython cimport NULL
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
        self._sim = <MarketSimulator*>NULL
        self._price_data = <double*>NULL

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
        self._price_data = &price[0]

        self._sim = new MarketSimulator(&price[0], &open_[0], &high[0], &low[0], &volume[0], self._n_steps, seed)
        if self._sim == NULL:
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
        if self._sim != NULL:
            del self._sim
            self._sim = <MarketSimulator*>NULL
        self._price_data = <double*>NULL

    def set_seed(self, seed):
        cdef uint64_t c_seed = <uint64_t>seed
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")

        self._sim.set_seed(c_seed)
        self._last_price = 0.0
        self._last_shock = False
        self._last_step = 0

    def enable_random_shocks(self, enable, probability=0.01):
        cdef bint c_enable = <bint>enable
        cdef double c_probability = float(probability)
        if c_probability < 0.0:
            c_probability = 0.0
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")

        self._sim.enable_random_shocks(c_enable, c_probability)
        self._random_shocks_enabled = c_enable
        self._flash_probability = c_probability

    def step(self, step_index, black_swan_probability, is_training_mode):
        cdef int c_step_index = int(step_index)
        if c_step_index < 0:
            raise ValueError("step_index must be non-negative")
        cdef double c_prob = float(black_swan_probability)
        cdef bint c_training = <bint>is_training_mode
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")

        self._last_step = <size_t>c_step_index
        self._last_price = self._sim.step(self._last_step, c_prob, c_training)
        self._last_shock = self._sim.shock_triggered(self._last_step) != 0
        return self._last_price

    def shock_triggered(self, step_idx=-1):
        cdef size_t idx
        cdef long c_step_idx = <long>step_idx
        if c_step_idx < 0:
            idx = self._last_step
        else:
            idx = <size_t>c_step_idx
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")

        return self._sim.shock_triggered(idx) != 0

    def get_last_price(self):
        if self._last_step < self._n_steps and self._price_data != NULL:
            return self._price_data[self._last_step]
        return self._last_price

    def get_ma5(self):
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")
        return self._sim.get_ma5(self._last_step)

    def get_ma20(self):
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")
        return self._sim.get_ma20(self._last_step)

    def get_atr(self):
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")
        return self._sim.get_atr(self._last_step)

    def get_rsi(self):
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")
        return self._sim.get_rsi(self._last_step)

    def get_macd(self):
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")
        return self._sim.get_macd(self._last_step)

    def get_macd_signal(self):
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")
        return self._sim.get_macd_signal(self._last_step)

    def get_momentum(self):
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")
        return self._sim.get_momentum(self._last_step)

    def get_cci(self):
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")
        return self._sim.get_cci(self._last_step)

    def get_obv(self):
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")
        return self._sim.get_obv(self._last_step)

    def get_bb_lower(self):
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")
        return self._sim.get_bb_lower(self._last_step)

    def get_bb_upper(self):
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")
        return self._sim.get_bb_upper(self._last_step)

    def set_regime_distribution(self, probabilities):
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
        cdef ArrayDouble4 c_probs = ArrayDouble4()
        cdef double* probs_ptr = <double*>c_probs.data()
        for i in range(4):
            self._regime_probs[i] = probs[i] / total
            probs_ptr[i] = self._regime_probs[i]
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")

        self._sim.set_regime_distribution(c_probs)

    def get_regime_distribution(self):
        cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(4, dtype=np.float64)
        cdef int i
        for i in range(4):
            out[i] = self._regime_probs[i]
        return out

    def set_liquidity_seasonality(self, multipliers):
        cdef np.ndarray[np.float64_t, ndim=1] mult = np.ascontiguousarray(multipliers, dtype=np.float64)
        if mult.size != 168:
            raise ValueError("liquidity seasonality must contain 168 hourly multipliers")
        cdef int i
        cdef ArrayDouble168 c_mult = ArrayDouble168()
        cdef double* mult_ptr = <double*>c_mult.data()
        for i in range(168):
            self._liquidity_multipliers[i] = _clamp_non_negative(mult[i])
            mult_ptr[i] = self._liquidity_multipliers[i]
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")

        self._sim.set_liquidity_seasonality(c_mult)

    def get_liquidity_seasonality(self):
        cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(168, dtype=np.float64)
        cdef int i
        for i in range(168):
            out[i] = self._liquidity_multipliers[i]
        return out

    def force_market_regime(self, regime, start=0, duration=0):
        cdef MarketRegime regime_code
        try:
            regime_code = <MarketRegime><int>regime
        except Exception:
            raise ValueError("regime must be convertible to MarketRegime enum")
        cdef int c_start = int(start)
        cdef int c_duration = int(duration)
        if c_start < 0 or c_duration < 0:
            raise ValueError("start and duration must be non-negative")
        if self._sim == NULL:
            raise RuntimeError("MarketSimulator instance not initialised")

        self._sim.force_market_regime(regime_code, <size_t>c_start, <size_t>c_duration)
