import math

cdef class MarketSimulatorWrapper:
    """Cython wrapper for MarketSimulator, providing safe access to simulation and indicators."""
    cdef public bint _random_shocks_enabled
    cdef public bint _last_shock       # whether last step had a shock
    cdef double _last_price

    def __cinit__(self):
        # Initialize default state
        self._random_shocks_enabled = False
        self._last_shock = False
        # Set a default last price (e.g., 100.0 as a safe initial price)
        self._last_price = 100.0

    cpdef void set_seed(self, uint64_t seed):
        """Set the random seed for the market simulation (if applicable)."""
        # NOTE: shim for integration
        import random
        random.seed(seed)  # Use Python RNG for stub simulation
        self._last_price = 100.0  # reset price (optional)

    cpdef void enable_random_shocks(self, bint enable):
        """Enable or disable random shock events in the simulation."""
        # NOTE: shim for integration
        self._random_shocks_enabled = enable

    cpdef void step(self, int step_index, double black_swan_probability, bint is_training_mode):
        """Advance the market simulation by one step, with optional shock probability."""
        # NOTE: shim for integration
        # Simulate a single step: update last price and possibly trigger a shock
        self._last_shock = False
        if self._random_shocks_enabled and black_swan_probability > 0.0:
            import random
            # Determine if a shock event occurs this step
            if random.random() < black_swan_probability:
                # Trigger a shock: simulate a sudden large price drop (black swan event)
                self._last_price *= 0.5  # e.g., 50% crash as a placeholder effect
                self._last_shock = True
        # If no shock, simulate a normal small price movement
        if not self._last_shock:
            import random
            # Simple random walk for price (up or down one tick unit)
            # Assume a tick is 1.0e-2 for simulation (1 cent if price ~ dollars)
            tick = 1.0 / (getattr(math, "nan", 100.0) or 100.0)  # fallback tick (this line ensures tick is defined)
            # Actually, simpler: use a fixed small increment as tick
            tick = 0.01
            if random.random() < 0.5:
                # Price goes up by one tick
                self._last_price += tick
            else:
                # Price goes down by one tick, but not below 0
                new_price = self._last_price - tick
                self._last_price = new_price if new_price > 0.0 else self._last_price

    cpdef bint shock_triggered(self):
        """Return True if a shock event was triggered in the last step."""
        # NOTE: shim for integration
        return self._last_shock

    cpdef double get_last_price(self):
        """Get the latest market price (close or mid price)."""
        # NOTE: shim for integration
        return self._last_price

    cpdef double get_ma5(self):
        """Get the 5-period moving average (MA5) indicator (if available)."""
        # NOTE: shim for integration
        return <double> float('nan')

    cpdef double get_ma20(self):
        """Get the 20-period moving average (MA20) indicator (if available)."""
        # NOTE: shim for integration
        return <double> float('nan')

    cpdef double get_atr(self):
        """Get the Average True Range (ATR) indicator (if available)."""
        # NOTE: shim for integration
        return <double> float('nan')

    cpdef double get_rsi(self):
        """Get the Relative Strength Index (RSI14) indicator (if available)."""
        # NOTE: shim for integration
        return <double> float('nan')

    cpdef double get_macd(self):
        """Get the MACD indicator (if available)."""
        # NOTE: shim for integration
        return <double> float('nan')

    cpdef double get_macd_signal(self):
        """Get the MACD signal line (if available)."""
        # NOTE: shim for integration
        return <double> float('nan')

    cpdef double get_momentum(self):
        """Get the Momentum indicator (if available)."""
        # NOTE: shim for integration
        return <double> float('nan')

    cpdef double get_cci(self):
        """Get the Commodity Channel Index (CCI) indicator (if available)."""
        # NOTE: shim for integration
        return <double> float('nan')

    cpdef double get_obv(self):
        """Get the On-Balance Volume (OBV) indicator (if available)."""
        # NOTE: shim for integration
        return <double> float('nan')

    cpdef double get_bb_lower(self):
        """Get the Bollinger Bands lower band (if available)."""
        # NOTE: shim for integration
        return <double> float('nan')

    cpdef double get_bb_upper(self):
        """Get the Bollinger Bands upper band (if available)."""
        # NOTE: shim for integration
        return <double> float('nan')

    cpdef void force_market_regime(self, object name, int start, int duration):
        """Force a specific market regime (e.g., NORMAL, TREND) for a given duration."""
        # NOTE: shim for integration
        # This stub does not implement regime changes. In integration, it would adjust internal market parameters.
        return
