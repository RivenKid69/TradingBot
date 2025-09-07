from libc.math cimport tanh, log, log1p, fabs

# Inline helper functions for numeric safety and clipping
cdef inline double _tanh_clip(double x, double clip):
    cdef double t = tanh(x)
    if t > clip:
        t = clip
    elif t < -clip:
        t = -clip
    return t

cdef inline double _safe_div(double a, double b, double eps):
    if fabs(b) < eps:
        return 0.0
    return a / b

cdef int compute_n_features(list layout):
    """Return the total number of features given a layout definition."""
    cdef int total = 0
    cdef dict block
    for block in layout:
        total += <int>block["size"]
    return total

cdef void build_observation_vector_c(const EnvState* state, MarketSimulatorWrapper* market,
                                     float[::1] ext_norm_cols, float[::1] out_vec) nogil:
    """Build the observation feature vector (C-contiguous) under nogil."""
    cdef int i
    cdef double price = 0.0
    cdef int trade_count = 0
    cdef float rel_vol = 1.0
    cdef double vol_log = 0.0
    cdef double last_price = 0.0
    cdef double prev_price_static = 0.0
    cdef double prev_units_static = 0.0
    cdef double return_raw = 0.0
    cdef double atr_val = 0.0
    cdef double ofi_value = 0.0
    cdef double q_ratio = 0.0
    cdef double micro_offset = 0.0
    cdef double cash_frac = 0.0
    cdef double pos_frac = 0.0
    cdef double vol_imbalance = 0.0
    cdef double vol_imb_frac = 0.0
    cdef double event_hours = 0.0
    cdef float* out_ptr = &out_vec[0]
    cdef float* ext_ptr = NULL

    # Lazy initialization of static variables (with GIL) on first call
    cdef extern double PRICE_HISTORY[60]
    cdef extern bint history_initialized
    cdef extern double prev_price
    cdef extern double prev_units
    cdef extern int last_trade_count
    # The above extern declarations assume module-level static definitions exist for persistence.

    if not history_initialized:
        # Acquire GIL to import feature_config and initialize static data
        with gil:
            import obs.feature_config as feature_config
        # Initialize price history and prev values
        history_initialized = True
        prev_price = 0.0
        prev_units = 0.0
        last_trade_count = 0
        for i in range(60):
            PRICE_HISTORY[i] = 0.0

    # Compute bar-level data
    price = market.get_last_price()
    trade_count = <int>market.get_trade_count()
    # log-normalized volume
    vol_log = log1p(trade_count)
    # relative volume (current vs previous step)
    if state.step_idx == 0:
        rel_vol = 1.0
    else:
        if last_trade_count < 1:
            rel_vol = <float>trade_count
        else:
            rel_vol = <float>(trade_count / <double>last_trade_count)
    last_trade_count = trade_count

    # Fill bar block (price, log_volume_norm, rel_volume)
    out_ptr[0] = <float>price
    out_ptr[1] = <float>vol_log
    out_ptr[2] = rel_vol

    # Derived features: 1h return and volatility proxy
    # 1h return (tanh normalized)
    if state.step_idx == 0:
        # Initialize history
        PRICE_HISTORY[0] = price
        return_raw = 0.0
    elif state.step_idx < 60:
        return_raw = price / (PRICE_HISTORY[0] + 1e-8) - 1.0
        PRICE_HISTORY[state.step_idx % 60] = price
    else:
        prev_price_static = PRICE_HISTORY[state.step_idx % 60]
        return_raw = price / (prev_price_static + 1e-8) - 1.0
        PRICE_HISTORY[state.step_idx % 60] = price
    out_ptr[3] = <float>_tanh_clip(return_raw, 0.999)
    # ATR volatility proxy: log1p(ATR/price)
    atr_val = market.get_atr()
    out_ptr[4] = <float>(log1p(atr_val / (price + 1e-8)))

    # Technical indicators block (13 features)
    cdef double ma5 = market.get_ma5()
    cdef double ma20 = market.get_ma20()
    cdef double rsi14 = market.get_rsi()
    cdef double macd_val = market.get_macd()
    cdef double macd_signal = market.get_macd_signal()
    cdef double momentum_val = market.get_momentum()
    # We reuse atr_val computed above for ATR indicator
    cdef double cci_val = market.get_cci()
    cdef double obv_val = market.get_obv()
    cdef double bb_lower = market.get_bb_lower()
    cdef double bb_upper = market.get_bb_upper()
    # Validity flags for MA5 and MA20
    cdef float ma5_valid = 0.0
    cdef float ma20_valid = 0.0
    if state.step_idx >= 4:
        ma5_valid = 1.0
    if state.step_idx >= 19:
        ma20_valid = 1.0
    # Fill indicator block
    out_ptr[5] = <float>ma5
    out_ptr[6] = <float>ma20
    out_ptr[7] = ma5_valid
    out_ptr[8] = ma20_valid
    out_ptr[9] = <float>rsi14
    out_ptr[10] = <float>macd_val
    out_ptr[11] = <float>macd_signal
    out_ptr[12] = <float>momentum_val
    out_ptr[13] = <float>atr_val
    out_ptr[14] = <float>cci_val
    out_ptr[15] = <float>obv_val
    out_ptr[16] = <float>bb_lower
    out_ptr[17] = <float>bb_upper

    # Microstructure proxies block (3 features)
    # Order Flow Imbalance proxy = sign(return) * volume intensity
    cdef int sign_ret = 0
    if state.step_idx > 0:
        if price > prev_price:
            sign_ret = 1
        elif price < prev_price:
            sign_ret = -1
    ofi_value = sign_ret * rel_vol
    # Queue imbalance (tanh normalized ratio of top volumes)
    cdef double best_bid = market.get_best_bid()
    cdef double best_ask = market.get_best_ask()
    cdef double bid_vol = market.get_best_bid_volume()
    cdef double ask_vol = market.get_best_ask_volume()
    q_ratio = _safe_div(bid_vol - ask_vol, bid_vol + ask_vol, 1e-8)
    # Microprice offset from mid (in price units)
    cdef double mid_price = (best_bid + best_ask) / 2.0
    micro_offset = _safe_div(bid_vol - ask_vol, bid_vol + ask_vol, 1e-8) * ((best_ask - best_bid) / 2.0)
    # Fill microstructure block
    out_ptr[18] = <float>ofi_value
    out_ptr[19] = <float>_tanh_clip(q_ratio, 0.999)
    out_ptr[20] = <float>micro_offset

    # Agent state features block (6 features)
    # Fractions of cash and position in equity (tanh clipped)
    cash_frac = _safe_div(state.cash, state.net_worth, 1e-8)
    pos_frac = _safe_div(state._position_value, state.net_worth, 1e-8)
    out_ptr[21] = <float>_tanh_clip(cash_frac, 0.999)
    out_ptr[22] = <float>_tanh_clip(pos_frac, 0.999)
    # Last volume imbalance (buy-sell volume) as fraction of equity (tanh clipped)
    if state.step_idx == 0:
        prev_units = state.units
    vol_imbalance = state.units - prev_units
    prev_units = state.units
    # Convert volume imbalance to notional (volume * price) and fraction of net worth
    vol_imb_frac = _safe_div(vol_imbalance * price, state.net_worth, 1e-8)
    out_ptr[23] = <float>_tanh_clip(vol_imb_frac, 0.999)
    # Trade intensity (total number of trades this step)
    out_ptr[24] = <float>trade_count
    # Realized spread proxy (half of final bid-ask spread)
    cdef double realized_spread_val = (best_ask - best_bid) / 2.0
    out_ptr[25] = <float>realized_spread_val
    # Last agent fill ratio (actual filled vs requested volume)
    cdef double fill_ratio = state.last_agent_fill_ratio
    out_ptr[26] = <float>fill_ratio

    # Metadata features block (event importance, time since event, optional fear/greed)
    cdef float event_imp_feat = 0.0
    cdef float time_since_feat = 0.0
    cdef float fear_greed_feat = 0.0
    # Event importance (direct)
    event_imp_feat = <float>state.last_event_importance if hasattr(state, "last_event_importance") else 0.0
    # Time since event (tanh normalized days)
    if hasattr(state, "time_since_event"):
        event_hours = state.time_since_event
    elif hasattr(state, "last_event_step"):
        event_hours = _safe_div(state.step_idx - state.last_event_step, 60.0, 1e-8)
    else:
        event_hours = 1e6  # assume a very long time (no event)
    time_since_feat = <float>_tanh_clip(event_hours / 24.0, 0.999)
    # Fear/Greed index (if dynamic risk used)
    if state.use_dynamic_risk:
        fear_greed_feat = <float>state.fear_greed_value
    # Fill metadata block
    out_ptr[27] = event_imp_feat
    out_ptr[28] = time_since_feat
    if state.use_dynamic_risk:
        out_ptr[29] = fear_greed_feat

    # External normalized columns (if present)
    if ext_norm_cols.shape[0] > 0:
        ext_ptr = &ext_norm_cols[0]
    if ext_ptr != NULL:
        for i in range(ext_norm_cols.shape[0]):
            out_ptr[27 + (3 if state.use_dynamic_risk else 2) + i] = ext_ptr[i]

    # Token one-hot encoding
    cdef int token_id = 0
    if hasattr(state, "token_index"):
        token_id = <int>state.token_index
    cdef int token_offset = 27 + (3 if state.use_dynamic_risk else 2)
    token_offset += ext_norm_cols.shape[0] if ext_ptr != NULL else 0
    # Zero out all token slots
    for i in range(token_offset, token_offset + <int>feature_config.MAX_NUM_TOKENS):
        out_ptr[i] = 0.0
    if token_id < feature_config.MAX_NUM_TOKENS:
        out_ptr[token_offset + token_id] = 1.0

    # Update previous price for next step's OFI calculation
    prev_price = price
