# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport tanh, log1p, isnan


cdef inline float _clipf(double value, double lower, double upper) nogil:
    if value < lower:
        value = lower
    elif value > upper:
        value = upper
    return <float>value


cdef int compute_n_features(list layout):
    """Utility used by legacy Python code to count feature slots."""
    cdef int total = 0
    cdef dict block
    for block in layout:
        total += <int>block.get("size", 0)
    return total


cpdef void build_observation_vector_c(
    float price,
    float prev_price,
    float log_volume_norm,
    float rel_volume,
    float ma5,
    float ma20,
    float rsi14,
    float macd,
    float macd_signal,
    float momentum,
    float atr,
    float cci,
    float obv,
    float bb_lower,
    float bb_upper,
    float is_high_importance,
    float time_since_event,
    float fear_greed_value,
    bint has_fear_greed,
    bint risk_off_flag,
    float cash,
    float units,
    float last_vol_imbalance,
    float last_trade_intensity,
    float last_realized_spread,
    float last_agent_fill_ratio,
    int token_id,
    int max_num_tokens,
    int num_tokens,
    float[::1] norm_cols_values,
    float[::1] out_features
) noexcept nogil:
    """Populate ``out_features`` with the observation vector without acquiring the GIL."""

    cdef int feature_idx = 0
    cdef float feature_val
    cdef double price_d = price
    cdef double prev_price_d = prev_price
    cdef double position_value
    cdef double total_worth
    cdef double ret_1h
    cdef double vol_proxy
    cdef double mid_ret
    cdef double vol_intensity
    cdef double ofi_proxy
    cdef double qimb
    cdef double micro_dev
    cdef double bb_width
    cdef bint ma5_valid
    cdef bint ma20_valid
    cdef bint bb_valid
    cdef double min_bb_width
    cdef int padded_tokens
    cdef int i

    # --- Bar level block ---------------------------------------------------
    out_features[feature_idx] = price
    feature_idx += 1
    out_features[feature_idx] = log_volume_norm
    feature_idx += 1
    out_features[feature_idx] = rel_volume
    feature_idx += 1

    ma5_valid = not isnan(ma5)
    out_features[feature_idx] = ma5 if ma5_valid else 0.0
    feature_idx += 1
    out_features[feature_idx] = 1.0 if ma5_valid else 0.0
    feature_idx += 1

    ma20_valid = not isnan(ma20)
    out_features[feature_idx] = ma20 if ma20_valid else 0.0
    feature_idx += 1
    out_features[feature_idx] = 1.0 if ma20_valid else 0.0
    feature_idx += 1

    out_features[feature_idx] = rsi14
    feature_idx += 1
    out_features[feature_idx] = macd
    feature_idx += 1
    out_features[feature_idx] = macd_signal
    feature_idx += 1
    out_features[feature_idx] = momentum
    feature_idx += 1
    out_features[feature_idx] = atr
    feature_idx += 1
    out_features[feature_idx] = cci
    feature_idx += 1
    out_features[feature_idx] = obv
    feature_idx += 1

    # Derived price/volatility signals
    ret_1h = tanh((price_d - prev_price_d) / (prev_price_d + 1e-8))
    out_features[feature_idx] = <float>ret_1h
    feature_idx += 1

    vol_proxy = tanh(log1p(atr / (price_d + 1e-8)))
    out_features[feature_idx] = <float>vol_proxy
    feature_idx += 1

    # --- Agent state block -------------------------------------------------
    position_value = units * price_d
    total_worth = cash + position_value

    if total_worth <= 1e-8:
        feature_val = 1.0
    else:
        feature_val = _clipf(cash / total_worth, 0.0, 1.0)
    out_features[feature_idx] = feature_val
    feature_idx += 1

    if total_worth <= 1e-8:
        feature_val = 0.0
    else:
        feature_val = <float>tanh(position_value / (total_worth + 1e-8))
    out_features[feature_idx] = feature_val
    feature_idx += 1

    out_features[feature_idx] = <float>tanh(last_vol_imbalance)
    feature_idx += 1
    out_features[feature_idx] = <float>tanh(last_trade_intensity)
    feature_idx += 1

    feature_val = _clipf(last_realized_spread, -0.1, 0.1)
    out_features[feature_idx] = feature_val
    feature_idx += 1

    out_features[feature_idx] = last_agent_fill_ratio
    feature_idx += 1

    # --- Microstructure proxies --------------------------------------------
    mid_ret = tanh((price_d - prev_price_d) / (prev_price_d + 1e-8))
    vol_intensity = tanh(rel_volume)
    ofi_proxy = mid_ret * vol_intensity
    out_features[feature_idx] = <float>ofi_proxy
    feature_idx += 1

    qimb = tanh(last_vol_imbalance)
    out_features[feature_idx] = <float>qimb
    feature_idx += 1

    micro_dev = 0.5 * last_realized_spread * qimb
    out_features[feature_idx] = <float>micro_dev
    feature_idx += 1

    # --- Bollinger band context -------------------------------------------
    bb_width = bb_upper - bb_lower
    bb_valid = not isnan(bb_lower)
    min_bb_width = price_d * 0.0001
    if (not bb_valid) or bb_width <= min_bb_width:
        feature_val = 0.5
    else:
        feature_val = _clipf((price_d - bb_lower) / (bb_width + 1e-9), -1.0, 2.0)
    out_features[feature_idx] = feature_val
    feature_idx += 1

    if bb_valid:
        feature_val = _clipf(bb_width / (price_d + 1e-8), 0.0, 10.0)
    else:
        feature_val = 0.0
    out_features[feature_idx] = feature_val
    feature_idx += 1

    # --- Event metadata ----------------------------------------------------
    out_features[feature_idx] = is_high_importance
    feature_idx += 1

    out_features[feature_idx] = <float>tanh(time_since_event / 24.0)
    feature_idx += 1

    out_features[feature_idx] = 1.0 if risk_off_flag else 0.0
    feature_idx += 1

    # --- External normalised columns --------------------------------------
    for i in range(norm_cols_values.shape[0]):
        feature_val = _clipf(norm_cols_values[i], -3.0, 3.0)
        out_features[feature_idx] = feature_val
        feature_idx += 1

    # --- Fear & greed (optional) ------------------------------------------
    if has_fear_greed:
        feature_val = _clipf(fear_greed_value / 100.0, -3.0, 3.0)
        out_features[feature_idx] = feature_val
        feature_idx += 1

    # --- Token one-hot -----------------------------------------------------
    padded_tokens = max_num_tokens
    if padded_tokens <= 0:
        padded_tokens = num_tokens
    elif padded_tokens < num_tokens:
        padded_tokens = num_tokens

    if padded_tokens > 0:
        for i in range(padded_tokens):
            out_features[feature_idx + i] = 0.0
        if 0 <= token_id < num_tokens and token_id < padded_tokens:
            out_features[feature_idx + token_id] = 1.0
        feature_idx += padded_tokens
