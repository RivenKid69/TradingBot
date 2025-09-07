# metrics/info_builder.pyx

from libc.math cimport fabs
cdef inline double _safe_div(double a, double b) nogil:
    """Safe division: returns 0.0 if b is zero, otherwise a/b"""
    return a / b if b != 0.0 else 0.0

cdef double compute_vol_imbalance_nogil(double agent_net_taker_flow) nogil:
    """Compute vol_imbalance: agent taker flow imbalance (buy volume – sell volume):contentReference[oaicite:11]{index=11}"""
    # Just return the net taker volume (positive if net buy, negative if net sell)
    return agent_net_taker_flow

cdef double compute_trade_intensity_nogil(int total_trades_count) nogil:
    """Compute trade_intensity: total number of trades in the order book this step:contentReference[oaicite:12]{index=12}"""
    return <double>total_trades_count

cdef double compute_realized_spread_nogil(long long best_bid, long long best_ask, double price_scale) nogil:
    """Compute realized_spread: half of the bid-ask spread at step end (final BBO):contentReference[oaicite:13]{index=13}"""
    if best_bid > 0 and best_ask > 0:
        # (ask - bid) / 2, convert from ticks by dividing by price_scale
        return _safe_div(best_ask - best_bid, 2.0 * price_scale)
    else:
        # If either side is empty (no best bid or ask), define spread as 0
        return 0.0

cdef double compute_agent_fill_ratio_nogil(double actual_taker_vol, double intended_taker_vol) nogil:
    """Compute agent_fill_ratio: actual taker volume filled vs intended volume"""
    if intended_taker_vol <= 0.0:
        # If no taker order was intended this step, define fill ratio as 1 (no shortfall):contentReference[oaicite:16]{index=16}
        return 1.0
    # Return fraction of intended volume that was actually filled (<= 1.0)
    return _safe_div(actual_taker_vol, intended_taker_vol)

cdef double compute_slippage_bps_nogil(long long initial_bid, long long initial_ask,
                                       long long final_bid, long long final_ask,
                                       double price_scale, double agent_net_taker_flow) nogil:
    """Compute slippage in basis points (bps) for agent's taker order execution."""
    # If no taker trade occurred, slippage is 0
    if agent_net_taker_flow == 0.0:
        return 0.0
    # Compute initial mid-price (in actual price units)
    cdef double initial_mid_price = 0.0
    if initial_bid > 0 and initial_ask > 0:
        initial_mid_price = (initial_bid + initial_ask) / (2.0 * price_scale)
    else:
        # If initial bid/ask missing, cannot compute slippage meaningfully
        return 0.0
    # Estimate average fill price for agent's taker trade
    cdef double avg_fill_price = 0.0
    if agent_net_taker_flow > 0.0:
        # Agent was net buyer. If final ask moved up from initial ask, assume multiple levels filled.
        if final_ask > initial_ask:
            # Estimate avg fill price roughly as midpoint of initial and final ask price levels (in actual units)
            avg_fill_price = ((initial_ask / price_scale) + (final_ask / price_scale)) / 2.0
        else:
            # Final ask unchanged -> partial fill at initial ask price
            avg_fill_price = initial_ask / price_scale
    elif agent_net_taker_flow < 0.0:
        # Agent was net seller. If final bid moved down from initial bid, multiple levels consumed.
        if final_bid < initial_bid:
            avg_fill_price = ((initial_bid / price_scale) + (final_bid / price_scale)) / 2.0
        else:
            # Final bid unchanged -> partial fill at initial bid price
            avg_fill_price = initial_bid / price_scale
    else:
        return 0.0  # no taker volume
    # Compute slippage fraction relative to initial mid-price
    cdef double slippage_frac = 0.0
    if agent_net_taker_flow > 0.0:
        # For buys: (avg fill price – initial mid) / initial mid
        slippage_frac = _safe_div(avg_fill_price - initial_mid_price, initial_mid_price)
    else:
        # For sells: (initial mid – avg fill price) / initial mid
        slippage_frac = _safe_div(initial_mid_price - avg_fill_price, initial_mid_price)
    # Convert to basis points (1 bps = 1e-4)
    return slippage_frac * 10000.0

cpdef dict build_info_dict(EnvState* state,
                           double agent_intended_taker_vol,
                           double agent_actual_taker_vol,
                           double agent_net_taker_flow,
                           int total_trades_count,
                           long long initial_best_bid,
                           long long initial_best_ask,
                           long long final_best_bid,
                           long long final_best_ask,
                           ClosedReason closed_reason):
    """
    Build the info dictionary with step metrics:
    - "slippage_bps"
    - "agent_fill_ratio"
    - "trade_intensity"
    - "vol_imbalance"
    - "realized_spread"
    - "closed" (dict with reason or None):contentReference[oaicite:18]{index=18}
    """
    cdef double slippage_bps, fill_ratio, trade_intens, vol_imb, realized_spread
    # Compute metrics in nogil (no Python operations)
    with nogil:
        vol_imb = compute_vol_imbalance_nogil(agent_net_taker_flow)
        trade_intens = compute_trade_intensity_nogil(total_trades_count)
        realized_spread = compute_realized_spread_nogil(final_best_bid, final_best_ask, state.price_scale)
        fill_ratio = compute_agent_fill_ratio_nogil(agent_actual_taker_vol, agent_intended_taker_vol)
        slippage_bps = compute_slippage_bps_nogil(initial_best_bid, initial_best_ask,
                                                  final_best_bid, final_best_ask,
                                                  state.price_scale, agent_net_taker_flow)
    # Prepare the info dict (acquire GIL again for Python object creation)
    cdef dict info = {
        "vol_imbalance": vol_imb,
        "trade_intensity": trade_intens,
        "realized_spread": realized_spread,
        "agent_fill_ratio": fill_ratio,
        "slippage_bps": slippage_bps
    }
    # Determine closed position info
    cdef dict closed_dict = None
    if closed_reason != ClosedReason.NONE:
        # Map ClosedReason to string labels:contentReference[oaicite:19]{index=19}
        cdef str reason_str
        if closed_reason == ClosedReason.ATR_SL_LONG:
            reason_str = "atr_sl_long"
        elif closed_reason == ClosedReason.ATR_SL_SHORT:
            reason_str = "atr_sl_short"
        elif closed_reason == ClosedReason.TRAILING_SL_LONG:
            reason_str = "trailing_sl_long"
        elif closed_reason == ClosedReason.TRAILING_SL_SHORT:
            reason_str = "trailing_sl_short"
        elif closed_reason == ClosedReason.STATIC_TP_LONG:
            reason_str = "static_tp_long"
        elif closed_reason == ClosedReason.STATIC_TP_SHORT:
            reason_str = "static_tp_short"
        elif closed_reason == ClosedReason.BANKRUPTCY:
            reason_str = "bankruptcy"
        elif closed_reason == ClosedReason.MAX_DRAWDOWN:
            reason_str = "max_drawdown"
        else:
            reason_str = "none"
        closed_dict = {"reason": reason_str}
    # Include the 'closed' key in info (None if no closure):contentReference[oaicite:20]{index=20}
    info["closed"] = closed_dict
    return info
