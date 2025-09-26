# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
"""Helper utilities to build the info dictionary for environment metrics."""

from lob_state_cython cimport EnvState
from risk_enums cimport ClosedReason


cdef inline double _safe_div(double a, double b) nogil:
    if b != 0.0:
        return a / b
    return 0.0


cdef inline double compute_vol_imbalance_nogil(double agent_net_taker_flow) nogil:
    return agent_net_taker_flow


cdef inline double compute_trade_intensity_nogil(int total_trades_count) nogil:
    return <double>total_trades_count


cdef inline double compute_realized_spread_nogil(long long best_bid,
                                                 long long best_ask,
                                                 double price_scale) nogil:
    cdef double spread_ticks
    if best_bid > 0 and best_ask > 0:
        spread_ticks = <double>(best_ask - best_bid)
        return _safe_div(spread_ticks, 2.0 * price_scale)
    return 0.0


cdef inline double compute_agent_fill_ratio_nogil(double actual_taker_vol,
                                                  double intended_taker_vol) nogil:
    if intended_taker_vol <= 0.0:
        return 1.0
    return _safe_div(actual_taker_vol, intended_taker_vol)


cdef inline double compute_slippage_bps_nogil(long long initial_bid,
                                              long long initial_ask,
                                              long long final_bid,
                                              long long final_ask,
                                              double price_scale,
                                              double agent_net_taker_flow) nogil:
    cdef double initial_mid_price
    cdef double avg_fill_price
    cdef double slippage_frac

    if agent_net_taker_flow == 0.0:
        return 0.0

    if initial_bid > 0 and initial_ask > 0:
        initial_mid_price = (<double>initial_bid + <double>initial_ask) / (2.0 * price_scale)
    else:
        return 0.0

    avg_fill_price = 0.0
    if agent_net_taker_flow > 0.0:
        if final_ask > initial_ask:
            avg_fill_price = ((<double>initial_ask / price_scale) + (<double>final_ask / price_scale)) / 2.0
        else:
            avg_fill_price = <double>initial_ask / price_scale
    elif agent_net_taker_flow < 0.0:
        if final_bid < initial_bid:
            avg_fill_price = ((<double>initial_bid / price_scale) + (<double>final_bid / price_scale)) / 2.0
        else:
            avg_fill_price = <double>initial_bid / price_scale
    else:
        return 0.0

    if agent_net_taker_flow > 0.0:
        slippage_frac = _safe_div(avg_fill_price - initial_mid_price, initial_mid_price)
    else:
        slippage_frac = _safe_div(initial_mid_price - avg_fill_price, initial_mid_price)

    return slippage_frac * 10000.0


CLOSED_REASON_LABELS = {
    int(ClosedReason.ATR_SL_LONG): "atr_sl_long",
    int(ClosedReason.ATR_SL_SHORT): "atr_sl_short",
    int(ClosedReason.TRAILING_SL_LONG): "trailing_sl_long",
    int(ClosedReason.TRAILING_SL_SHORT): "trailing_sl_short",
    int(ClosedReason.STATIC_TP_LONG): "static_tp_long",
    int(ClosedReason.STATIC_TP_SHORT): "static_tp_short",
    int(ClosedReason.BANKRUPTCY): "bankruptcy",
    int(ClosedReason.MAX_DRAWDOWN): "max_drawdown",
}


cpdef dict build_info_dict(EnvState state,
                           double agent_intended_taker_vol,
                           double agent_actual_taker_vol,
                           double agent_net_taker_flow,
                           int total_trades_count,
                           long long initial_best_bid,
                           long long initial_best_ask,
                           long long final_best_bid,
                           long long final_best_ask,
                           ClosedReason closed_reason):
    cdef double slippage_bps
    cdef double fill_ratio
    cdef double trade_intensity
    cdef double vol_imbalance
    cdef double realized_spread
    cdef dict info
    cdef object closed_value
    cdef str reason_str

    with nogil:
        vol_imbalance = compute_vol_imbalance_nogil(agent_net_taker_flow)
        trade_intensity = compute_trade_intensity_nogil(total_trades_count)
        realized_spread = compute_realized_spread_nogil(final_best_bid, final_best_ask, state.price_scale)
        fill_ratio = compute_agent_fill_ratio_nogil(agent_actual_taker_vol, agent_intended_taker_vol)
        slippage_bps = compute_slippage_bps_nogil(initial_best_bid, initial_best_ask,
                                                  final_best_bid, final_best_ask,
                                                  state.price_scale, agent_net_taker_flow)

    info = {
        "vol_imbalance": vol_imbalance,
        "trade_intensity": trade_intensity,
        "realized_spread": realized_spread,
        "agent_fill_ratio": fill_ratio,
        "slippage_bps": slippage_bps,
    }

    closed_value = None
    if closed_reason != ClosedReason.NONE:
        reason_str = CLOSED_REASON_LABELS.get(<int>closed_reason, "none")
        closed_value = {"reason": reason_str}

    info["closed"] = closed_value
    return info
