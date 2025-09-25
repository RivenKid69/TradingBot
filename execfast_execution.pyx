# cython: language_level=3
import core_constants as constants
from coreworkspace cimport SimulationWorkspace


cdef long long _resolve_timestamp(object state, SimulationWorkspace ws):
    cdef object value = getattr(ws, "step_index", None)
    if value is not None:
        try:
            return <long long> value
        except Exception:
            pass

    value = getattr(state, "step_index", None)
    if value is not None:
        try:
            return <long long> value
        except Exception:
            pass

    return 0


def execute_market_fast(state, tracker, params, SimulationWorkspace ws, int side, int qty, double price):
    """
    Fast execution model for market order: immediately fill at given price.
    side: 1 for buy, -1 for sell. qty: volume to trade. price: execution price in actual currency.
    Records the trade in SimulationWorkspace.
    """
    if qty <= 0:
        return
    cdef double exec_price = price
    cdef long long ts = _resolve_timestamp(state, ws)
    ws.push_trade(exec_price, qty, <char> side, <char> constants.AGENT_TAKER, ts)
    # Note: No order remains open in fast execution (market order fully executed immediately).

def execute_limit_fast(state, tracker, params, SimulationWorkspace ws, int side, int qty, double price):
    """
    Fast execution model for limit order: assume it gets filled by end of step if price is reached.
    If not reached, it is canceled (no partial persistence in this simple model).
    side: 1 for buy limit, -1 for sell limit. price: limit price in actual currency.
    """
    if qty <= 0:
        return False  # no trade
    cdef bint filled = False
    # Simple model: assume limit order always executes at desired price by step end for simplicity
    filled = True
    cdef double exec_price
    cdef long long ts
    if filled:
        exec_price = price
        ts = _resolve_timestamp(state, ws)
        ws.push_trade(exec_price, qty, <char> side, <char> constants.AGENT_MAKER, ts)
        # No open order to carry since it was filled
    return filled
