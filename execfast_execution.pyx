# cython: language_level=3
from core.constants cimport PRICE_SCALE
from core.workspace cimport SimulationWorkspace

def execute_market_fast(state, tracker, params, SimulationWorkspace ws, int side, int qty, double price):
    """
    Fast execution model for market order: immediately fill at given price.
    side: 1 for buy, -1 for sell. qty: volume to trade. price: execution price in actual currency.
    Records the trade in SimulationWorkspace.
    """
    if qty <= 0:
        return
    cdef double exec_price = price
    # Record trade
    ws.ensure_capacity(ws.trade_count + 1)
    ws.trade_price[ws.trade_count] = <double> (exec_price * PRICE_SCALE)  # convert price to scaled ticks
    ws.trade_volume[ws.trade_count] = qty
    ws.trade_side[ws.trade_count] = side
    ws.trade_agent_marker[ws.trade_count] = 2  # agent is taker in fast execution (market order)
    ws.trade_timestamp[ws.trade_count] = ws.step_index
    ws.trade_count += 1
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
    if filled:
        cdef double exec_price = price
        ws.ensure_capacity(ws.trade_count + 1)
        ws.trade_price[ws.trade_count] = <double> (exec_price * PRICE_SCALE)
        ws.trade_volume[ws.trade_count] = qty
        ws.trade_side[ws.trade_count] = side
        ws.trade_agent_marker[ws.trade_count] = 1  # assume agent as maker if limit executed
        ws.trade_timestamp[ws.trade_count] = ws.step_index
        ws.trade_count += 1
        # No open order to carry since it was filled
    return filled
