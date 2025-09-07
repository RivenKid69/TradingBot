"""Constants and enumerations for the trading simulation module.

This module defines global constants such as price scale and default limits 
for events and trades per simulation step, as well as enumeration values (flags) 
for order execution types and trade sides. These constants are used throughout 
the simulation to avoid magic numbers and ensure consistency.

Attributes:
    PRICE_SCALE (int): The price scale (number of ticks in one unit of price). 
        For example, if PRICE_SCALE = 10000, one tick = 0.0001 in price. 
        This acts as a global scale for converting between float prices and integer ticks.
    DEFAULT_MAX_TRADES_PER_STEP (int): Default safe upper limit on the number of trades that can occur in a single step.
    DEFAULT_MAX_GENERATED_EVENTS_PER_TYPE (int): Default limit on the number of generated events of a single type per step (to avoid runaway event generation).
    
Constants/Flags:
    ORDER_TYPE_MARKET (int): Code for market order execution mode.
    ORDER_TYPE_LIMIT (int): Code for limit order execution mode.
    SIDE_BUY (int): Trade side code for buy (typically used as 1).
    SIDE_SELL (int): Trade side code for sell (typically used as -1).
    AGENT_MAKER (int): Flag indicating the agent was the maker in a trade (1 for maker).
    AGENT_TAKER (int): Flag indicating the agent was the taker in a trade (0 for taker).
"""
from enum import IntEnum, IntFlag  # (if needed for additional enum types, otherwise not used)

# Price scale: number of ticks per unit of price (e.g., 10000 means 1 tick = 0.0001).
PRICE_SCALE: int = 10000

# Default limits for simulation events and trades per step.
DEFAULT_MAX_TRADES_PER_STEP: int = 10000
DEFAULT_MAX_GENERATED_EVENTS_PER_TYPE: int = 5000

# Execution mode constants (e.g., order types for execution).
ORDER_TYPE_MARKET: int = 1
ORDER_TYPE_LIMIT: int = 2

# Trade side constants.
SIDE_BUY: int = 1
SIDE_SELL: int = -1

# Trade role flags for agent's maker/taker status.
AGENT_MAKER: int = 1
AGENT_TAKER: int = 0

# (Additional enums or flags can be added here as needed for execution modes, event types, etc.)
