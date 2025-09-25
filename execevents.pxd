cpdef enum EventType:
    AGENT_LIMIT_ADD = 0
    AGENT_MARKET_MATCH = 1
    AGENT_CANCEL_SPECIFIC = 2
    PUBLIC_LIMIT_ADD = 3
    PUBLIC_MARKET_MATCH = 4
    PUBLIC_CANCEL_RANDOM = 5

cpdef enum Side:
    BUY = 1
    SELL = -1

cdef struct MarketEvent:
    EventType type
    Side side
    int price    # price in ticks (for limit events; 0 if not applicable)
    int qty      # quantity (volume) of order or trade
    int order_id # order identifier (for add events, id to assign; for cancel, id to remove; unused for pure market events)
