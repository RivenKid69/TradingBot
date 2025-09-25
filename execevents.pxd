cdef extern from "include/execevents_types.h":
    ctypedef enum EventType:
        AGENT_LIMIT_ADD
        AGENT_MARKET_MATCH
        AGENT_CANCEL_SPECIFIC
        PUBLIC_LIMIT_ADD
        PUBLIC_MARKET_MATCH
        PUBLIC_CANCEL_RANDOM

    ctypedef enum Side:
        BUY
        SELL

    ctypedef struct MarketEvent:
        EventType type
        Side side
        int price
        int qty
        int order_id
