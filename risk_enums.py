"""Python helpers mirroring the C-level risk enums."""
from enum import IntEnum


class ClosedReason(IntEnum):
    NONE = 0
    ATR_SL_LONG = 1
    ATR_SL_SHORT = 2
    TRAILING_SL_LONG = 3
    TRAILING_SL_SHORT = 4
    STATIC_TP_LONG = 5
    STATIC_TP_SHORT = 6
    BANKRUPTCY = 7
    MAX_DRAWDOWN = 8


__all__ = ["ClosedReason"]
