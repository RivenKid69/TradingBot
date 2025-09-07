# -*- coding: utf-8 -*-
"""
market_data_port.py
Единые контракты источников рыночных данных для офлайна и Binance WS.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union, Protocol
from enum import Enum
from datetime import datetime, timezone
from utils_time import parse_time_to_ms


class EventKind(str, Enum):
    BAR = "BAR"
    QUOTE = "QUOTE"
    TRADE = "TRADE"


@dataclass(frozen=True)
class Bar:
    ts: int                    # миллисекунды Unix
    symbol: str
    timeframe: str             # "1s"|"1m"|"5m"|"1h"|...
    open: float
    high: float
    low: float
    close: float
    volume: float
    # необязательные поля
    bid: Optional[float] = None
    ask: Optional[float] = None
    n_trades: Optional[int] = None
    vendor: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Quote:
    ts: int
    symbol: str
    bid: float
    ask: float
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    vendor: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Trade:
    ts: int
    symbol: str
    price: float
    qty: float
    is_buy: Optional[bool] = None
    vendor: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketEvent:
    kind: EventKind
    bar: Optional[Bar] = None
    quote: Optional[Quote] = None
    trade: Optional[Trade] = None

    def to_dict(self) -> Dict[str, Any]:
        if self.kind == EventKind.BAR and self.bar is not None:
            d = self.bar.to_dict()
        elif self.kind == EventKind.QUOTE and self.quote is not None:
            d = self.quote.to_dict()
        elif self.kind == EventKind.TRADE and self.trade is not None:
            d = self.trade.to_dict()
        else:
            d = {}
        return {"kind": self.kind.value, **d}


class MarketDataSource(Protocol):
    """
    Минимальный синхронный контракт источника событий рынка.
    """
    def open(self) -> None: ...
    def subscribe(self, symbols: Sequence[str], *, timeframe: Optional[str] = None) -> None: ...
    def iter_events(self) -> Iterator[MarketEvent]: ...
    def close(self) -> None: ...


# Утилиты времени/таймфрейма

_VALID_TF = {"1s", "5s", "10s", "15s", "30s",
             "1m", "3m", "5m", "15m", "30m",
             "1h", "2h", "4h", "6h", "8h", "12h",
             "1d"}

def ensure_timeframe(tf: str) -> str:
    tf = str(tf).lower()
    if tf not in _VALID_TF:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return tf

def to_ms(dt: Union[int, float, str, datetime]) -> int:
    """Return Unix timestamp in milliseconds.

    Numeric inputs may represent either seconds or milliseconds; values
    below ``10_000_000_000`` are assumed to be seconds and are multiplied by
    ``1000``. String inputs are delegated to :func:`utils_time.parse_time_to_ms`
    for parsing, and :class:`datetime` objects are converted assuming UTC if
    timezone information is missing.
    """
    if isinstance(dt, (int, float)):
        v = float(dt)
        if v < 10_000_000_000:
            v *= 1000
        return int(v)
    if isinstance(dt, str):
        return parse_time_to_ms(dt)
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    raise TypeError(f"Unsupported datetime type: {type(dt)}")


def binance_tf(tf: str) -> str:
    """
    Преобразование таймфрейма проекта в интервал Binance kline.
    """
    tf = ensure_timeframe(tf)
    return tf.lower()  # совпадает по обозначениям
