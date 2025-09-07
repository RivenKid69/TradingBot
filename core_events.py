# -*- coding: utf-8 -*-
"""
core_events.py
Единые типы событий и структуры обмена между компонентами.
Совместимы с core_models.* и используются симулятором, сервисами live и бэктестом.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, Mapping

from core_models import Bar, Tick, Order, ExecReport


class EventType(str, Enum):
    """
    Унифицированные типы событий.
    MARKET_DATA_* — входящие данные рынка.
    ORDER_* — события жизненного цикла ордеров.
    EXEC_* — сделки (частичное или полное исполнение).
    SERVICE_* — служебные события.
    RISK_EVENT — событие risk-логики.
    """

    MARKET_DATA_BAR = "MARKET_DATA_BAR"
    MARKET_DATA_TICK = "MARKET_DATA_TICK"

    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_CANCELED = "ORDER_CANCELED"
    ORDER_REJECTED = "ORDER_REJECTED"

    EXEC_PARTIAL = "EXEC_PARTIAL"
    EXEC_FILLED = "EXEC_FILLED"

    RISK_EVENT = "RISK_EVENT"
    HEARTBEAT = "HEARTBEAT"


def _as_plain_dict(obj: Any) -> Dict[str, Any]:
    from decimal import Decimal
    from enum import Enum as _Enum
    def _conv(v: Any):
        if isinstance(v, Decimal):
            return str(v)
        if isinstance(v, _Enum):
            return v.value
        if hasattr(v, "__dataclass_fields__"):
            return {k: _conv(val) for k, val in asdict(v).items()}
        if isinstance(v, dict):
            return {k: _conv(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_conv(x) for x in v]
        return v
    return {k: _conv(v) for k, v in asdict(obj).items()}


@dataclass(frozen=True)
class MarketEvent:
    """
    Универсальный конверт рыночных данных.
    Ровно одно из полей bar или tick должно быть задано.
    """
    etype: EventType
    ts: int
    bar: Optional[Bar] = None
    tick: Optional[Tick] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _as_plain_dict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "MarketEvent":
        et = EventType(d["etype"]) if not isinstance(d.get("etype"), EventType) else d["etype"]
        bar = Bar.from_dict(d["bar"]) if d.get("bar") is not None else None
        tick = Tick.from_dict(d["tick"]) if d.get("tick") is not None else None
        return MarketEvent(etype=et, ts=int(d["ts"]), bar=bar, tick=tick, meta=dict(d.get("meta", {})))


@dataclass(frozen=True)
class OrderEvent:
    """
    Событие, описывающее подачу/отмену/отклонение ордера.
    Для ORDER_SUBMITTED поле order содержит намерение. Для ORDER_CANCELED/REJECTED — оригинальный ордер.
    """
    etype: EventType
    ts: int
    order: Order
    reason: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _as_plain_dict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "OrderEvent":
        et = EventType(d["etype"]) if not isinstance(d.get("etype"), EventType) else d["etype"]
        return OrderEvent(etype=et, ts=int(d["ts"]), order=Order.from_dict(d["order"]), reason=d.get("reason"), meta=dict(d.get("meta", {})))


@dataclass(frozen=True)
class FillEvent:
    """
    Событие сделки. Для частичного/полного исполнения используйте etype: EXEC_PARTIAL/EXEC_FILLED.
    """
    etype: EventType
    ts: int
    exec_report: ExecReport
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _as_plain_dict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "FillEvent":
        et = EventType(d["etype"]) if not isinstance(d.get("etype"), EventType) else d["etype"]
        return FillEvent(etype=et, ts=int(d["ts"]), exec_report=ExecReport.from_dict(d["exec_report"]), meta=dict(d.get("meta", {})))
