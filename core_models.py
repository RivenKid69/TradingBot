# -*- coding: utf-8 -*-
"""
core_models.py
Единая доменная модель для среды обучения и инструментов построения среднечастотного сигнального бота.

Содержимое:
- Перечисления: Side, OrderType, TimeInForce, Liquidity, ExecStatus.
- Базовые сущности: Instrument, Bar, Tick.
- Ордера и исполнения: Order, ExecReport.
- Позиции и ограничения портфеля: Position, PortfolioLimits.
- Логи и отчёты: TradeLogRow, EquityPoint.
- Утилиты сериализации: as_dict(), to_json(), from_dict().

Договорённости:
- Время: int миллисекунды UNIX UTC, поле "ts" или "ts_ms" (в этом файле везде "ts").
- Денежные величины: Decimal в валюте котировки или базовой, явно помечены.
- Количества (qty): Decimal в базовом активе.
- Цена (price): Decimal в валюте котировки за 1 единицу базового.
- Строковые enum в CSV/JSON — значения .value соответствующих Enum.
- Все dataclass "frozen=True" для неизменяемости; изменяемые поля вынесены в "meta: Dict[str, Any]".
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from enum import Enum
from typing import Optional, Dict, Any, Mapping, List, Union
from decimal import Decimal, InvalidOperation
import json
import uuid

# =========================
# Перечисления
# =========================

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class Liquidity(str, Enum):
    MAKER = "MAKER"
    TAKER = "TAKER"
    UNKNOWN = "UNKNOWN"


class ExecStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


# =========================
# Базовые сущности
# =========================

@dataclass(frozen=True)
class OrderIntent:
    """
    Стандартное «намерение ордера», которое выходит из стратегии.
    Не фиксирует абсолютное количество. Используется как унифицированный выход,
    а затем может быть трансформировано в Order с учётом контекста (квантование, лимиты).
    """
    ts: int
    symbol: str
    side: Side
    order_type: OrderType
    volume_frac: Decimal            # доля от максимально допустимой позиции по базовому активу ([-1..1])
    price_offset_ticks: int = 0     # смещение цены в тиках для LIMIT; 0 = по референсу
    time_in_force: TimeInForce = TimeInForce.GTC
    client_tag: str = field(default_factory=lambda: str(uuid.uuid4()))
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return as_dict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "OrderIntent":
        return OrderIntent(
            ts=int(d["ts"]),
            symbol=str(d["symbol"]),
            side=Side(str(d["side"])),
            order_type=OrderType(str(d["order_type"])),
            volume_frac=to_decimal(d["volume_frac"]),
            price_offset_ticks=int(d.get("price_offset_ticks", 0)),
            time_in_force=TimeInForce(str(d.get("time_in_force", "GTC"))),
            client_tag=str(d.get("client_tag") or ""),
            meta=dict(d.get("meta", {})),
        )

@dataclass(frozen=True)
class Instrument:
    """
    Описание торгового инструмента.
    """
    symbol: str
    base_asset: str
    quote_asset: str
    tick_size: Decimal                   # минимальный шаг цены
    step_size: Decimal                   # минимальный шаг количества
    min_notional: Decimal                # минимальный ноционал в quote
    price_scale: int = 0                 # количество знаков после запятой у цены (опционально)
    qty_scale: int = 0                   # количество знаков после запятой у qty (опционально)
    filters: Dict[str, Any] = field(default_factory=dict)  # сырой snapshot фильтров биржи

    def to_dict(self) -> Dict[str, Any]:
        return as_dict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Instrument":
        return Instrument(
            symbol=str(d["symbol"]),
            base_asset=str(d["base_asset"]),
            quote_asset=str(d["quote_asset"]),
            tick_size=to_decimal(d["tick_size"]),
            step_size=to_decimal(d["step_size"]),
            min_notional=to_decimal(d["min_notional"]),
            price_scale=int(d.get("price_scale", 0)),
            qty_scale=int(d.get("qty_scale", 0)),
            filters=dict(d.get("filters", {})),
        )


@dataclass(frozen=True)
class Bar:
    """
    Свеча OHLCV. Все цены и объёмы — Decimal.
    """
    ts: int                        # unix ms (UTC)
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume_base: Optional[Decimal] = None   # объём в базовом активе (например BTC)
    volume_quote: Optional[Decimal] = None  # объём в котировочной валюте (например USDT)
    trades: Optional[int] = None
    vwap: Optional[Decimal] = None
    is_final: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return as_dict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Bar":
        return Bar(
            ts=int(d["ts"]),
            symbol=str(d["symbol"]),
            open=to_decimal(d["open"]),
            high=to_decimal(d["high"]),
            low=to_decimal(d["low"]),
            close=to_decimal(d["close"]),
            volume_base=to_decimal_opt(d.get("volume_base")),
            volume_quote=to_decimal_opt(d.get("volume_quote")),
            trades=int(d["trades"]) if d.get("trades") is not None else None,
            vwap=to_decimal_opt(d.get("vwap")),
            is_final=bool(d.get("is_final", True)),
        )


@dataclass(frozen=True)
class Tick:
    """
    Снимок BBO/сделки.
    """
    ts: int
    symbol: str
    price: Optional[Decimal] = None        # последняя цена сделки
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    bid_qty: Optional[Decimal] = None
    ask_qty: Optional[Decimal] = None
    is_final: bool = True
    spread_bps: Optional[Decimal] = None

    def mid(self) -> Optional[Decimal]:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / Decimal("2")
        return None

    def to_dict(self) -> Dict[str, Any]:
        return as_dict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Tick":
        return Tick(
            ts=int(d["ts"]),
            symbol=str(d["symbol"]),
            price=to_decimal_opt(d.get("price")),
            bid=to_decimal_opt(d.get("bid")),
            ask=to_decimal_opt(d.get("ask")),
            bid_qty=to_decimal_opt(d.get("bid_qty")),
            ask_qty=to_decimal_opt(d.get("ask_qty")),
            is_final=bool(d.get("is_final", True)),
            spread_bps=to_decimal_opt(d.get("spread_bps")),
        )


# =========================
# Ордер и исполнение
# =========================

@dataclass(frozen=True)
class Order:
    """
    Запрос на размещение/модификацию ордера.
    Для MARKET price может быть None.
    """
    ts: int
    symbol: str
    side: Side
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reduce_only: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return as_dict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Order":
        return Order(
            ts=int(d["ts"]),
            symbol=str(d["symbol"]),
            side=Side(d["side"]),
            order_type=OrderType(d["order_type"]),
            quantity=to_decimal(d["quantity"]),
            price=to_decimal_opt(d.get("price")),
            time_in_force=TimeInForce(d.get("time_in_force", "GTC")),
            client_order_id=str(d.get("client_order_id") or uuid.uuid4()),
            reduce_only=bool(d.get("reduce_only", False)),
            meta=dict(d.get("meta", {})),
        )


@dataclass(frozen=True)
class ExecReport:
    """
    Единый отчёт об исполнении. Используется и в симуляторе, и в live.
    Один отчёт — одна сделка/частичное исполнение.
    """
    ts: int
    run_id: str
    symbol: str
    side: Side
    order_type: OrderType
    price: Decimal
    quantity: Decimal
    fee: Decimal
    fee_asset: Optional[str]
    exec_status: ExecStatus = ExecStatus.FILLED
    liquidity: Liquidity = Liquidity.UNKNOWN
    client_order_id: Optional[str] = None
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    pnl: Optional[Decimal] = None          # реализованный PnL по этой сделке, если считается на лету
    execution_profile: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return as_dict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ExecReport":
        return ExecReport(
            ts=int(d["ts"]),
            run_id=str(d["run_id"]),
            symbol=str(d["symbol"]),
            side=Side(d["side"]),
            order_type=OrderType(d["order_type"]),
            price=to_decimal(d["price"]),
            quantity=to_decimal(d["quantity"]),
            fee=to_decimal(d["fee"]),
            fee_asset=str(d["fee_asset"]) if d.get("fee_asset") is not None else None,
            exec_status=ExecStatus(d.get("exec_status", "FILLED")),
            liquidity=Liquidity(d.get("liquidity", "UNKNOWN")),
            client_order_id=str(d["client_order_id"]) if d.get("client_order_id") is not None else None,
            order_id=str(d["order_id"]) if d.get("order_id") is not None else None,
            trade_id=str(d["trade_id"]) if d.get("trade_id") is not None else None,
            pnl=to_decimal_opt(d.get("pnl")),
            execution_profile=str(d.get("execution_profile")) if d.get("execution_profile") is not None else None,
            meta=dict(d.get("meta", {})),
        )


# =========================
# Позиции, риски, отчёты
# =========================

@dataclass(frozen=True)
class Position:
    """
    Аггрегированная позиция по символу.
    avg_entry_price == 0 при qty == 0.
    """
    symbol: str
    qty: Decimal
    avg_entry_price: Decimal
    realized_pnl: Decimal = Decimal("0")
    fee_paid: Decimal = Decimal("0")
    ts: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def with_update(self, **kwargs) -> "Position":
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return as_dict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Position":
        return Position(
            symbol=str(d["symbol"]),
            qty=to_decimal(d["qty"]),
            avg_entry_price=to_decimal(d["avg_entry_price"]),
            realized_pnl=to_decimal(d.get("realized_pnl", "0")),
            fee_paid=to_decimal(d.get("fee_paid", "0")),
            ts=int(d["ts"]) if d.get("ts") is not None else None,
            meta=dict(d.get("meta", {})),
        )


@dataclass(frozen=True)
class PortfolioLimits:
    """
    Ограничения портфеля для RiskGuard/симулятора.
    """
    max_notional: Optional[Decimal] = None          # абсолютный лимит по ноционалу портфеля
    max_position_qty: Optional[Decimal] = None      # лимит по количеству на символ
    max_orders_per_min: Optional[int] = None
    max_drawdown_bps: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return as_dict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "PortfolioLimits":
        return PortfolioLimits(
            max_notional=to_decimal_opt(d.get("max_notional")),
            max_position_qty=to_decimal_opt(d.get("max_position_qty")),
            max_orders_per_min=int(d["max_orders_per_min"]) if d.get("max_orders_per_min") is not None else None,
            max_drawdown_bps=int(d["max_drawdown_bps"]) if d.get("max_drawdown_bps") is not None else None,
            meta=dict(d.get("meta", {})),
        )


@dataclass(frozen=True)
class TradeLogRow:
    """
    Строка лога сделок. Совпадает по схеме с ExecReport + обязательные поля для удобства анализа.
    """
    ts: int
    run_id: str
    symbol: str
    side: Side
    order_type: OrderType
    price: Decimal
    quantity: Decimal
    fee: Decimal
    fee_asset: Optional[str]
    exec_status: ExecStatus
    liquidity: Liquidity
    client_order_id: Optional[str] = None
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    pnl: Optional[Decimal] = None
    mark_price: Optional[Decimal] = None       # mark price в момент трейда
    equity: Optional[Decimal] = None           # equity после трейда
    notional: Optional[Decimal] = None         # absolute price*quantity
    drawdown: Optional[Decimal] = None         # drawdown после трейда
    slippage_bps: Optional[Decimal] = None     # slippage в bps
    spread_bps: Optional[Decimal] = None       # спред в bps
    latency_ms: Optional[int] = None           # латентность запроса
    tif: Optional[str] = None                  # Time in force
    ttl_steps: Optional[int] = None            # TTL в шагах
    execution_profile: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_exec(er: ExecReport) -> "TradeLogRow":
        return TradeLogRow(
            ts=er.ts,
            run_id=er.run_id,
            symbol=er.symbol,
            side=er.side,
            order_type=er.order_type,
            price=er.price,
            quantity=er.quantity,
            fee=er.fee,
            fee_asset=er.fee_asset,
            exec_status=er.exec_status,
            liquidity=er.liquidity,
            client_order_id=er.client_order_id,
            order_id=er.order_id,
            trade_id=er.trade_id,
            pnl=er.pnl,
            notional=er.price * er.quantity,
            mark_price=to_decimal_opt(er.meta.get("mark_price")),
            equity=to_decimal_opt(er.meta.get("equity")),
            drawdown=to_decimal_opt(er.meta.get("drawdown")),
            slippage_bps=to_decimal_opt(er.meta.get("slippage_bps")),
            spread_bps=to_decimal_opt(er.meta.get("spread_bps")),
            latency_ms=int(er.meta.get("latency_ms", 0)) if er.meta.get("latency_ms") is not None else None,
            tif=str(er.meta.get("tif")) if er.meta.get("tif") is not None else None,
            ttl_steps=int(er.meta.get("ttl_steps", 0)) if er.meta.get("ttl_steps") is not None else None,
            execution_profile=er.execution_profile,
            meta=dict(er.meta),
        )

    def to_dict(self) -> Dict[str, Any]:
        return as_dict(self)


@dataclass(frozen=True)
class EquityPoint:
    """
    Точка эквити для отчёта: кумулятивная стоимость портфеля в quote.
    """
    ts: int
    run_id: str
    symbol: str
    fee_total: Decimal
    position_qty: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    equity: Decimal
    mark_price: Decimal
    notional: Optional[Decimal] = None               # текущий ноционал позиции (qty*mark_price)
    drawdown: Optional[Decimal] = None
    risk_paused_until_ms: int = 0
    risk_events_count: int = 0
    funding_events_count: int = 0
    funding_cashflow: Optional[Decimal] = None
    cash: Optional[Decimal] = None
    execution_profile: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return as_dict(self)


# =========================
# Утилиты сериализации
# =========================

JSONDict = Dict[str, Any]

def to_decimal(x: Union[str, float, int, Decimal]) -> Decimal:
    if isinstance(x, Decimal):
        return x
    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"Невалидное десятичное значение: {x!r}")

def to_decimal_opt(x: Any) -> Optional[Decimal]:
    return None if x is None else to_decimal(x)

def as_dict(obj: Any) -> JSONDict:
    """
    Рекурсивная сериализация:
    - Decimal -> str
    - Enum -> value
    - dataclass -> dict
    - list/dict -> обход по элементам
    """
    def _convert(v: Any) -> Any:
        if isinstance(v, Decimal):
            return str(v)
        if isinstance(v, Enum):
            return v.value
        if hasattr(v, "__dataclass_fields__"):
            return as_dict(v)
        if isinstance(v, list):
            return [_convert(x) for x in v]
        if isinstance(v, dict):
            return {k: _convert(val) for k, val in v.items()}
        return v

    d = asdict(obj)
    return {k: _convert(v) for k, v in d.items()}

def to_json(obj: Any) -> str:
    """
    JSON строка с Decimal как str, Enum как value.
    """
    return json.dumps(as_dict(obj), ensure_ascii=False, separators=(",", ":"))

def from_dict(cls, data: Mapping[str, Any]):
    """
    Универсальная точка входа: cls.from_dict(data) если есть, иначе cls(**data).
    """
    if hasattr(cls, "from_dict") and callable(getattr(cls, "from_dict")):
        return getattr(cls, "from_dict")(data)
    return cls(**data)  # type: ignore[misc]


