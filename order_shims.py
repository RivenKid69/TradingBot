# -*- coding: utf-8 -*-
"""
order_shims.py
Адаптеры совместимости для унификации входов стратегии/симуляции в core_models.Order.

Сценарии:
- Преобразование ActionProto (или словаря в его формате) → Order.
- Преобразование legacy-решений вида {"kind": "MARKET", "side": "BUY", "volume_frac": 0.25, ...} → Order.

Политика объёма:
- volume_frac ∈ [-1.0, 1.0] — доля от max_position_abs по базовому активу.
- Требуются контекстные параметры: ts_ms, symbol, ref_price, max_position_abs_base, round_qty_fn (опционально).
- Если ref_price отсутствует, для MARKET создаётся Order с quantity по знаку и величине, price=None.
- Для LIMIT допускается price_offset_ticks вместе с tick_size, либо abs_price.

Замечание:
- Квантизация до биржевых фильтров производится позже (на стороне исполнителя).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Mapping, Optional, Callable, Sequence, List

from core_models import Order, OrderIntent, Side, OrderType, TimeInForce
from core_strategy import Decision
try:
    from action_proto import ActionProto, ActionType
except Exception:
    ActionProto = None  # type: ignore
    class ActionType:  # minimal fallback
        HOLD = 0; MARKET = 1; LIMIT = 2; CANCEL_ALL = 3

DecimalLike = Decimal | float | int | str

@dataclass(frozen=True)
class OrderContext:
    ts_ms: int
    symbol: str
    ref_price: Optional[DecimalLike]
    max_position_abs_base: DecimalLike
    tick_size: Optional[DecimalLike] = None
    price_offset_ticks: int = 0
    tif: str = "GTC"
    client_tag: Optional[str] = None
    round_qty_fn: Optional[Callable[[Decimal], Decimal]] = None

def _to_dec(x: DecimalLike) -> Decimal:
    return Decimal(str(x))

def _side_from_volume_frac(volume_frac: float) -> Side:
    return Side.BUY if float(volume_frac) > 0 else Side.SELL

def _qty_from_volume_frac(volume_frac: float, max_abs_base: DecimalLike) -> Decimal:
    v = abs(float(volume_frac))
    m = float(max_abs_base)
    q = Decimal(str(v * m))
    return q

def _price_from_offset(ref_price: Optional[DecimalLike], offset_ticks: int, tick_size: Optional[DecimalLike]) -> Optional[Decimal]:
    if ref_price is None:
        return None
    if offset_ticks == 0:
        return _to_dec(ref_price)
    ts = Decimal(str(tick_size)) if tick_size is not None else Decimal("0")
    return _to_dec(ref_price) + (ts * Decimal(int(offset_ticks)))

def actionproto_to_order(action: Any, ctx: OrderContext) -> Optional[Order]:
    """
    Преобразует ActionProto или dict с полями ActionProto в core_models.Order.
    Возвращает None для HOLD/CANCEL_ALL.
    """
    # извлечь поля из ActionProto или dict
    if ActionProto is not None and isinstance(action, ActionProto):
        ap = {
            "type": int(action.action_type),
            "volume_frac": float(action.volume_frac),
            "price_offset_ticks": int(getattr(action, "price_offset_ticks", 0)),
            "tif": str(getattr(action, "tif", "GTC")),
            "client_tag": getattr(action, "client_tag", None),
        }
    elif isinstance(action, Mapping):
        ap = {
            "type": int(action.get("type", action.get("action_type", 0))),
            "volume_frac": float(action.get("volume_frac", 0.0)),
            "price_offset_ticks": int(action.get("price_offset_ticks", 0)),
            "tif": str(action.get("tif", "GTC")),
            "client_tag": action.get("client_tag"),
        }
    else:
        return None

    t = int(ap["type"])
    if t in (ActionType.HOLD, ActionType.CANCEL_ALL):
        return None

    side = _side_from_volume_frac(ap["volume_frac"])
    qty = _qty_from_volume_frac(ap["volume_frac"], ctx.max_position_abs_base)
    if ctx.round_qty_fn is not None:
        qty = ctx.round_qty_fn(qty)

    tif = TimeInForce(str(ap["tif"]))
    if t == ActionType.MARKET:
        return Order(
            ts=ctx.ts_ms,
            symbol=ctx.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=qty,
            price=None,
            time_in_force=tif,
            client_order_id=(ctx.client_tag or ""),
            meta={"source": "ActionProto"}
        )
    # LIMIT
    price = _price_from_offset(ctx.ref_price, int(ap["price_offset_ticks"]), ctx.tick_size)
    return Order(
        ts=ctx.ts_ms,
        symbol=ctx.symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=qty,
        price=price,
        time_in_force=tif,
        client_order_id=(ctx.client_tag or ""),
        meta={"source": "ActionProto"}
    )

def legacy_decision_to_order(decision: Mapping[str, Any], ctx: OrderContext) -> Optional[Order]:
    """
    Поддержка старого формата решений сим-адаптера:
    {"kind": "MARKET"|"LIMIT"|"HOLD", "side": "BUY"|"SELL", "volume_frac": float, "price_offset_ticks": int?}
    """
    kind = str(decision.get("kind", "HOLD")).upper()
    if kind in ("HOLD", "CANCEL_ALL"):
        return None
    side_str = str(decision.get("side", "BUY")).upper()
    side = Side.BUY if side_str == "BUY" else Side.SELL
    volume_frac = float(decision.get("volume_frac", 0.0))
    qty = _qty_from_volume_frac(volume_frac, ctx.max_position_abs_base)
    if ctx.round_qty_fn is not None:
        qty = ctx.round_qty_fn(qty)
    tif = TimeInForce(str(decision.get("tif", "GTC")))
    if kind == "MARKET":
        return Order(
            ts=ctx.ts_ms, symbol=ctx.symbol, side=side,
            order_type=OrderType.MARKET, quantity=qty, price=None,
            time_in_force=tif, client_order_id=(ctx.client_tag or ""),
            meta={"source": "legacy_decision"}
        )
    # LIMIT
    price = _price_from_offset(ctx.ref_price, int(decision.get("price_offset_ticks", 0)), ctx.tick_size)
    return Order(
        ts=ctx.ts_ms, symbol=ctx.symbol, side=side,
        order_type=OrderType.LIMIT, quantity=qty, price=price,
        time_in_force=tif, client_order_id=(ctx.client_tag or ""),
        meta={"source": "legacy_decision"}
    )


# --- Новые конвертеры: ActionProto/legacy Decision -> OrderIntent и далее -> Order ---

def actionproto_to_order_intent(action: Any, ctx: OrderContext) -> Optional[OrderIntent]:
    """
    Преобразует ActionProto или dict с полями ActionProto в core_models.OrderIntent.
    Возвращает None для HOLD/CANCEL_ALL.
    """
    if ActionProto is not None and isinstance(action, ActionProto):
        ap = {
            "type": int(action.action_type),
            "volume_frac": float(action.volume_frac),
            "price_offset_ticks": int(getattr(action, "price_offset_ticks", 0)),
            "tif": str(getattr(action, "tif", "GTC")),
            "client_tag": getattr(action, "client_tag", None),
        }
    elif isinstance(action, Mapping):
        ap = {
            "type": int(action.get("type", 0)),
            "volume_frac": float(action.get("volume_frac", 0.0)),
            "price_offset_ticks": int(action.get("price_offset_ticks", 0)),
            "tif": str(action.get("tif", "GTC")),
            "client_tag": action.get("client_tag", None),
        }
    else:
        return None

    t = int(ap["type"])
    if t in (int(ActionType.HOLD), int(ActionType.CANCEL_ALL)):
        return None

    side = _side_from_volume_frac(ap["volume_frac"])
    return OrderIntent(
        ts=ctx.ts_ms,
        symbol=ctx.symbol,
        side=side,
        order_type=(OrderType.MARKET if t == int(ActionType.MARKET) else OrderType.LIMIT),
        volume_frac=_to_dec(ap["volume_frac"]),
        price_offset_ticks=int(ap.get("price_offset_ticks", 0)),
        time_in_force=TimeInForce(str(ap.get("tif", "GTC"))),
        client_tag=(ctx.client_tag or ""),
        meta={"source": "ActionProto"},
    )


def legacy_decision_to_order_intent(decision: Mapping[str, Any], ctx: OrderContext) -> Optional[OrderIntent]:
    """
    Поддержка старого формата решений сим-адаптера → OrderIntent.
    {"kind": "MARKET"|"LIMIT"|"HOLD", "side": "BUY"|"SELL", "volume_frac": float, "price_offset_ticks": int?}
    """
    kind = str(decision.get("kind", "HOLD")).upper()
    if kind in ("HOLD", "CANCEL_ALL"):
        return None
    side_str = str(decision.get("side", "BUY")).upper()
    side = Side.BUY if side_str == "BUY" else Side.SELL
    volume_frac = float(decision.get("volume_frac", 0.0))
    return OrderIntent(
        ts=ctx.ts_ms,
        symbol=ctx.symbol,
        side=side,
        order_type=(OrderType.MARKET if kind == "MARKET" else OrderType.LIMIT),
        volume_frac=_to_dec(volume_frac),
        price_offset_ticks=int(decision.get("price_offset_ticks", 0)),
        time_in_force=TimeInForce(str(decision.get("tif", "GTC"))),
        client_tag=(ctx.client_tag or ""),
        meta={"source": "legacy_decision"},
    )


def intent_to_order(intent: OrderIntent, ctx: OrderContext) -> Order:
    """
    Преобразование стандартизированного OrderIntent → конкретный Order с количеством.
    Использует ctx.max_position_abs_base и ctx.round_qty_fn для квантизации.
    """
    qty = _qty_from_volume_frac(float(intent.volume_frac), ctx.max_position_abs_base)
    if ctx.round_qty_fn is not None:
        qty = ctx.round_qty_fn(qty)

    if intent.order_type == OrderType.MARKET:
        price = None
    else:
        price = _price_from_offset(ctx.ref_price, int(intent.price_offset_ticks), ctx.tick_size)

    return Order(
        ts=ctx.ts_ms,
        symbol=intent.symbol,
        side=intent.side,
        order_type=intent.order_type,
        quantity=qty,
        price=price,
        time_in_force=intent.time_in_force,
        client_order_id=intent.client_tag or (ctx.client_tag or ""),
        meta={**(intent.meta or {}), "source": "OrderIntent"},
    )


def orders_to_order_intents(orders: Sequence[Order], ctx: OrderContext) -> List[OrderIntent]:
    """Преобразование Order -> OrderIntent.

    Использует ``ctx.max_position_abs_base`` для вычисления ``volume_frac`` и
    ``ctx.ref_price``/``ctx.tick_size`` для восстановления ``price_offset_ticks``.
    """
    out: List[OrderIntent] = []
    max_abs = _to_dec(ctx.max_position_abs_base)
    ref = _to_dec(ctx.ref_price) if ctx.ref_price is not None else None
    tick = _to_dec(ctx.tick_size) if ctx.tick_size is not None else None

    for o in orders:
        try:
            vf = (o.quantity / max_abs) if max_abs != 0 else Decimal("0")
            if o.side == Side.SELL:
                vf = -abs(vf)
            else:
                vf = abs(vf)

            if o.meta and "price_offset_ticks" in o.meta:
                po = int(o.meta.get("price_offset_ticks", 0))
            elif ref is not None and tick is not None and o.price is not None and tick != 0:
                po = int((o.price - ref) / tick)
            else:
                po = 0

            out.append(
                OrderIntent(
                    ts=o.ts,
                    symbol=o.symbol,
                    side=o.side,
                    order_type=o.order_type,
                    volume_frac=vf,
                    price_offset_ticks=po,
                    time_in_force=o.time_in_force,
                    client_tag=o.client_order_id,
                    meta=o.meta,
                )
            )
        except Exception:
            continue
    return out
