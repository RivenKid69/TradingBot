# -*- coding: utf-8 -*-
"""
compat_shims.py
Мост совместимости между различными формами отчётов симулятора/адаптеров и единой моделью core_models.ExecReport.

Назначение:
- Принять отчёт из ExecutionSimulator.to_dict() или схожих источников.
- Превратить каждый trade-дикт в CoreExecReport по унифицированной схеме.
- Быть толерантным к разным ключам: 'price'|'avg_price'|'p', 'qty'|'quantity'|'filled_qty'|'q', 'side'|'is_buy' и т.п.
- Если комиссия известна только суммарно (fee_total), распределить её пропорционально нотио (price*qty).

ВНИМАНИЕ: этот модуль предполагает, что все файлы проекта лежат в одной папке.
Импорты идут напрямую по именам модулей.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from core_models import (
    ExecReport as CoreExecReport,
    ExecStatus,
    Liquidity,
    Side,
    OrderType,
    as_dict,
)

def _dec(x: Any, *, default: str = "0") -> Decimal:
    if isinstance(x, Decimal):
        return x
    try:
        return Decimal(str(x))
    except Exception:
        try:
            return Decimal(default)
        except InvalidOperation:
            return Decimal("0")

def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _as_side(trade: Dict[str, Any]) -> Side:
    v = _get(trade, "side", "SIDE", "s", "buy_sell", default=None)
    if v is None:
        is_buy = bool(_get(trade, "is_buy", "buyer", default=False))
        return Side.BUY if is_buy else Side.SELL
    if isinstance(v, str):
        v = v.upper()
        if v in ("B", "BUY", "LONG", "OPEN_LONG"):
            return Side.BUY
        if v in ("S", "SELL", "SHORT", "OPEN_SHORT"):
            return Side.SELL
    try:
        iv = int(v)
        return Side.BUY if iv > 0 else Side.SELL
    except Exception:
        return Side.BUY

def _as_liquidity(trade: Dict[str, Any]) -> Liquidity:
    v = _get(trade, "liquidity", "L", default=None)
    if isinstance(v, str):
        u = v.upper()
        if "MAKER" in u:
            return Liquidity.MAKER
        if "TAKER" in u:
            return Liquidity.TAKER
    is_maker = _get(trade, "is_maker", "maker", default=None)
    if isinstance(is_maker, bool):
        return Liquidity.MAKER if is_maker else Liquidity.TAKER
    return Liquidity.UNKNOWN

def _as_ordertype(trade: Dict[str, Any], *, parent: Dict[str, Any]) -> OrderType:
    v = _get(trade, "order_type", "type", default=None)
    if isinstance(v, str):
        u = v.upper()
        if "LIMIT" in u:
            return OrderType.LIMIT
        if "MARKET" in u:
            return OrderType.MARKET
    # эвристика: наличие abs_price/limit_price/price_offset_ticks => LIMIT, иначе MARKET
    if _get(trade, "abs_price", "limit_price", default=None) is not None:
        return OrderType.LIMIT
    if _get(parent, "execution", default=None) == "LIMIT":
        return OrderType.LIMIT
    return OrderType.MARKET

def _price_and_qty(trade: Dict[str, Any]) -> Tuple[Decimal, Decimal]:
    price = _dec(_get(trade, "price", "avg_price", "p", "match_price", "fill_price", "limit_price", default="0"))
    qty = _dec(_get(trade, "qty", "quantity", "filled_qty", "q", default="0"))
    # величина qty — абсолютная; знак в core задаёт side
    qty = qty.copy_abs()
    return price, qty

def _ts(trade: Dict[str, Any], parent: Dict[str, Any]) -> int:
    v = _get(trade, "ts", "timestamp", "T", default=None)
    if v is not None:
        try:
            return int(v)
        except Exception:
            pass
    v = _get(parent, "ts", "timestamp", default=0)
    try:
        return int(v)
    except Exception:
        return 0

def _order_and_trade_ids(trade: Dict[str, Any], parent: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    oid = _get(trade, "order_id", "oid", default=None)
    if oid is None:
        # ExecutionSimulator.to_dict() может отдавать список new_order_ids
        noids = parent.get("new_order_ids") or parent.get("order_ids")
        if isinstance(noids, list) and noids:
            oid = str(noids[0])
    tid = _get(trade, "trade_id", "tid", "id", default=None)
    return (str(oid) if oid is not None else None, str(tid) if tid is not None else None)

def trade_dict_to_core_exec_report(
    trade: Dict[str, Any],
    *,
    parent: Dict[str, Any],
    symbol: str,
    run_id: str,
    client_order_id: Optional[str] = None,
) -> CoreExecReport:
    price, qty = _price_and_qty(trade)
    side = _as_side(trade)
    order_type = _as_ordertype(trade, parent=parent)
    fee = _dec(_get(trade, "fee", "commission", default="0"))
    fee_asset = _get(trade, "fee_asset", "commissionAsset", default=None)
    liquidity = _as_liquidity(trade)
    ts_ms = _ts(trade, parent)
    order_id, trade_id = _order_and_trade_ids(trade, parent)
    return CoreExecReport(
        ts=ts_ms,
        run_id=run_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        price=price,
        quantity=qty,
        fee=fee,
        fee_asset=(None if fee_asset is None else str(fee_asset)),
        exec_status=ExecStatus.FILLED,
        liquidity=liquidity,
        client_order_id=(None if client_order_id is None else str(client_order_id)),
        order_id=order_id,
        trade_id=trade_id,
        pnl=None,
        meta={"raw": trade},
    )

def _distribute_fee(total_fee: Decimal, trades: List[CoreExecReport]) -> List[CoreExecReport]:
    if total_fee is None:
        return trades
    try:
        tf = Decimal(str(total_fee))
    except Exception:
        return trades
    if tf == 0 or not trades:
        return trades
    notionals = [t.price * t.quantity for t in trades]
    s = sum(notionals) or Decimal("1")
    out: List[CoreExecReport] = []
    for t, w in zip(trades, notionals):
        share = (w / s) * tf
        out.append(CoreExecReport(
            ts=t.ts,
            run_id=t.run_id,
            symbol=t.symbol,
            side=t.side,
            order_type=t.order_type,
            price=t.price,
            quantity=t.quantity,
            fee=share,
            fee_asset=t.fee_asset,
            exec_status=t.exec_status,
            liquidity=t.liquidity,
            client_order_id=t.client_order_id,
            order_id=t.order_id,
            trade_id=t.trade_id,
            pnl=t.pnl,
            meta=t.meta,
        ))
    return out

def sim_report_dict_to_core_exec_reports(
    d: Dict[str, Any],
    *,
    symbol: str,
    run_id: str = "sim",
    client_order_id: Optional[str] = None,
) -> List[CoreExecReport]:
    """
    Преобразует dict от ExecutionSimulator.to_dict() в список CoreExecReport — по одной записи на сделку.
    Если комиссия не указана помарочно, но есть 'fee_total', распределяет её пропорционально нотио.
    """
    trades_src = d.get("trades") or d.get("fills") or []
    if not isinstance(trades_src, list):
        trades_src = []
    reports: List[CoreExecReport] = [
        trade_dict_to_core_exec_report(t, parent=d, symbol=symbol, run_id=run_id, client_order_id=client_order_id)
        for t in trades_src
    ]
    # Если у сделок нет индивидуальной комиссии, а в отчёте есть fee_total — распределим
    if reports and all((t.fee is None or t.fee == Decimal("0")) for t in reports):
        total_fee = _dec(d.get("fee_total", "0"))
        reports = _distribute_fee(total_fee, reports)
    return reports
