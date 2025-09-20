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
import math
from typing import Any, Dict, List, Optional, Tuple, Mapping

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


def _float_or_none(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _extract_capacity_meta(source: Mapping[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for field in ("cap_base_per_bar", "used_base_before", "used_base_after", "fill_ratio"):
        if field in source:
            f_val = _float_or_none(source.get(field))
            if f_val is not None:
                meta[field] = f_val
    reason = source.get("capacity_reason")
    if reason not in (None, ""):
        meta["capacity_reason"] = str(reason)
    exec_status_val = source.get("exec_status")
    if exec_status_val not in (None, ""):
        meta["exec_status"] = str(exec_status_val)
    return meta


def _attach_capacity_meta(meta: Dict[str, Any], source: Mapping[str, Any]) -> None:
    capacity_meta = _extract_capacity_meta(source)
    if not capacity_meta:
        return
    existing = meta.get("bar_capacity_base")
    if isinstance(existing, Mapping):
        merged = dict(existing)
        merged.update(capacity_meta)
        meta["bar_capacity_base"] = merged
    else:
        meta["bar_capacity_base"] = capacity_meta


def _normalize_filter_reason(reason: Any) -> Optional[Any]:
    if reason is None:
        return None
    if isinstance(reason, Mapping):
        normalized: Dict[str, Any] = {}
        primary = reason.get("primary")
        which = reason.get("which") if primary is None else None
        if primary is not None:
            normalized["primary"] = str(primary)
        elif which is not None:
            normalized["primary"] = str(which)
        details = None
        if "details" in reason:
            details = _normalize_filter_reason(reason.get("details"))
            normalized["details"] = details
        elif "detail" in reason:
            details = _normalize_filter_reason(reason.get("detail"))
            normalized["details"] = details
        if "extra" in reason:
            normalized["extra"] = _normalize_filter_reason(reason.get("extra"))
        if "rejections" in reason:
            normalized["rejections"] = _normalize_filter_reason(reason.get("rejections"))
        if "counts" in reason:
            normalized["counts"] = _normalize_filter_reason(reason.get("counts"))
        for key, value in reason.items():
            key_str = str(key)
            if key_str in normalized:
                continue
            normalized[key_str] = _normalize_filter_reason(value)
        return normalized
    if isinstance(reason, (list, tuple, set)):
        return [_normalize_filter_reason(item) for item in reason]
    return reason


def _extract_reason_price(reason: Any) -> Optional[float]:
    if isinstance(reason, Mapping):
        for key, value in reason.items():
            key_lower = str(key).lower()
            if key_lower in {"price", "ref_price", "limit_price", "match_price"}:
                hint = _float_or_none(value)
                if hint is not None:
                    return hint
            nested = _extract_reason_price(value)
            if nested is not None:
                return nested
    elif isinstance(reason, list):
        for item in reason:
            nested = _extract_reason_price(item)
            if nested is not None:
                return nested
    return None


def _attach_filter_reason_meta(report: CoreExecReport, reason: Any) -> None:
    if reason is None:
        return
    meta = dict(report.meta or {})
    meta["filter_rejection"] = _normalize_filter_reason(reason)
    report.meta = meta


def _make_reject_report(
    parent: Dict[str, Any],
    *,
    symbol: str,
    run_id: str,
    client_order_id: Optional[str],
    reason: Any,
) -> CoreExecReport:
    normalized_reason = _normalize_filter_reason(reason)
    price_hint = _extract_reason_price(normalized_reason)
    trade_stub: Dict[str, Any] = {
        "price": price_hint if price_hint is not None else "0",
        "qty": "0",
        "status": "REJECTED",
        "exec_status": "REJECTED",
    }
    if client_order_id:
        trade_stub["client_order_id"] = client_order_id
    if normalized_reason is not None:
        trade_stub["reason"] = _normalize_filter_reason(normalized_reason)
    report = trade_dict_to_core_exec_report(
        trade_stub,
        parent=parent,
        symbol=symbol,
        run_id=run_id,
        client_order_id=client_order_id,
    )
    report.exec_status = ExecStatus.REJECTED
    report.quantity = Decimal("0")
    _attach_filter_reason_meta(report, normalized_reason)
    return report


def _derive_exec_status(trade: Dict[str, Any]) -> Tuple[ExecStatus, Optional[str]]:
    raw = _get(trade, "exec_status", default=None)
    if raw is None:
        raw = _get(trade, "status", default=None)
    if raw is None:
        return ExecStatus.FILLED, None
    raw_str = str(raw)
    up = raw_str.upper()
    if "PART" in up:
        return ExecStatus.PARTIALLY_FILLED, raw_str
    if "REJECT" in up:
        return ExecStatus.REJECTED, raw_str
    if "CANCEL" in up:
        return ExecStatus.CANCELED, raw_str
    if up in ("NEW", "NONE"):
        return ExecStatus.NEW, raw_str
    return ExecStatus.FILLED, raw_str

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
    exec_status_enum, exec_status_raw = _derive_exec_status(trade)
    meta: Dict[str, Any] = {"raw": trade}
    _attach_capacity_meta(meta, trade)
    if exec_status_raw:
        base_meta = meta.setdefault("bar_capacity_base", {})  # type: ignore[assignment]
        if isinstance(base_meta, dict) and "exec_status" not in base_meta:
            base_meta["exec_status"] = exec_status_raw
    return CoreExecReport(
        ts=ts_ms,
        run_id=run_id,
        symbol=symbol,
        execution_profile=str(parent.get("execution_profile")) if parent.get("execution_profile") is not None else None,
        side=side,
        order_type=order_type,
        price=price,
        quantity=qty,
        fee=fee,
        fee_asset=(None if fee_asset is None else str(fee_asset)),
        exec_status=exec_status_enum,
        liquidity=liquidity,
        client_order_id=(None if client_order_id is None else str(client_order_id)),
        order_id=order_id,
        trade_id=trade_id,
        pnl=None,
        meta=meta,
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
        meta = dict(t.meta)
        raw_payload = meta.get("raw")
        if isinstance(raw_payload, Mapping):
            _attach_capacity_meta(meta, raw_payload)
        out.append(CoreExecReport(
            ts=t.ts,
            run_id=t.run_id,
            symbol=t.symbol,
            execution_profile=t.execution_profile,
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
            meta=meta,
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
    status_val = str(d.get("status") or "").upper()
    if status_val == "REJECTED_BY_FILTER":
        reason_payload = d.get("reason")
        if not reports or all(t.quantity == Decimal("0") for t in reports):
            return [
                _make_reject_report(
                    d,
                    symbol=symbol,
                    run_id=run_id,
                    client_order_id=client_order_id,
                    reason=reason_payload,
                )
            ]
        if reason_payload is not None:
            for report in reports:
                _attach_filter_reason_meta(report, reason_payload)
    # Если у сделок нет индивидуальной комиссии, а в отчёте есть fee_total — распределим
    if reports and all((t.fee is None or t.fee == Decimal("0")) for t in reports):
        total_fee = _dec(d.get("fee_total", "0"))
        reports = _distribute_fee(total_fee, reports)
    return reports
