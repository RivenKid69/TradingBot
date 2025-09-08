# -*- coding: utf-8 -*-
"""
impl_sim_executor.py
Реализация исполнителя торгов (TradeExecutor) поверх ExecutionSimulator из execution_sim.py.

Контракт:
- Принимает core_models.Order.
- Возвращает core_models.ExecReport(ы) через compat_shims.sim_report_dict_to_core_exec_reports().
- cancel() — no-op (симулятор ордеров «в книге» моделирует через new_order_ids, а не реальную книгу заявок).
- get_open_positions() — строит Position из состояния симулятора (position_qty, _avg_entry_price, realized_pnl_cum, fees_cum).

Важно:
- Все файлы лежат в одной папке; импорты — по именам модулей.
"""

from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional, Sequence, Mapping, List, Any

from core_models import Order, ExecReport, Position, as_dict
from core_contracts import TradeExecutor
from compat_shims import sim_report_dict_to_core_exec_reports
from execution_sim import ExecutionSimulator, SimStepReport  # type: ignore
from action_proto import ActionProto, ActionType

# новые компонентные имплементации
from impl_quantizer import QuantizerImpl, QuantizerConfig
from impl_fees import FeesImpl, FeesConfig
from impl_slippage import SlippageImpl, SlippageCfg
from impl_latency import LatencyImpl, LatencyCfg
from impl_risk_basic import RiskBasicImpl, RiskBasicCfg


@dataclass
class _SimCtx:
    symbol: str
    # базовая единица позиции для volume_frac. Если 0, будем интерпретировать quantity как долю 1.0
    max_position_abs_base: float = 1.0


class SimExecutor(TradeExecutor):
    """
    Обёртка над ExecutionSimulator с интерфейсом TradeExecutor.
    """
    def __init__(
        self,
        sim: ExecutionSimulator,
        *,
        symbol: str,
        max_position_abs_base: float = 1.0,
        quantizer: QuantizerImpl | None = None,
        risk: RiskBasicImpl | None = None,
        latency: LatencyImpl | None = None,
        slippage: SlippageImpl | None = None,
        fees: FeesImpl | None = None,
        run_config: Any = None,
    ) -> None:
        self._sim = sim
        self._run_id = str(getattr(run_config, "run_id", "sim") or "sim")
        self._ctx = _SimCtx(symbol=str(symbol), max_position_abs_base=float(max_position_abs_base))

        # последовательное подключение компонентов к симулятору
        if quantizer is not None:
            quantizer.attach_to(self._sim)
        if risk is not None:
            risk.attach_to(self._sim)
        if latency is not None:
            latency.attach_to(self._sim)
        if slippage is not None:
            slippage.attach_to(self._sim)
        if fees is not None:
            fees.attach_to(self._sim)

    @staticmethod
    def from_config(
        *,
        symbol: str,
        max_position_abs_base: float = 1.0,
        sim: ExecutionSimulator,
        filters_cfg: dict | None = None,
        risk_cfg: dict | None = None,
        latency_cfg: dict | None = None,
        slippage_cfg: dict | None = None,
        fees_cfg: dict | None = None,
        run_config: Any = None,
    ) -> "SimExecutor":
        q_impl = QuantizerImpl.from_dict(filters_cfg or {}) if filters_cfg is not None else None
        r_impl = RiskBasicImpl.from_dict(risk_cfg or {}) if risk_cfg is not None else None
        l_impl = LatencyImpl.from_dict(latency_cfg or {}) if latency_cfg is not None else None
        s_impl = SlippageImpl.from_dict(slippage_cfg or {}) if slippage_cfg is not None else None
        f_impl = FeesImpl.from_dict(fees_cfg or {}) if fees_cfg is not None else None
        return SimExecutor(
            sim,
            symbol=symbol,
            max_position_abs_base=max_position_abs_base,
            quantizer=q_impl,
            risk=r_impl,
            latency=l_impl,
            slippage=s_impl,
            fees=f_impl,
            run_config=run_config,
        )

    # ---- вспомогательное: Order -> (ActionType, ActionProto) ----
    def _order_to_action(self, order: Order) -> tuple[int, object]:
        """
        Преобразует core_models.Order к (ActionType, ActionProto) симулятора.
        Интерпретация:
        - MARKET: volume_frac = sign(quantity) * min(1.0, abs(quantity) / max_position_abs_base)
        - LIMIT:  volume_frac аналогично; если price задан — кладём в proto.abs_price
        """
        qty = float(order.quantity)
        vol_frac = 0.0
        base = float(self._ctx.max_position_abs_base) if self._ctx.max_position_abs_base > 0 else 1.0
        if base <= 0:
            base = 1.0
        vol_frac = max(0.0, abs(qty) / base)
        if str(order.side).upper().endswith("SELL"):
            vol_frac = -vol_frac

        tif = str(getattr(order, "time_in_force", "GTC"))

        if str(order.order_type).upper().endswith("MARKET"):
            proto = ActionProto(action_type=ActionType.MARKET, volume_frac=float(vol_frac))
            setattr(proto, "tif", tif)
            setattr(proto, "client_tag", getattr(order, "client_order_id", "") or "")
            return ActionType.MARKET, proto

        # LIMIT
        proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=float(vol_frac))
        setattr(proto, "tif", tif)
        setattr(proto, "client_tag", getattr(order, "client_order_id", "") or "")
        price = order.price
        if price is not None:
            # execution_sim понимает proto.abs_price
            setattr(proto, "abs_price", float(price))
        return ActionType.LIMIT, proto

    # ---- интерфейс TradeExecutor ----
    def execute(self, order: Order) -> ExecReport:
        """
        Синхронно исполняет ордер через ExecutionSimulator и возвращает первый ExecReport.
        Если сделок не было — возвращает ExecReport с нулевым qty и статусом 'NONE' (в рамках схемы compat).
        """
        atype, proto = self._order_to_action(order)

        # обновим символ и реф. цену в симуляторе, если нужно
        try:
            self._sim.set_symbol(self._ctx.symbol)
        except Exception:
            pass

        # прогон шага симуляции с одним действием
        rep: SimStepReport = self._sim.run_step(
            ts=int(order.ts),
            ref_price=float(order.price) if getattr(order, "price", None) is not None else getattr(self._sim, "_last_ref_price", None),
            bid=None,
            ask=None,
            vol_factor=None,
            liquidity=None,
            actions=[(atype, proto)],
        )  # type: ignore

        d = rep.to_dict()
        core_reports: List[ExecReport] = sim_report_dict_to_core_exec_reports(
            d,
            symbol=self._ctx.symbol,
            run_id=self._run_id,
            client_order_id=str(getattr(order, "client_order_id", "") or ""),
        )
        # Возвращаем первый отчёт; при необходимости вызывающая сторона может получить остальные из d
        return core_reports[0] if core_reports else ExecReport.from_dict({
            "ts": int(order.ts),
            "symbol": self._ctx.symbol,
            "side": "BUY" if float(order.quantity) >= 0 else "SELL",
            "order_type": "MARKET" if str(order.order_type).upper().endswith("MARKET") else "LIMIT",
            "price": float(order.price) if getattr(order, "price", None) is not None else 0.0,
            "quantity": 0.0,
            "fee": 0.0,
            "fee_asset": None,
            "pnl": 0.0,
            "exec_status": "NONE",
            "liquidity": "UNKNOWN",
            "client_order_id": str(getattr(order, "client_order_id", "") or ""),
            "order_id": None,
            "meta": {},
        })

    def cancel(self, client_order_id: str) -> None:
        """
        В простом симуляторе отмена моделируется через new_order_ids/new_order_pos или LOB-заглушку.
        Здесь — no-op.
        """
        return None

    def get_open_positions(self, symbols: Optional[Sequence[str]] = None) -> Mapping[str, Position]:
        """
        Строит Position из внутренних полей симулятора.
        """
        sym = self._ctx.symbol
        if symbols is not None and len(symbols) > 0 and sym not in set(map(str, symbols)):
            return {}

        qty = float(getattr(self._sim, "position_qty", 0.0))
        avg = float(getattr(self._sim, "_avg_entry_price", 0.0) or 0.0)
        realized_pnl = float(getattr(self._sim, "realized_pnl_cum", 0.0) or 0.0)
        fee_paid = float(getattr(self._sim, "fees_cum", 0.0) or 0.0)

        pos = Position(
            symbol=str(sym),
            qty=Decimal(str(qty)),
            avg_entry_price=Decimal(str(avg if qty != 0 else 0.0)),
            realized_pnl=Decimal(str(realized_pnl)),
            fee_paid=Decimal(str(fee_paid)),
            ts=None,
            meta={},
        )
        return {sym: pos}
