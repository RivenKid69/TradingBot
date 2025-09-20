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
from core_config import ExecutionProfile, ExecutionParams
from config import DataDegradationConfig

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

    @staticmethod
    def _latency_dict(cfg: Any) -> Dict[str, Any]:
        if cfg is None:
            return {}
        if hasattr(cfg, "dict"):
            try:
                payload = cfg.dict(exclude_unset=False)  # type: ignore[call-arg]
            except Exception:
                payload = {}
            else:
                if isinstance(payload, dict):
                    return dict(payload)
        if isinstance(cfg, dict):
            return dict(cfg)
        try:
            return dict(cfg)  # type: ignore[arg-type]
        except Exception:
            return {}

    @staticmethod
    def _execution_dict(cfg: Any) -> Dict[str, Any]:
        if cfg is None:
            return {}
        if hasattr(cfg, "dict"):
            try:
                payload = cfg.dict(exclude_unset=False)  # type: ignore[call-arg]
            except Exception:
                payload = {}
            else:
                if isinstance(payload, dict):
                    return dict(payload)
        if isinstance(cfg, dict):
            return dict(cfg)
        try:
            return dict(cfg)  # type: ignore[arg-type]
        except Exception:
            return {}

    @staticmethod
    def _dynamic_spread_enabled(cfg: Any) -> bool:
        if cfg is None:
            return False
        if isinstance(cfg, Mapping):
            dyn_block = cfg.get("dynamic")
            if dyn_block is None:
                dyn_block = cfg.get("dynamic_spread")
        else:
            dyn_block = getattr(cfg, "dynamic", None)
            if dyn_block is None:
                dyn_block = getattr(cfg, "dynamic_spread", None)
        if dyn_block is None:
            return False
        if isinstance(dyn_block, Mapping):
            enabled_value = dyn_block.get("enabled")
        else:
            enabled_value = getattr(dyn_block, "enabled", None)
        try:
            return bool(enabled_value)
        except Exception:
            return False

    @staticmethod
    def _fees_dict(cfg: Any) -> Dict[str, Any]:
        if cfg is None:
            return {}
        if hasattr(cfg, "dict"):
            try:
                payload = cfg.dict(exclude_unset=False)  # type: ignore[call-arg]
            except Exception:
                payload = {}
            else:
                if isinstance(payload, dict):
                    return dict(payload)
        if isinstance(cfg, Mapping):
            return dict(cfg)
        try:
            return dict(cfg)  # type: ignore[arg-type]
        except Exception:
            return {}

    @staticmethod
    def _build_fee_config(
        raw_cfg: Any,
        *,
        run_config: Any | None,
        symbol: str,
    ) -> Dict[str, Any]:
        payload = SimExecutor._fees_dict(raw_cfg)
        if symbol and "symbol" not in payload:
            payload["symbol"] = symbol

        def _get_attr(source: Any, key: str) -> Any:
            if source is None:
                return None
            if isinstance(source, Mapping):
                return source.get(key)
            return getattr(source, key, None)

        share_block = payload.get("maker_taker_share")
        if share_block is None and run_config is not None:
            direct_share = _get_attr(run_config, "maker_taker_share")
            if direct_share is not None:
                if hasattr(direct_share, "dict"):
                    try:
                        share_payload = direct_share.dict(exclude_unset=False)  # type: ignore[call-arg]
                    except Exception:
                        share_payload = None
                    else:
                        if isinstance(share_payload, dict):
                            direct_share = share_payload
                payload["maker_taker_share"] = direct_share

        override_keys = {
            "maker_taker_share_enabled": "maker_taker_share_enabled",
            "maker_taker_share_mode": "maker_taker_share_mode",
            "maker_share_default": "maker_share_default",
            "spread_cost_maker_bps": "spread_cost_maker_bps",
            "spread_cost_taker_bps": "spread_cost_taker_bps",
            "taker_fee_override_bps": "taker_fee_override_bps",
        }
        for attr_name, key in override_keys.items():
            if key in payload:
                continue
            override_value = _get_attr(run_config, attr_name)
            if override_value is None:
                continue
            payload[key] = override_value

        return payload

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
        data_degradation: DataDegradationConfig | None = None,
        run_config: Any | None = None,
    ) -> None:
        """Создать исполнителя поверх :class:`ExecutionSimulator`.

        Параметры ``quantizer``, ``risk``, ``latency``, ``slippage`` и ``fees`` могут
        быть переданы явно как соответствующие реализации. Если они отсутствуют,
        попытка построения происходит из блоков ``run_config``: ``quantizer``,
        ``fees``, ``slippage``, ``latency``, ``risk`` и ``no_trade``. При отсутствии
        этих блоков используются значения по умолчанию.
        """

        self._sim = sim
        self._run_id = str(getattr(run_config, "run_id", "sim") or "sim")
        self._ctx = _SimCtx(symbol=str(symbol), max_position_abs_base=float(max_position_abs_base))

        rc_quantizer = getattr(run_config, "quantizer", {}) if run_config else {}
        rc_risk = getattr(run_config, "risk", None) if run_config else None
        rc_latency = getattr(run_config, "latency", None) if run_config else None
        rc_slippage = getattr(run_config, "slippage", {}) if run_config else {}
        rc_fees = getattr(run_config, "fees", {}) if run_config else {}
        rc_degradation = getattr(run_config, "data_degradation", {}) if run_config else {}
        self._no_trade_cfg = getattr(run_config, "no_trade", {}) if run_config else {}
        self._exec_profile: ExecutionProfile = (
            getattr(run_config, "execution_profile", ExecutionProfile.MKT_OPEN_NEXT_H1)
            if run_config is not None
            else ExecutionProfile.MKT_OPEN_NEXT_H1
        )
        self._exec_params: ExecutionParams = (
            getattr(run_config, "execution_params", ExecutionParams())
            if run_config is not None
            else ExecutionParams()
        )
        self._execution_cfg = getattr(run_config, "execution", None) if run_config else None
        exec_cfg_payload: Dict[str, Any] = {}
        if run_config is not None:
            exec_cfg_payload = self._execution_dict(self._execution_cfg)
        if exec_cfg_payload:
            try:
                setattr(self._sim, "_execution_intrabar_cfg", dict(exec_cfg_payload))
            except Exception:
                pass
            bridge_payload = exec_cfg_payload.get("bridge")
            if isinstance(bridge_payload, dict):
                try:
                    setattr(self._sim, "_execution_bridge_cfg", dict(bridge_payload))
                except Exception:
                    pass
        if run_config is not None:
            try:
                setattr(self._sim, "_execution_runtime_cfg", self._execution_cfg)
            except Exception:
                pass
        if data_degradation is None:
            data_degradation = (
                DataDegradationConfig.from_dict(rc_degradation)
                if rc_degradation
                else DataDegradationConfig.default()
            )
        self._data_degradation = data_degradation

        if quantizer is None:
            quantizer = QuantizerImpl.from_dict(rc_quantizer)
        if risk is None:
            risk = RiskBasicImpl.from_dict(rc_risk)
        if latency is None:
            cfg_lat = self._latency_dict(rc_latency)
            if run_config is not None:
                lat_path = getattr(run_config, "latency_seasonality_path", None)
                if lat_path and not cfg_lat.get("latency_seasonality_path"):
                    cfg_lat.setdefault("latency_seasonality_path", lat_path)
            sim_lat_cfg = getattr(sim, "latency_config_payload", None)
            if sim_lat_cfg:
                sim_lat_dict = self._latency_dict(sim_lat_cfg)
                for key, value in sim_lat_dict.items():
                    cfg_lat.setdefault(key, value)
            cfg_lat.setdefault("symbol", symbol)
            latency = LatencyImpl.from_dict(cfg_lat)
        if slippage is None:
            slippage = SlippageImpl.from_dict(rc_slippage, run_config=run_config)
        if fees is None:
            fee_cfg_payload = self._build_fee_config(
                rc_fees,
                run_config=run_config,
                symbol=str(symbol),
            )
            fees = FeesImpl.from_dict(fee_cfg_payload)

        dyn_cfg_source: Any = None
        if run_config is not None:
            dyn_cfg_source = getattr(run_config, "slippage", None)
        if dyn_cfg_source is None:
            dyn_cfg_source = rc_slippage
        dyn_spread_enabled = self._dynamic_spread_enabled(dyn_cfg_source)

        # последовательное подключение компонентов к симулятору
        if quantizer is not None:
            quantizer.attach_to(
                self._sim,
                strict=quantizer.cfg.strict,
                enforce_percent_price_by_side=quantizer.cfg.enforce_percent_price_by_side,
            )
        if risk is not None:
            risk.attach_to(self._sim)
        if latency is not None:
            latency.attach_to(self._sim)
        if slippage is not None:
            slippage.attach_to(self._sim)
            if dyn_spread_enabled:
                profile = getattr(slippage, "dynamic_profile", None)
                if profile is not None:
                    try:
                        setattr(self._sim, "slippage_dynamic_profile", profile)
                    except Exception:
                        pass
        if fees is not None:
            fees.attach_to(self._sim)

        try:
            if hasattr(self._sim, "set_execution_profile"):
                self._sim.set_execution_profile(str(self._exec_profile), self._exec_params.dict())
            else:
                setattr(self._sim, "execution_profile", str(self._exec_profile))
                setattr(self._sim, "execution_params", self._exec_params.dict())
        except Exception:
            pass

    @staticmethod
    def from_config(
        *,
        symbol: str,
        max_position_abs_base: float = 1.0,
        sim: ExecutionSimulator,
        run_config: Any | None = None,
    ) -> "SimExecutor":
        """Сконструировать :class:`SimExecutor` из ``run_config``.

        Извлекает блоки ``quantizer``, ``fees``, ``slippage``, ``latency``, ``risk`` и
        ``no_trade`` из ``run_config`` и создаёт соответствующие реализации.
        Значения по умолчанию используются, если блок отсутствует.
        """

        q_impl = QuantizerImpl.from_dict(getattr(run_config, "quantizer", {}) or {})
        fee_cfg_payload = SimExecutor._build_fee_config(
            getattr(run_config, "fees", {}) or {},
            run_config=run_config,
            symbol=str(symbol),
        )
        f_impl = FeesImpl.from_dict(fee_cfg_payload)
        s_impl = SlippageImpl.from_dict(
            getattr(run_config, "slippage", {}) or {}, run_config=run_config
        )
        l_cfg = SimExecutor._latency_dict(getattr(run_config, "latency", None))
        lat_path = getattr(run_config, "latency_seasonality_path", None)
        if lat_path and not l_cfg.get("latency_seasonality_path"):
            l_cfg.setdefault("latency_seasonality_path", lat_path)
        sim_lat_cfg = getattr(sim, "latency_config_payload", None)
        if sim_lat_cfg:
            sim_lat_dict = SimExecutor._latency_dict(sim_lat_cfg)
            for key, value in sim_lat_dict.items():
                l_cfg.setdefault(key, value)
        l_cfg.setdefault("symbol", symbol)
        l_impl = LatencyImpl.from_dict(l_cfg)
        r_impl = RiskBasicImpl.from_dict(getattr(run_config, "risk", None))
        d_impl = DataDegradationConfig.from_dict(
            getattr(run_config, "data_degradation", {}) or {}
        )

        execution_cfg = getattr(run_config, "execution", None)
        exec_cfg_payload = SimExecutor._execution_dict(execution_cfg)
        if exec_cfg_payload:
            try:
                setattr(sim, "_execution_intrabar_cfg", dict(exec_cfg_payload))
            except Exception:
                pass
            bridge_payload = exec_cfg_payload.get("bridge")
            if isinstance(bridge_payload, dict):
                try:
                    setattr(sim, "_execution_bridge_cfg", dict(bridge_payload))
                except Exception:
                    pass
        if run_config is not None:
            try:
                setattr(sim, "_execution_runtime_cfg", execution_cfg)
            except Exception:
                pass

        if q_impl is not None:
            q_impl.attach_to(
                sim,
                strict=q_impl.cfg.strict,
                enforce_percent_price_by_side=q_impl.cfg.enforce_percent_price_by_side,
            )
        if r_impl is not None:
            r_impl.attach_to(sim)
        if l_impl is not None:
            l_impl.attach_to(sim)
        if s_impl is not None:
            s_impl.attach_to(sim)
        if f_impl is not None:
            f_impl.attach_to(sim)

        return SimExecutor(
            sim,
            symbol=symbol,
            max_position_abs_base=max_position_abs_base,
            quantizer=q_impl,
            risk=r_impl,
            latency=l_impl,
            slippage=s_impl,
            fees=f_impl,
            data_degradation=d_impl,
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
        base = float(self._ctx.max_position_abs_base) if self._ctx.max_position_abs_base > 0 else 1.0
        if base <= 0:
            base = 1.0
        vol_frac = max(0.0, abs(qty) / base)
        if str(order.side).upper().endswith("SELL"):
            vol_frac = -vol_frac

        tif = str(self._exec_params.tif)
        ttl_steps = int(self._exec_params.ttl_steps)

        profile = getattr(self, "_exec_profile", ExecutionProfile.MKT_OPEN_NEXT_H1)

        if profile == ExecutionProfile.LIMIT_MID_BPS:
            proto = ActionProto(action_type=ActionType.LIMIT, volume_frac=float(vol_frac))
            setattr(proto, "tif", tif)
            setattr(proto, "ttl_steps", ttl_steps)
            setattr(proto, "client_tag", getattr(order, "client_order_id", "") or "")
            price = getattr(order, "price", None)
            if price is None:
                mid = getattr(self._sim, "_last_ref_price", None)
                if mid is None:
                    bid = getattr(self._sim, "_last_bid", None)
                    ask = getattr(self._sim, "_last_ask", None)
                    if bid is not None and ask is not None:
                        mid = (float(bid) + float(ask)) / 2.0
                if mid is not None:
                    off = float(self._exec_params.limit_offset_bps) / 1e4
                    if vol_frac > 0:
                        price = mid * (1 - off)
                    else:
                        price = mid * (1 + off)
            if price is not None:
                setattr(proto, "abs_price", float(price))
            return ActionType.LIMIT, proto

        proto = ActionProto(action_type=ActionType.MARKET, volume_frac=float(vol_frac))
        setattr(proto, "tif", tif)
        setattr(proto, "ttl_steps", ttl_steps)
        setattr(proto, "client_tag", getattr(order, "client_order_id", "") or "")
        return ActionType.MARKET, proto

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
            "execution_profile": str(self._exec_profile),
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
