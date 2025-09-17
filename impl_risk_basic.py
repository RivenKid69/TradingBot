# -*- coding: utf-8 -*-
"""
impl_risk_basic.py
Обёртка над risk.RiskManager/RiskConfig. Подключает риск в симулятор.
Учтены сезонные коэффициенты ликвидности/латентности, которые могут
масштабировать лимиты RiskManager через параметры ``liquidity_mult`` и
``latency_mult`` соответствующих методов.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

try:  # pragma: no cover - optional dependency used for type checks
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    BaseModel = None  # type: ignore

try:
    from risk import RiskManager, RiskConfig
except Exception:  # pragma: no cover
    RiskManager = None  # type: ignore
    RiskConfig = None  # type: ignore


@dataclass
class RiskBasicCfg:
    enabled: bool = True
    max_abs_position_qty: float = 0.0
    max_abs_position_notional: float = 0.0
    max_order_notional: float = 0.0
    max_orders_per_min: int = 60
    max_orders_window_s: int = 60
    daily_loss_limit: float = 0.0
    pause_seconds_on_violation: int = 300
    daily_reset_utc_hour: int = 0
    max_entries_per_day: Optional[int] = None
    max_total_notional: Optional[float] = None
    max_total_exposure_pct: Optional[float] = None
    exposure_buffer_frac: float = 0.0


class RiskBasicImpl:
    def __init__(self, cfg: RiskBasicCfg) -> None:
        self.cfg = cfg
        payload = {
            "enabled": bool(cfg.enabled),
            "max_abs_position_qty": float(cfg.max_abs_position_qty),
            "max_abs_position_notional": float(cfg.max_abs_position_notional),
            "max_order_notional": float(cfg.max_order_notional),
            "max_orders_per_min": int(cfg.max_orders_per_min),
            "max_orders_window_s": int(cfg.max_orders_window_s),
            "daily_loss_limit": float(cfg.daily_loss_limit),
            "pause_seconds_on_violation": int(cfg.pause_seconds_on_violation),
            "daily_reset_utc_hour": int(cfg.daily_reset_utc_hour),
            "max_entries_per_day": (
                None if cfg.max_entries_per_day is None else int(cfg.max_entries_per_day)
            ),
            "exposure_buffer_frac": float(cfg.exposure_buffer_frac or 0.0),
        }
        if cfg.max_total_notional is not None:
            payload["max_total_notional"] = float(cfg.max_total_notional)
        if cfg.max_total_exposure_pct is not None:
            payload["max_total_exposure_pct"] = float(cfg.max_total_exposure_pct)

        self._manager = (
            RiskManager(RiskConfig.from_dict(payload))
            if (RiskManager is not None and RiskConfig is not None)
            else None
        )

    @property
    def manager(self):
        return self._manager

    def attach_to(self, sim) -> None:
        if self._manager is not None:
            setattr(sim, "risk", self._manager)

    @staticmethod
    def from_dict(d: Mapping[str, Any] | Any | None) -> "RiskBasicImpl":
        data: Dict[str, Any]
        exposure_defaults: Dict[str, Any] = {}

        def _extract_limits(obj: Any) -> Dict[str, Any]:
            limits = getattr(obj, "exposure_limits", None)
            if callable(limits):
                try:
                    candidate = limits()
                except TypeError:
                    candidate = limits  # callable but without args; treat as value
            else:
                candidate = limits
            if isinstance(candidate, Mapping):
                return dict(candidate)
            return {}

        if d is None:
            data = {}
        elif isinstance(d, RiskBasicCfg):
            data = {
                "enabled": d.enabled,
                "max_abs_position_qty": d.max_abs_position_qty,
                "max_abs_position_notional": d.max_abs_position_notional,
                "max_order_notional": d.max_order_notional,
                "max_orders_per_min": d.max_orders_per_min,
                "max_orders_window_s": d.max_orders_window_s,
                "daily_loss_limit": d.daily_loss_limit,
                "pause_seconds_on_violation": d.pause_seconds_on_violation,
                "daily_reset_utc_hour": d.daily_reset_utc_hour,
                "max_entries_per_day": d.max_entries_per_day,
                "max_total_notional": d.max_total_notional,
                "max_total_exposure_pct": d.max_total_exposure_pct,
                "exposure_buffer_frac": d.exposure_buffer_frac,
            }
        elif BaseModel is not None and isinstance(d, BaseModel):  # type: ignore[arg-type]
            data = dict(d.dict())
            exposure_defaults = _extract_limits(d)
        elif hasattr(d, "component_params"):
            data = dict(d.component_params())  # type: ignore[call-arg]
            exposure_defaults = _extract_limits(d)
        elif isinstance(d, Mapping):
            data = dict(d)
        else:
            try:
                data = dict(d)  # type: ignore[arg-type]
            except Exception:
                data = {}

        max_total_notional = exposure_defaults.get("max_total_notional", data.get("max_total_notional"))
        max_total_exposure_pct = exposure_defaults.get("max_total_exposure_pct", data.get("max_total_exposure_pct"))
        exposure_buffer_frac = exposure_defaults.get("exposure_buffer_frac", data.get("exposure_buffer_frac", 0.0))

        clean_data = dict(data)
        clean_data.pop("max_total_notional", None)
        clean_data.pop("max_total_exposure_pct", None)
        clean_data.pop("exposure_buffer_frac", None)

        return RiskBasicImpl(RiskBasicCfg(
            enabled=bool(clean_data.get("enabled", True)),
            max_abs_position_qty=float(clean_data.get("max_abs_position_qty", 0.0)),
            max_abs_position_notional=float(clean_data.get("max_abs_position_notional", 0.0)),
            max_order_notional=float(clean_data.get("max_order_notional", 0.0)),
            max_orders_per_min=int(clean_data.get("max_orders_per_min", 60)),
            max_orders_window_s=int(clean_data.get("max_orders_window_s", 60)),
            daily_loss_limit=float(clean_data.get("daily_loss_limit", 0.0)),
            pause_seconds_on_violation=int(clean_data.get("pause_seconds_on_violation", 300)),
            daily_reset_utc_hour=int(clean_data.get("daily_reset_utc_hour", 0)),
            max_entries_per_day=(
                None
                if clean_data.get("max_entries_per_day") is None
                else int(clean_data.get("max_entries_per_day"))
            ),
            max_total_notional=(
                None if max_total_notional is None else float(max_total_notional)
            ),
            max_total_exposure_pct=(
                None if max_total_exposure_pct is None else float(max_total_exposure_pct)
            ),
            exposure_buffer_frac=float(exposure_buffer_frac or 0.0),
        ))
