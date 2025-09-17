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
from typing import Dict, Any, Optional

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


class RiskBasicImpl:
    def __init__(self, cfg: RiskBasicCfg) -> None:
        self.cfg = cfg
        self._manager = RiskManager(RiskConfig.from_dict({
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
        })) if (RiskManager is not None and RiskConfig is not None) else None

    @property
    def manager(self):
        return self._manager

    def attach_to(self, sim) -> None:
        if self._manager is not None:
            setattr(sim, "risk", self._manager)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RiskBasicImpl":
        return RiskBasicImpl(RiskBasicCfg(
            enabled=bool(d.get("enabled", True)),
            max_abs_position_qty=float(d.get("max_abs_position_qty", 0.0)),
            max_abs_position_notional=float(d.get("max_abs_position_notional", 0.0)),
            max_order_notional=float(d.get("max_order_notional", 0.0)),
            max_orders_per_min=int(d.get("max_orders_per_min", 60)),
            max_orders_window_s=int(d.get("max_orders_window_s", 60)),
            daily_loss_limit=float(d.get("daily_loss_limit", 0.0)),
            pause_seconds_on_violation=int(d.get("pause_seconds_on_violation", 300)),
            daily_reset_utc_hour=int(d.get("daily_reset_utc_hour", 0)),
            max_entries_per_day=(
                None
                if d.get("max_entries_per_day") is None
                else int(d.get("max_entries_per_day"))
            ),
        ))
