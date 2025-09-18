# -*- coding: utf-8 -*-
"""
impl_slippage.py
Обёртка над slippage.SlippageConfig и функциями оценки. Подключает конфиг к симулятору.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from slippage import SlippageConfig, DynamicSpreadConfig
except Exception:  # pragma: no cover
    SlippageConfig = None  # type: ignore
    DynamicSpreadConfig = None  # type: ignore


@dataclass
class SlippageCfg:
    k: float = 0.8
    min_half_spread_bps: float = 0.0
    default_spread_bps: float = 2.0
    eps: float = 1e-12
    dynamic_spread: Optional[Any] = None


class SlippageImpl:
    def __init__(self, cfg: SlippageCfg) -> None:
        self.cfg = cfg
        cfg_dict: Dict[str, Any] = {
            "k": float(cfg.k),
            "min_half_spread_bps": float(cfg.min_half_spread_bps),
            "default_spread_bps": float(cfg.default_spread_bps),
            "eps": float(cfg.eps),
        }
        if cfg.dynamic_spread is not None:
            dyn = cfg.dynamic_spread
            if hasattr(dyn, "to_dict"):
                dyn_dict = dyn.to_dict()
            elif isinstance(dyn, dict):
                dyn_dict = dict(dyn)
            else:
                dyn_dict = None
            if dyn_dict is not None:
                cfg_dict["dynamic_spread"] = dyn_dict

        self._cfg_obj = (
            SlippageConfig.from_dict(cfg_dict)
            if SlippageConfig is not None
            else None
        )

    @property
    def config(self):
        return self._cfg_obj

    def attach_to(self, sim) -> None:
        if self._cfg_obj is not None:
            setattr(sim, "slippage_cfg", self._cfg_obj)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SlippageImpl":
        dyn_block = d.get("dynamic_spread")
        dyn_cfg: Optional[Any] = None
        if isinstance(dyn_block, dict):
            if DynamicSpreadConfig is not None:
                dyn_cfg = DynamicSpreadConfig.from_dict(dyn_block)
            else:
                dyn_cfg = dict(dyn_block)

        return SlippageImpl(
            SlippageCfg(
                k=float(d.get("k", 0.8)),
                min_half_spread_bps=float(d.get("min_half_spread_bps", 0.0)),
                default_spread_bps=float(d.get("default_spread_bps", 2.0)),
                eps=float(d.get("eps", 1e-12)),
                dynamic_spread=dyn_cfg,
            )
        )
