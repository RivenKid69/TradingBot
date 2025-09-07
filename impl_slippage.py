# -*- coding: utf-8 -*-
"""
impl_slippage.py
Обёртка над slippage.SlippageConfig и функциями оценки. Подключает конфиг к симулятору.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

try:
    from slippage import SlippageConfig
except Exception:  # pragma: no cover
    SlippageConfig = None  # type: ignore


@dataclass
class SlippageCfg:
    k: float = 0.8
    min_half_spread_bps: float = 0.0
    default_spread_bps: float = 2.0
    eps: float = 1e-12


class SlippageImpl:
    def __init__(self, cfg: SlippageCfg) -> None:
        self.cfg = cfg
        self._cfg_obj = SlippageConfig.from_dict({
            "k": float(cfg.k),
            "min_half_spread_bps": float(cfg.min_half_spread_bps),
            "default_spread_bps": float(cfg.default_spread_bps),
            "eps": float(cfg.eps),
        }) if SlippageConfig is not None else None

    @property
    def config(self):
        return self._cfg_obj

    def attach_to(self, sim) -> None:
        if self._cfg_obj is not None:
            setattr(sim, "slippage_cfg", self._cfg_obj)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SlippageImpl":
        return SlippageImpl(SlippageCfg(
            k=float(d.get("k", 0.8)),
            min_half_spread_bps=float(d.get("min_half_spread_bps", 0.0)),
            default_spread_bps=float(d.get("default_spread_bps", 2.0)),
            eps=float(d.get("eps", 1e-12)),
        ))
