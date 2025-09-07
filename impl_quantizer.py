# -*- coding: utf-8 -*-
"""
impl_quantizer.py
Обёртка над quantizer. Строит Quantizer из JSON-фильтров Binance и умеет подключаться к ExecutionSimulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

try:
    from quantizer import Quantizer, load_filters
except Exception as e:  # pragma: no cover
    Quantizer = None  # type: ignore
    load_filters = None  # type: ignore


@dataclass
class QuantizerConfig:
    path: str
    strict: bool = True
    enforce_percent_price_by_side: bool = True  # передаётся в симулятор как enforce_ppbs


class QuantizerImpl:
    def __init__(self, cfg: QuantizerConfig) -> None:
        self.cfg = cfg
        self._quantizer = None
        if Quantizer is not None and load_filters is not None and cfg.path:
            filters = load_filters(cfg.path)
            if filters:
                self._quantizer = Quantizer(filters, strict=bool(cfg.strict))

    @property
    def quantizer(self):
        return self._quantizer

    def attach_to(self, sim) -> None:
        """
        Подключает к симулятору:
          - sim.quantizer = Quantizer(...)
          - sim.enforce_ppbs = cfg.enforce_percent_price_by_side
          - sim.strict_filters = cfg.strict
        """
        if self._quantizer is not None:
            setattr(sim, "quantizer", self._quantizer)
        setattr(sim, "enforce_ppbs", bool(self.cfg.enforce_percent_price_by_side))
        setattr(sim, "strict_filters", bool(self.cfg.strict))

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "QuantizerImpl":
        return QuantizerImpl(QuantizerConfig(
            path=str(d.get("path", "")),
            strict=bool(d.get("strict", True)),
            enforce_percent_price_by_side=bool(d.get("enforce_percent_price_by_side", True)),
        ))
