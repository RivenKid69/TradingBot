# -*- coding: utf-8 -*-
"""
impl_fees.py
Обёртка над fees.FeesModel. Создаёт модель комиссий и подключает к симулятору.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

try:
    from fees import FeesModel
except Exception:  # pragma: no cover
    FeesModel = None  # type: ignore


@dataclass
class FeesConfig:
    maker_bps: float = 1.0
    taker_bps: float = 5.0
    use_bnb_discount: bool = False
    maker_discount_mult: Optional[float] = None
    taker_discount_mult: Optional[float] = None

    def __post_init__(self) -> None:
        if self.use_bnb_discount:
            if self.maker_discount_mult is None:
                self.maker_discount_mult = 0.75
            if self.taker_discount_mult is None:
                self.taker_discount_mult = 0.75
        else:
            if self.maker_discount_mult is None:
                self.maker_discount_mult = 1.0
            if self.taker_discount_mult is None:
                self.taker_discount_mult = 1.0


class FeesImpl:
    def __init__(self, cfg: FeesConfig) -> None:
        self.cfg = cfg
        self._model = FeesModel.from_dict({
            "maker_bps": float(cfg.maker_bps),
            "taker_bps": float(cfg.taker_bps),
            "use_bnb_discount": bool(cfg.use_bnb_discount),
            "maker_discount_mult": float(cfg.maker_discount_mult),
            "taker_discount_mult": float(cfg.taker_discount_mult),
        }) if FeesModel is not None else None

    @property
    def model(self):
        return self._model

    def attach_to(self, sim) -> None:
        if self._model is not None:
            setattr(sim, "fees", self._model)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FeesImpl":
        use_bnb = bool(d.get("use_bnb_discount", False))
        maker_mult = d.get("maker_discount_mult")
        taker_mult = d.get("taker_discount_mult")
        return FeesImpl(FeesConfig(
            maker_bps=float(d.get("maker_bps", 1.0)),
            taker_bps=float(d.get("taker_bps", 5.0)),
            use_bnb_discount=use_bnb,
            maker_discount_mult=float(maker_mult) if maker_mult is not None else None,
            taker_discount_mult=float(taker_mult) if taker_mult is not None else None,
        ))
