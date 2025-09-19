# -*- coding: utf-8 -*-
"""
impl_fees.py
Обёртка над fees.FeesModel. Создаёт модель комиссий и подключает к симулятору.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Mapping

try:
    from fees import FeesModel
except Exception:  # pragma: no cover
    FeesModel = None  # type: ignore

from services.costs import MakerTakerShareSettings


@dataclass
class FeesConfig:
    maker_bps: float = 1.0
    taker_bps: float = 5.0
    use_bnb_discount: bool = False
    maker_discount_mult: Optional[float] = None
    taker_discount_mult: Optional[float] = None
    maker_taker_share: Optional[Dict[str, Any]] = None

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
        share_cfg = MakerTakerShareSettings.parse(self.maker_taker_share)
        if share_cfg is not None:
            self.maker_taker_share = share_cfg.as_dict()
        elif isinstance(self.maker_taker_share, Mapping):
            self.maker_taker_share = dict(self.maker_taker_share)
        else:
            self.maker_taker_share = None


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
        self.base_fee_bps: Dict[str, float] = {
            "maker_fee_bps": float(cfg.maker_bps) * float(cfg.maker_discount_mult),
            "taker_fee_bps": float(cfg.taker_bps) * float(cfg.taker_discount_mult),
        }
        share_cfg = MakerTakerShareSettings.parse(cfg.maker_taker_share)
        self.maker_taker_share_cfg = share_cfg
        self.maker_taker_share_expected: Optional[Dict[str, float]] = None
        if share_cfg is not None:
            self.maker_taker_share_expected = share_cfg.expected_fee_breakdown(
                self.base_fee_bps["maker_fee_bps"], self.base_fee_bps["taker_fee_bps"]
            )
        self.expected_fee_bps: Dict[str, float] = dict(self.base_fee_bps)
        if self.maker_taker_share_expected is not None:
            self.expected_fee_bps.update(self.maker_taker_share_expected)
        self.maker_taker_share_raw: Optional[Dict[str, Any]] = (
            dict(cfg.maker_taker_share) if isinstance(cfg.maker_taker_share, dict) else None
        )

    @property
    def model(self):
        return self._model

    def attach_to(self, sim) -> None:
        if self._model is not None:
            setattr(sim, "fees", self._model)
        payload = None
        if self.maker_taker_share_cfg is not None:
            payload = self.maker_taker_share_cfg.to_sim_payload(
                self.base_fee_bps["maker_fee_bps"],
                self.base_fee_bps["taker_fee_bps"],
            )
        setattr(sim, "_maker_taker_share_cfg", payload)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FeesImpl":
        use_bnb = bool(d.get("use_bnb_discount", False))
        maker_mult = d.get("maker_discount_mult")
        taker_mult = d.get("taker_discount_mult")
        share_block = d.get("maker_taker_share")
        share_payload: Optional[Dict[str, Any]] = None
        share_cfg = MakerTakerShareSettings.parse(share_block)
        if share_cfg is not None:
            share_payload = share_cfg.as_dict()
        elif isinstance(share_block, Mapping):
            share_payload = dict(share_block)
        return FeesImpl(FeesConfig(
            maker_bps=float(d.get("maker_bps", 1.0)),
            taker_bps=float(d.get("taker_bps", 5.0)),
            use_bnb_discount=use_bnb,
            maker_discount_mult=float(maker_mult) if maker_mult is not None else None,
            taker_discount_mult=float(taker_mult) if taker_mult is not None else None,
            maker_taker_share=share_payload,
        ))
