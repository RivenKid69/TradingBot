# -*- coding: utf-8 -*-
"""Utility helpers for trade cost configuration blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class MakerTakerShareSettings:
    """Container for maker/taker share configuration.

    The structure normalises raw configuration values and provides helpers for
    downstream components (fees, slippage, simulator) to operate on the same
    canonical representation.
    """

    enabled: bool = False
    mode: str = "fixed"
    maker_share_default: float = 0.5
    spread_cost_maker_bps: float = 0.0
    spread_cost_taker_bps: float = 0.0
    taker_fee_override_bps: Optional[float] = None

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(num):
            return default
        return num

    @staticmethod
    def _coerce_optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(num):
            return None
        return num

    @staticmethod
    def _normalise_mode(mode: Any) -> str:
        if isinstance(mode, str):
            candidate = mode.strip()
            if candidate:
                return candidate.lower()
        return "fixed"

    @classmethod
    def _normalise_share(cls, value: Any) -> float:
        share = cls._coerce_float(value, 0.5)
        if share < 0.0:
            share = 0.0
        elif share > 1.0:
            share = 1.0
        return share

    @classmethod
    def parse(cls, data: Any) -> Optional["MakerTakerShareSettings"]:
        """Best-effort parsing from an arbitrary payload."""

        if data is None:
            return None
        if isinstance(data, cls):
            return data
        if isinstance(data, Mapping):
            return cls.from_dict(data)
        return None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MakerTakerShareSettings":
        enabled = bool(data.get("enabled", False))
        mode = cls._normalise_mode(data.get("mode"))
        maker_share_default = cls._normalise_share(data.get("maker_share_default"))
        spread_cost_maker_bps = cls._coerce_float(
            data.get("spread_cost_maker_bps"), 0.0
        )
        spread_cost_taker_bps = cls._coerce_float(
            data.get("spread_cost_taker_bps"), 0.0
        )
        taker_fee_override_bps = cls._coerce_optional_float(
            data.get("taker_fee_override_bps")
        )
        return cls(
            enabled=enabled,
            mode=mode,
            maker_share_default=maker_share_default,
            spread_cost_maker_bps=spread_cost_maker_bps,
            spread_cost_taker_bps=spread_cost_taker_bps,
            taker_fee_override_bps=taker_fee_override_bps,
        )

    @property
    def maker_share(self) -> float:
        return self.maker_share_default

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "maker_share_default": float(self.maker_share_default),
            "spread_cost_maker_bps": float(self.spread_cost_maker_bps),
            "spread_cost_taker_bps": float(self.spread_cost_taker_bps),
            "taker_fee_override_bps": (
                float(self.taker_fee_override_bps)
                if self.taker_fee_override_bps is not None
                else None
            ),
        }

    def effective_maker_fee_bps(self, maker_fee_bps: float) -> float:
        return float(maker_fee_bps) + float(self.spread_cost_maker_bps)

    def effective_taker_fee_bps(self, taker_fee_bps: float) -> float:
        base = (
            float(taker_fee_bps)
            if self.taker_fee_override_bps is None
            else float(self.taker_fee_override_bps)
        )
        return base + float(self.spread_cost_taker_bps)

    def expected_fee_breakdown(
        self, maker_fee_bps: float, taker_fee_bps: float
    ) -> Dict[str, float]:
        maker_fee = self.effective_maker_fee_bps(maker_fee_bps)
        taker_fee = self.effective_taker_fee_bps(taker_fee_bps)
        share = float(self.maker_share_default)
        expected_fee = share * maker_fee + (1.0 - share) * taker_fee
        return {
            "maker_fee_bps": maker_fee,
            "taker_fee_bps": taker_fee,
            "maker_share": share,
            "expected_fee_bps": expected_fee,
        }

    def to_sim_payload(
        self, maker_fee_bps: float, taker_fee_bps: float
    ) -> Dict[str, Any]:
        breakdown = self.expected_fee_breakdown(maker_fee_bps, taker_fee_bps)
        payload: Dict[str, Any] = {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "maker_share": breakdown["maker_share"],
            "maker_share_default": float(self.maker_share_default),
            "maker_fee_bps": breakdown["maker_fee_bps"],
            "taker_fee_bps": breakdown["taker_fee_bps"],
            "expected_fee_bps": breakdown["expected_fee_bps"],
            "spread_cost_maker_bps": float(self.spread_cost_maker_bps),
            "spread_cost_taker_bps": float(self.spread_cost_taker_bps),
            "taker_fee_override_bps": (
                float(self.taker_fee_override_bps)
                if self.taker_fee_override_bps is not None
                else None
            ),
        }
        return payload


__all__ = ["MakerTakerShareSettings"]
