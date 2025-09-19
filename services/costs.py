# -*- coding: utf-8 -*-
"""Utility helpers for trade cost configuration blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
    distance_to_mid: Optional[float] = None
    latency: Optional[float] = None
    coefficients: Dict[str, float] = field(default_factory=dict)

    _VALID_MODES = {"fixed", "model", "predictor"}

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

    @classmethod
    def _normalise_mode(cls, mode: Any) -> str:
        if isinstance(mode, str):
            candidate = mode.strip().lower()
            if candidate in cls._VALID_MODES:
                return candidate
        return "fixed"

    @staticmethod
    def _clip_probability(value: float, default: float = 0.5) -> float:
        if not math.isfinite(value):
            return default
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return float(value)

    @classmethod
    def _normalise_feature(
        cls,
        value: Any,
        *,
        minimum: Optional[float] = 0.0,
        maximum: Optional[float] = None,
        default: Optional[float] = None,
    ) -> Optional[float]:
        fallback = default
        candidate = value
        min_bound = minimum
        max_bound = maximum
        if isinstance(value, Mapping):
            candidate = value.get("value")
            fb = value.get("fallback")
            if fb is not None:
                fallback = cls._coerce_optional_float(fb)
            min_candidate = value.get("min")
            max_candidate = value.get("max")
            min_normalised = cls._coerce_optional_float(min_candidate)
            max_normalised = cls._coerce_optional_float(max_candidate)
            if min_normalised is not None:
                min_bound = min_normalised
            if max_normalised is not None:
                max_bound = max_normalised
        normalised = cls._coerce_optional_float(candidate)
        if normalised is None:
            normalised = fallback
        if normalised is None:
            return None
        if min_bound is not None and max_bound is not None and max_bound < min_bound:
            max_bound = None
        if min_bound is not None and normalised < min_bound:
            normalised = min_bound
        if max_bound is not None and normalised > max_bound:
            normalised = max_bound
        return float(normalised)

    @staticmethod
    def _normalise_coefficients(value: Any) -> Dict[str, float]:
        if not isinstance(value, Mapping):
            return {}
        normalised: Dict[str, float] = {}
        for key, raw in value.items():
            if not isinstance(key, str):
                continue
            num = MakerTakerShareSettings._coerce_optional_float(raw)
            if num is None:
                continue
            normalised[key.strip().lower()] = float(num)
        return normalised

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
        model_block_raw = data.get("model")
        model_block = model_block_raw if isinstance(model_block_raw, Mapping) else None
        distance_raw = data.get("distance_to_mid")
        latency_raw = data.get("latency")
        coeffs_raw = data.get("coefficients")
        if isinstance(model_block, Mapping):
            distance_raw = model_block.get("distance_to_mid", distance_raw)
            latency_raw = model_block.get("latency", latency_raw)
            coeffs_candidate = model_block.get("coefficients")
            if coeffs_candidate is not None:
                coeffs_raw = coeffs_candidate
        distance_to_mid = cls._normalise_feature(
            distance_raw,
            minimum=0.0,
            maximum=None,
        )
        latency = cls._normalise_feature(
            latency_raw,
            minimum=0.0,
            maximum=None,
        )
        coefficients = cls._normalise_coefficients(coeffs_raw)
        return cls(
            enabled=enabled,
            mode=mode,
            maker_share_default=maker_share_default,
            spread_cost_maker_bps=spread_cost_maker_bps,
            spread_cost_taker_bps=spread_cost_taker_bps,
            taker_fee_override_bps=taker_fee_override_bps,
            distance_to_mid=distance_to_mid,
            latency=latency,
            coefficients=coefficients,
        )

    @property
    def maker_share(self) -> float:
        return self._resolve_maker_share()

    def _resolve_maker_share(self) -> float:
        base_share = float(self.maker_share_default)
        if not self.enabled:
            return self._clip_probability(base_share, base_share)
        if self.mode in {"model", "predictor"}:
            predicted = self._predict_maker_share()
            if predicted is not None:
                return self._clip_probability(predicted, base_share)
        return self._clip_probability(base_share, base_share)

    def _predict_maker_share(self) -> Optional[float]:
        coeffs = self.coefficients
        if not coeffs:
            return None
        base_share = float(self.maker_share_default)
        score = 0.0
        used = False
        intercept = coeffs.get("intercept") or coeffs.get("bias")
        if intercept is not None:
            score = float(intercept)
            used = True
        features = {
            "distance_to_mid": self.distance_to_mid,
            "latency": self.latency,
        }
        aliases = {
            "distance_to_mid": (
                "distance_to_mid",
                "distance",
                "dist",
                "distance_bps",
            ),
            "latency": ("latency", "latency_ms", "lat"),
        }
        for feature, value in features.items():
            if value is None or not math.isfinite(value):
                continue
            coef = None
            for alias in aliases[feature]:
                coef = coeffs.get(alias)
                if coef is not None:
                    break
            if coef is None:
                continue
            score += float(coef) * float(value)
            used = True
        if not used:
            return None
        try:
            probability = 1.0 / (1.0 + math.exp(-score))
        except OverflowError:
            probability = 0.0 if score < 0.0 else 1.0
        if not math.isfinite(probability):
            return None
        min_share = coeffs.get("min_share")
        max_share = coeffs.get("max_share")
        probability = self._clip_probability(probability, base_share)
        if min_share is not None:
            min_val = self._clip_probability(float(min_share), base_share)
            probability = max(probability, min_val)
        if max_share is not None:
            max_val = self._clip_probability(float(max_share), base_share)
            probability = min(probability, max_val)
        return probability

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
            "distance_to_mid": (
                float(self.distance_to_mid) if self.distance_to_mid is not None else None
            ),
            "latency": float(self.latency) if self.latency is not None else None,
            "coefficients": {
                str(key): float(value)
                for key, value in self.coefficients.items()
            },
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
        share = self._resolve_maker_share()
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
            "distance_to_mid": (
                float(self.distance_to_mid) if self.distance_to_mid is not None else None
            ),
            "latency": float(self.latency) if self.latency is not None else None,
            "coefficients": {
                str(key): float(value)
                for key, value in self.coefficients.items()
            },
        }
        return payload


__all__ = ["MakerTakerShareSettings"]
