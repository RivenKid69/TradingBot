# -*- coding: utf-8 -*-
"""
impl_slippage.py
Обёртка над slippage.SlippageConfig и функциями оценки. Подключает конфиг к симулятору.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

try:
    from slippage import (
        SlippageConfig,
        DynamicSpreadConfig,
        DynamicImpactConfig,
        TailShockConfig,
        AdvConfig,
    )
except Exception:  # pragma: no cover
    SlippageConfig = None  # type: ignore
    DynamicSpreadConfig = None  # type: ignore
    DynamicImpactConfig = None  # type: ignore
    TailShockConfig = None  # type: ignore
    AdvConfig = None  # type: ignore

try:
    from utils_time import get_hourly_multiplier, watch_seasonality_file
except Exception:  # pragma: no cover
    def get_hourly_multiplier(ts_ms, multipliers, *, interpolate=False):  # type: ignore
        return 1.0

    def watch_seasonality_file(path, callback, *, poll_interval=60.0):  # type: ignore
        return None


logger = logging.getLogger(__name__)


def _coerce_sequence(values: Iterable[float]) -> tuple[float, ...]:
    res = []
    for raw in values:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            raise ValueError("multipliers must be numeric") from None
        if not math.isfinite(val):
            raise ValueError("multipliers must be finite")
        res.append(val)
    if not res:
        raise ValueError("multipliers must be non-empty")
    return tuple(res)


def _as_iterable(values: Any) -> Optional[Iterable[Any]]:
    if isinstance(values, Mapping):
        return None
    if isinstance(values, (str, bytes, bytearray)):
        return None
    if isinstance(values, Sequence):
        return values
    if hasattr(values, "__iter__"):
        return values  # type: ignore[return-value]
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


class _DynamicSpreadProfile:
    """Maintain hourly spread multipliers with optional smoothing."""

    def __init__(
        self,
        *,
        cfg: DynamicSpreadConfig,
        default_spread_bps: float,
    ) -> None:
        self._cfg = cfg
        self._base_spread_bps = float(default_spread_bps)
        self._min_spread_bps = (
            float(cfg.min_spread_bps)
            if getattr(cfg, "min_spread_bps", None) is not None
            else None
        )
        self._max_spread_bps = (
            float(cfg.max_spread_bps)
            if getattr(cfg, "max_spread_bps", None) is not None
            else None
        )
        alpha = getattr(cfg, "smoothing_alpha", None)
        if alpha is None:
            self._smoothing_alpha: Optional[float] = None
        else:
            try:
                alpha_val = float(alpha)
            except (TypeError, ValueError):
                alpha_val = 0.0
            if alpha_val <= 0.0:
                self._smoothing_alpha = None
            elif alpha_val >= 1.0:
                self._smoothing_alpha = 1.0
            else:
                self._smoothing_alpha = alpha_val
        self._prev_smoothed: Optional[float] = None
        self._lock = threading.Lock()
        self._last_mtime: Dict[str, float] = {}
        self._multipliers: tuple[float, ...] = (1.0,)
        self._load_initial()
        watch_paths = {
            p
            for p in (
                getattr(cfg, "path", None),
                getattr(cfg, "override_path", None),
            )
            if p
        }
        for path in watch_paths:
            try:
                watch_seasonality_file(path, self._handle_reload)
            except Exception:  # pragma: no cover - watcher is optional
                logger.exception("Failed to start seasonality watcher for %s", path)

    def _load_initial(self) -> None:
        inline = self._load_inline()
        if inline is not None:
            self._set_multipliers(inline)
            return
        base_path = getattr(self._cfg, "path", None)
        base = self._load_from_path(base_path) if base_path else None
        if base is not None:
            self._set_multipliers(base)
        override_path = getattr(self._cfg, "override_path", None)
        override = self._load_from_path(override_path) if override_path else None
        if override is not None:
            self._set_multipliers(override)

    def _set_multipliers(self, values: Sequence[float]) -> None:
        try:
            arr = _coerce_sequence(values)
        except ValueError as exc:
            logger.warning("Invalid spread multipliers: %s", exc)
            return
        with self._lock:
            self._multipliers = arr

    def _load_inline(self) -> Optional[Sequence[float]]:
        values = getattr(self._cfg, "multipliers", None)
        if values is None:
            return None
        seq = _as_iterable(values)
        if seq is not None:
            return tuple(float(v) for v in seq)
        logger.warning("Dynamic spread multipliers must be a sequence; got %r", values)
        return None

    def _select_payload(self, payload: Any) -> Optional[Sequence[float]]:
        seq = _as_iterable(payload)
        if seq is not None:
            return seq  # type: ignore[return-value]
        if isinstance(payload, Mapping):
            profile = getattr(self._cfg, "profile_kind", None)
            if profile and profile in payload:
                return self._select_payload(payload[profile])
            for key in ("spread", "multipliers"):
                if key in payload:
                    return self._select_payload(payload[key])
            for val in payload.values():
                res = self._select_payload(val)
                if res is not None:
                    return res
        return None

    def _load_from_path(self, path: Optional[str]) -> Optional[Sequence[float]]:
        if not path:
            return None
        try:
            mtime = os.path.getmtime(path)
        except (OSError, TypeError, ValueError):
            return None
        if path in self._last_mtime and mtime <= self._last_mtime[path]:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            logger.exception("Failed to load dynamic spread multipliers from %s", path)
            return None
        values = self._select_payload(payload)
        if values is None:
            logger.warning("No spread multipliers found in %s", path)
            return None
        self._last_mtime[path] = mtime
        return values

    def _handle_reload(self, data: Dict[str, Any]) -> None:
        profile = getattr(self._cfg, "profile_kind", None)
        candidate: Optional[Iterable[float]] = None
        if profile and profile in data:
            candidate = data.get(profile)
        elif "spread" in data:
            candidate = data.get("spread")
        elif data:
            # use the first array-like payload
            for value in data.values():
                seq = _as_iterable(value)
                if seq is not None:
                    candidate = seq
                    break
        if candidate is None:
            return
        self._set_multipliers(candidate)  # type: ignore[arg-type]

    def _seasonal_multiplier(self, ts_ms: Any) -> float:
        with self._lock:
            multipliers = self._multipliers
        try:
            ts_val = int(ts_ms)
        except (TypeError, ValueError):
            return 1.0
        try:
            return float(get_hourly_multiplier(ts_val, multipliers))
        except Exception:
            return 1.0

    def _apply_clamp(self, spread_bps: float) -> float:
        if self._min_spread_bps is not None:
            spread_bps = max(self._min_spread_bps, spread_bps)
        if self._max_spread_bps is not None:
            spread_bps = min(self._max_spread_bps, spread_bps)
        return spread_bps

    def _apply_smoothing(self, spread_bps: float) -> float:
        alpha = self._smoothing_alpha
        if alpha is None:
            with self._lock:
                self._prev_smoothed = spread_bps
            return spread_bps
        with self._lock:
            prev = self._prev_smoothed
            if prev is None:
                self._prev_smoothed = spread_bps
                return spread_bps
            smoothed = alpha * spread_bps + (1.0 - alpha) * prev
            self._prev_smoothed = smoothed
        return smoothed

    def seasonal_multiplier(self, ts_ms: Any) -> float:
        """Public wrapper used by the dynamic spread helper."""

        return self._seasonal_multiplier(ts_ms)

    def process_spread(self, spread_bps: Any, *, already_clamped: bool = False) -> float:
        """Clamp and smooth the supplied spread value."""

        candidate = _safe_float(spread_bps)
        if candidate is None:
            candidate = self._base_spread_bps
        if not already_clamped:
            candidate = self._apply_clamp(candidate)
        return self._apply_smoothing(candidate)

    def compute(
        self,
        *,
        ts_ms: Any,
        base_spread_bps: float,
        vol_multiplier: float,
    ) -> float:
        try:
            base = float(base_spread_bps)
        except (TypeError, ValueError):
            base = self._base_spread_bps
        else:
            if not math.isfinite(base) or base <= 0.0:
                base = self._base_spread_bps
        seasonal = self._seasonal_multiplier(ts_ms)
        spread = base * seasonal * float(vol_multiplier)
        return self.process_spread(spread)


def _calc_dynamic_spread(
    *,
    cfg: DynamicSpreadConfig,
    default_spread_bps: float,
    bar_high: Any,
    bar_low: Any,
    mid_price: Any,
    vol_metrics: Optional[Mapping[str, Any]] = None,
    seasonal_multiplier: float = 1.0,
    vol_multiplier: float = 1.0,
    profile: Optional[_DynamicSpreadProfile] = None,
) -> Optional[float]:
    """Compute a dynamic spread using the range-based heuristic.

    The helper returns ``None`` when the supplied inputs are insufficient,
    allowing callers to fall back to the default spread behaviour.
    """

    alpha = _safe_float(getattr(cfg, "alpha_bps", None))
    if alpha is None:
        alpha = _safe_float(default_spread_bps) or 0.0
    beta = _safe_float(getattr(cfg, "beta_coef", None))
    if beta is None:
        beta = 0.0

    high = _safe_float(bar_high)
    low = _safe_float(bar_low)
    mid = _safe_float(mid_price)
    range_ratio_bps: Optional[float] = None
    if high is not None and low is not None and mid is not None and mid > 0.0:
        price_range = high - low
        if price_range < 0.0:
            logger.debug(
                "Dynamic spread received inverted bar range: high=%s low=%s",
                high,
                low,
            )
            price_range = abs(price_range)
        ratio = price_range / mid if mid > 0.0 else None
        if ratio is not None and math.isfinite(ratio):
            range_ratio_bps = max(ratio, 0.0) * 1e4

    if range_ratio_bps is None and vol_metrics and isinstance(vol_metrics, Mapping):
        vol_key = getattr(cfg, "vol_metric", None)
        candidates = []
        if vol_key and vol_key in vol_metrics:
            candidates.append(vol_metrics[vol_key])
        if "range_ratio_bps" in vol_metrics:
            candidates.append(vol_metrics["range_ratio_bps"])
        for candidate in candidates:
            candidate_val = _safe_float(candidate)
            if candidate_val is not None and candidate_val >= 0.0:
                range_ratio_bps = candidate_val
                break

    if range_ratio_bps is None:
        logger.debug(
            "Dynamic spread inputs missing (high=%r, low=%r, mid=%r, vol_metric=%r)",
            bar_high,
            bar_low,
            mid_price,
            getattr(cfg, "vol_metric", None),
        )
        return None

    spread = alpha + beta * range_ratio_bps

    seasonal = _safe_float(seasonal_multiplier)
    if seasonal is None or seasonal <= 0.0:
        seasonal = 1.0
    vol_mult = _safe_float(vol_multiplier)
    if vol_mult is None or vol_mult <= 0.0:
        vol_mult = 1.0
    spread *= seasonal * vol_mult

    if profile is not None:
        return profile.process_spread(spread)

    min_spread = _safe_float(getattr(cfg, "min_spread_bps", None))
    max_spread = _safe_float(getattr(cfg, "max_spread_bps", None))
    if min_spread is not None:
        spread = max(min_spread, spread)
    if max_spread is not None:
        spread = min(max_spread, spread)
    return spread


@dataclass
class SlippageCfg:
    k: float = 0.8
    min_half_spread_bps: float = 0.0
    default_spread_bps: float = 2.0
    eps: float = 1e-12
    dynamic: Optional[Any] = None
    dynamic_spread: Optional[Any] = None
    dynamic_impact: Optional[Any] = None
    tail_shock: Optional[Any] = None
    adv: Optional[Any] = None

    def get_dynamic_block(self) -> Optional[Any]:
        if self.dynamic is not None:
            return self.dynamic
        return self.dynamic_spread


class SlippageImpl:
    def __init__(self, cfg: SlippageCfg) -> None:
        self.cfg = cfg
        self._dynamic_profile: Optional[_DynamicSpreadProfile] = None
        dyn_cfg_obj: Optional[DynamicSpreadConfig] = None
        cfg_dict: Dict[str, Any] = {
            "k": float(cfg.k),
            "min_half_spread_bps": float(cfg.min_half_spread_bps),
            "default_spread_bps": float(cfg.default_spread_bps),
            "eps": float(cfg.eps),
        }
        if hasattr(cfg, "get_dynamic_block"):
            dyn_block = cfg.get_dynamic_block()
        else:
            dyn_block = getattr(cfg, "dynamic", None)
            if dyn_block is None:
                dyn_block = getattr(cfg, "dynamic_spread", None)
        dyn_dict: Optional[Dict[str, Any]] = None
        if dyn_block is not None:
            if DynamicSpreadConfig is not None and isinstance(
                dyn_block, DynamicSpreadConfig
            ):
                dyn_cfg_obj = dyn_block
                try:
                    dyn_dict = dyn_block.to_dict()
                except Exception:
                    dyn_dict = None
            elif hasattr(dyn_block, "to_dict"):
                try:
                    payload = dyn_block.to_dict()
                except Exception:
                    payload = None
                if isinstance(payload, Mapping):
                    dyn_dict = dict(payload)
                    if DynamicSpreadConfig is not None:
                        try:
                            dyn_cfg_obj = DynamicSpreadConfig.from_dict(dyn_dict)
                        except Exception:
                            logger.exception("Failed to parse dynamic spread config")
            elif isinstance(dyn_block, Mapping):
                dyn_dict = dict(dyn_block)
                if DynamicSpreadConfig is not None:
                    try:
                        dyn_cfg_obj = DynamicSpreadConfig.from_dict(dyn_dict)
                    except Exception:
                        logger.exception("Failed to parse dynamic spread config")
        if dyn_dict is not None:
            cfg_dict["dynamic"] = dict(dyn_dict)
            cfg_dict.setdefault("dynamic_spread", dict(dyn_dict))

        def _normalise_section(
            block: Any, cfg_cls: Optional[type]
        ) -> Optional[Dict[str, Any]]:
            if block is None:
                return None
            if cfg_cls is not None and isinstance(block, cfg_cls):
                try:
                    payload = block.to_dict()
                except Exception:
                    return None
                else:
                    return dict(payload)
            if hasattr(block, "to_dict"):
                try:
                    payload = block.to_dict()
                except Exception:
                    payload = None
                if isinstance(payload, Mapping):
                    return dict(payload)
            if isinstance(block, Mapping):
                return dict(block)
            return None

        extra_sections = (
            ("dynamic_impact", getattr(cfg, "dynamic_impact", None), DynamicImpactConfig),
            ("tail_shock", getattr(cfg, "tail_shock", None), TailShockConfig),
            ("adv", getattr(cfg, "adv", None), AdvConfig),
        )
        for key, block, cfg_cls in extra_sections:
            payload = _normalise_section(block, cfg_cls)
            if payload is not None:
                cfg_dict[key] = payload

        self._cfg_obj = (
            SlippageConfig.from_dict(cfg_dict)
            if SlippageConfig is not None
            else None
        )
        if dyn_cfg_obj is None and self._cfg_obj is not None:
            dyn_cfg_obj = getattr(self._cfg_obj, "dynamic_spread", None)
        if dyn_cfg_obj is not None and getattr(dyn_cfg_obj, "enabled", False):
            try:
                self._dynamic_profile = _DynamicSpreadProfile(
                    cfg=dyn_cfg_obj,
                    default_spread_bps=float(cfg.default_spread_bps),
                )
            except Exception:
                logger.exception("Failed to initialise dynamic spread profile")
                self._dynamic_profile = None

    @property
    def config(self):
        return self._cfg_obj

    @property
    def dynamic_profile(self) -> Optional[_DynamicSpreadProfile]:
        return self._dynamic_profile

    def attach_to(self, sim) -> None:
        if self._cfg_obj is not None:
            setattr(sim, "slippage_cfg", self._cfg_obj)
        try:
            setattr(sim, "_slippage_get_spread", self.get_spread_bps)
        except Exception:
            logger.exception("Failed to attach _slippage_get_spread to simulator")
        try:
            setattr(sim, "get_spread_bps", self.get_spread_bps)
        except Exception:
            logger.exception("Failed to attach get_spread_bps to simulator")

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SlippageImpl":
        dyn_cfg: Optional[Any] = None
        dyn_block: Optional[Any] = None
        for key in ("dynamic", "dynamic_spread"):
            candidate = d.get(key)
            if candidate is not None:
                dyn_block = candidate
                break
        if isinstance(dyn_block, dict):
            if DynamicSpreadConfig is not None:
                dyn_cfg = DynamicSpreadConfig.from_dict(dyn_block)
            else:
                dyn_cfg = dict(dyn_block)
        elif DynamicSpreadConfig is not None and isinstance(
            dyn_block, DynamicSpreadConfig
        ):
            dyn_cfg = dyn_block

        def _parse_section(block: Any, cfg_cls: Optional[type]) -> Optional[Any]:
            if block is None:
                return None
            if cfg_cls is not None and isinstance(block, cfg_cls):
                return block
            if isinstance(block, Mapping):
                if cfg_cls is not None:
                    return cfg_cls.from_dict(block)
                return dict(block)
            return None

        impact_cfg = _parse_section(d.get("dynamic_impact"), DynamicImpactConfig)
        tail_cfg = _parse_section(d.get("tail_shock"), TailShockConfig)
        adv_cfg = _parse_section(d.get("adv"), AdvConfig)

        return SlippageImpl(
            SlippageCfg(
                k=float(d.get("k", 0.8)),
                min_half_spread_bps=float(d.get("min_half_spread_bps", 0.0)),
                default_spread_bps=float(d.get("default_spread_bps", 2.0)),
                eps=float(d.get("eps", 1e-12)),
                dynamic=dyn_cfg,
                dynamic_spread=dyn_cfg,
                dynamic_impact=impact_cfg,
                tail_shock=tail_cfg,
                adv=adv_cfg,
            )
        )

    def get_spread_bps(
        self,
        *,
        ts_ms: Any,
        base_spread_bps: Optional[float] = None,
        vol_factor: Optional[float] = None,
        bar_high: Any = None,
        bar_low: Any = None,
        mid_price: Any = None,
        vol_metrics: Optional[Mapping[str, Any]] = None,
    ) -> float:
        base = float(self.cfg.default_spread_bps)
        if base_spread_bps is not None:
            try:
                candidate = float(base_spread_bps)
            except (TypeError, ValueError):
                candidate = base
            else:
                if math.isfinite(candidate) and candidate > 0.0:
                    base = candidate
        vol_multiplier = 1.0
        if vol_factor is not None:
            try:
                vf = float(vol_factor)
            except (TypeError, ValueError):
                vf = 1.0
            else:
                if math.isfinite(vf) and vf > 0.0:
                    vol_multiplier = vf
        profile = self._dynamic_profile
        if profile is not None and getattr(profile._cfg, "enabled", False):
            seasonal_multiplier = profile.seasonal_multiplier(ts_ms)
            try:
                dynamic_spread = _calc_dynamic_spread(
                    cfg=profile._cfg,
                    default_spread_bps=base,
                    bar_high=bar_high,
                    bar_low=bar_low,
                    mid_price=mid_price,
                    vol_metrics=vol_metrics,
                    seasonal_multiplier=seasonal_multiplier,
                    vol_multiplier=vol_multiplier,
                    profile=profile,
                )
            except Exception:
                logger.exception("Dynamic spread computation failed")
            else:
                if dynamic_spread is not None:
                    return float(dynamic_spread)
            logger.debug(
                "Dynamic spread fell back to default profile computation for ts=%r",
                ts_ms,
            )
            try:
                return float(
                    profile.compute(
                        ts_ms=ts_ms,
                        base_spread_bps=base,
                        vol_multiplier=vol_multiplier,
                    )
                )
            except Exception:
                logger.exception("Dynamic spread fallback computation failed")
        return float(base * vol_multiplier)
