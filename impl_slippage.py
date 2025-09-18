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
    from slippage import SlippageConfig, DynamicSpreadConfig
except Exception:  # pragma: no cover
    SlippageConfig = None  # type: ignore
    DynamicSpreadConfig = None  # type: ignore

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
        spread = self._apply_clamp(spread)
        return self._apply_smoothing(spread)


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
        self._dynamic_profile: Optional[_DynamicSpreadProfile] = None
        dyn_cfg_obj: Optional[DynamicSpreadConfig] = None
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
                if DynamicSpreadConfig is not None and isinstance(
                    dyn, DynamicSpreadConfig
                ):
                    dyn_cfg_obj = dyn
            elif isinstance(dyn, dict):
                dyn_dict = dict(dyn)
                if DynamicSpreadConfig is not None:
                    try:
                        dyn_cfg_obj = DynamicSpreadConfig.from_dict(dyn_dict)
                    except Exception:
                        logger.exception("Failed to parse dynamic spread config")
            else:
                dyn_dict = None
            if dyn_dict is not None:
                cfg_dict["dynamic_spread"] = dyn_dict
        elif DynamicSpreadConfig is not None and hasattr(cfg, "dynamic_spread"):
            dyn_val = getattr(cfg, "dynamic_spread")
            if isinstance(dyn_val, DynamicSpreadConfig):
                dyn_cfg_obj = dyn_val

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

    def attach_to(self, sim) -> None:
        if self._cfg_obj is not None:
            setattr(sim, "slippage_cfg", self._cfg_obj)
        try:
            setattr(sim, "get_spread_bps", self.get_spread_bps)
        except Exception:
            logger.exception("Failed to attach get_spread_bps to simulator")

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

    def get_spread_bps(
        self,
        *,
        ts_ms: Any,
        base_spread_bps: Optional[float] = None,
        vol_factor: Optional[float] = None,
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
        if profile is not None:
            try:
                return float(
                    profile.compute(
                        ts_ms=ts_ms,
                        base_spread_bps=base,
                        vol_multiplier=vol_multiplier,
                    )
                )
            except Exception:
                logger.exception("Dynamic spread computation failed")
        return float(base * vol_multiplier)
