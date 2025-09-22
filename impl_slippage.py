# -*- coding: utf-8 -*-
"""
impl_slippage.py
Обёртка над slippage.SlippageConfig и функциями оценки. Подключает конфиг к симулятору.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from adv_store import ADVStore

try:  # pragma: no cover - optional dependency when YAML support is unavailable
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

try:
    from slippage import (
        SlippageConfig,
        DynamicSpreadConfig,
        DynamicImpactConfig,
        TailShockConfig,
        AdvConfig,
        CalibratedProfilesConfig,
        SymbolCalibratedProfile,
        CalibratedHourlyProfile,
        CalibratedRegimeOverride,
        _estimate_calibrated_slippage,
    )
except Exception:  # pragma: no cover
    SlippageConfig = None  # type: ignore
    DynamicSpreadConfig = None  # type: ignore
    DynamicImpactConfig = None  # type: ignore
    TailShockConfig = None  # type: ignore
    AdvConfig = None  # type: ignore
    CalibratedProfilesConfig = None  # type: ignore
    SymbolCalibratedProfile = None  # type: ignore
    CalibratedHourlyProfile = None  # type: ignore
    CalibratedRegimeOverride = None  # type: ignore
    _estimate_calibrated_slippage = None  # type: ignore

try:  # pragma: no cover - optional during tests
    from core_config import AdvRuntimeConfig
except Exception:  # pragma: no cover
    AdvRuntimeConfig = None  # type: ignore

try:
    from utils_time import get_hourly_multiplier, watch_seasonality_file
except Exception:  # pragma: no cover
    def get_hourly_multiplier(ts_ms, multipliers, *, interpolate=False):  # type: ignore
        return 1.0

    def watch_seasonality_file(path, callback, *, poll_interval=60.0):  # type: ignore
        return None


from services.costs import MakerTakerShareSettings


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


def _safe_non_negative_float(value: Any, default: float = 0.0) -> float:
    num = _safe_float(value)
    if num is None or num < 0.0:
        return float(default)
    return float(num)


def _safe_share_value(value: Any, default: float = 0.5) -> float:
    share = _safe_float(value)
    if share is None:
        share = float(default)
    if share < 0.0:
        share = 0.0
    elif share > 1.0:
        share = 1.0
    return float(share)


def _safe_positive_int(value: Any) -> Optional[int]:
    try:
        num = int(value)
    except (TypeError, ValueError):
        return None
    if num <= 0:
        return None
    return num


def _parse_generated_timestamp(value: Any) -> Optional[int]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    cleaned = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    try:
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def load_calibration_artifact(
    path: str,
    *,
    default_symbol: Optional[str] = None,
    symbols: Optional[Sequence[str]] = None,
    enabled: bool = True,
) -> Optional[Dict[str, Any]]:
    """Load a calibrated slippage artifact and normalise it for ``SlippageImpl``.

    The helper accepts JSON or YAML payloads produced by ``calibrate_live_slippage``
    and converts them into a structure consumable by
    :class:`CalibratedProfilesConfig`.
    """

    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw_text = fh.read()
    except FileNotFoundError:
        logger.warning("Slippage calibration artifact not found: %s", path)
        return None
    except Exception:
        logger.exception("Failed to read slippage calibration artifact: %s", path)
        return None

    data: Any
    try:
        lower = path.lower()
        if lower.endswith((".yaml", ".yml")) and yaml is not None:
            data = yaml.safe_load(raw_text)  # type: ignore[call-arg]
        else:
            data = json.loads(raw_text)
    except Exception:
        logger.exception("Failed to parse slippage calibration artifact: %s", path)
        return None

    if not isinstance(data, Mapping):
        logger.warning(
            "Slippage calibration artifact %s has unexpected structure (expected mapping)",
            path,
        )
        return None

    symbols_block = data.get("symbols")
    if not isinstance(symbols_block, Mapping) or not symbols_block:
        logger.warning(
            "Slippage calibration artifact %s does not contain symbol profiles", path
        )
        return None

    symbols_filter: Optional[set[str]] = None
    if symbols:
        symbols_filter = {
            str(sym).upper()
            for sym in symbols
            if sym is not None and str(sym).strip()
        }
        if not symbols_filter:
            symbols_filter = None

    normalised_symbols: Dict[str, Dict[str, Any]] = {}
    for key, entry in symbols_block.items():
        if not isinstance(entry, Mapping):
            continue
        symbol_key = str(key).upper()
        if symbols_filter and symbol_key not in symbols_filter:
            continue
        metadata: Dict[str, Any] = {}
        for meta_key in ("samples", "impact_mean_bps", "impact_std_bps"):
            if meta_key in entry:
                metadata[meta_key] = entry[meta_key]
        exec_counts = entry.get("execution_profile_counts")
        if isinstance(exec_counts, Mapping):
            try:
                metadata["execution_profile_counts"] = dict(exec_counts)
            except Exception:
                metadata["execution_profile_counts"] = exec_counts
        profile_payload: Dict[str, Any] = {"symbol": symbol_key}
        if entry.get("notional_curve") is not None:
            profile_payload["notional_curve"] = entry.get("notional_curve")
        if entry.get("hourly_multipliers") is not None:
            profile_payload["hourly_multipliers"] = entry.get("hourly_multipliers")
        if entry.get("regime_multipliers") is not None:
            profile_payload["regime_multipliers"] = entry.get("regime_multipliers")
        for key_name in ("k", "default_spread_bps", "min_half_spread_bps"):
            if entry.get(key_name) is not None:
                profile_payload[key_name] = entry.get(key_name)
        if metadata:
            profile_payload["metadata"] = metadata
        normalised_symbols[symbol_key] = profile_payload

    has_profiles = bool(normalised_symbols)
    if not has_profiles:
        logger.warning(
            "Slippage calibration artifact %s did not yield usable symbol profiles",
            path,
        )

    generated_at = data.get("generated_at")
    last_refresh_ts = _parse_generated_timestamp(generated_at)
    metadata_payload = {
        "generated_at": generated_at,
        "source_files": data.get("source_files"),
        "regime_column": data.get("regime_column"),
        "total_samples": data.get("total_samples"),
        "artifact_path": path,
    }

    config_payload: Dict[str, Any] = {
        "enabled": bool(enabled) and has_profiles,
        "path": path,
        "symbols": normalised_symbols,
        "metadata": metadata_payload,
    }
    if last_refresh_ts is not None:
        config_payload["last_refresh_ts"] = last_refresh_ts

    default_candidate = None
    if default_symbol:
        candidate = str(default_symbol).upper()
        if candidate in normalised_symbols:
            default_candidate = candidate
    if default_candidate is None and len(normalised_symbols) == 1:
        default_candidate = next(iter(normalised_symbols))
    if default_candidate is not None:
        config_payload["default_symbol"] = default_candidate

    return config_payload


def _cfg_attr(block: Any, key: str, default: Any = None) -> Any:
    if block is None:
        return default
    if isinstance(block, Mapping):
        return block.get(key, default)
    return getattr(block, key, default)


def _lookup_metric(metrics: Mapping[str, Any], key: str) -> Any:
    if key in metrics:
        return metrics[key]
    key_lower = key.lower()
    for metric_key, value in metrics.items():
        if isinstance(metric_key, str) and metric_key.lower() == key_lower:
            return value
    return None


def _clamp(value: float, minimum: Optional[float], maximum: Optional[float]) -> float:
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


@dataclass
class _TradeCostState:
    impact_cfg: Optional[Any] = None
    tail_cfg: Optional[Any] = None
    adv_cfg: Optional[Any] = None
    adv_store: Optional[ADVStore] = None
    vol_window: Optional[int] = None
    participation_window: Optional[int] = None
    zscore_clip: Optional[float] = None
    smoothing_alpha: Optional[float] = None
    vol_metric: Optional[str] = None
    participation_metric: Optional[str] = None
    vol_history: deque[float] = field(default_factory=deque)
    participation_history: deque[float] = field(default_factory=deque)
    k_ema: Optional[float] = None
    adv_cache: Dict[str, float] = field(default_factory=dict)

    def reset(self, *, reset_store: bool = True) -> None:
        self.vol_history.clear()
        self.participation_history.clear()
        self.k_ema = None
        self.adv_cache.clear()
        if reset_store and self.adv_store is not None:
            try:
                self.adv_store.reset_runtime_state()
            except Exception:
                logger.exception("Failed to reset ADV store runtime state")

    def _normalise(
        self,
        history: deque[float],
        window: Optional[int],
        value: Optional[float],
    ) -> Optional[float]:
        if value is None:
            return None
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(val):
            return None
        if window is None or window <= 1:
            history.clear()
            return val
        history.append(val)
        if len(history) > window:
            history.popleft()
        if len(history) <= 1:
            return 0.0
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / max(len(history) - 1, 1)
        std = math.sqrt(variance) if variance > 0.0 else 0.0
        if std <= 0.0:
            return 0.0
        return (val - mean) / std

    def normalise_vol(self, value: Optional[float]) -> Optional[float]:
        norm = self._normalise(self.vol_history, self.vol_window, value)
        if norm is None:
            return None
        clip = self.zscore_clip
        if clip is not None and clip > 0.0:
            norm = max(-clip, min(clip, norm))
        return norm

    def normalise_part(self, value: Optional[float]) -> Optional[float]:
        norm = self._normalise(
            self.participation_history, self.participation_window, value
        )
        if norm is None:
            return None
        clip = self.zscore_clip
        if clip is not None and clip > 0.0:
            norm = max(-clip, min(clip, norm))
        return norm

    def apply_k_smoothing(self, value: float) -> float:
        try:
            k_val = float(value)
        except (TypeError, ValueError):
            return value
        alpha = self.smoothing_alpha
        if alpha is None or alpha <= 0.0:
            self.k_ema = k_val
            return k_val
        if alpha >= 1.0:
            self.k_ema = k_val
            return k_val
        prev = self.k_ema
        if prev is None:
            self.k_ema = k_val
            return k_val
        smoothed = alpha * k_val + (1.0 - alpha) * prev
        self.k_ema = smoothed
        return smoothed

def _tail_rng_seed(
    *,
    symbol: Optional[str],
    ts: Any,
    side: Any,
    order_seq: Any,
    seed: Any,
) -> int:
    try:
        ts_val = int(ts) if ts is not None else 0
    except (TypeError, ValueError):
        ts_val = 0
    try:
        seq_val = int(order_seq) if order_seq is not None else 0
    except (TypeError, ValueError):
        seq_val = 0
    try:
        seed_val = int(seed) if seed is not None else 0
    except (TypeError, ValueError):
        seed_val = 0
    key = "|".join(
        (
            str(symbol or ""),
            str(ts_val),
            str(side).upper(),
            str(seq_val),
            str(seed_val),
        )
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _tail_percentile_sample(
    extra: Mapping[str, Any], rng: random.Random, default_bps: float
) -> float:
    percentiles_raw = extra.get("percentiles") if isinstance(extra, Mapping) else None
    weights_raw = extra.get("weights") if isinstance(extra, Mapping) else None
    choices: list[tuple[float, float]] = []
    if isinstance(percentiles_raw, Mapping):
        for key, value in percentiles_raw.items():
            candidate = _safe_float(value)
            if candidate is None:
                continue
            weight_val = 1.0
            if isinstance(weights_raw, Mapping) and key in weights_raw:
                weight_candidate = _safe_float(weights_raw[key])
                if weight_candidate is not None and weight_candidate > 0.0:
                    weight_val = weight_candidate
            if weight_val <= 0.0:
                continue
            choices.append((float(candidate), float(weight_val)))
    if not choices:
        return float(default_bps)
    total_weight = sum(weight for _, weight in choices)
    if total_weight <= 0.0:
        return float(default_bps)
    pick = rng.random() * total_weight
    cumulative = 0.0
    for value, weight in choices:
        cumulative += weight
        if pick <= cumulative:
            return float(value)
    return float(choices[-1][0])


def _tail_gaussian_sample(
    extra: Mapping[str, Any], rng: random.Random, default_bps: float
) -> float:
    mean = _safe_float(extra.get("gaussian_mean_bps")) if isinstance(extra, Mapping) else None
    std = _safe_float(extra.get("gaussian_std_bps")) if isinstance(extra, Mapping) else None
    if mean is None:
        mean = float(default_bps)
    if std is None or std <= 0.0:
        std = abs(float(default_bps)) if default_bps else 0.0
    sample = float(mean)
    if std > 0.0:
        sample = rng.gauss(float(mean), float(std))
    clip_low: Optional[float] = None
    clip_high: Optional[float] = None
    if isinstance(extra, Mapping):
        clip_block = extra.get("gaussian_clip_bps")
        if isinstance(clip_block, Mapping):
            clip_low = _safe_float(
                clip_block.get("min")
                or clip_block.get("low")
                or clip_block.get("p05")
                or clip_block.get("p5")
            )
            clip_high = _safe_float(
                clip_block.get("max")
                or clip_block.get("high")
                or clip_block.get("p95")
                or clip_block.get("p99")
            )
        if clip_low is None or clip_high is None:
            percentiles_block = extra.get("percentiles")
            if isinstance(percentiles_block, Mapping):
                if clip_low is None:
                    for key in ("p05", "p5", "low", "min"):
                        if key in percentiles_block:
                            clip_low = _safe_float(percentiles_block[key])
                            if clip_low is not None:
                                break
                if clip_high is None:
                    for key in ("p95", "p99", "high", "max"):
                        if key in percentiles_block:
                            clip_high = _safe_float(percentiles_block[key])
                            if clip_high is not None:
                                break
    if clip_low is not None:
        sample = max(clip_low, sample)
    if clip_high is not None:
        sample = min(clip_high, sample)
    return float(sample)


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

    def _finalise_spread(self, spread_bps: float, *, already_clamped: bool = False) -> float:
        candidate = float(spread_bps)
        if not already_clamped:
            candidate = self._apply_clamp(candidate)
        return self._apply_smoothing(candidate)

    def seasonal_multiplier(self, ts_ms: Any) -> float:
        """Public wrapper used by the dynamic spread helper."""

        return self._seasonal_multiplier(ts_ms)

    def process_spread(self, spread_bps: Any, *, already_clamped: bool = False) -> float:
        """Clamp and smooth the supplied spread value."""

        candidate = _safe_float(spread_bps)
        if candidate is None:
            candidate = self._base_spread_bps
        result = _calc_dynamic_spread(
            cfg=self._cfg,
            default_spread_bps=self._base_spread_bps,
            bar_high=None,
            bar_low=None,
            mid_price=None,
            seasonal_multiplier=1.0,
            vol_multiplier=1.0,
            profile=self,
            raw_spread_bps=candidate,
            already_clamped=already_clamped,
        )
        if result is None:
            # fall back to direct smoothing if helper rejected the value
            return self._finalise_spread(candidate, already_clamped=already_clamped)
        return float(result)

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
        result = _calc_dynamic_spread(
            cfg=self._cfg,
            default_spread_bps=self._base_spread_bps,
            bar_high=None,
            bar_low=None,
            mid_price=None,
            seasonal_multiplier=seasonal,
            vol_multiplier=vol_multiplier,
            profile=self,
            raw_spread_bps=base,
        )
        if result is None:
            spread = base * seasonal * float(vol_multiplier)
            return self._finalise_spread(spread)
        return float(result)


    def metadata(self) -> Mapping[str, Any]:
        with self._lock:
            self._ensure_loaded_locked()
            return dict(self._meta)


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
    raw_spread_bps: Optional[float] = None,
    already_clamped: bool = False,
) -> Optional[float]:
    """Compute a dynamic spread using the range-based heuristic.

    The helper returns ``None`` when the supplied inputs are insufficient or the
    resulting value is invalid, allowing callers to fall back to default
    behaviour.
    """

    def _finalise_value(value: float) -> Optional[float]:
        if profile is not None:
            result = profile._finalise_spread(value, already_clamped=already_clamped)
        else:
            candidate = float(value)
            if not already_clamped:
                min_spread = _safe_float(getattr(cfg, "min_spread_bps", None))
                max_spread = _safe_float(getattr(cfg, "max_spread_bps", None))
                if min_spread is not None:
                    candidate = max(min_spread, candidate)
                if max_spread is not None:
                    candidate = min(max_spread, candidate)
            smoothing_alpha = _safe_float(getattr(cfg, "smoothing_alpha", None))
            if smoothing_alpha is not None:
                if smoothing_alpha <= 0.0:
                    smoothing_alpha = None
                elif smoothing_alpha >= 1.0:
                    smoothing_alpha = 1.0
            if smoothing_alpha is not None:
                prev = getattr(cfg, "_ema_prev_spread", None)
                if prev is None or not math.isfinite(prev):
                    ema_val = candidate
                else:
                    ema_val = smoothing_alpha * candidate + (1.0 - smoothing_alpha) * prev
                setattr(cfg, "_ema_prev_spread", float(ema_val))
                candidate = ema_val
            result = candidate
        if not math.isfinite(result):
            return None
        if result < 0.0:
            result = 0.0
        return float(result)

    alpha = _safe_float(getattr(cfg, "alpha_bps", None))
    if alpha is None:
        alpha = _safe_float(default_spread_bps) or 0.0
    beta = _safe_float(getattr(cfg, "beta_coef", None))
    if beta is None:
        beta = 0.0

    base_spread: Optional[float]
    if raw_spread_bps is not None:
        base_spread = _safe_float(raw_spread_bps)
        if base_spread is None:
            return None
    else:
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

        base_spread = alpha + beta * range_ratio_bps

    if base_spread is None:
        return None

    seasonal = _safe_float(seasonal_multiplier)
    if seasonal is None or seasonal <= 0.0:
        seasonal = 1.0
    vol_mult = _safe_float(vol_multiplier)
    if vol_mult is None or vol_mult <= 0.0:
        vol_mult = 1.0

    adjusted = base_spread * seasonal * vol_mult
    if not math.isfinite(adjusted):
        return None

    return _finalise_value(adjusted)


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

    @staticmethod
    def _normalise_dynamic(block: Any) -> Optional[Any]:
        if block is None:
            return None
        if DynamicSpreadConfig is None:
            return block
        if isinstance(block, DynamicSpreadConfig):
            return block
        candidate: Any = block
        if hasattr(block, "to_dict"):
            try:
                payload = block.to_dict()
            except Exception:
                payload = None
            else:
                if isinstance(payload, Mapping):
                    candidate = dict(payload)
        if isinstance(candidate, Mapping):
            try:
                return DynamicSpreadConfig.from_dict(dict(candidate))
            except Exception:
                logger.exception("Failed to normalise dynamic spread config")
                return dict(candidate)
        return block

    def __post_init__(self) -> None:
        dyn_source: Any = self.dynamic if self.dynamic is not None else self.dynamic_spread
        normalised = self._normalise_dynamic(dyn_source)
        if isinstance(normalised, DynamicSpreadConfig):
            self.dynamic = normalised
            self.dynamic_spread = normalised
        else:
            if self.dynamic is None and self.dynamic_spread is not None:
                self.dynamic = self.dynamic_spread
            if self.dynamic_spread is None and self.dynamic is not None:
                self.dynamic_spread = self.dynamic

    def get_dynamic_block(self) -> Optional[Any]:
        if self.dynamic is not None:
            return self.dynamic
        return self.dynamic_spread

    def dynamic_trade_cost_enabled(self) -> bool:
        block = self.get_dynamic_block()
        if block is None:
            return False
        enabled = _cfg_attr(block, "enabled")
        try:
            return bool(enabled)
        except Exception:
            return False


class SlippageImpl:
    def __init__(self, cfg: SlippageCfg, *, run_config: Any | None = None) -> None:
        self.cfg = cfg
        self._symbol: Optional[str] = None
        self._dynamic_profile: Optional[_DynamicSpreadProfile] = None
        self._adv_store: Optional[ADVStore] = None
        self._adv_runtime_cfg: Optional[Any] = None
        self._maker_taker_share_raw: Optional[Dict[str, Any]] = None
        self.maker_taker_share_cfg: Optional[MakerTakerShareSettings] = None
        self._maker_taker_share_enabled: bool = False
        self._maker_share_default: float = 0.5
        self._spread_cost_maker_bps_default: float = 0.0
        self._spread_cost_taker_bps_default: float = 0.0
        self._last_trade_cost_meta: Dict[str, Any] = {}
        self._calibrated_cfg: Optional[Any] = None
        self._calibration_symbols: Dict[str, SymbolCalibratedProfile] = {}
        self._calibration_global_hourly: Optional[CalibratedHourlyProfile] = None
        self._calibration_global_regime: Dict[str, CalibratedRegimeOverride] = {}
        self._calibration_base_dir: Optional[str] = None
        self._calibration_lock = threading.Lock()
        self._current_market_regime: Any = None
        self._calibration_enabled: bool = False
        dyn_cfg_obj: Optional[DynamicSpreadConfig] = None
        adv_cfg_obj: Optional[Any] = None
        impact_cfg_obj: Optional[Any] = None
        tail_cfg_obj: Optional[Any] = None
        calibrated_cfg_obj: Optional[Any] = None
        adv_runtime_payload: Dict[str, Any] = {}
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

        share_block: Any = None
        if run_config is not None:
            if isinstance(run_config, Mapping):
                share_block = run_config.get("maker_taker_share")
            else:
                share_block = getattr(run_config, "maker_taker_share", None)
        share_cfg = MakerTakerShareSettings.parse(share_block)
        self.maker_taker_share_cfg = share_cfg
        if share_cfg is not None:
            self._maker_taker_share_raw = share_cfg.as_dict()
        elif isinstance(share_block, Mapping):
            self._maker_taker_share_raw = dict(share_block)

        share_enabled = False
        maker_share_default = self._maker_share_default
        maker_spread_cost = self._spread_cost_maker_bps_default
        taker_spread_cost = self._spread_cost_taker_bps_default
        if share_cfg is not None:
            share_enabled = bool(share_cfg.enabled)
            maker_share_default = _safe_share_value(
                share_cfg.maker_share_default, maker_share_default
            )
            maker_spread_cost = _safe_non_negative_float(
                share_cfg.spread_cost_maker_bps, maker_spread_cost
            )
            taker_spread_cost = _safe_non_negative_float(
                share_cfg.spread_cost_taker_bps, taker_spread_cost
            )
        elif isinstance(share_block, Mapping):
            share_enabled = bool(share_block.get("enabled", False))
            maker_share_default = _safe_share_value(
                share_block.get("maker_share_default"), maker_share_default
            )
            maker_spread_cost = _safe_non_negative_float(
                share_block.get("spread_cost_maker_bps"), maker_spread_cost
            )
            taker_spread_cost = _safe_non_negative_float(
                share_block.get("spread_cost_taker_bps"), taker_spread_cost
            )
        self._maker_taker_share_enabled = share_enabled
        self._maker_share_default = maker_share_default
        self._spread_cost_maker_bps_default = maker_spread_cost
        self._spread_cost_taker_bps_default = taker_spread_cost
        if self._maker_taker_share_raw is not None:
            self._maker_taker_share_raw["enabled"] = share_enabled
            self._maker_taker_share_raw["maker_share_default"] = maker_share_default
            self._maker_taker_share_raw["spread_cost_maker_bps"] = maker_spread_cost
            self._maker_taker_share_raw["spread_cost_taker_bps"] = taker_spread_cost

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
            (
                "calibrated_profiles",
                getattr(cfg, "calibrated_profiles", None),
                CalibratedProfilesConfig,
            ),
        )
        for key, block, cfg_cls in extra_sections:
            payload = _normalise_section(block, cfg_cls)
            if payload is not None:
                cfg_dict[key] = payload
            if key == "dynamic_impact":
                if DynamicImpactConfig is not None and isinstance(block, DynamicImpactConfig):
                    impact_cfg_obj = block
                elif isinstance(payload, Mapping) and DynamicImpactConfig is not None:
                    try:
                        impact_cfg_obj = DynamicImpactConfig.from_dict(dict(payload))
                    except Exception:
                        logger.exception("Failed to parse dynamic impact config")
                        impact_cfg_obj = dict(payload)
                elif isinstance(payload, Mapping) and impact_cfg_obj is None:
                    impact_cfg_obj = dict(payload)
            elif key == "tail_shock":
                if TailShockConfig is not None and isinstance(block, TailShockConfig):
                    tail_cfg_obj = block
                elif isinstance(payload, Mapping) and TailShockConfig is not None:
                    try:
                        tail_cfg_obj = TailShockConfig.from_dict(dict(payload))
                    except Exception:
                        logger.exception("Failed to parse tail shock config")
                        tail_cfg_obj = dict(payload)
                elif isinstance(payload, Mapping) and tail_cfg_obj is None:
                    tail_cfg_obj = dict(payload)
            if key == "adv":
                if AdvConfig is not None and isinstance(block, AdvConfig):
                    adv_cfg_obj = block
                elif isinstance(payload, Mapping) and AdvConfig is not None:
                    try:
                        adv_cfg_obj = AdvConfig.from_dict(dict(payload))
                    except Exception:
                        logger.exception("Failed to parse ADV config")
                        adv_cfg_obj = None
                elif isinstance(payload, Mapping) and adv_cfg_obj is None:
                    adv_cfg_obj = dict(payload)
            elif key == "calibrated_profiles":
                if CalibratedProfilesConfig is not None and isinstance(
                    block, CalibratedProfilesConfig
                ):
                    calibrated_cfg_obj = block
                elif isinstance(payload, Mapping) and CalibratedProfilesConfig is not None:
                    try:
                        calibrated_cfg_obj = CalibratedProfilesConfig.from_dict(dict(payload))
                    except Exception:
                        logger.exception("Failed to parse calibrated slippage profiles")
                        calibrated_cfg_obj = dict(payload)
                elif isinstance(payload, Mapping) and calibrated_cfg_obj is None:
                    calibrated_cfg_obj = dict(payload)

        self._cfg_obj = (
            SlippageConfig.from_dict(cfg_dict)
            if SlippageConfig is not None
            else None
        )
        if dyn_cfg_obj is None and self._cfg_obj is not None:
            dyn_cfg_obj = getattr(self._cfg_obj, "dynamic_spread", None)
        self._adv_cfg = adv_cfg_obj
        if self._adv_cfg is None and self._cfg_obj is not None:
            candidate = getattr(self._cfg_obj, "adv", None)
            if candidate is not None:
                self._adv_cfg = candidate
        self._impact_cfg = impact_cfg_obj
        self._tail_cfg = tail_cfg_obj
        calibrated_candidate = calibrated_cfg_obj
        if calibrated_candidate is None and self._cfg_obj is not None:
            calibrated_candidate = getattr(self._cfg_obj, "calibrated_profiles", None)
        if CalibratedProfilesConfig is not None and isinstance(
            calibrated_candidate, CalibratedProfilesConfig
        ):
            self._calibrated_cfg = calibrated_candidate
        elif isinstance(calibrated_candidate, Mapping) and CalibratedProfilesConfig is not None:
            try:
                self._calibrated_cfg = CalibratedProfilesConfig.from_dict(
                    dict(calibrated_candidate)
                )
            except Exception:
                logger.exception("Failed to interpret calibrated profiles mapping")
                self._calibrated_cfg = None
        elif isinstance(calibrated_candidate, CalibratedProfilesConfig):
            self._calibrated_cfg = calibrated_candidate
        else:
            self._calibrated_cfg = None
        legacy_adv_overrides = self._adv_cfg

        def _adv_runtime_dict(block: Any) -> Dict[str, Any]:
            if block is None:
                return {}
            if AdvRuntimeConfig is not None and isinstance(block, AdvRuntimeConfig):
                try:
                    payload = block.dict(exclude_unset=False)
                except Exception:
                    payload = {}
                if isinstance(payload, Mapping):
                    return dict(payload)
                return {}
            if hasattr(block, "dict"):
                try:
                    payload = block.dict(exclude_unset=False)  # type: ignore[call-arg]
                except Exception:
                    payload = {}
                if isinstance(payload, Mapping):
                    return dict(payload)
            if isinstance(block, Mapping):
                return dict(block)
            return {}

        def _legacy_adv_runtime(block: Any) -> Dict[str, Any]:
            overrides: Dict[str, Any] = {}
            if block is None:
                return overrides
            enabled_val = _cfg_attr(block, "enabled")
            if enabled_val is not None:
                try:
                    overrides["enabled"] = bool(enabled_val)
                except Exception:
                    pass
            fallback_adv = _safe_float(_cfg_attr(block, "fallback_adv"))
            if fallback_adv is not None and fallback_adv > 0.0:
                overrides.setdefault("default_quote", fallback_adv)
            min_adv = _safe_float(_cfg_attr(block, "min_adv"))
            if min_adv is not None and min_adv > 0.0:
                overrides.setdefault("floor_quote", min_adv)
            refresh_days = _safe_positive_int(_cfg_attr(block, "refresh_days"))
            if refresh_days is not None:
                overrides.setdefault("refresh_days", refresh_days)
            window_days = _safe_positive_int(_cfg_attr(block, "window_days"))
            if window_days is not None:
                overrides.setdefault("window_days", window_days)
            seasonality_path = _cfg_attr(block, "seasonality_path")
            if seasonality_path is not None:
                overrides.setdefault("seasonality_path", seasonality_path)
            seasonality_profile = _cfg_attr(block, "profile_kind")
            if seasonality_profile is not None:
                overrides.setdefault("seasonality_profile", seasonality_profile)
            override_path = _cfg_attr(block, "override_path")
            if override_path is not None:
                overrides["path"] = override_path
            extra = _cfg_attr(block, "extra")
            if isinstance(extra, Mapping):
                for key in ("quote_path", "adv_path", "path", "data_path"):
                    candidate = extra.get(key)
                    if candidate is not None and "path" not in overrides:
                        overrides["path"] = candidate
                dataset = extra.get("dataset")
                if dataset is not None:
                    overrides.setdefault("dataset", dataset)
                missing_policy = extra.get("missing_symbol_policy")
                if missing_policy is not None:
                    overrides.setdefault("missing_symbol_policy", missing_policy)
                default_quote = _safe_float(extra.get("default_quote"))
                if default_quote is not None and default_quote > 0.0:
                    overrides.setdefault("default_quote", default_quote)
                floor_quote = _safe_float(extra.get("floor_quote"))
                if floor_quote is not None and floor_quote > 0.0:
                    overrides.setdefault("floor_quote", floor_quote)
                refresh_extra = _safe_positive_int(extra.get("refresh_days"))
                if refresh_extra is not None and "refresh_days" not in overrides:
                    overrides["refresh_days"] = refresh_extra
                auto_refresh = _safe_positive_int(extra.get("auto_refresh_days"))
                if auto_refresh is not None and "refresh_days" not in overrides:
                    overrides["refresh_days"] = auto_refresh
                window_extra = _safe_positive_int(extra.get("window_days"))
                if window_extra is not None and "window_days" not in overrides:
                    overrides["window_days"] = window_extra
                seasonality_profile_extra = extra.get("seasonality_profile")
                if seasonality_profile_extra is not None:
                    overrides.setdefault("seasonality_profile", seasonality_profile_extra)
            return overrides

        adv_runtime_payload.update(_legacy_adv_runtime(legacy_adv_overrides))
        if run_config is not None:
            adv_block = getattr(run_config, "adv", None)
            for key, value in _adv_runtime_dict(adv_block).items():
                if value is None:
                    continue
                adv_runtime_payload[key] = value
        adv_runtime_cfg_obj: Optional[Any] = None
        if adv_runtime_payload:
            if AdvRuntimeConfig is not None:
                try:
                    adv_runtime_cfg_obj = AdvRuntimeConfig.parse_obj(adv_runtime_payload)
                except Exception:
                    logger.exception("Failed to parse runtime ADV config")
                    adv_runtime_cfg_obj = dict(adv_runtime_payload)
            else:
                adv_runtime_cfg_obj = dict(adv_runtime_payload)
        self._adv_runtime_cfg = adv_runtime_cfg_obj
        adv_enabled_val = None
        if self._adv_runtime_cfg is not None:
            adv_enabled_val = _cfg_attr(self._adv_runtime_cfg, "enabled")
        if adv_enabled_val is None:
            adv_enabled_val = _cfg_attr(self._adv_cfg, "enabled")
        try:
            adv_enabled = bool(adv_enabled_val)
        except Exception:
            adv_enabled = False
        if adv_enabled and (self._adv_runtime_cfg is not None or adv_runtime_payload):
            store_source: Any = self._adv_runtime_cfg
            if store_source is None:
                store_source = dict(adv_runtime_payload)
            try:
                self._adv_store = ADVStore(store_source)
            except Exception:
                logger.exception("Failed to initialise ADV store")
                self._adv_store = None
        if dyn_cfg_obj is not None and getattr(dyn_cfg_obj, "enabled", False):
            try:
                self._dynamic_profile = _DynamicSpreadProfile(
                    cfg=dyn_cfg_obj,
                    default_spread_bps=float(cfg.default_spread_bps),
                )
            except Exception:
                logger.exception("Failed to initialise dynamic spread profile")
                self._dynamic_profile = None
        impact_vol_metric = _cfg_attr(self._impact_cfg, "vol_metric")
        part_metric = _cfg_attr(self._impact_cfg, "participation_metric")
        def _normalise_str(value: Any) -> Optional[str]:
            if value is None:
                return None
            try:
                text = str(value).strip()
            except Exception:
                return None
            return text.lower() if text else None

        def _positive_int(value: Any) -> Optional[int]:
            window = _safe_positive_int(value)
            return window

        smoothing_alpha = _safe_float(_cfg_attr(self._impact_cfg, "smoothing_alpha"))
        if smoothing_alpha is not None:
            if smoothing_alpha <= 0.0:
                smoothing_alpha = None
            elif smoothing_alpha >= 1.0:
                smoothing_alpha = 1.0
        zscore_clip = _safe_float(_cfg_attr(self._impact_cfg, "zscore_clip"))
        if zscore_clip is not None and zscore_clip <= 0.0:
            zscore_clip = None
        self._trade_cost_state = _TradeCostState(
            impact_cfg=self._impact_cfg,
            tail_cfg=self._tail_cfg,
            adv_cfg=self._adv_cfg,
            adv_store=self._adv_store,
            vol_window=_positive_int(_cfg_attr(self._impact_cfg, "vol_window")),
            participation_window=_positive_int(
                _cfg_attr(self._impact_cfg, "participation_window")
            ),
            zscore_clip=zscore_clip,
            smoothing_alpha=smoothing_alpha,
            vol_metric=_normalise_str(impact_vol_metric),
            participation_metric=_normalise_str(part_metric),
        )
        self._initialise_calibrations()

    @property
    def config(self):
        return self._cfg_obj

    @property
    def dynamic_profile(self) -> Optional[_DynamicSpreadProfile]:
        return self._dynamic_profile

    def _consume_trade_cost_meta(self) -> Dict[str, Any]:
        meta = dict(self._last_trade_cost_meta)
        self._last_trade_cost_meta = {}
        return meta

    def _get_maker_taker_share_info(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self._maker_taker_share_enabled),
            "maker_share": float(self._maker_share_default),
            "spread_cost_maker_bps": float(self._spread_cost_maker_bps_default),
            "spread_cost_taker_bps": float(self._spread_cost_taker_bps_default),
        }

    def _resolve_calibration_path(self, path: Any) -> Optional[str]:
        if path is None:
            return None
        try:
            text = str(path).strip()
        except Exception:
            return None
        if not text:
            return None
        if os.path.isabs(text):
            return text
        base = self._calibration_base_dir
        if base is None:
            try:
                base = os.getcwd()
            except Exception:
                base = None
        if base is None:
            return text
        return os.path.normpath(os.path.join(base, text))

    def _load_calibration_payload(self, path: str) -> Optional[Any]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read()
        except Exception:
            logger.exception("Failed to read calibration file: %s", path)
            return None
        try:
            return json.loads(raw)
        except Exception:
            try:
                import yaml  # type: ignore

                return yaml.safe_load(raw)
            except Exception:
                logger.exception("Failed to parse calibration file: %s", path)
        return None

    def _profiles_from_payload(
        self, payload: Any
    ) -> Dict[str, SymbolCalibratedProfile]:
        profiles: Dict[str, SymbolCalibratedProfile] = {}
        if SymbolCalibratedProfile is None:
            return profiles
        if not isinstance(payload, Mapping):
            return profiles
        mapping: Mapping[str, Any]
        symbols_block = payload.get("symbols") if isinstance(payload, Mapping) else None
        if isinstance(symbols_block, Mapping):
            mapping = symbols_block
        else:
            mapping = payload
        for key, value in mapping.items():
            if not isinstance(value, Mapping):
                continue
            try:
                profile = SymbolCalibratedProfile.from_dict(value)
            except Exception:
                logger.exception("Failed to parse calibrated profile for %s", key)
                continue
            symbol_key = self._normalise_symbol(key) or profile.symbol
            if symbol_key is None:
                continue
            profile.symbol = symbol_key
            profiles[symbol_key] = profile
        return profiles

    def _apply_profile_update(
        self, target: SymbolCalibratedProfile, source: SymbolCalibratedProfile
    ) -> None:
        if SymbolCalibratedProfile is None:
            return
        if source.symbol is not None:
            target.symbol = self._normalise_symbol(source.symbol)
        if source.path is not None:
            target.path = source.path
        if source.curve_path is not None:
            target.curve_path = source.curve_path
        if source.hourly_path is not None:
            target.hourly_path = source.hourly_path
        if source.regime_path is not None:
            target.regime_path = source.regime_path
        if source.impact_curve:
            target.impact_curve = source.impact_curve
        if source.hourly_multipliers is not None:
            target.hourly_multipliers = source.hourly_multipliers
        if source.regime_overrides:
            merged = dict(target.regime_overrides)
            merged.update(source.regime_overrides)
            target.regime_overrides = merged
        if source.k is not None:
            target.k = source.k
        if source.default_spread_bps is not None:
            target.default_spread_bps = source.default_spread_bps
        if source.min_half_spread_bps is not None:
            target.min_half_spread_bps = source.min_half_spread_bps
        if source.metadata:
            meta = dict(target.metadata)
            meta.update(source.metadata)
            target.metadata = meta
        if source.extra:
            extra = dict(target.extra)
            extra.update(source.extra)
            target.extra = extra

    def _extract_symbol_payload(
        self, payload: Any, symbol: Optional[str]
    ) -> Optional[Mapping[str, Any]]:
        if not isinstance(payload, Mapping):
            return None
        symbol_key = self._normalise_symbol(symbol)
        mapping = payload
        symbols_block = payload.get("symbols") if isinstance(payload, Mapping) else None
        if isinstance(symbols_block, Mapping):
            mapping = symbols_block
        if symbol_key:
            if symbol_key in mapping and isinstance(mapping[symbol_key], Mapping):
                return mapping[symbol_key]
            for key, value in mapping.items():
                if isinstance(key, str) and key.upper() == symbol_key and isinstance(value, Mapping):
                    return value
        return payload if isinstance(payload, Mapping) else None

    def _ensure_profile_loaded(self, profile: SymbolCalibratedProfile) -> None:
        if SymbolCalibratedProfile is None:
            return
        if profile is None:
            return
        base_dir = self._calibration_base_dir
        if profile.path and not getattr(profile, "_path_loaded", False):
            resolved = self._resolve_calibration_path(profile.path)
            if resolved:
                payload = self._load_calibration_payload(resolved)
                block = self._extract_symbol_payload(payload, profile.symbol)
                if isinstance(block, Mapping):
                    try:
                        update = SymbolCalibratedProfile.from_dict(block)
                    except Exception:
                        logger.exception("Failed to parse calibration profile from %s", resolved)
                    else:
                        self._apply_profile_update(profile, update)
            profile._path_loaded = True
        if profile.curve_path and not getattr(profile, "_curve_loaded", False):
            resolved = self._resolve_calibration_path(profile.curve_path)
            if resolved:
                payload = self._load_calibration_payload(resolved)
                block = self._extract_symbol_payload(payload, profile.symbol)
                if isinstance(block, Mapping):
                    curve_block = (
                        block.get("impact_curve")
                        or block.get("notional_curve")
                        or block.get("buckets")
                    )
                    if isinstance(curve_block, Sequence):
                        profile.impact_curve = tuple(
                            dict(entry)
                            for entry in curve_block
                            if isinstance(entry, Mapping)
                        )
            profile._curve_loaded = True
        if profile.hourly_path and not getattr(profile, "_hourly_loaded", False):
            resolved = self._resolve_calibration_path(profile.hourly_path)
            if resolved:
                payload = self._load_calibration_payload(resolved)
                block = self._extract_symbol_payload(payload, profile.symbol)
                if isinstance(block, Mapping):
                    try:
                        profile.hourly_multipliers = CalibratedHourlyProfile.from_dict(block)
                    except Exception:
                        logger.exception("Failed to parse hourly multipliers from %s", resolved)
            profile._hourly_loaded = True
        if profile.regime_path and not getattr(profile, "_regime_loaded", False):
            resolved = self._resolve_calibration_path(profile.regime_path)
            if resolved:
                payload = self._load_calibration_payload(resolved)
                block = self._extract_symbol_payload(payload, profile.symbol)
                if isinstance(block, Mapping):
                    overrides: Dict[str, CalibratedRegimeOverride] = {}
                    for key, value in block.items():
                        if not isinstance(value, Mapping):
                            continue
                        try:
                            overrides[self._normalise_symbol(key) or str(key)] = (
                                CalibratedRegimeOverride.from_dict(value)
                            )
                        except Exception:
                            logger.exception("Failed to parse regime override for %s", key)
                    if overrides:
                        merged = dict(profile.regime_overrides)
                        merged.update(overrides)
                        profile.regime_overrides = merged
            profile._regime_loaded = True

    def _initialise_calibrations(self) -> None:
        if CalibratedProfilesConfig is None or SymbolCalibratedProfile is None:
            self._calibrated_cfg = None
            self._calibration_symbols = {}
            return
        cfg_obj = self._calibrated_cfg
        if cfg_obj is None and self._cfg_obj is not None:
            candidate = getattr(self._cfg_obj, "calibrated_profiles", None)
            if isinstance(candidate, CalibratedProfilesConfig):
                cfg_obj = candidate
        if cfg_obj is None:
            self._calibration_symbols = {}
            return
        self._calibrated_cfg = cfg_obj
        self._calibration_global_hourly = getattr(cfg_obj, "hourly_multipliers", None)
        regime_map = getattr(cfg_obj, "regime_overrides", None)
        if isinstance(regime_map, Mapping):
            self._calibration_global_regime = dict(regime_map)
        else:
            self._calibration_global_regime = {}
        symbols: Dict[str, SymbolCalibratedProfile] = {}
        base_path = getattr(cfg_obj, "path", None)
        if base_path is not None:
            resolved = self._resolve_calibration_path(base_path)
            if resolved:
                self._calibration_base_dir = os.path.dirname(resolved)
                payload = self._load_calibration_payload(resolved)
                for sym, profile in self._profiles_from_payload(payload).items():
                    symbols[sym] = profile
        inline_symbols = getattr(cfg_obj, "symbols", None)
        if isinstance(inline_symbols, Mapping):
            for key, profile in inline_symbols.items():
                if not isinstance(profile, SymbolCalibratedProfile):
                    continue
                symbol_key = self._normalise_symbol(key) or profile.symbol
                if symbol_key is None:
                    continue
                profile.symbol = symbol_key
                if symbol_key in symbols:
                    self._apply_profile_update(symbols[symbol_key], profile)
                else:
                    symbols[symbol_key] = profile
        cfg_obj.symbols = symbols
        self._calibration_symbols = symbols
        if self._cfg_obj is not None:
            try:
                setattr(self._cfg_obj, "calibrated_profiles", cfg_obj)
            except Exception:
                logger.exception("Failed to attach calibrated profiles to config")
        enabled_flag = False
        if cfg_obj is not None:
            try:
                enabled_flag = bool(getattr(cfg_obj, "enabled"))
            except Exception:
                enabled_flag = False
        self._calibration_enabled = enabled_flag and bool(self._calibration_symbols)

    def _get_calibrated_profile(
        self, symbol: Optional[str]
    ) -> Optional[SymbolCalibratedProfile]:
        if self._calibrated_cfg is None:
            return None
        profile: Optional[SymbolCalibratedProfile] = None
        if hasattr(self._calibrated_cfg, "get_symbol_profile"):
            try:
                profile = self._calibrated_cfg.get_symbol_profile(symbol)
            except Exception:
                profile = None
        if profile is None and symbol is not None:
            symbol_key = self._normalise_symbol(symbol)
            if symbol_key and symbol_key in self._calibration_symbols:
                profile = self._calibration_symbols[symbol_key]
        if profile is None and not symbol and len(self._calibration_symbols) == 1:
            profile = next(iter(self._calibration_symbols.values()))
        if profile is not None:
            self._ensure_profile_loaded(profile)
        return profile

    def set_market_regime(self, regime: Any) -> None:
        """Update the cached market regime used by calibrated profiles."""

        self._current_market_regime = regime

    def get_calibrated_trade_cost_bps(
        self,
        *,
        side: Any,
        qty: Any,
        spread_bps: Any = None,
        liquidity: Any = None,
        vol_factor: Any = None,
        ts_ms: Any = None,
        market_regime: Any = None,
        mid: Any = None,
        notional: Any = None,
        hour_of_week: Any = None,
        symbol: Any = None,
    ) -> Optional[float]:
        if _estimate_calibrated_slippage is None:
            return None
        cfg_obj = self._cfg_obj
        if cfg_obj is None:
            return None
        size_val = _safe_float(qty)
        if size_val is None:
            return None
        liq_val = _safe_float(liquidity)
        vf_val = _safe_float(vol_factor)
        spread_val = _safe_float(spread_bps)
        hour_idx: Optional[int] = None
        if hour_of_week is not None:
            try:
                hour_idx = int(hour_of_week)
            except (TypeError, ValueError):
                hour_idx = None
        notional_val = _safe_float(notional)
        if notional_val is None and mid is not None:
            mid_val = _safe_float(mid)
            if mid_val is not None:
                try:
                    notional_val = abs(float(size_val)) * float(mid_val)
                except Exception:
                    notional_val = None
        symbol_key = symbol
        if symbol_key is None:
            symbol_key = self._symbol
        else:
            symbol_key = self._normalise_symbol(symbol_key)
        self._get_calibrated_profile(symbol_key)
        result = _estimate_calibrated_slippage(
            cfg=cfg_obj,
            symbol=symbol_key,
            spread_bps=spread_val,
            size=float(size_val),
            liquidity=liq_val,
            vol_factor=vf_val,
            ts_ms=ts_ms,
            market_regime=market_regime,
            hour_of_week=hour_idx,
            notional=notional_val,
        )
        if result is None:
            return None
        return float(result)

    def attach_to(self, sim) -> None:
        prev_symbol = self._symbol
        try:
            symbol_attr = getattr(sim, "symbol", None)
        except Exception:
            symbol_attr = None
        new_symbol = self._normalise_symbol(symbol_attr)
        self._symbol = new_symbol
        symbol_changed = prev_symbol != new_symbol
        state = self._trade_cost_state
        if symbol_changed:
            state.reset()
        else:
            state.reset(reset_store=False)
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
        try:
            setattr(sim, "get_adv_quote", self.get_adv_quote)
            setattr(sim, "_slippage_get_adv_quote", self.get_adv_quote)
        except Exception:
            logger.exception("Failed to attach get_adv_quote to simulator")
        if self._adv_store is not None:
            try:
                setattr(sim, "get_bar_capacity_quote", self.get_bar_capacity_quote)
                setattr(
                    sim, "_slippage_get_bar_capacity_quote", self.get_bar_capacity_quote
                )
            except Exception:
                logger.exception("Failed to attach get_bar_capacity_quote to simulator")
        try:
            setattr(sim, "get_trade_cost_bps", self.get_trade_cost_bps)
            setattr(sim, "_slippage_get_trade_cost", self.get_trade_cost_bps)
        except Exception:
            logger.exception("Failed to attach get_trade_cost_bps to simulator")
        try:
            setattr(sim, "get_calibrated_trade_cost_bps", self.get_calibrated_trade_cost_bps)
        except Exception:
            logger.exception("Failed to attach get_calibrated_trade_cost_bps to simulator")
        try:
            setattr(
                sim,
                "_slippage_consume_trade_cost_meta",
                self._consume_trade_cost_meta,
            )
        except Exception:
            logger.exception("Failed to attach trade cost meta consumer to simulator")
        try:
            setattr(
                sim,
                "_slippage_get_maker_taker_share_info",
                self._get_maker_taker_share_info,
            )
        except Exception:
            logger.exception("Failed to attach maker/taker share info provider")
        if self._cfg_obj is not None:
            try:
                setattr(self._cfg_obj, "get_trade_cost_bps", self.get_trade_cost_bps)
            except Exception:
                logger.exception("Failed to attach get_trade_cost_bps to config")
            try:
                setattr(
                    self._cfg_obj,
                    "get_calibrated_trade_cost_bps",
                    self.get_calibrated_trade_cost_bps,
                )
            except Exception:
                logger.exception(
                    "Failed to attach get_calibrated_trade_cost_bps to config"
                )

    @staticmethod
    def from_dict(d: Dict[str, Any], *, run_config: Any | None = None) -> "SlippageImpl":
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
            ),
            run_config=run_config,
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
                if dynamic_spread is not None and math.isfinite(dynamic_spread):
                    return float(dynamic_spread)
            logger.debug(
                "Dynamic spread fell back to default profile computation for ts=%r",
                ts_ms,
            )
            try:
                fallback = profile.compute(
                    ts_ms=ts_ms,
                    base_spread_bps=base,
                    vol_multiplier=vol_multiplier,
                )
            except Exception:
                logger.exception("Dynamic spread fallback computation failed")
            else:
                if math.isfinite(fallback):
                    return float(fallback)

        dyn_cfg: Optional[DynamicSpreadConfig] = None
        if profile is not None:
            dyn_cfg = profile._cfg
        elif isinstance(self.cfg.get_dynamic_block(), DynamicSpreadConfig):
            dyn_cfg = self.cfg.get_dynamic_block()  # type: ignore[assignment]

        if dyn_cfg is not None and getattr(dyn_cfg, "enabled", False):
            try:
                processed = _calc_dynamic_spread(
                    cfg=dyn_cfg,
                    default_spread_bps=base,
                    bar_high=None,
                    bar_low=None,
                    mid_price=None,
                    seasonal_multiplier=1.0,
                    vol_multiplier=vol_multiplier,
                    profile=profile,
                    raw_spread_bps=base,
                )
            except Exception:
                logger.exception("Dynamic spread base processing failed")
            else:
                if processed is not None and math.isfinite(processed):
                    return float(processed)

        return float(base * vol_multiplier)

    def _normalise_symbol(self, symbol: Any) -> Optional[str]:
        if symbol is None:
            return None
        try:
            text = str(symbol).strip().upper()
        except Exception:
            return None
        return text or None

    def _resolve_adv_default_quote(self) -> Optional[float]:
        store = self._adv_store
        if store is not None:
            default_quote = store.default_quote
            if default_quote is not None and default_quote > 0.0:
                return float(default_quote)
        runtime_cfg = self._adv_runtime_cfg
        if runtime_cfg is not None:
            default_quote = _safe_float(_cfg_attr(runtime_cfg, "default_quote"))
            if default_quote is not None and default_quote > 0.0:
                return float(default_quote)
            extra = _cfg_attr(runtime_cfg, "extra")
            if isinstance(extra, Mapping):
                extra_default = _safe_float(extra.get("default_quote"))
                if extra_default is not None and extra_default > 0.0:
                    return float(extra_default)
        adv_cfg = self._adv_cfg
        if adv_cfg is not None:
            fallback = _safe_float(_cfg_attr(adv_cfg, "fallback_adv"))
            if fallback is not None and fallback > 0.0:
                return float(fallback)
        return None

    def _resolve_adv_floor(self) -> Optional[float]:
        floor_candidates: list[float] = []
        store = self._adv_store
        if store is not None:
            floor_quote = store.floor_quote
            if floor_quote is not None and floor_quote > 0.0:
                floor_candidates.append(float(floor_quote))
        runtime_cfg = self._adv_runtime_cfg
        if runtime_cfg is not None:
            runtime_floor = _safe_float(_cfg_attr(runtime_cfg, "floor_quote"))
            if runtime_floor is not None and runtime_floor > 0.0:
                floor_candidates.append(float(runtime_floor))
            extra = _cfg_attr(runtime_cfg, "extra")
            if isinstance(extra, Mapping):
                extra_floor = _safe_float(extra.get("floor_quote"))
                if extra_floor is not None and extra_floor > 0.0:
                    floor_candidates.append(float(extra_floor))
        adv_cfg = self._adv_cfg
        if adv_cfg is not None:
            min_adv = _safe_float(_cfg_attr(adv_cfg, "min_adv"))
            if min_adv is not None and min_adv > 0.0:
                floor_candidates.append(float(min_adv))
        if not floor_candidates:
            return None
        return max(floor_candidates)

    def get_adv_quote(self, symbol: Any) -> Optional[float]:
        symbol_key = self._normalise_symbol(symbol)
        if not symbol_key:
            return self._resolve_adv_default_quote()
        state = self._trade_cost_state
        cached = state.adv_cache.get(symbol_key)
        if cached is not None and math.isfinite(cached) and cached > 0.0:
            return cached
        value: Optional[float] = None
        if self._adv_store is not None:
            try:
                value = self._adv_store.get_adv_quote(symbol_key)
            except Exception:
                logger.exception("Failed to query ADV store for %s", symbol_key)
                value = None
        if value is None:
            value = self._resolve_adv_default_quote()
        candidate = _safe_float(value)
        if candidate is None or candidate <= 0.0:
            return None
        floor_quote = self._resolve_adv_floor()
        if floor_quote is not None:
            candidate = max(candidate, floor_quote)
        adv_cfg = self._adv_cfg
        max_adv = _safe_float(_cfg_attr(adv_cfg, "max_adv")) if adv_cfg is not None else None
        if max_adv is not None and max_adv > 0.0:
            candidate = min(candidate, max_adv)
        buffer = _safe_float(_cfg_attr(adv_cfg, "liquidity_buffer")) if adv_cfg is not None else None
        if buffer is not None and buffer > 0.0 and buffer != 1.0:
            candidate = candidate * buffer
        if candidate <= 0.0 or not math.isfinite(candidate):
            return None
        state.adv_cache[symbol_key] = float(candidate)
        return float(candidate)

    def get_bar_capacity_quote(self, symbol: Any) -> Optional[float]:
        symbol_key = self._normalise_symbol(symbol)
        if not symbol_key:
            return self._resolve_adv_default_quote()
        if self._adv_store is None:
            return self._resolve_adv_default_quote()
        try:
            capacity = self._adv_store.get_bar_capacity_quote(symbol_key)
        except Exception:
            logger.exception("Failed to query ADV bar capacity for %s", symbol_key)
            return self._resolve_adv_default_quote()
        candidate = _safe_float(capacity)
        if candidate is None or candidate <= 0.0:
            return self._resolve_adv_default_quote()
        floor_quote = self._resolve_adv_floor()
        if floor_quote is not None:
            candidate = max(candidate, floor_quote)
        if not math.isfinite(candidate) or candidate <= 0.0:
            return None
        return float(candidate)

    def _resolve_adv_value(
        self,
        *,
        symbol: Optional[str],
        metrics: Mapping[str, Any],
    ) -> Optional[float]:
        state = self._trade_cost_state
        adv_hint = _safe_float(metrics.get("adv"))
        if adv_hint is not None and adv_hint > 0.0 and math.isfinite(adv_hint):
            return adv_hint
        symbol_key = self._normalise_symbol(symbol)
        if symbol_key:
            cached = state.adv_cache.get(symbol_key)
            if cached is not None and math.isfinite(cached) and cached > 0.0:
                return cached
            adv_val = self.get_adv_quote(symbol_key)
        else:
            adv_val = self._resolve_adv_default_quote()
        if adv_val is not None and adv_val > 0.0 and math.isfinite(adv_val):
            if symbol_key:
                state.adv_cache[symbol_key] = adv_val
            return adv_val
        liquidity_hint = _safe_float(metrics.get("liquidity"))
        if liquidity_hint is not None and liquidity_hint > 0.0:
            return liquidity_hint
        return None

    def _evaluate_tail_shock(
        self,
        *,
        side: Any,
        bar_close_ts: Any,
        order_seq: Any,
    ) -> tuple[float, float]:
        cfg = self._tail_cfg
        if cfg is None:
            return 1.0, 0.0
        enabled = _cfg_attr(cfg, "enabled")
        try:
            if not bool(enabled):
                return 1.0, 0.0
        except Exception:
            return 1.0, 0.0
        probability = _safe_float(_cfg_attr(cfg, "probability"))
        if probability is None or probability <= 0.0:
            return 1.0, 0.0
        probability = max(0.0, min(1.0, probability))
        rng_seed = _tail_rng_seed(
            symbol=self._symbol,
            ts=bar_close_ts,
            side=side,
            order_seq=order_seq,
            seed=_cfg_attr(cfg, "seed"),
        )
        rng = random.Random(rng_seed)
        if rng.random() > probability:
            return 1.0, 0.0
        base_bps = _safe_float(_cfg_attr(cfg, "shock_bps"))
        if base_bps is None:
            base_bps = 0.0
        extra = _cfg_attr(cfg, "extra")
        mode = ""
        if isinstance(extra, Mapping):
            raw_mode = extra.get("mode")
            if raw_mode is not None:
                try:
                    mode = str(raw_mode).strip().lower()
                except Exception:
                    mode = ""
        tail_bps = float(base_bps)
        if isinstance(extra, Mapping):
            if mode == "percentile":
                tail_bps = _tail_percentile_sample(extra, rng, tail_bps)
            elif mode == "gaussian":
                tail_bps = _tail_gaussian_sample(extra, rng, tail_bps)
        multiplier = _safe_float(_cfg_attr(cfg, "shock_multiplier"))
        if multiplier is None or multiplier <= 0.0:
            multiplier = 1.0
        min_mult = _safe_float(_cfg_attr(cfg, "min_multiplier"))
        max_mult = _safe_float(_cfg_attr(cfg, "max_multiplier"))
        multiplier = _clamp(float(multiplier), min_mult, max_mult)
        if not math.isfinite(tail_bps):
            tail_bps = base_bps if math.isfinite(base_bps) else 0.0
        return float(multiplier), float(tail_bps)

    def get_trade_cost_bps(
        self,
        *,
        side: Any,
        qty: Any,
        mid: Any,
        spread_bps: Any = None,
        bar_close_ts: Any = None,
        order_seq: Any = None,
        vol_metrics: Optional[Mapping[str, Any]] = None,
        market_regime: Any = None,
        symbol: Any = None,
        ts_ms: Any = None,
        hour_of_week: Any = None,
        notional: Any = None,
    ) -> float:
        base_spread = _safe_float(spread_bps)
        if base_spread is None or base_spread < 0.0:
            base_spread = float(self.cfg.default_spread_bps)
        half_spread = max(0.5 * float(base_spread), float(self.cfg.min_half_spread_bps))
        qty_val = _safe_float(qty)
        if qty_val is None or qty_val <= 0.0:
            return float(half_spread)
        metrics: Dict[str, Any] = {}
        if isinstance(vol_metrics, Mapping):
            try:
                metrics = dict(vol_metrics)
            except Exception:
                metrics = {}
        mid_val = _safe_float(mid)
        if mid_val is None or mid_val <= 0.0:
            mid_candidate = _safe_float(metrics.get("mid"))
            if mid_candidate is not None and mid_candidate > 0.0:
                mid_val = mid_candidate
        order_notional = None
        if mid_val is not None and mid_val > 0.0:
            order_notional = qty_val * mid_val
        else:
            notional_hint = _safe_float(metrics.get("notional"))
            if notional_hint is not None and notional_hint > 0.0:
                order_notional = notional_hint
        symbol_key = self._normalise_symbol(symbol) if symbol is not None else None
        if symbol_key is None:
            symbol_key = self._symbol
        adv_val = self._resolve_adv_value(symbol=symbol_key, metrics=metrics)
        participation_ratio: Optional[float] = None
        if order_notional is not None and adv_val is not None and adv_val > 0.0:
            try:
                participation_ratio = float(order_notional) / float(adv_val)
            except (TypeError, ValueError, ZeroDivisionError):
                participation_ratio = None
        if participation_ratio is None:
            liquidity_hint = _safe_float(metrics.get("liquidity"))
            if liquidity_hint is not None and liquidity_hint > 0.0:
                participation_ratio = qty_val / liquidity_hint
        if participation_ratio is None:
            participation_ratio = qty_val
        participation_ratio = max(float(participation_ratio), float(self.cfg.eps))

        regime_value = market_regime
        if regime_value is not None:
            self._current_market_regime = regime_value
        else:
            regime_value = self._current_market_regime

        ts_hint = None
        if ts_ms is not None:
            try:
                ts_hint = int(ts_ms)
            except (TypeError, ValueError):
                ts_hint = None
        if ts_hint is None and bar_close_ts is not None:
            try:
                ts_hint = int(bar_close_ts)
            except (TypeError, ValueError):
                ts_hint = None

        hour_hint = None
        if hour_of_week is not None:
            try:
                hour_hint = int(hour_of_week)
            except (TypeError, ValueError):
                hour_hint = None

        notional_override = _safe_float(notional)
        if notional_override is None:
            notional_override = order_notional

        liq_for_calibration = _safe_float(metrics.get("liquidity"))
        if liq_for_calibration is None and adv_val is not None:
            liq_for_calibration = adv_val

        vol_factor_val = _safe_float(metrics.get("vol_factor"))
        if vol_factor_val is None:
            vol_factor_val = _safe_float(metrics.get("sigma"))
        if vol_factor_val is None:
            vol_factor_val = _safe_float(self._last_vol_factor)

        calibration_meta: Dict[str, Any] = {}
        total_cost: Optional[float] = None
        taker_cost: float

        if self._calibration_enabled:
            calibration_cost = self.get_calibrated_trade_cost_bps(
                side=side,
                qty=qty_val,
                spread_bps=base_spread,
                liquidity=liq_for_calibration,
                vol_factor=vol_factor_val,
                ts_ms=ts_hint,
                market_regime=regime_value,
                hour_of_week=hour_hint,
                notional=notional_override,
                mid=mid_val,
                symbol=symbol_key,
            )
            if calibration_cost is not None:
                total_cost = float(calibration_cost)
                taker_cost = float(total_cost)
                calibration_meta["calibration_profile"] = True
                if symbol_key is not None:
                    calibration_meta["calibration_symbol"] = symbol_key
                if regime_value is not None:
                    regime_repr = getattr(regime_value, "name", None) or str(regime_value)
                    calibration_meta["calibration_regime"] = regime_repr
                if ts_hint is not None:
                    calibration_meta["calibration_ts_ms"] = int(ts_hint)

        if total_cost is None:
            impact_cfg = self._impact_cfg
            k_base = float(self.cfg.k)
            k_effective = k_base
            vol_mult = 1.0
            metrics_available = False
            if impact_cfg is not None:
                enabled = _cfg_attr(impact_cfg, "enabled")
                try:
                    impact_enabled = bool(enabled)
                except Exception:
                    impact_enabled = False
                if impact_enabled:
                    state = self._trade_cost_state
                    vol_value = None
                    if state.vol_metric and metrics:
                        metric_val = _lookup_metric(metrics, state.vol_metric)
                        vol_value = _safe_float(metric_val)
                    if vol_value is None:
                        vol_value = _safe_float(metrics.get("vol_factor"))
                    part_metric_value = None
                    if state.participation_metric and metrics:
                        metric_val = _lookup_metric(metrics, state.participation_metric)
                        part_metric_value = _safe_float(metric_val)
                    if part_metric_value is None and participation_ratio is not None:
                        part_metric_value = float(participation_ratio)
                    vol_norm = state.normalise_vol(vol_value)
                    part_norm = state.normalise_part(part_metric_value)
                    beta_vol = _safe_float(_cfg_attr(impact_cfg, "beta_vol")) or 0.0
                    beta_part = _safe_float(_cfg_attr(impact_cfg, "beta_participation")) or 0.0
                    if vol_norm is not None:
                        vol_mult += beta_vol * vol_norm
                        metrics_available = True
                    if part_norm is not None:
                        vol_mult += beta_part * part_norm
                        metrics_available = True
                    if not math.isfinite(vol_mult):
                        vol_mult = 1.0
                    if vol_mult < 0.0:
                        vol_mult = 0.0
                    if metrics_available:
                        k_effective = k_base * vol_mult
                    else:
                        fallback_k = _safe_float(_cfg_attr(impact_cfg, "fallback_k"))
                        if fallback_k is not None and fallback_k > 0.0:
                            k_effective = fallback_k
                    min_k = _safe_float(_cfg_attr(impact_cfg, "min_k"))
                    max_k = _safe_float(_cfg_attr(impact_cfg, "max_k"))
                    if min_k is not None or max_k is not None:
                        k_effective = _clamp(k_effective, min_k, max_k)
                    k_effective = self._trade_cost_state.apply_k_smoothing(k_effective)

            impact_term = k_effective * math.sqrt(
                max(participation_ratio, float(self.cfg.eps))
            )
            base_cost = half_spread + impact_term
            tail_mult, tail_bps = self._evaluate_tail_shock(
                side=side, bar_close_ts=bar_close_ts, order_seq=order_seq
            )
            total_cost = base_cost * tail_mult + tail_bps
            if not math.isfinite(total_cost):
                total_cost = base_cost
            if total_cost < 0.0:
                total_cost = 0.0
            taker_cost = float(total_cost)

        trade_meta = dict(calibration_meta)
        if self._maker_taker_share_enabled:
            share_metric: Any = None
            maker_cost_metric: Any = None
            taker_cost_metric: Any = None
            if metrics:
                share_metric = _lookup_metric(metrics, "maker_share")
                if share_metric is None:
                    share_metric = metrics.get("maker_share_default")
                maker_cost_metric = _lookup_metric(metrics, "spread_cost_maker_bps")
                taker_cost_metric = _lookup_metric(metrics, "spread_cost_taker_bps")
            share_value = _safe_share_value(share_metric, self._maker_share_default)
            maker_cost = _safe_non_negative_float(
                maker_cost_metric, self._spread_cost_maker_bps_default
            )
            taker_cost_effective = _safe_non_negative_float(
                taker_cost_metric, taker_cost
            )
            expected_spread = (
                share_value * maker_cost + (1.0 - share_value) * taker_cost_effective
            )
            if not math.isfinite(expected_spread):
                expected_spread = taker_cost_effective
            if expected_spread < 0.0:
                expected_spread = 0.0
            trade_meta.update(
                {
                    "maker_share": float(share_value),
                    "spread_cost_maker_bps": float(maker_cost),
                    "spread_cost_taker_bps": float(taker_cost_effective),
                    "taker_spread_bps": float(taker_cost_effective),
                    "expected_spread_bps": float(expected_spread),
                }
            )
            self._last_trade_cost_meta = trade_meta
        else:
            self._last_trade_cost_meta = trade_meta
        if total_cost is None:
            total_cost = float(half_spread)
        return float(total_cost)
