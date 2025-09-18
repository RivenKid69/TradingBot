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


def _safe_positive_int(value: Any) -> Optional[int]:
    try:
        num = int(value)
    except (TypeError, ValueError):
        return None
    if num <= 0:
        return None
    return num


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
    adv_loader: Optional["_AdvDataset"] = None
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

    def reset(self) -> None:
        self.vol_history.clear()
        self.participation_history.clear()
        self.k_ema = None
        self.adv_cache.clear()

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


class _AdvDataset:
    """Load and cache ADV dataset from disk."""

    def __init__(self, cfg: AdvConfig) -> None:  # type: ignore[name-defined]
        self._cfg = cfg
        self._path = self._extract_path(cfg)
        self._refresh_days = self._extract_refresh_days(cfg)
        self._cache: dict[str, float] = {}
        self._meta: dict[str, Any] = {}
        self._mtime: float | None = None
        self._stale = False
        self._lock = threading.Lock()

    @staticmethod
    def _extract_path(cfg: AdvConfig) -> Optional[str]:  # type: ignore[name-defined]
        candidates: list[Any] = []
        extra = getattr(cfg, "extra", None)
        if isinstance(extra, Mapping):
            for key in ("quote_path", "adv_path", "path", "dataset", "data_path"):
                if key in extra:
                    candidates.append(extra[key])
        inline = getattr(cfg, "path", None)
        if inline is not None:
            candidates.insert(0, inline)
        for candidate in candidates:
            if candidate is None:
                continue
            text = str(candidate).strip()
            if text:
                return text
        return None

    @staticmethod
    def _extract_refresh_days(cfg: AdvConfig) -> Optional[int]:  # type: ignore[name-defined]
        extra = getattr(cfg, "extra", None)
        candidate: Any = None
        if isinstance(extra, Mapping):
            for key in ("refresh_days", "auto_refresh_days"):
                if key in extra:
                    candidate = extra[key]
                    break
        if candidate is None:
            candidate = getattr(cfg, "refresh_days", None)
        return _safe_positive_int(candidate)

    def _ensure_loaded_locked(self) -> None:
        path = self._path
        if not path:
            self._cache.clear()
            self._meta = {}
            self._stale = False
            return
        try:
            mtime = os.path.getmtime(path)
        except (OSError, TypeError, ValueError):
            if not self._cache:
                logger.warning("ADV dataset file %s is not accessible", path)
            self._stale = False
            return
        if self._mtime is not None and self._cache and mtime <= self._mtime:
            self._check_refresh_locked()
            return
        payload = self._read_payload(path)
        if payload is None:
            self._check_refresh_locked()
            return
        data, meta = payload
        self._cache = data
        self._meta = meta
        self._mtime = mtime
        self._check_refresh_locked()

    def _read_payload(self, path: str) -> tuple[dict[str, float], dict[str, Any]] | None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except FileNotFoundError:
            logger.warning("ADV dataset file %s not found", path)
            return {}, {}
        except Exception:
            logger.exception("Failed to load ADV dataset from %s", path)
            return None
        if not isinstance(payload, Mapping):
            logger.warning("ADV dataset %s must be a JSON object", path)
            return {}, {}
        meta_raw = payload.get("meta")
        data_raw = payload.get("data")
        if not isinstance(data_raw, Mapping):
            logger.warning("ADV dataset %s is missing 'data' mapping", path)
            data_raw = {}
        dataset: dict[str, float] = {}
        for key, value in data_raw.items():
            symbol = str(key).strip().upper()
            if not symbol:
                continue
            if isinstance(value, Mapping):
                candidate = value.get("adv_quote")
            else:
                candidate = value
            adv_val = _safe_float(candidate)
            if adv_val is None or adv_val <= 0.0:
                continue
            dataset[symbol] = float(adv_val)
        meta: dict[str, Any] = dict(meta_raw) if isinstance(meta_raw, Mapping) else {}
        meta.setdefault("path", path)
        meta.setdefault("symbol_count", len(dataset))
        return dataset, meta

    def _extract_timestamp_locked(self) -> Optional[int]:
        meta = self._meta
        candidates = [
            meta.get("generated_at_ms"),
            meta.get("generated_ms"),
            meta.get("timestamp_ms"),
            meta.get("end_ms"),
        ]
        for candidate in candidates:
            ts_val = _safe_positive_int(candidate)
            if ts_val is not None:
                return ts_val
        for key in ("generated_at", "end_at"):
            value = meta.get(key)
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not text:
                continue
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(text)
            except ValueError:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        if self._mtime is not None:
            return int(self._mtime * 1000)
        return None

    def _check_refresh_locked(self) -> None:
        refresh_days = self._refresh_days
        if not refresh_days:
            self._stale = False
            return
        ts_ms = self._extract_timestamp_locked()
        if ts_ms is None:
            self._stale = True
            logger.warning("ADV dataset %s lacks timestamp metadata", self._path)
            return
        age_ms = int(time.time() * 1000) - ts_ms
        if age_ms <= 0:
            self._stale = False
            return
        age_days = age_ms / 86_400_000
        if age_days > float(refresh_days):
            self._stale = True
            logger.warning(
                "ADV dataset %s is older than %.1f days (threshold=%s)",
                self._path,
                age_days,
                refresh_days,
            )
        else:
            self._stale = False

    def get(self, symbol: str) -> Optional[float]:
        if not symbol:
            return None
        with self._lock:
            self._ensure_loaded_locked()
            if self._stale:
                return None
            if not self._cache:
                return None
            return self._cache.get(str(symbol).strip().upper())

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
    def __init__(self, cfg: SlippageCfg) -> None:
        self.cfg = cfg
        self._symbol: Optional[str] = None
        self._dynamic_profile: Optional[_DynamicSpreadProfile] = None
        dyn_cfg_obj: Optional[DynamicSpreadConfig] = None
        self._adv_loader: Optional[_AdvDataset] = None
        adv_cfg_obj: Optional[Any] = None
        impact_cfg_obj: Optional[Any] = None
        tail_cfg_obj: Optional[Any] = None
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
        if self._adv_cfg is not None and getattr(self._adv_cfg, "enabled", False):
            try:
                self._adv_loader = _AdvDataset(self._adv_cfg)
            except Exception:
                logger.exception("Failed to initialise ADV dataset loader")
                self._adv_loader = None
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
            adv_loader=self._adv_loader,
            vol_window=_positive_int(_cfg_attr(self._impact_cfg, "vol_window")),
            participation_window=_positive_int(
                _cfg_attr(self._impact_cfg, "participation_window")
            ),
            zscore_clip=zscore_clip,
            smoothing_alpha=smoothing_alpha,
            vol_metric=_normalise_str(impact_vol_metric),
            participation_metric=_normalise_str(part_metric),
        )

    @property
    def config(self):
        return self._cfg_obj

    @property
    def dynamic_profile(self) -> Optional[_DynamicSpreadProfile]:
        return self._dynamic_profile

    def attach_to(self, sim) -> None:
        try:
            symbol = getattr(sim, "symbol", None)
        except Exception:
            symbol = None
        if symbol is not None:
            try:
                self._symbol = str(symbol)
            except Exception:
                self._symbol = None
        else:
            self._symbol = None
        self._trade_cost_state.reset()
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
        try:
            setattr(sim, "get_trade_cost_bps", self.get_trade_cost_bps)
            setattr(sim, "_slippage_get_trade_cost", self.get_trade_cost_bps)
        except Exception:
            logger.exception("Failed to attach get_trade_cost_bps to simulator")
        if self._cfg_obj is not None:
            try:
                setattr(self._cfg_obj, "get_trade_cost_bps", self.get_trade_cost_bps)
            except Exception:
                logger.exception("Failed to attach get_trade_cost_bps to config")

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

    def get_adv_quote(self, symbol: Any) -> Optional[float]:
        adv_cfg = getattr(self, "_adv_cfg", None)
        loader = self._adv_loader
        value: Optional[float] = None
        if loader is not None and symbol:
            try:
                value = loader.get(str(symbol))
            except Exception:
                logger.exception("Failed to query ADV dataset for %s", symbol)
                value = None
        def _cfg_lookup(key: str) -> Any:
            if isinstance(adv_cfg, Mapping):
                return adv_cfg.get(key)
            return getattr(adv_cfg, key, None)

        if value is None and adv_cfg is not None:
            fallback = _safe_float(_cfg_lookup("fallback_adv"))
            if fallback is not None and fallback > 0.0:
                value = fallback
        if value is None:
            return None
        candidate = _safe_float(value)
        if candidate is None or candidate <= 0.0:
            return None
        min_adv = _safe_float(_cfg_lookup("min_adv")) if adv_cfg is not None else None
        max_adv = _safe_float(_cfg_lookup("max_adv")) if adv_cfg is not None else None
        if min_adv is not None:
            candidate = max(min_adv, candidate)
        if max_adv is not None:
            candidate = min(max_adv, candidate)
        buffer = _safe_float(_cfg_lookup("liquidity_buffer")) if adv_cfg is not None else None
        if buffer is not None and buffer > 0.0 and buffer != 1.0:
            candidate = candidate * buffer
        if candidate <= 0.0 or not math.isfinite(candidate):
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
        if adv_hint is not None and adv_hint > 0.0:
            return adv_hint
        if symbol:
            cached = state.adv_cache.get(symbol)
            if cached is not None and math.isfinite(cached) and cached > 0.0:
                return cached
        adv_val = self.get_adv_quote(symbol) if symbol else None
        if adv_val is not None and adv_val > 0.0 and math.isfinite(adv_val) and symbol:
            state.adv_cache[symbol] = adv_val
        if adv_val is not None and adv_val > 0.0 and math.isfinite(adv_val):
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
        symbol = self._symbol
        adv_val = self._resolve_adv_value(symbol=symbol, metrics=metrics)
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

        impact_term = k_effective * math.sqrt(max(participation_ratio, float(self.cfg.eps)))
        base_cost = half_spread + impact_term
        tail_mult, tail_bps = self._evaluate_tail_shock(
            side=side, bar_close_ts=bar_close_ts, order_seq=order_seq
        )
        total_cost = base_cost * tail_mult + tail_bps
        if not math.isfinite(total_cost):
            total_cost = base_cost
        if total_cost < 0.0:
            total_cost = 0.0
        return float(total_cost)
