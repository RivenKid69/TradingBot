# sim/slippage.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Sequence, Mapping, Iterable


@dataclass
class DynamicSpreadConfig:
    enabled: bool = False
    profile_kind: Optional[str] = None
    multipliers: Optional[tuple[float, ...]] = None
    path: Optional[str] = None
    override_path: Optional[str] = None
    hash: Optional[str] = None
    alpha_bps: Optional[float] = None
    beta_coef: Optional[float] = None
    min_spread_bps: Optional[float] = None
    max_spread_bps: Optional[float] = None
    smoothing_alpha: Optional[float] = None
    vol_metric: Optional[str] = None
    vol_window: Optional[int] = None
    use_volatility: bool = False
    gamma: Optional[float] = None
    zscore_clip: Optional[float] = None
    refresh_warn_days: Optional[int] = None
    refresh_fail_days: Optional[int] = None
    refresh_on_start: bool = False
    last_refresh_ts: Optional[int] = None
    fallback_spread_bps: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DynamicSpreadConfig":
        if not isinstance(d, dict):
            raise TypeError("dynamic spread config must be a dict")

        multipliers_raw = d.get("multipliers")
        multipliers: Optional[tuple[float, ...]] = None
        if multipliers_raw is not None:
            if isinstance(multipliers_raw, Sequence) and not isinstance(
                multipliers_raw, (str, bytes, bytearray)
            ):
                multipliers = tuple(float(x) for x in multipliers_raw)
            else:
                try:
                    multipliers = (float(multipliers_raw),)
                except (TypeError, ValueError):
                    multipliers = None

        known_keys = {
            "enabled",
            "profile_kind",
            "multipliers",
            "path",
            "override_path",
            "hash",
            "alpha_bps",
            "beta_coef",
            "min_spread_bps",
            "max_spread_bps",
            "smoothing_alpha",
            "vol_metric",
            "vol_window",
            "use_volatility",
            "gamma",
            "zscore_clip",
            "refresh_warn_days",
            "refresh_fail_days",
            "refresh_on_start",
            "last_refresh_ts",
            "fallback_spread_bps",
            # legacy aliases kept for backwards compatibility
            "alpha",
            "beta",
            "volatility_metric",
            "volatility_window",
        }

        extra = {k: v for k, v in d.items() if k not in known_keys}

        def _first_non_null(*keys: str) -> Any:
            for key in keys:
                if key in d and d[key] is not None:
                    return d[key]
            return None

        alpha_bps_val = _first_non_null("alpha_bps", "alpha")
        beta_coef_val = _first_non_null("beta_coef", "beta")
        vol_metric_val = _first_non_null("vol_metric", "volatility_metric")
        vol_window_val = _first_non_null("vol_window", "volatility_window")

        return cls(
            enabled=bool(d.get("enabled", False)),
            profile_kind=str(d["profile_kind"]) if d.get("profile_kind") is not None else None,
            multipliers=multipliers,
            path=str(d["path"]) if d.get("path") is not None else None,
            override_path=str(d["override_path"]) if d.get("override_path") is not None else None,
            hash=str(d["hash"]) if d.get("hash") is not None else None,
            alpha_bps=float(alpha_bps_val) if alpha_bps_val is not None else None,
            beta_coef=float(beta_coef_val) if beta_coef_val is not None else None,
            min_spread_bps=float(d["min_spread_bps"]) if d.get("min_spread_bps") is not None else None,
            max_spread_bps=float(d["max_spread_bps"]) if d.get("max_spread_bps") is not None else None,
            smoothing_alpha=float(d["smoothing_alpha"]) if d.get("smoothing_alpha") is not None else None,
            vol_metric=str(vol_metric_val) if vol_metric_val is not None else None,
            vol_window=int(vol_window_val) if vol_window_val is not None else None,
            use_volatility=bool(d.get("use_volatility", False)),
            gamma=float(d["gamma"]) if d.get("gamma") is not None else None,
            zscore_clip=float(d["zscore_clip"]) if d.get("zscore_clip") is not None else None,
            refresh_warn_days=int(d["refresh_warn_days"]) if d.get("refresh_warn_days") is not None else None,
            refresh_fail_days=int(d["refresh_fail_days"]) if d.get("refresh_fail_days") is not None else None,
            refresh_on_start=bool(d.get("refresh_on_start", False)),
            last_refresh_ts=int(d["last_refresh_ts"]) if d.get("last_refresh_ts") is not None else None,
            fallback_spread_bps=float(d["fallback_spread_bps"]) if d.get("fallback_spread_bps") is not None else None,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = dict(self.extra)
        data["enabled"] = bool(self.enabled)
        if self.profile_kind is not None:
            data["profile_kind"] = str(self.profile_kind)
        if self.multipliers is not None:
            data["multipliers"] = [float(x) for x in self.multipliers]
        if self.path is not None:
            data["path"] = str(self.path)
        if self.override_path is not None:
            data["override_path"] = str(self.override_path)
        if self.hash is not None:
            data["hash"] = str(self.hash)
        if self.alpha_bps is not None:
            data["alpha_bps"] = float(self.alpha_bps)
        if self.beta_coef is not None:
            data["beta_coef"] = float(self.beta_coef)
        if self.min_spread_bps is not None:
            data["min_spread_bps"] = float(self.min_spread_bps)
        if self.max_spread_bps is not None:
            data["max_spread_bps"] = float(self.max_spread_bps)
        if self.smoothing_alpha is not None:
            data["smoothing_alpha"] = float(self.smoothing_alpha)
        if self.vol_metric is not None:
            data["vol_metric"] = str(self.vol_metric)
        if self.vol_window is not None:
            data["vol_window"] = int(self.vol_window)
        data["use_volatility"] = bool(self.use_volatility)
        if self.gamma is not None:
            data["gamma"] = float(self.gamma)
        if self.zscore_clip is not None:
            data["zscore_clip"] = float(self.zscore_clip)
        if self.refresh_warn_days is not None:
            data["refresh_warn_days"] = int(self.refresh_warn_days)
        if self.refresh_fail_days is not None:
            data["refresh_fail_days"] = int(self.refresh_fail_days)
        data["refresh_on_start"] = bool(self.refresh_on_start)
        if self.last_refresh_ts is not None:
            data["last_refresh_ts"] = int(self.last_refresh_ts)
        if self.fallback_spread_bps is not None:
            data["fallback_spread_bps"] = float(self.fallback_spread_bps)
        return data


@dataclass
class DynamicImpactConfig:
    enabled: bool = False
    beta_vol: float = 0.0
    beta_participation: float = 0.0
    min_k: Optional[float] = None
    max_k: Optional[float] = None
    fallback_k: Optional[float] = None
    vol_metric: Optional[str] = None
    vol_window: Optional[int] = None
    participation_metric: Optional[str] = None
    participation_window: Optional[int] = None
    smoothing_alpha: Optional[float] = None
    zscore_clip: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DynamicImpactConfig":
        if not isinstance(d, dict):
            raise TypeError("dynamic impact config must be a dict")

        known_keys = {
            "enabled",
            "beta_vol",
            "beta_participation",
            "min_k",
            "max_k",
            "fallback_k",
            "vol_metric",
            "vol_window",
            "participation_metric",
            "participation_window",
            "smoothing_alpha",
            "zscore_clip",
        }

        extra = {k: v for k, v in d.items() if k not in known_keys}

        return cls(
            enabled=bool(d.get("enabled", False)),
            beta_vol=float(d.get("beta_vol", 0.0)),
            beta_participation=float(d.get("beta_participation", 0.0)),
            min_k=float(d["min_k"]) if d.get("min_k") is not None else None,
            max_k=float(d["max_k"]) if d.get("max_k") is not None else None,
            fallback_k=float(d["fallback_k"]) if d.get("fallback_k") is not None else None,
            vol_metric=str(d["vol_metric"]) if d.get("vol_metric") is not None else None,
            vol_window=int(d["vol_window"]) if d.get("vol_window") is not None else None,
            participation_metric=str(d["participation_metric"]) if d.get("participation_metric") is not None else None,
            participation_window=int(d["participation_window"]) if d.get("participation_window") is not None else None,
            smoothing_alpha=float(d["smoothing_alpha"]) if d.get("smoothing_alpha") is not None else None,
            zscore_clip=float(d["zscore_clip"]) if d.get("zscore_clip") is not None else None,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = dict(self.extra)
        data["enabled"] = bool(self.enabled)
        data["beta_vol"] = float(self.beta_vol)
        data["beta_participation"] = float(self.beta_participation)
        if self.min_k is not None:
            data["min_k"] = float(self.min_k)
        if self.max_k is not None:
            data["max_k"] = float(self.max_k)
        if self.fallback_k is not None:
            data["fallback_k"] = float(self.fallback_k)
        if self.vol_metric is not None:
            data["vol_metric"] = str(self.vol_metric)
        if self.vol_window is not None:
            data["vol_window"] = int(self.vol_window)
        if self.participation_metric is not None:
            data["participation_metric"] = str(self.participation_metric)
        if self.participation_window is not None:
            data["participation_window"] = int(self.participation_window)
        if self.smoothing_alpha is not None:
            data["smoothing_alpha"] = float(self.smoothing_alpha)
        if self.zscore_clip is not None:
            data["zscore_clip"] = float(self.zscore_clip)
        return data


@dataclass
class TailShockConfig:
    enabled: bool = False
    probability: float = 0.0
    shock_bps: float = 0.0
    shock_multiplier: float = 1.0
    decay_halflife_bars: Optional[int] = None
    min_multiplier: Optional[float] = None
    max_multiplier: Optional[float] = None
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TailShockConfig":
        if not isinstance(d, dict):
            raise TypeError("tail shock config must be a dict")

        known_keys = {
            "enabled",
            "probability",
            "shock_bps",
            "shock_multiplier",
            "decay_halflife_bars",
            "min_multiplier",
            "max_multiplier",
            "seed",
        }

        extra = {k: v for k, v in d.items() if k not in known_keys}

        return cls(
            enabled=bool(d.get("enabled", False)),
            probability=float(d.get("probability", 0.0)),
            shock_bps=float(d.get("shock_bps", 0.0)),
            shock_multiplier=float(d.get("shock_multiplier", 1.0)),
            decay_halflife_bars=int(d["decay_halflife_bars"]) if d.get("decay_halflife_bars") is not None else None,
            min_multiplier=float(d["min_multiplier"]) if d.get("min_multiplier") is not None else None,
            max_multiplier=float(d["max_multiplier"]) if d.get("max_multiplier") is not None else None,
            seed=int(d["seed"]) if d.get("seed") is not None else None,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = dict(self.extra)
        data["enabled"] = bool(self.enabled)
        data["probability"] = float(self.probability)
        data["shock_bps"] = float(self.shock_bps)
        data["shock_multiplier"] = float(self.shock_multiplier)
        if self.decay_halflife_bars is not None:
            data["decay_halflife_bars"] = int(self.decay_halflife_bars)
        if self.min_multiplier is not None:
            data["min_multiplier"] = float(self.min_multiplier)
        if self.max_multiplier is not None:
            data["max_multiplier"] = float(self.max_multiplier)
        if self.seed is not None:
            data["seed"] = int(self.seed)
        return data


@dataclass
class AdvConfig:
    enabled: bool = False
    window_days: int = 30
    smoothing_alpha: Optional[float] = None
    fallback_adv: Optional[float] = None
    min_adv: Optional[float] = None
    max_adv: Optional[float] = None
    seasonality_path: Optional[str] = None
    override_path: Optional[str] = None
    hash: Optional[str] = None
    profile_kind: Optional[str] = None
    multipliers: Optional[tuple[float, ...]] = None
    zscore_clip: Optional[float] = None
    liquidity_buffer: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdvConfig":
        if not isinstance(d, dict):
            raise TypeError("adv config must be a dict")

        multipliers_raw = d.get("multipliers")
        multipliers: Optional[tuple[float, ...]] = None
        if multipliers_raw is not None:
            if isinstance(multipliers_raw, Sequence) and not isinstance(
                multipliers_raw, (str, bytes, bytearray)
            ):
                multipliers = tuple(float(x) for x in multipliers_raw)
            else:
                try:
                    multipliers = (float(multipliers_raw),)
                except (TypeError, ValueError):
                    multipliers = None

        known_keys = {
            "enabled",
            "window_days",
            "smoothing_alpha",
            "fallback_adv",
            "min_adv",
            "max_adv",
            "seasonality_path",
            "override_path",
            "hash",
            "profile_kind",
            "multipliers",
            "zscore_clip",
            "liquidity_buffer",
        }

        extra = {k: v for k, v in d.items() if k not in known_keys}

        return cls(
            enabled=bool(d.get("enabled", False)),
            window_days=int(d.get("window_days", 30)),
            smoothing_alpha=float(d["smoothing_alpha"]) if d.get("smoothing_alpha") is not None else None,
            fallback_adv=float(d["fallback_adv"]) if d.get("fallback_adv") is not None else None,
            min_adv=float(d["min_adv"]) if d.get("min_adv") is not None else None,
            max_adv=float(d["max_adv"]) if d.get("max_adv") is not None else None,
            seasonality_path=str(d["seasonality_path"]) if d.get("seasonality_path") is not None else None,
            override_path=str(d["override_path"]) if d.get("override_path") is not None else None,
            hash=str(d["hash"]) if d.get("hash") is not None else None,
            profile_kind=str(d["profile_kind"]) if d.get("profile_kind") is not None else None,
            multipliers=multipliers,
            zscore_clip=float(d["zscore_clip"]) if d.get("zscore_clip") is not None else None,
            liquidity_buffer=float(d.get("liquidity_buffer", 1.0)),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = dict(self.extra)
        data["enabled"] = bool(self.enabled)
        data["window_days"] = int(self.window_days)
        if self.smoothing_alpha is not None:
            data["smoothing_alpha"] = float(self.smoothing_alpha)
        if self.fallback_adv is not None:
            data["fallback_adv"] = float(self.fallback_adv)
        if self.min_adv is not None:
            data["min_adv"] = float(self.min_adv)
        if self.max_adv is not None:
            data["max_adv"] = float(self.max_adv)
        if self.seasonality_path is not None:
            data["seasonality_path"] = str(self.seasonality_path)
        if self.override_path is not None:
            data["override_path"] = str(self.override_path)
        if self.hash is not None:
            data["hash"] = str(self.hash)
        if self.profile_kind is not None:
            data["profile_kind"] = str(self.profile_kind)
        if self.multipliers is not None:
            data["multipliers"] = [float(x) for x in self.multipliers]
        if self.zscore_clip is not None:
            data["zscore_clip"] = float(self.zscore_clip)
        data["liquidity_buffer"] = float(self.liquidity_buffer)
        return data


def _safe_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _coerce_float_tuple(values: Iterable[Any]) -> tuple[float, ...]:
    result: list[float] = []
    for raw in values:
        num = _safe_float(raw)
        if num is None:
            continue
        result.append(float(num))
    return tuple(result)


def _coerce_int_tuple(values: Iterable[Any]) -> tuple[int, ...]:
    result: list[int] = []
    for raw in values:
        try:
            num = int(raw)
        except (TypeError, ValueError):
            continue
        result.append(int(num))
    return tuple(result)


def _normalise_symbol_key(symbol: Optional[str]) -> Optional[str]:
    if symbol is None:
        return None
    try:
        text = str(symbol).strip().upper()
    except Exception:
        return None
    return text or None


@dataclass
class CalibratedHourlyProfile:
    multipliers: tuple[float, ...] = ()
    hours: tuple[int, ...] = ()
    counts: tuple[int, ...] = ()
    path: Optional[str] = None
    default_multiplier: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibratedHourlyProfile":
        if not isinstance(data, Mapping):
            raise TypeError("hourly profile must be a mapping")
        multipliers_raw = data.get("multipliers")
        hours_raw = data.get("hour_of_week") or data.get("hours")
        counts_raw = data.get("counts")
        multipliers: tuple[float, ...] = ()
        if isinstance(multipliers_raw, Iterable) and not isinstance(
            multipliers_raw, (str, bytes, bytearray)
        ):
            multipliers = _coerce_float_tuple(multipliers_raw)
        hours: tuple[int, ...] = ()
        if isinstance(hours_raw, Iterable) and not isinstance(hours_raw, (str, bytes, bytearray)):
            hours = _coerce_int_tuple(hours_raw)
        counts: tuple[int, ...] = ()
        if isinstance(counts_raw, Iterable) and not isinstance(counts_raw, (str, bytes, bytearray)):
            counts = _coerce_int_tuple(counts_raw)
        default_multiplier = _safe_float(data.get("default_multiplier"))
        extra = {
            key: value
            for key, value in data.items()
            if key
            not in {
                "multipliers",
                "hour_of_week",
                "hours",
                "counts",
                "path",
                "default_multiplier",
            }
        }
        return cls(
            multipliers=multipliers,
            hours=hours,
            counts=counts,
            path=str(data.get("path")) if data.get("path") is not None else None,
            default_multiplier=float(default_multiplier) if default_multiplier is not None else 1.0,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(self.extra)
        if self.multipliers:
            payload["multipliers"] = [float(x) for x in self.multipliers]
        if self.hours:
            payload["hour_of_week"] = [int(x) for x in self.hours]
        if self.counts:
            payload["counts"] = [int(x) for x in self.counts]
        if self.path is not None:
            payload["path"] = str(self.path)
        if self.default_multiplier != 1.0:
            payload["default_multiplier"] = float(self.default_multiplier)
        return payload

    def get_multiplier(self, hour_of_week: Optional[int]) -> float:
        if hour_of_week is None:
            return float(self.default_multiplier)
        if not self.multipliers:
            return float(self.default_multiplier)
        if self.hours and len(self.hours) == len(self.multipliers):
            for idx, hour in enumerate(self.hours):
                if hour == hour_of_week:
                    return float(self.multipliers[idx])
        if len(self.multipliers) == 168:
            try:
                return float(self.multipliers[hour_of_week % 168])
            except Exception:
                return float(self.default_multiplier)
        try:
            return float(self.multipliers[hour_of_week % len(self.multipliers)])
        except Exception:
            return float(self.default_multiplier)


@dataclass
class CalibratedRegimeOverride:
    multiplier: Optional[float] = None
    impact_mean_bps: Optional[float] = None
    k: Optional[float] = None
    default_spread_bps: Optional[float] = None
    min_half_spread_bps: Optional[float] = None
    count: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibratedRegimeOverride":
        if not isinstance(data, Mapping):
            raise TypeError("regime override must be a mapping")
        known = {
            "multiplier",
            "impact_mean_bps",
            "impact_bps",
            "impact",
            "k",
            "default_spread_bps",
            "min_half_spread_bps",
            "count",
        }
        impact_mean = _safe_float(
            data.get("impact_mean_bps")
            or data.get("impact_bps")
            or data.get("impact")
        )
        count_val = data.get("count")
        count_int = None
        if isinstance(count_val, (int, float)):
            try:
                count_int = int(count_val)
            except (TypeError, ValueError):
                count_int = None
        return cls(
            multiplier=_safe_float(data.get("multiplier")),
            impact_mean_bps=float(impact_mean) if impact_mean is not None else None,
            k=_safe_float(data.get("k")),
            default_spread_bps=_safe_float(data.get("default_spread_bps")),
            min_half_spread_bps=_safe_float(data.get("min_half_spread_bps")),
            count=count_int,
            extra={key: value for key, value in data.items() if key not in known},
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(self.extra)
        if self.multiplier is not None:
            payload["multiplier"] = float(self.multiplier)
        if self.impact_mean_bps is not None:
            payload["impact_mean_bps"] = float(self.impact_mean_bps)
        if self.k is not None:
            payload["k"] = float(self.k)
        if self.default_spread_bps is not None:
            payload["default_spread_bps"] = float(self.default_spread_bps)
        if self.min_half_spread_bps is not None:
            payload["min_half_spread_bps"] = float(self.min_half_spread_bps)
        if self.count is not None:
            payload["count"] = int(self.count)
        return payload


@dataclass
class SymbolCalibratedProfile:
    symbol: Optional[str] = None
    path: Optional[str] = None
    curve_path: Optional[str] = None
    hourly_path: Optional[str] = None
    regime_path: Optional[str] = None
    impact_curve: tuple[Mapping[str, Any], ...] = ()
    hourly_multipliers: Optional[CalibratedHourlyProfile] = None
    regime_overrides: Dict[str, CalibratedRegimeOverride] = field(default_factory=dict)
    k: Optional[float] = None
    default_spread_bps: Optional[float] = None
    min_half_spread_bps: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    _path_loaded: bool = field(default=False, init=False, repr=False)
    _curve_loaded: bool = field(default=False, init=False, repr=False)
    _hourly_loaded: bool = field(default=False, init=False, repr=False)
    _regime_loaded: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SymbolCalibratedProfile":
        if not isinstance(data, Mapping):
            raise TypeError("symbol profile must be a mapping")
        impact_curve_raw = data.get("impact_curve")
        if impact_curve_raw is None:
            impact_curve_raw = data.get("notional_curve") or data.get("buckets")
        impact_curve: tuple[Mapping[str, Any], ...] = ()
        if isinstance(impact_curve_raw, Iterable) and not isinstance(
            impact_curve_raw, (str, bytes, bytearray)
        ):
            impact_curve = tuple(
                dict(bucket)
                for bucket in impact_curve_raw
                if isinstance(bucket, Mapping)
            )
        hourly_block = data.get("hourly_multipliers") or data.get("hourly")
        hourly_profile: Optional[CalibratedHourlyProfile] = None
        if isinstance(hourly_block, Mapping):
            hourly_profile = CalibratedHourlyProfile.from_dict(hourly_block)
        regime_block = data.get("regime_overrides") or data.get("regime_multipliers")
        regime_values: Dict[str, CalibratedRegimeOverride] = {}
        if isinstance(regime_block, Mapping):
            values = regime_block.get("values") if isinstance(regime_block.get("values"), Mapping) else regime_block
            if isinstance(values, Mapping):
                for key, value in values.items():
                    if not isinstance(value, Mapping):
                        continue
                    regime_key = _normalise_symbol_key(key) or str(key)
                    regime_values[regime_key] = CalibratedRegimeOverride.from_dict(value)
        known = {
            "symbol",
            "path",
            "curve_path",
            "hourly_path",
            "regime_path",
            "impact_curve",
            "notional_curve",
            "buckets",
            "hourly_multipliers",
            "hourly",
            "regime_overrides",
            "regime_multipliers",
            "k",
            "default_spread_bps",
            "min_half_spread_bps",
            "metadata",
        }
        metadata_block = data.get("metadata")
        metadata_dict: Dict[str, Any] = {}
        if isinstance(metadata_block, Mapping):
            metadata_dict = dict(metadata_block)
        return cls(
            symbol=_normalise_symbol_key(data.get("symbol")),
            path=str(data.get("path")) if data.get("path") is not None else None,
            curve_path=str(data.get("curve_path")) if data.get("curve_path") is not None else None,
            hourly_path=str(data.get("hourly_path")) if data.get("hourly_path") is not None else None,
            regime_path=str(data.get("regime_path")) if data.get("regime_path") is not None else None,
            impact_curve=impact_curve,
            hourly_multipliers=hourly_profile,
            regime_overrides={k: v for k, v in regime_values.items() if k is not None},
            k=_safe_float(data.get("k")),
            default_spread_bps=_safe_float(data.get("default_spread_bps")),
            min_half_spread_bps=_safe_float(data.get("min_half_spread_bps")),
            metadata=metadata_dict,
            extra={key: value for key, value in data.items() if key not in known},
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(self.extra)
        if self.symbol is not None:
            payload["symbol"] = str(self.symbol)
        if self.path is not None:
            payload["path"] = str(self.path)
        if self.curve_path is not None:
            payload["curve_path"] = str(self.curve_path)
        if self.hourly_path is not None:
            payload["hourly_path"] = str(self.hourly_path)
        if self.regime_path is not None:
            payload["regime_path"] = str(self.regime_path)
        if self.impact_curve:
            payload["impact_curve"] = [dict(bucket) for bucket in self.impact_curve]
        if self.hourly_multipliers is not None:
            payload["hourly_multipliers"] = self.hourly_multipliers.to_dict()
        if self.regime_overrides:
            payload["regime_overrides"] = {
                key: value.to_dict() for key, value in self.regime_overrides.items()
            }
        if self.k is not None:
            payload["k"] = float(self.k)
        if self.default_spread_bps is not None:
            payload["default_spread_bps"] = float(self.default_spread_bps)
        if self.min_half_spread_bps is not None:
            payload["min_half_spread_bps"] = float(self.min_half_spread_bps)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass
class CalibratedProfilesConfig:
    enabled: bool = False
    path: Optional[str] = None
    symbols: Dict[str, SymbolCalibratedProfile] = field(default_factory=dict)
    default_symbol: Optional[str] = None
    hourly_multipliers: Optional[CalibratedHourlyProfile] = None
    regime_overrides: Dict[str, CalibratedRegimeOverride] = field(default_factory=dict)
    last_refresh_ts: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibratedProfilesConfig":
        if not isinstance(data, Mapping):
            raise TypeError("calibrated profile config must be a mapping")
        symbols_block = data.get("symbols")
        symbols: Dict[str, SymbolCalibratedProfile] = {}
        if isinstance(symbols_block, Mapping):
            for key, value in symbols_block.items():
                if isinstance(value, SymbolCalibratedProfile):
                    profile = value
                elif isinstance(value, Mapping):
                    profile = SymbolCalibratedProfile.from_dict(value)
                else:
                    continue
                symbol_key = _normalise_symbol_key(key) or profile.symbol
                if symbol_key is None:
                    continue
                profile.symbol = symbol_key
                symbols[symbol_key] = profile
        hourly_block = data.get("hourly_multipliers") or data.get("hourly")
        hourly_profile: Optional[CalibratedHourlyProfile] = None
        if isinstance(hourly_block, CalibratedHourlyProfile):
            hourly_profile = hourly_block
        elif isinstance(hourly_block, Mapping):
            hourly_profile = CalibratedHourlyProfile.from_dict(hourly_block)
        regime_block = data.get("regime_overrides")
        regime_overrides: Dict[str, CalibratedRegimeOverride] = {}
        if isinstance(regime_block, Mapping):
            for key, value in regime_block.items():
                if isinstance(value, CalibratedRegimeOverride):
                    override = value
                elif isinstance(value, Mapping):
                    override = CalibratedRegimeOverride.from_dict(value)
                else:
                    continue
                regime_key = _normalise_symbol_key(key) or str(key)
                regime_overrides[regime_key] = override
        known = {
            "enabled",
            "path",
            "data_path",
            "symbols",
            "default_symbol",
            "hourly_multipliers",
            "hourly",
            "regime_overrides",
            "last_refresh_ts",
        }
        extra = {key: value for key, value in data.items() if key not in known}
        path_value = data.get("path") or data.get("data_path")
        last_refresh = data.get("last_refresh_ts")
        last_refresh_int = None
        if isinstance(last_refresh, (int, float)):
            try:
                last_refresh_int = int(last_refresh)
            except (TypeError, ValueError):
                last_refresh_int = None
        cfg = cls(
            enabled=bool(data.get("enabled", bool(symbols))),
            path=str(path_value) if path_value is not None else None,
            symbols=symbols,
            default_symbol=_normalise_symbol_key(data.get("default_symbol")),
            hourly_multipliers=hourly_profile,
            regime_overrides=regime_overrides,
            last_refresh_ts=last_refresh_int,
            extra=extra,
        )
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(self.extra)
        payload["enabled"] = bool(self.enabled)
        if self.path is not None:
            payload["path"] = str(self.path)
        if self.symbols:
            payload["symbols"] = {key: value.to_dict() for key, value in self.symbols.items()}
        if self.default_symbol is not None:
            payload["default_symbol"] = str(self.default_symbol)
        if self.hourly_multipliers is not None:
            payload["hourly_multipliers"] = self.hourly_multipliers.to_dict()
        if self.regime_overrides:
            payload["regime_overrides"] = {
                key: value.to_dict() for key, value in self.regime_overrides.items()
            }
        if self.last_refresh_ts is not None:
            payload["last_refresh_ts"] = int(self.last_refresh_ts)
        return payload

    def get_symbol_profile(self, symbol: Optional[str]) -> Optional[SymbolCalibratedProfile]:
        if not self.symbols:
            return None
        symbol_key = _normalise_symbol_key(symbol)
        if symbol_key and symbol_key in self.symbols:
            return self.symbols[symbol_key]
        if symbol_key:
            for key, value in self.symbols.items():
                if key.upper() == symbol_key:
                    return value
        if self.default_symbol and self.default_symbol in self.symbols:
            return self.symbols[self.default_symbol]
        if len(self.symbols) == 1:
            return next(iter(self.symbols.values()))
        return None

    def get_hourly_profile(self, symbol_profile: Optional[SymbolCalibratedProfile]) -> Optional[CalibratedHourlyProfile]:
        if symbol_profile and symbol_profile.hourly_multipliers is not None:
            return symbol_profile.hourly_multipliers
        return self.hourly_multipliers

    def get_regime_override(
        self,
        symbol_profile: Optional[SymbolCalibratedProfile],
        regime: Any,
    ) -> Optional[CalibratedRegimeOverride]:
        regime_key_candidates: list[str] = []
        if isinstance(regime, str):
            text = regime.strip()
            if text:
                regime_key_candidates.extend({text, text.upper(), text.lower()})
        elif isinstance(regime, (int, float)):
            try:
                regime_key_candidates.append(str(int(regime)))
            except (TypeError, ValueError):
                pass
        else:
            name = getattr(regime, "name", None)
            value = getattr(regime, "value", None)
            if name:
                text = str(name)
                regime_key_candidates.extend({text, text.upper(), text.lower()})
            if value is not None:
                try:
                    regime_key_candidates.append(str(int(value)))
                except (TypeError, ValueError):
                    pass
        if not regime_key_candidates:
            return None
        search_maps: list[Dict[str, CalibratedRegimeOverride]] = []
        if symbol_profile is not None and symbol_profile.regime_overrides:
            search_maps.append(symbol_profile.regime_overrides)
        if self.regime_overrides:
            search_maps.append(self.regime_overrides)
        for mapping in search_maps:
            for candidate in regime_key_candidates:
                for key, value in mapping.items():
                    if key == candidate or (isinstance(key, str) and key.lower() == candidate.lower()):
                        return value
        return None


@dataclass
class SlippageConfig:
    """
    Конфиг слиппеджа «среднего уровня реализма» для среднечастотного бота.
    Формула (в bps):
        slippage_bps = half_spread_bps + k * vol_factor * sqrt( max(size, eps) / max(liquidity, eps) )
    где:
      - half_spread_bps = max(spread_bps * 0.5, min_half_spread_bps)
      - vol_factor: ATR/σ/др. масштаб волатильности, нормированный (например, ATR% за бар)
      - size: абсолютное торгуемое количество (в базовой валюте, штуках)
      - liquidity: прокси ликвидности (например, rolling_volume_shares или ADV в штуках)
    """
    k: float = 0.8
    min_half_spread_bps: float = 0.0
    default_spread_bps: float = 2.0
    eps: float = 1e-12
    dynamic_spread: Optional[DynamicSpreadConfig] = None
    dynamic_impact: DynamicImpactConfig = field(default_factory=DynamicImpactConfig)
    tail_shock: TailShockConfig = field(default_factory=TailShockConfig)
    adv: AdvConfig = field(default_factory=AdvConfig)
    calibrated_profiles: Optional[CalibratedProfilesConfig] = None

    def get_dynamic_block(self) -> Optional[Any]:
        dyn = getattr(self, "dynamic_spread", None)
        if dyn is not None:
            return dyn
        return getattr(self, "dynamic", None)

    def dynamic_trade_cost_enabled(self) -> bool:
        block = self.get_dynamic_block()
        if block is None:
            return False
        if isinstance(block, DynamicSpreadConfig):
            return bool(block.enabled)
        if isinstance(block, dict):
            return bool(block.get("enabled"))
        return bool(getattr(block, "enabled", False))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SlippageConfig":
        dynamic_cfg: Optional[DynamicSpreadConfig] = None
        for key in ("dynamic", "dynamic_spread"):
            block = d.get(key)
            if isinstance(block, dict):
                dynamic_cfg = DynamicSpreadConfig.from_dict(block)
                break
            if isinstance(block, DynamicSpreadConfig):
                dynamic_cfg = block
                break

        dynamic_impact_cfg: DynamicImpactConfig
        impact_block = d.get("dynamic_impact")
        if isinstance(impact_block, DynamicImpactConfig):
            dynamic_impact_cfg = impact_block
        elif isinstance(impact_block, dict):
            dynamic_impact_cfg = DynamicImpactConfig.from_dict(impact_block)
        else:
            dynamic_impact_cfg = DynamicImpactConfig()

        tail_shock_cfg: TailShockConfig
        tail_block = d.get("tail_shock")
        if isinstance(tail_block, TailShockConfig):
            tail_shock_cfg = tail_block
        elif isinstance(tail_block, dict):
            tail_shock_cfg = TailShockConfig.from_dict(tail_block)
        else:
            tail_shock_cfg = TailShockConfig()

        adv_cfg: AdvConfig
        adv_block = d.get("adv")
        if isinstance(adv_block, AdvConfig):
            adv_cfg = adv_block
        elif isinstance(adv_block, dict):
            adv_cfg = AdvConfig.from_dict(adv_block)
        else:
            adv_cfg = AdvConfig()

        calibrated_cfg: Optional[CalibratedProfilesConfig] = None
        calib_block = (
            d.get("calibrated_profiles")
            or d.get("calibrated")
            or d.get("calibration")
        )
        if isinstance(calib_block, CalibratedProfilesConfig):
            calibrated_cfg = calib_block
        elif isinstance(calib_block, Mapping):
            calibrated_cfg = CalibratedProfilesConfig.from_dict(calib_block)

        return cls(
            k=float(d.get("k", 0.8)),
            min_half_spread_bps=float(d.get("min_half_spread_bps", 0.0)),
            default_spread_bps=float(d.get("default_spread_bps", 2.0)),
            eps=float(d.get("eps", 1e-12)),
            dynamic_spread=dynamic_cfg,
            dynamic_impact=dynamic_impact_cfg,
            tail_shock=tail_shock_cfg,
            adv=adv_cfg,
            calibrated_profiles=calibrated_cfg,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "k": float(self.k),
            "min_half_spread_bps": float(self.min_half_spread_bps),
            "default_spread_bps": float(self.default_spread_bps),
            "eps": float(self.eps),
        }
        if self.dynamic_spread is not None:
            dyn_dict = self.dynamic_spread.to_dict()
            data["dynamic"] = dict(dyn_dict)
            data.setdefault("dynamic_spread", dict(dyn_dict))
        if self.dynamic_impact is not None:
            data["dynamic_impact"] = self.dynamic_impact.to_dict()
        if self.tail_shock is not None:
            data["tail_shock"] = self.tail_shock.to_dict()
        if self.adv is not None:
            data["adv"] = self.adv.to_dict()
        if self.calibrated_profiles is not None:
            payload = self.calibrated_profiles.to_dict()
            data["calibrated_profiles"] = payload
            data.setdefault("calibrated", dict(payload))
        return data

    @classmethod
    def from_file(cls, path: str) -> "SlippageConfig":
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):  # pragma: no cover - sanity check
            raise ValueError("slippage config file must contain a JSON object")
        return cls.from_dict(data)


def _hour_of_week_from_timestamp(ts_ms: Optional[Any]) -> Optional[int]:
    if ts_ms is None:
        return None
    try:
        ts_val = int(ts_ms)
    except (TypeError, ValueError):
        return None
    try:
        dt = datetime.fromtimestamp(ts_val / 1000.0, tz=timezone.utc)
    except Exception:
        return None
    return int(dt.weekday()) * 24 + int(dt.hour)


def _curve_impact(
    curve: Sequence[Mapping[str, Any]],
    *,
    notional: float,
    half_spread_bps: float,
) -> Optional[float]:
    if not curve:
        return None
    buckets: list[tuple[float, Optional[float], Optional[float], Mapping[str, Any]]] = []
    for bucket in curve:
        if not isinstance(bucket, Mapping):
            continue
        lower = _safe_float(
            bucket.get("lower_notional")
            or bucket.get("lower")
            or bucket.get("min_notional")
            or bucket.get("lower_size")
        )
        upper = _safe_float(
            bucket.get("upper_notional")
            or bucket.get("upper")
            or bucket.get("max_notional")
            or bucket.get("upper_size")
        )
        mean_notional = _safe_float(
            bucket.get("mean_notional") or bucket.get("median_notional")
        )
        key = lower
        if key is None:
            key = mean_notional
        if key is None:
            key = 0.0
        buckets.append((float(key), lower, upper, bucket))
    if not buckets:
        return None
    buckets.sort(key=lambda item: item[0])
    selected: Optional[Mapping[str, Any]] = None
    value = float(notional)
    for _, lower, upper, bucket in buckets:
        lower_bound = lower if lower is not None else None
        upper_bound = upper if upper is not None else None
        if lower_bound is not None and value < lower_bound:
            continue
        if upper_bound is not None and value > upper_bound:
            continue
        selected = bucket
        break
    if selected is None:
        selected = buckets[-1][3]
    if selected is None:
        return None
    impact = _safe_float(
        selected.get("mean_impact_bps")
        or selected.get("impact_mean_bps")
        or selected.get("impact_bps")
    )
    if impact is not None:
        return float(max(0.0, impact))
    slip = _safe_float(
        selected.get("mean_slippage_bps") or selected.get("slippage_bps")
    )
    if slip is None:
        return None
    value = float(slip) - float(half_spread_bps)
    if value < 0.0:
        value = 0.0
    return float(value)


def _estimate_calibrated_slippage(
    *,
    cfg: SlippageConfig,
    symbol: Optional[str],
    spread_bps: Optional[float],
    size: float,
    liquidity: Optional[float],
    vol_factor: Optional[float],
    ts_ms: Optional[Any],
    market_regime: Optional[Any],
    hour_of_week: Optional[int],
    notional: Optional[float],
) -> Optional[float]:
    calibrated = getattr(cfg, "calibrated_profiles", None)
    if calibrated is None:
        return None
    if not getattr(calibrated, "enabled", False) and not calibrated.symbols:
        return None
    profile = calibrated.get_symbol_profile(symbol)
    if profile is None:
        return None
    spread_default = _safe_float(spread_bps)
    if spread_default is None:
        if profile.default_spread_bps is not None:
            spread_default = float(profile.default_spread_bps)
        else:
            spread_default = float(cfg.default_spread_bps)
    min_half_candidates = [float(cfg.min_half_spread_bps)]
    if profile.min_half_spread_bps is not None:
        min_half_candidates.append(float(profile.min_half_spread_bps))
    regime_override = calibrated.get_regime_override(profile, market_regime)
    regime_multiplier = 1.0
    override_k = None
    override_impact = None
    if regime_override is not None:
        if regime_override.default_spread_bps is not None:
            spread_default = float(regime_override.default_spread_bps)
        if regime_override.min_half_spread_bps is not None:
            min_half_candidates.append(float(regime_override.min_half_spread_bps))
        if regime_override.multiplier is not None:
            regime_multiplier = float(regime_override.multiplier)
        if regime_override.k is not None:
            override_k = float(regime_override.k)
        if regime_override.impact_mean_bps is not None:
            override_impact = float(regime_override.impact_mean_bps)
    min_half_spread = max(float(candidate) for candidate in min_half_candidates if candidate is not None)
    base_spread = float(spread_default)
    half_spread = max(0.5 * base_spread, float(min_half_spread))
    notional_val = _safe_float(notional)
    if notional_val is None:
        notional_val = _safe_float(size)
    if notional_val is None:
        notional_val = 0.0
    impact = _curve_impact(
        profile.impact_curve,
        notional=abs(float(notional_val)),
        half_spread_bps=half_spread,
    )
    hour_profile = calibrated.get_hourly_profile(profile)
    hour_index = hour_of_week if hour_of_week is not None else _hour_of_week_from_timestamp(ts_ms)
    hour_multiplier = 1.0
    if hour_profile is not None:
        hour_multiplier = float(hour_profile.get_multiplier(hour_index))
    vf = _safe_float(vol_factor)
    if vf is None or vf <= 0.0:
        vf = 1.0
    if impact is None:
        if override_impact is not None:
            impact = max(0.0, float(override_impact))
        else:
            size_val = _safe_float(size)
            liquidity_val = _safe_float(liquidity)
            if size_val is None or size_val <= 0.0:
                size_val = 0.0
            if liquidity_val is None or liquidity_val <= 0.0:
                liquidity_val = 1.0
            k_base = profile.k if profile.k is not None else cfg.k
            if override_k is not None:
                k_base = override_k
            k_base = float(k_base)
            if regime_multiplier != 1.0:
                k_base *= float(regime_multiplier)
            impact = k_base * float(vf) * math.sqrt(
                max(abs(float(size_val)), cfg.eps) / max(abs(float(liquidity_val)), cfg.eps)
            )
    else:
        impact = float(impact) * float(vf)
        if regime_multiplier != 1.0:
            impact *= float(regime_multiplier)
    if hour_multiplier != 1.0:
        impact *= float(hour_multiplier)
    total = half_spread + float(impact)
    if not math.isfinite(total):
        return None
    if total < 0.0:
        total = 0.0
    return float(total)


def estimate_slippage_bps(
    *,
    spread_bps: Optional[float],
    size: float,
    liquidity: Optional[float],
    vol_factor: Optional[float],
    cfg: SlippageConfig,
    symbol: Optional[str] = None,
    ts_ms: Optional[Any] = None,
    market_regime: Optional[Any] = None,
    hour_of_week: Optional[int] = None,
    notional: Optional[float] = None,
) -> float:
    """
    Оценка слиппеджа в bps по простой калибруемой формуле.
    Если spread_bps/liquidity/vol_factor отсутствуют — используются дефолты/единицы.
    """
    calibrated_estimate = _estimate_calibrated_slippage(
        cfg=cfg,
        symbol=symbol,
        spread_bps=spread_bps,
        size=size,
        liquidity=liquidity,
        vol_factor=vol_factor,
        ts_ms=ts_ms,
        market_regime=market_regime,
        hour_of_week=hour_of_week,
        notional=notional,
    )
    if calibrated_estimate is not None:
        return float(calibrated_estimate)
    try:
        size_val = float(size)
    except (TypeError, ValueError):
        size_val = 0.0

    delegate = getattr(cfg, "get_trade_cost_bps", None)
    dynamic_enabled = False
    if callable(delegate):
        detector = getattr(cfg, "dynamic_trade_cost_enabled", None)
        if callable(detector):
            try:
                dynamic_enabled = bool(detector())
            except Exception:
                dynamic_enabled = False
        if not dynamic_enabled:
            block: Any = None
            getter = getattr(cfg, "get_dynamic_block", None)
            if callable(getter):
                try:
                    block = getter()
                except Exception:
                    block = None
            if block is None:
                block = getattr(cfg, "dynamic_spread", None)
            if block is None:
                block = getattr(cfg, "dynamic", None)
            if block is not None:
                if isinstance(block, dict):
                    dynamic_enabled = bool(block.get("enabled"))
                else:
                    dynamic_enabled = bool(getattr(block, "enabled", False))
        if dynamic_enabled:
            side = "BUY" if size_val >= 0.0 else "SELL"
            qty = abs(size_val)
            vol_metrics_payload: Optional[Dict[str, float]] = None
            vol_payload: Dict[str, float] = {}
            if vol_factor is not None:
                try:
                    vf_val = float(vol_factor)
                except (TypeError, ValueError):
                    vf_val = None
                else:
                    if math.isfinite(vf_val):
                        vol_payload["vol_factor"] = vf_val
            if liquidity is not None:
                try:
                    liq_val = float(liquidity)
                except (TypeError, ValueError):
                    liq_val = None
                else:
                    if math.isfinite(liq_val):
                        vol_payload["liquidity"] = liq_val
            if vol_payload:
                vol_metrics_payload = vol_payload
            kwargs: Dict[str, Any] = {
                "side": side,
                "qty": qty,
                "mid": None,
                "spread_bps": spread_bps,
                "bar_close_ts": None,
                "order_seq": 0,
            }
            if vol_metrics_payload is not None:
                kwargs["vol_metrics"] = vol_metrics_payload
            try:
                result = delegate(**kwargs)
            except TypeError:
                kwargs.pop("order_seq", None)
                try:
                    result = delegate(**kwargs)
                except TypeError:
                    kwargs.pop("vol_metrics", None)
                    result = delegate(**kwargs)
                except Exception:
                    result = None
            except Exception:
                result = None
            if result is not None:
                try:
                    candidate = float(result)
                except (TypeError, ValueError):
                    candidate = None
                else:
                    if math.isfinite(candidate):
                        return float(candidate)

    sbps = (
        float(spread_bps)
        if (spread_bps is not None and math.isfinite(float(spread_bps)))
        else float(cfg.default_spread_bps)
    )
    half_spread_bps = max(0.5 * sbps, float(cfg.min_half_spread_bps))

    vf = float(vol_factor) if (vol_factor is not None and math.isfinite(float(vol_factor))) else 1.0
    liq = float(liquidity) if (liquidity is not None and float(liquidity) > 0.0 and math.isfinite(float(liquidity))) else 1.0
    sz = abs(size_val)

    impact_term = float(cfg.k) * vf * math.sqrt(max(sz, cfg.eps) / max(liq, cfg.eps))
    return float(half_spread_bps + impact_term)


def apply_slippage_price(*, side: str, quote_price: float, slippage_bps: float) -> float:
    """
    Применить слиппедж к котировке:
      - для BUY цена ухудшается (увеличивается)
      - для SELL цена ухудшается (уменьшается)
    """
    q = float(quote_price)
    bps = float(slippage_bps) / 1e4
    if str(side).upper() == "BUY":
        return float(q * (1.0 + bps))
    else:
        return float(q * (1.0 - bps))


def compute_spread_bps_from_quotes(*, bid: Optional[float], ask: Optional[float], cfg: SlippageConfig) -> float:
    """
    Рассчитать spread_bps из котировок. Если данных нет — вернуть cfg.default_spread_bps.
    """
    if bid is None or ask is None:
        return float(cfg.default_spread_bps)
    b = float(bid)
    a = float(ask)
    if not (math.isfinite(b) and math.isfinite(a)) or a <= 0.0:
        return float(cfg.default_spread_bps)
    return float((a - b) / a * 1e4)


def mid_from_quotes(*, bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    b = float(bid)
    a = float(ask)
    if not (math.isfinite(b) and math.isfinite(a)):
        return None
    return float((a + b) / 2.0)


def model_curve(
    participations: Sequence[float],
    *,
    cfg: SlippageConfig,
    spread_bps: float,
    vol_factor: float = 1.0,
) -> list[float]:
    """Return expected slippage for a range of participation rates.

    ``participations`` are interpreted as size/liquidity ratios.  The model is
    evaluated with ``liquidity=1`` and ``size=participation`` for each value.
    """

    out = []
    for p in participations:
        s = estimate_slippage_bps(
            spread_bps=spread_bps,
            size=float(p),
            liquidity=1.0,
            vol_factor=vol_factor,
            cfg=cfg,
        )
        out.append(float(s))
    return out
