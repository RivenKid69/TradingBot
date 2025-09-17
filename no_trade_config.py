from __future__ import annotations

"""Utility helpers for loading no-trade configuration.

This module provides :func:`get_no_trade_config` which reads the ``no_trade``
section from a YAML file and returns a :class:`NoTradeConfig` object.  All
consumers should use this function so that the configuration is loaded from a
single source of truth.
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


DEFAULT_NO_TRADE_STATE_PATH = Path("state/no_trade_state.json")


class DynamicGuardMetricConfig(BaseModel):
    """Base configuration for a dynamic metric window."""

    window: Optional[int] = None
    min_periods: Optional[int] = None
    pctile_window: Optional[int] = None
    pctile_min_periods: Optional[int] = None
    abs: Optional[float] = None
    pctile: Optional[float] = None
    upper_multiplier: Optional[float] = None
    lower_multiplier: Optional[float] = None
    upper_pctile: Optional[float] = None
    lower_pctile: Optional[float] = None
    cooldown_bars: Optional[int] = None


class DynamicGuardSpreadConfig(DynamicGuardMetricConfig):
    """Spread specific extensions."""

    abs_bps: Optional[float] = None


class DynamicGuardVolatilityConfig(DynamicGuardMetricConfig):
    """Volatility specific extensions."""

    pass


class DynamicGuardConfig(BaseModel):
    """Configuration for a dynamic no-trade guard."""

    enable: bool = False
    requested_enable: Optional[bool] = None
    enabled: Optional[bool] = None
    sigma_window: Optional[int] = None
    sigma_min_periods: Optional[int] = None
    vol_pctile_window: Optional[int] = None
    vol_pctile_min_periods: Optional[int] = None
    atr_window: Optional[int] = None
    atr_min_periods: Optional[int] = None
    spread_pctile_window: Optional[int] = None
    spread_pctile_min_periods: Optional[int] = None
    vol_abs: Optional[float] = None
    vol_pctile: Optional[float] = None
    spread_abs_bps: Optional[float] = None
    spread_pctile: Optional[float] = None
    hysteresis: Optional[float] = None
    cooldown_bars: int = 0
    next_bars_block: Dict[str, int] = Field(default_factory=dict)
    log_reason: bool = False
    volatility: DynamicGuardVolatilityConfig = Field(default_factory=DynamicGuardVolatilityConfig)
    spread: DynamicGuardSpreadConfig = Field(default_factory=DynamicGuardSpreadConfig)


class DynamicHysteresisConfig(BaseModel):
    """Tuning for guard release behaviour."""

    ratio: Optional[float] = None
    cooldown_bars: Optional[int] = None


class DynamicConfig(BaseModel):
    """Structured dynamic guard configuration."""

    enabled: Optional[bool] = None
    guard: DynamicGuardConfig = Field(default_factory=DynamicGuardConfig)
    hysteresis: DynamicHysteresisConfig = Field(default_factory=DynamicHysteresisConfig)
    next_bars_block: Dict[str, int] = Field(default_factory=dict)


class MaintenanceConfig(BaseModel):
    """Time windows for scheduled maintenance."""

    format: str = "HH:MM-HH:MM"
    path: Optional[str] = None
    max_age_sec: Optional[int] = None
    max_age_hours: Optional[float] = None
    funding_buffer_min: int = 0
    daily_utc: List[str] = Field(default_factory=list)
    custom_ms: List[Dict[str, int]] = Field(default_factory=list)


class NoTradeConfig(BaseModel):
    """Pydantic model for the ``no_trade`` section."""

    funding_buffer_min: int = 0
    daily_utc: List[str] = Field(default_factory=list)
    custom_ms: List[Dict[str, int]] = Field(default_factory=list)
    dynamic_guard: DynamicGuardConfig = Field(default_factory=DynamicGuardConfig)
    maintenance: MaintenanceConfig = Field(default_factory=MaintenanceConfig)
    dynamic: DynamicConfig = Field(default_factory=DynamicConfig)


class NoTradeState(BaseModel):
    """Persisted state for online anomaly-driven no-trade rules."""

    anomaly_block_until_ts: Dict[str, int] = Field(default_factory=dict)
    dynamic_guard: Dict[str, Any] = Field(default_factory=dict)


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(value)
    except Exception:  # pragma: no cover - defensive branch
        return None


def _ensure_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_no_trade_payload(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge legacy and structured configuration layouts."""

    maintenance = _ensure_mapping(raw.get("maintenance"))
    if "format" in maintenance:
        fmt = maintenance.get("format")
        maintenance["format"] = str(fmt) if fmt is not None else "HH:MM-HH:MM"
    else:
        maintenance["format"] = "HH:MM-HH:MM"

    for key in ("funding_buffer_min", "daily_utc", "custom_ms", "path", "max_age_sec", "max_age_hours"):
        if key not in maintenance and raw.get(key) is not None:
            maintenance[key] = raw[key]

    funding_buffer = maintenance.get("funding_buffer_min")
    maintenance["funding_buffer_min"] = int(funding_buffer or 0)
    maintenance.setdefault("daily_utc", [])
    maintenance.setdefault("custom_ms", [])
    maintenance["daily_utc"] = list(maintenance.get("daily_utc") or [])
    maintenance["custom_ms"] = list(maintenance.get("custom_ms") or [])
    if maintenance.get("path"):
        maintenance["path"] = str(maintenance.get("path"))
    else:
        maintenance["path"] = None

    max_age_sec = _coerce_int(maintenance.get("max_age_sec"))
    max_age_hours = _coerce_float(maintenance.get("max_age_hours"))
    maintenance["max_age_sec"] = max_age_sec
    maintenance["max_age_hours"] = max_age_hours

    dynamic = _ensure_mapping(raw.get("dynamic"))
    legacy_guard = _ensure_mapping(raw.get("dynamic_guard"))
    nested_guard = _ensure_mapping(dynamic.get("guard"))
    guard_data: Dict[str, Any] = {**legacy_guard, **nested_guard}

    # Normalise enable flags
    if "enabled" in guard_data and "enable" not in guard_data:
        guard_data["enable"] = guard_data.get("enabled")
    guard_enable = bool(guard_data.get("enable", False))
    guard_data["enable"] = guard_enable
    guard_data["requested_enable"] = guard_enable
    guard_data["enabled"] = guard_enable

    # Extract structured metric configuration
    volatility_cfg = _ensure_mapping(guard_data.pop("volatility", {}))
    spread_cfg = _ensure_mapping(guard_data.pop("spread", {}))

    def _extract_metric(
        *,
        block: Mapping[str, Any],
        window_key: str,
        min_periods_key: str,
        pctile_window_key: str,
        pctile_min_key: str,
        abs_key: str,
        abs_alias: Optional[str] = None,
        pctile_key: str,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[float],
        Optional[float],
        Dict[str, Any],
    ]:
        window_val = _coerce_int(block.get("window"))
        if window_val is None:
            window_val = _coerce_int(guard_data.get(window_key))

        min_periods_val = _coerce_int(block.get("min_periods"))
        if min_periods_val is None:
            min_periods_val = _coerce_int(guard_data.get(min_periods_key))

        pctile_window_val = _coerce_int(block.get("pctile_window"))
        if pctile_window_val is None:
            pctile_window_val = _coerce_int(guard_data.get(pctile_window_key))

        pctile_min_val = _coerce_int(block.get("pctile_min_periods"))
        if pctile_min_val is None:
            pctile_min_val = _coerce_int(guard_data.get(pctile_min_key))

        abs_val: Optional[float] = None
        if abs_alias and block.get(abs_alias) is not None:
            abs_val = _coerce_float(block.get(abs_alias))
        if abs_val is None:
            abs_val = _coerce_float(block.get("abs"))
        if abs_val is None:
            abs_val = _coerce_float(guard_data.get(abs_key))

        pctile_val = _coerce_float(block.get("pctile"))
        if pctile_val is None:
            pctile_val = _coerce_float(guard_data.get(pctile_key))

        extras: Dict[str, Any] = {}

        def _pick_float(*keys: str) -> Optional[float]:
            for key in keys:
                if key in block and block.get(key) is not None:
                    val = _coerce_float(block.get(key))
                    if val is not None:
                        return val
            for key in keys:
                if key in guard_data and guard_data.get(key) is not None:
                    val = _coerce_float(guard_data.get(key))
                    if val is not None:
                        return val
            return None

        def _pick_int(*keys: str) -> Optional[int]:
            for key in keys:
                if key in block and block.get(key) is not None:
                    val = _coerce_int(block.get(key))
                    if val is not None:
                        return val
            for key in keys:
                if key in guard_data and guard_data.get(key) is not None:
                    val = _coerce_int(guard_data.get(key))
                    if val is not None:
                        return val
            return None

        upper_mult = _pick_float("upper_multiplier", "upper_mult")
        if upper_mult is not None:
            extras["upper_multiplier"] = upper_mult

        lower_mult = _pick_float("lower_multiplier", "lower_mult")
        if lower_mult is not None:
            extras["lower_multiplier"] = lower_mult

        cooldown_override = _pick_int("cooldown_bars")
        if cooldown_override is not None:
            extras["cooldown_bars"] = cooldown_override

        upper_pct = _pick_float("upper_pctile", "upper_percentile")
        if upper_pct is not None:
            extras["upper_pctile"] = upper_pct

        lower_pct = _pick_float("lower_pctile", "lower_percentile")
        if lower_pct is not None:
            extras["lower_pctile"] = lower_pct

        return (
            window_val,
            min_periods_val,
            pctile_window_val,
            pctile_min_val,
            abs_val,
            pctile_val,
            extras,
        )

    (
        sigma_window,
        sigma_min_periods,
        vol_pctile_window,
        vol_pctile_min_periods,
        vol_abs,
        vol_pctile,
        vol_extras,
    ) = _extract_metric(
        block=volatility_cfg,
        window_key="sigma_window",
        min_periods_key="sigma_min_periods",
        pctile_window_key="vol_pctile_window",
        pctile_min_key="vol_pctile_min_periods",
        abs_key="vol_abs",
        pctile_key="vol_pctile",
    )

    (
        atr_window,
        atr_min_periods,
        spread_pctile_window,
        spread_pctile_min_periods,
        spread_abs,
        spread_pctile,
        spread_extras,
    ) = _extract_metric(
        block=spread_cfg,
        window_key="atr_window",
        min_periods_key="atr_min_periods",
        pctile_window_key="spread_pctile_window",
        pctile_min_key="spread_pctile_min_periods",
        abs_key="spread_abs_bps",
        abs_alias="abs_bps",
        pctile_key="spread_pctile",
    )

    guard_data["sigma_window"] = sigma_window
    guard_data["sigma_min_periods"] = sigma_min_periods
    guard_data["vol_pctile_window"] = vol_pctile_window
    guard_data["vol_pctile_min_periods"] = vol_pctile_min_periods
    guard_data["vol_abs"] = vol_abs
    guard_data["vol_pctile"] = vol_pctile

    guard_data["atr_window"] = atr_window
    guard_data["atr_min_periods"] = atr_min_periods
    guard_data["spread_pctile_window"] = spread_pctile_window
    guard_data["spread_pctile_min_periods"] = spread_pctile_min_periods
    guard_data["spread_abs_bps"] = spread_abs
    guard_data["spread_pctile"] = spread_pctile

    guard_data["volatility"] = {
        "window": sigma_window,
        "min_periods": sigma_min_periods,
        "pctile_window": vol_pctile_window,
        "pctile_min_periods": vol_pctile_min_periods,
        "abs": vol_abs,
        "pctile": vol_pctile,
    }
    guard_data["volatility"].update(vol_extras)
    guard_data["spread"] = {
        "window": atr_window,
        "min_periods": atr_min_periods,
        "pctile_window": spread_pctile_window,
        "pctile_min_periods": spread_pctile_min_periods,
        "abs": spread_abs,
        "abs_bps": spread_abs,
        "pctile": spread_pctile,
    }
    guard_data["spread"].update(spread_extras)

    # Extract hysteresis configuration
    hysteresis_cfg = _ensure_mapping(dynamic.get("hysteresis"))
    ratio = _coerce_float(hysteresis_cfg.get("ratio"))
    cooldown = _coerce_int(hysteresis_cfg.get("cooldown_bars"))

    if ratio is None and "hysteresis" in guard_data:
        ratio = _coerce_float(guard_data.get("hysteresis"))
    if cooldown is None and "cooldown_bars" in guard_data:
        cooldown = _coerce_int(guard_data.get("cooldown_bars"))

    if ratio is not None:
        guard_data["hysteresis"] = ratio
    else:
        guard_data.pop("hysteresis", None)

    guard_data["cooldown_bars"] = int(cooldown or 0)

    # Extract next bars block map
    next_block: Dict[str, int] = {}

    def _update_next_block(source: Any) -> None:
        mapping = _ensure_mapping(source)
        for key, value in mapping.items():
            ivalue = _coerce_int(value)
            if ivalue is not None:
                next_block[str(key)] = ivalue

    _update_next_block(raw.get("next_bars_block"))
    _update_next_block(dynamic.get("next_bars_block"))
    if "next_bars_block" in guard_data:
        _update_next_block(guard_data.pop("next_bars_block"))

    dynamic_enabled = dynamic.get("enabled")
    if dynamic_enabled is None:
        dynamic_enabled = guard_enable

    dynamic_payload: Dict[str, Any] = {
        "enabled": bool(dynamic_enabled) if dynamic_enabled is not None else bool(guard_enable),
        "guard": guard_data,
        "hysteresis": {},
        "next_bars_block": next_block,
    }
    if ratio is not None:
        dynamic_payload["hysteresis"]["ratio"] = ratio
    if cooldown is not None:
        dynamic_payload["hysteresis"]["cooldown_bars"] = cooldown

    return {
        "funding_buffer_min": maintenance["funding_buffer_min"],
        "daily_utc": maintenance["daily_utc"],
        "custom_ms": maintenance["custom_ms"],
        "dynamic_guard": dict(guard_data),
        "maintenance": maintenance,
        "dynamic": dynamic_payload,
    }


def get_no_trade_config(path: str) -> NoTradeConfig:
    """Load :class:`NoTradeConfig` from ``path``.

    Parameters
    ----------
    path:
        Path to a YAML file containing a top-level ``no_trade`` section.
    """

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_cfg = _ensure_mapping(data.get("no_trade"))
    normalised = _normalise_no_trade_payload(raw_cfg)
    config = NoTradeConfig(**normalised)

    config_path = Path(path)
    try:
        resolved_config_path = config_path.resolve()
    except Exception:  # pragma: no cover - defensive
        resolved_config_path = config_path
    base_dir = resolved_config_path.parent

    try:
        config.__dict__["_config_path"] = str(resolved_config_path)
        config.__dict__["_config_base_dir"] = str(base_dir)
    except Exception:  # pragma: no cover - defensive
        pass

    maintenance_cfg = config.maintenance
    try:
        maintenance_cfg.__dict__["_config_path"] = str(resolved_config_path)
        maintenance_cfg.__dict__["_config_base_dir"] = str(base_dir)
    except Exception:  # pragma: no cover - defensive
        pass

    if maintenance_cfg.path:
        try:
            maintenance_cfg.__dict__["_path_source"] = str(maintenance_cfg.path)
        except Exception:  # pragma: no cover - defensive
            maintenance_cfg.__dict__["_path_source"] = maintenance_cfg.path
        try:
            resolved = Path(str(maintenance_cfg.path))
            if not resolved.is_absolute():
                resolved = (base_dir / resolved).resolve(strict=False)
            else:
                resolved = resolved.resolve(strict=False)
            maintenance_cfg.path = str(resolved)
        except Exception:  # pragma: no cover - defensive
            maintenance_cfg.path = str(maintenance_cfg.path)
    else:
        try:
            maintenance_cfg.__dict__["_path_source"] = None
        except Exception:  # pragma: no cover - defensive
            pass

    if maintenance_cfg.max_age_sec is None and maintenance_cfg.max_age_hours is not None:
        try:
            maintenance_cfg.max_age_sec = int(float(maintenance_cfg.max_age_hours) * 3600)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            maintenance_cfg.max_age_sec = None
    elif maintenance_cfg.max_age_hours is None and maintenance_cfg.max_age_sec is not None:
        try:
            maintenance_cfg.max_age_hours = float(maintenance_cfg.max_age_sec) / 3600.0
        except (TypeError, ValueError):  # pragma: no cover - defensive
            maintenance_cfg.max_age_hours = None

    guard = config.dynamic.guard
    hysteresis_cfg = config.dynamic.hysteresis

    requested_enable = guard.requested_enable
    if requested_enable is None:
        requested_enable = bool(guard.enable)
    guard.requested_enable = bool(requested_enable)

    if config.dynamic.enabled is None:
        config.dynamic.enabled = bool(requested_enable)
    else:
        config.dynamic.enabled = bool(config.dynamic.enabled)

    effective_enable = bool(config.dynamic.enabled and requested_enable)
    guard.enabled = effective_enable
    guard.enable = effective_enable

    if hysteresis_cfg.ratio is None:
        hysteresis_cfg.ratio = guard.hysteresis
    else:
        guard.hysteresis = hysteresis_cfg.ratio

    if hysteresis_cfg.cooldown_bars is None:
        hysteresis_cfg.cooldown_bars = guard.cooldown_bars
    else:
        guard.cooldown_bars = hysteresis_cfg.cooldown_bars

    vol_cfg = guard.volatility or DynamicGuardVolatilityConfig()
    if vol_cfg.window is None:
        vol_cfg.window = guard.sigma_window
    elif guard.sigma_window is None:
        guard.sigma_window = vol_cfg.window

    if vol_cfg.min_periods is None:
        vol_cfg.min_periods = guard.sigma_min_periods
    elif guard.sigma_min_periods is None:
        guard.sigma_min_periods = vol_cfg.min_periods

    if vol_cfg.pctile_window is None:
        vol_cfg.pctile_window = guard.vol_pctile_window
    elif guard.vol_pctile_window is None:
        guard.vol_pctile_window = vol_cfg.pctile_window

    if vol_cfg.pctile_min_periods is None:
        vol_cfg.pctile_min_periods = guard.vol_pctile_min_periods
    elif guard.vol_pctile_min_periods is None:
        guard.vol_pctile_min_periods = vol_cfg.pctile_min_periods

    if vol_cfg.abs is None:
        vol_cfg.abs = guard.vol_abs
    elif guard.vol_abs is None:
        guard.vol_abs = vol_cfg.abs

    if vol_cfg.pctile is None:
        vol_cfg.pctile = guard.vol_pctile
    elif guard.vol_pctile is None:
        guard.vol_pctile = vol_cfg.pctile

    guard.volatility = vol_cfg

    spread_cfg = guard.spread or DynamicGuardSpreadConfig()
    if spread_cfg.window is None:
        spread_cfg.window = guard.atr_window
    elif guard.atr_window is None:
        guard.atr_window = spread_cfg.window

    if spread_cfg.min_periods is None:
        spread_cfg.min_periods = guard.atr_min_periods
    elif guard.atr_min_periods is None:
        guard.atr_min_periods = spread_cfg.min_periods

    if spread_cfg.pctile_window is None:
        spread_cfg.pctile_window = guard.spread_pctile_window
    elif guard.spread_pctile_window is None:
        guard.spread_pctile_window = spread_cfg.pctile_window

    if spread_cfg.pctile_min_periods is None:
        spread_cfg.pctile_min_periods = guard.spread_pctile_min_periods
    elif guard.spread_pctile_min_periods is None:
        guard.spread_pctile_min_periods = spread_cfg.pctile_min_periods

    spread_abs = guard.spread_abs_bps
    if spread_cfg.abs_bps is None and spread_cfg.abs is not None:
        spread_cfg.abs_bps = spread_cfg.abs
    if spread_cfg.abs_bps is None:
        spread_cfg.abs_bps = spread_abs
    elif spread_abs is None:
        spread_abs = spread_cfg.abs_bps

    if spread_cfg.abs is None and spread_cfg.abs_bps is not None:
        spread_cfg.abs = spread_cfg.abs_bps

    guard.spread_abs_bps = spread_abs

    if spread_cfg.pctile is None:
        spread_cfg.pctile = guard.spread_pctile
    elif guard.spread_pctile is None:
        guard.spread_pctile = spread_cfg.pctile

    guard.spread = spread_cfg

    if config.dynamic.next_bars_block:
        guard.next_bars_block = dict(config.dynamic.next_bars_block)
    else:
        config.dynamic.next_bars_block = dict(guard.next_bars_block)

    config.dynamic_guard = guard
    config.funding_buffer_min = config.maintenance.funding_buffer_min
    config.daily_utc = list(config.maintenance.daily_utc)
    config.custom_ms = list(config.maintenance.custom_ms)

    return config


def _iter_anomaly_entries(raw: Any) -> Iterable[Tuple[str, int]]:
    """Yield ``(symbol, timestamp)`` pairs from *raw* state payload."""

    if isinstance(raw, Mapping):
        for key, value in raw.items():
            ts = _coerce_int(value)
            symbol = _coerce_str(key)
            if symbol and ts is not None:
                yield symbol, ts
        return

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            symbol = _coerce_str(
                item.get("symbol")
                or item.get("pair")
                or item.get("instrument")
            )
            if not symbol:
                continue
            ts = (
                _coerce_int(item.get("block_until_ts"))
                or _coerce_int(item.get("ts"))
                or _coerce_int(item.get("timestamp"))
                or _coerce_int(item.get("timestamp_ms"))
            )
            if ts is None:
                continue
            yield symbol, ts


def _parse_anomaly_state(payload: Mapping[str, Any]) -> Dict[str, int]:
    """Normalise anomaly state from multiple legacy layouts."""

    candidates: List[Any] = []
    for key in (
        "anomaly_block_until_ts",
        "anomaly_block_until_ts_ms",
        "anomaly_block_until",
        "anomaly_state",
    ):
        if key in payload:
            candidates.append(payload[key])

    if not candidates:
        # Legacy format stored the map at the top level: ``{"BTCUSDT": 123}``.
        if all(_coerce_int(v) is not None for v in payload.values()):
            candidates.append(payload)

    result: Dict[str, int] = {}
    for candidate in candidates:
        for symbol, ts in _iter_anomaly_entries(candidate):
            result[symbol] = ts
    return result


def _parse_dynamic_guard_state(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract persisted dynamic guard metadata."""

    raw = payload.get("dynamic_guard") if isinstance(payload, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for symbol, entry in raw.items():
        if not isinstance(entry, Mapping):
            continue
        symbol_key = _coerce_str(symbol)
        if not symbol_key:
            continue
        parsed: Dict[str, Any] = {}
        if "blocked" in entry:
            parsed["blocked"] = bool(entry.get("blocked"))
        if "cooldown_left" in entry:
            value = _coerce_int(entry.get("cooldown_left"))
            if value is not None:
                parsed["cooldown_left"] = value
        if "next_block_left" in entry:
            value = _coerce_int(entry.get("next_block_left"))
            if value is not None:
                parsed["next_block_left"] = value
        if "block_until_ts" in entry:
            value = _coerce_int(entry.get("block_until_ts"))
            if value is not None:
                parsed["block_until_ts"] = value
        if "last_trigger" in entry and isinstance(entry.get("last_trigger"), list):
            parsed["last_trigger"] = [str(x) for x in entry.get("last_trigger", [])]
        if "last_snapshot" in entry and isinstance(entry.get("last_snapshot"), Mapping):
            parsed["last_snapshot"] = dict(entry.get("last_snapshot", {}))
        if parsed:
            result[symbol_key] = parsed
    return result


def load_no_trade_state(path: str | Path = DEFAULT_NO_TRADE_STATE_PATH) -> NoTradeState:
    """Load persisted no-trade state returning empty defaults on errors."""

    p = Path(path)
    if not p.exists():
        return NoTradeState()

    try:
        raw_text = p.read_text(encoding="utf-8")
    except OSError:
        return NoTradeState()

    if not raw_text.strip():
        return NoTradeState()

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return NoTradeState()

    if not isinstance(payload, Mapping):
        return NoTradeState()

    anomaly_map = _parse_anomaly_state(payload)
    dynamic_state = _parse_dynamic_guard_state(payload)
    return NoTradeState(
        anomaly_block_until_ts=anomaly_map,
        dynamic_guard=dynamic_state,
    )


def save_no_trade_state(
    state: NoTradeState,
    path: str | Path = DEFAULT_NO_TRADE_STATE_PATH,
) -> None:
    """Persist :class:`NoTradeState` to *path* in the canonical format."""

    data = {
        "anomaly_block_until_ts": dict(state.anomaly_block_until_ts or {}),
        "dynamic_guard": dict(state.dynamic_guard or {}),
    }
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload + "\n", encoding="utf-8")
