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


class DynamicGuardConfig(BaseModel):
    """Configuration for a dynamic no-trade guard."""

    enable: bool = False
    sigma_window: Optional[int] = None
    atr_window: Optional[int] = None
    vol_abs: Optional[float] = None
    vol_pctile: Optional[float] = None
    spread_abs_bps: Optional[float] = None
    spread_pctile: Optional[float] = None
    hysteresis: Optional[float] = None
    cooldown_bars: int = 0
    next_bars_block: Dict[str, int] = Field(default_factory=dict)
    log_reason: bool = False


class DynamicHysteresisConfig(BaseModel):
    """Tuning for guard release behaviour."""

    ratio: Optional[float] = None
    cooldown_bars: Optional[int] = None


class DynamicConfig(BaseModel):
    """Structured dynamic guard configuration."""

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

    dynamic_payload: Dict[str, Any] = {
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

    maintenance_cfg = config.maintenance
    if maintenance_cfg.path:
        try:
            base_dir = Path(path).resolve().parent
            resolved = Path(maintenance_cfg.path)
            if not resolved.is_absolute():
                resolved = (base_dir / resolved).resolve()
            maintenance_cfg.path = str(resolved)
        except Exception:  # pragma: no cover - defensive
            maintenance_cfg.path = str(maintenance_cfg.path)

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

    if hysteresis_cfg.ratio is None:
        hysteresis_cfg.ratio = guard.hysteresis
    else:
        guard.hysteresis = hysteresis_cfg.ratio

    if hysteresis_cfg.cooldown_bars is None:
        hysteresis_cfg.cooldown_bars = guard.cooldown_bars
    else:
        guard.cooldown_bars = hysteresis_cfg.cooldown_bars

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
    return NoTradeState(anomaly_block_until_ts=anomaly_map)


def save_no_trade_state(
    state: NoTradeState,
    path: str | Path = DEFAULT_NO_TRADE_STATE_PATH,
) -> None:
    """Persist :class:`NoTradeState` to *path* in the canonical format."""

    data = {
        "anomaly_block_until_ts": dict(state.anomaly_block_until_ts or {}),
    }
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload + "\n", encoding="utf-8")
