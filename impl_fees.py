# -*- coding: utf-8 -*-
"""Helpers for plugging the :mod:`fees` module into runtime components."""

from __future__ import annotations

import copy
import datetime as _dt
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Mapping, Tuple

from adapters.binance_spot_private import AccountFeeInfo, fetch_account_fee_info
from binance_fee_refresh import (
    DEFAULT_BNB_DISCOUNT_RATE,
    DEFAULT_UPDATE_THRESHOLD_DAYS,
    DEFAULT_VIP_TIER_LABEL,
    PUBLIC_FEE_URL,
    PublicFeeSnapshot,
    load_public_fee_snapshot,
    parse_timestamp,
)

try:
    from fees import FeesModel
except Exception:  # pragma: no cover
    FeesModel = None  # type: ignore

from services.costs import MakerTakerShareSettings


logger = logging.getLogger(__name__)

_FEE_TABLE_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_DEFAULT_FEE_TABLE_PATH = Path("data") / "fees" / "fees_by_symbol.json"
_DEFAULT_BINANCE_SAPI_BASE = "https://api.binance.com"


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not (num == num) or num in (float("inf"), float("-inf")):
        return default
    return num


def _safe_positive_int(value: Any) -> Optional[int]:
    try:
        num = int(value)
    except (TypeError, ValueError):
        return None
    if num < 0:
        return None
    return num


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text or None


def _safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return None
    if isinstance(value, (int, float)):
        if value == 1 or value == 1.0:
            return True
        if value == 0 or value == 0.0:
            return False
    return None


def _normalise_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        candidate = str(value).strip()
    except Exception:
        return None
    if not candidate:
        return None
    return os.path.expanduser(candidate)


def _plain_mapping(data: Any) -> Dict[str, Any]:
    if isinstance(data, Mapping):
        return {k: v for k, v in data.items()}
    return {}


def _safe_non_negative_float(value: Any) -> Optional[float]:
    result = _safe_float(value)
    if result is None:
        return None
    if result < 0.0:
        return None
    return result


def _normalise_rounding_options(
    payload: Any, *, fallback_step: Optional[float]
) -> Tuple[Optional[Dict[str, Any]], bool, Optional[float]]:
    mapping = dict(payload) if isinstance(payload, Mapping) else {}

    fallback_candidate = _safe_non_negative_float(fallback_step)
    has_input = bool(mapping) or (fallback_candidate is not None and fallback_candidate > 0.0)
    normalized: Dict[str, Any] = {}

    step_value = _safe_non_negative_float(mapping.get("step"))
    if step_value is not None and step_value <= 0.0:
        step_value = None

    fallback = fallback_candidate
    if fallback is not None and fallback <= 0.0:
        fallback = None
    if step_value is None:
        step_value = fallback

    mode = _safe_str(mapping.get("mode"))
    if mode:
        normalized["mode"] = mode.lower()

    precision = _safe_positive_int(mapping.get("precision"))
    if precision is not None:
        normalized["precision"] = precision

    decimals = _safe_positive_int(mapping.get("decimals"))
    if decimals is not None:
        normalized["decimals"] = decimals

    minimum = _safe_non_negative_float(
        mapping.get("minimum")
        or mapping.get("min_fee")
        or mapping.get("minimum_fee")
    )
    if minimum is not None:
        normalized["minimum_fee"] = float(minimum)

    maximum = _safe_non_negative_float(
        mapping.get("maximum")
        or mapping.get("max_fee")
        or mapping.get("maximum_fee")
    )
    if maximum is not None:
        normalized["maximum_fee"] = float(maximum)

    per_symbol_raw = mapping.get("per_symbol") or mapping.get("symbols")
    per_symbol: Dict[str, Any] = {}
    if isinstance(per_symbol_raw, Mapping):
        for symbol, cfg in per_symbol_raw.items():
            if not isinstance(symbol, str):
                continue
            nested_norm, nested_enabled, nested_step = _normalise_rounding_options(
                cfg, fallback_step=None
            )
            if nested_norm is None:
                nested_norm = {}
            if nested_enabled:
                if nested_step is not None and nested_step > 0.0:
                    nested_norm = dict(nested_norm)
                    nested_norm.setdefault("step", float(nested_step))
                    nested_norm.setdefault("enabled", True)
                else:
                    nested_norm = dict(nested_norm)
                    nested_norm.setdefault("enabled", True)
            else:
                nested_norm = dict(nested_norm)
                nested_norm["enabled"] = False
                nested_norm.pop("step", None)
            if nested_norm:
                per_symbol[symbol.upper()] = nested_norm
        if per_symbol:
            normalized["per_symbol"] = per_symbol
            has_input = True

    for key, value in mapping.items():
        if key in {
            "enabled",
            "step",
            "per_symbol",
            "symbols",
            "mode",
            "precision",
            "decimals",
            "minimum",
            "min_fee",
            "minimum_fee",
            "maximum",
            "max_fee",
            "maximum_fee",
        }:
            continue
        if isinstance(value, Mapping):
            normalized[key] = _plain_mapping(value)
            has_input = True
        elif isinstance(value, (str, int, float, bool)):
            normalized[key] = value
            has_input = True

    enabled_raw = mapping.get("enabled")
    if enabled_raw is None:
        enabled = bool(step_value)
    else:
        enabled = bool(enabled_raw)

    effective_step = step_value if enabled and step_value is not None else None

    if effective_step is not None:
        normalized["step"] = float(effective_step)
    elif "step" in normalized:
        normalized.pop("step", None)

    if enabled or has_input:
        normalized["enabled"] = bool(enabled)
    elif not has_input and enabled_raw is False:
        normalized["enabled"] = False

    if not normalized and not has_input:
        return None, bool(enabled), effective_step

    return normalized, bool(enabled), effective_step


def _normalise_settlement_options(payload: Any) -> Optional[Dict[str, Any]]:
    mapping = dict(payload) if isinstance(payload, Mapping) else {}
    if not mapping:
        return None

    normalized: Dict[str, Any] = {}

    if "enabled" in mapping:
        normalized["enabled"] = bool(mapping.get("enabled"))

    mode = (
        _safe_str(mapping.get("mode"))
        or _safe_str(mapping.get("type"))
        or _safe_str(mapping.get("settle_mode"))
    )
    if mode:
        normalized["mode"] = mode.lower()

    currency = (
        _safe_str(mapping.get("currency"))
        or _safe_str(mapping.get("asset"))
        or _safe_str(mapping.get("symbol"))
    )
    if currency:
        normalized["currency"] = currency.upper()

    fallback_currency = (
        _safe_str(mapping.get("fallback_currency"))
        or _safe_str(mapping.get("fallback_asset"))
    )
    if fallback_currency:
        normalized["fallback_currency"] = fallback_currency.upper()

    priority = _safe_positive_int(mapping.get("priority"))
    if priority is not None:
        normalized["priority"] = priority

    for key in (
        "prefer_discount_asset",
        "allow_conversion",
        "allow_external",
        "auto_convert",
    ):
        if key in mapping:
            normalized[key] = bool(mapping.get(key))

    per_symbol_raw = mapping.get("per_symbol") or mapping.get("symbols")
    if isinstance(per_symbol_raw, Mapping):
        per_symbol: Dict[str, Any] = {}
        for symbol, cfg in per_symbol_raw.items():
            if not isinstance(symbol, str):
                continue
            nested = _normalise_settlement_options(cfg)
            if nested:
                per_symbol[symbol.upper()] = nested
        if per_symbol:
            normalized["per_symbol"] = per_symbol

    for key, value in mapping.items():
        if key in {
            "enabled",
            "mode",
            "type",
            "settle_mode",
            "currency",
            "asset",
            "symbol",
            "fallback_currency",
            "fallback_asset",
            "priority",
            "prefer_discount_asset",
            "allow_conversion",
            "allow_external",
            "auto_convert",
            "per_symbol",
            "symbols",
        }:
            continue
        if isinstance(value, Mapping):
            normalized[key] = _plain_mapping(value)
        elif isinstance(value, (str, int, float, bool)):
            normalized[key] = value

    return normalized if normalized else None


@dataclass
class FeesConfig:
    """Normalised configuration for :class:`FeesImpl`."""

    enabled: bool = True
    path: Optional[str] = None
    refresh_days: Optional[int] = None
    maker_bps: Optional[float] = None
    taker_bps: Optional[float] = None
    use_bnb_discount: Optional[bool] = None
    maker_discount_mult: Optional[float] = None
    taker_discount_mult: Optional[float] = None
    vip_tier: Optional[int] = None
    fee_rounding_step: Optional[float] = None
    rounding: Dict[str, Any] = field(default_factory=dict)
    settlement: Dict[str, Any] = field(default_factory=dict)
    public_snapshot: Dict[str, Any] = field(default_factory=dict)
    symbol_fee_table: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    account_info: Dict[str, Any] = field(default_factory=dict)
    maker_taker_share: Optional[Dict[str, Any]] = None
    maker_taker_share_enabled: Optional[bool] = None
    maker_taker_share_mode: Optional[str] = None
    maker_share_default: Optional[float] = None
    spread_cost_maker_bps: Optional[float] = None
    spread_cost_taker_bps: Optional[float] = None
    taker_fee_override_bps: Optional[float] = None

    # filled during normalisation
    maker_taker_share_cfg: Optional[MakerTakerShareSettings] = field(
        init=False, default=None
    )
    rounding_options: Optional[Dict[str, Any]] = field(init=False, default=None)
    rounding_enabled: bool = field(init=False, default=False)
    rounding_step_effective: Optional[float] = field(init=False, default=None)
    settlement_options: Optional[Dict[str, Any]] = field(init=False, default=None)
    account_info_enabled: bool = field(init=False, default=False)
    account_info_endpoint: Optional[str] = field(init=False, default=None)
    account_info_recv_window_ms: Optional[int] = field(init=False, default=None)
    account_info_timeout_s: Optional[float] = field(init=False, default=None)
    account_info_api_key: Optional[str] = field(init=False, default=None)
    account_info_api_secret: Optional[str] = field(init=False, default=None)
    maker_bps_overridden: bool = field(init=False, default=False)
    taker_bps_overridden: bool = field(init=False, default=False)
    use_bnb_discount_overridden: bool = field(init=False, default=False)
    maker_discount_overridden: bool = field(init=False, default=False)
    taker_discount_overridden: bool = field(init=False, default=False)
    vip_tier_overridden: bool = field(init=False, default=False)
    auto_maker_bps: Optional[float] = field(init=False, default=None)
    auto_taker_bps: Optional[float] = field(init=False, default=None)
    auto_maker_discount_mult: Optional[float] = field(init=False, default=None)
    auto_taker_discount_mult: Optional[float] = field(init=False, default=None)
    auto_use_bnb_discount: Optional[bool] = field(init=False, default=None)
    auto_vip_tier: Optional[int] = field(init=False, default=None)
    auto_refresh_metadata: Dict[str, Any] = field(init=False, default_factory=dict)
    public_snapshot_use_bnb_discount: Optional[bool] = field(init=False, default=None)
    public_snapshot_maker_discount_mult: Optional[float] = field(init=False, default=None)
    public_snapshot_taker_discount_mult: Optional[float] = field(init=False, default=None)
    public_snapshot_vip_tier: Optional[int] = field(init=False, default=None)
    public_snapshot_vip_label: Optional[str] = field(init=False, default=None)
    public_snapshot_discount_rate: Optional[float] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.path = _normalise_path(self.path)
        self.refresh_days = _safe_positive_int(self.refresh_days)

        maker_input = self.maker_bps
        self.maker_bps_overridden = maker_input is not None
        maker_bps = _safe_float(maker_input)
        if maker_bps is None:
            maker_bps = 1.0
        self.maker_bps = maker_bps

        taker_input = self.taker_bps
        self.taker_bps_overridden = taker_input is not None
        taker_bps = _safe_float(taker_input)
        if taker_bps is None:
            taker_bps = 5.0
        self.taker_bps = taker_bps

        use_bnb_input = self.use_bnb_discount
        self.use_bnb_discount_overridden = use_bnb_input is not None
        use_bnb_flag = bool(use_bnb_input) if use_bnb_input is not None else False
        self.use_bnb_discount = use_bnb_flag

        maker_mult_input = self.maker_discount_mult
        self.maker_discount_overridden = maker_mult_input is not None
        maker_mult = _safe_float(maker_mult_input)
        try:
            default_discount_mult = 1.0 - float(DEFAULT_BNB_DISCOUNT_RATE)
        except (TypeError, ValueError):
            default_discount_mult = 1.0
        if not math.isfinite(default_discount_mult):
            default_discount_mult = 1.0
        default_discount_mult = max(0.0, min(default_discount_mult, 1.0))
        if maker_mult is None:
            maker_mult = default_discount_mult if self.use_bnb_discount else 1.0
        self.maker_discount_mult = maker_mult

        taker_mult_input = self.taker_discount_mult
        self.taker_discount_overridden = taker_mult_input is not None
        taker_mult = _safe_float(taker_mult_input)
        if taker_mult is None:
            taker_mult = default_discount_mult if self.use_bnb_discount else 1.0
        self.taker_discount_mult = taker_mult

        vip_input = self.vip_tier
        self.vip_tier_overridden = vip_input is not None
        self.vip_tier = _safe_positive_int(vip_input)

        step = _safe_float(self.fee_rounding_step)
        if step is not None and step <= 0.0:
            step = None
        self.fee_rounding_step = step

        rounding_norm, rounding_enabled, rounding_step = _normalise_rounding_options(
            self.rounding, fallback_step=step
        )
        if rounding_norm is not None:
            rounding_clean = copy.deepcopy(rounding_norm)
            self.rounding = rounding_clean
            self.rounding_options = copy.deepcopy(rounding_norm)
        else:
            self.rounding = {}
            self.rounding_options = None
        self.rounding_enabled = bool(rounding_enabled)
        if rounding_step is not None and rounding_enabled:
            self.fee_rounding_step = float(rounding_step)
        elif not rounding_enabled:
            self.fee_rounding_step = None
        self.rounding_step_effective = self.fee_rounding_step

        settlement_norm = _normalise_settlement_options(self.settlement)
        if settlement_norm is not None:
            settlement_clean = copy.deepcopy(settlement_norm)
            self.settlement = settlement_clean
            self.settlement_options = copy.deepcopy(settlement_norm)
        else:
            self.settlement = {}
            self.settlement_options = None

        if isinstance(self.symbol_fee_table, Mapping):
            table: Dict[str, Any] = {}
            for symbol, payload in self.symbol_fee_table.items():
                if not isinstance(symbol, str):
                    continue
                if isinstance(payload, Mapping):
                    table[symbol.upper()] = dict(payload)
            self.symbol_fee_table = table
        else:
            self.symbol_fee_table = {}

        if isinstance(self.public_snapshot, Mapping):
            snapshot_cfg = {k: v for k, v in self.public_snapshot.items()}
        else:
            snapshot_cfg = {}
        self.public_snapshot = snapshot_cfg

        ps_use = _safe_bool(snapshot_cfg.get("use_bnb_discount"))
        if ps_use is not None:
            self.public_snapshot_use_bnb_discount = ps_use

        ps_maker_mult = _safe_non_negative_float(snapshot_cfg.get("maker_discount_mult"))
        if ps_maker_mult is None:
            ps_maker_mult = _safe_non_negative_float(snapshot_cfg.get("maker_discount_multiplier"))
        if ps_maker_mult is not None:
            self.public_snapshot_maker_discount_mult = ps_maker_mult

        ps_taker_mult = _safe_non_negative_float(snapshot_cfg.get("taker_discount_mult"))
        if ps_taker_mult is None:
            ps_taker_mult = _safe_non_negative_float(snapshot_cfg.get("taker_discount_multiplier"))
        if ps_taker_mult is not None:
            self.public_snapshot_taker_discount_mult = ps_taker_mult

        vip_snapshot = snapshot_cfg.get("vip_tier")
        if vip_snapshot is None:
            vip_snapshot = snapshot_cfg.get("vip")
        vip_value = _safe_positive_int(vip_snapshot)
        if vip_value is not None:
            self.public_snapshot_vip_tier = vip_value

        vip_label = _safe_str(
            snapshot_cfg.get("vip_label") or snapshot_cfg.get("vip_tier_label")
        )
        if vip_label:
            self.public_snapshot_vip_label = vip_label

        discount_snapshot = snapshot_cfg.get("bnb_discount_rate")
        if discount_snapshot is None:
            discount_snapshot = snapshot_cfg.get("discount_rate")
        discount_value = _safe_non_negative_float(discount_snapshot)
        if discount_value is not None:
            self.public_snapshot_discount_rate = discount_value

        if isinstance(self.metadata, Mapping):
            self.metadata = dict(self.metadata)
        else:
            self.metadata = {}

        account_info_cfg: Dict[str, Any] = {}
        if isinstance(self.account_info, Mapping):
            account_info_cfg.update(self.account_info)
        self.account_info = account_info_cfg
        enabled_flag = account_info_cfg.get("enabled")
        if enabled_flag is None and account_info_cfg.get("enable") is not None:
            enabled_flag = account_info_cfg.get("enable")
        self.account_info_enabled = bool(enabled_flag)
        endpoint = account_info_cfg.get("endpoint") or account_info_cfg.get("base_url")
        endpoint_str = _safe_str(endpoint)
        if endpoint_str:
            endpoint_str = endpoint_str.rstrip("/")
        self.account_info_endpoint = endpoint_str
        recv_candidate = account_info_cfg.get("recv_window_ms")
        if recv_candidate is None:
            recv_candidate = account_info_cfg.get("recv_window")
        self.account_info_recv_window_ms = _safe_positive_int(recv_candidate)
        timeout_candidate = account_info_cfg.get("timeout_s")
        if timeout_candidate is None:
            timeout_candidate = account_info_cfg.get("timeout")
        timeout_value = _safe_float(timeout_candidate)
        if timeout_value is not None and timeout_value <= 0.0:
            timeout_value = None
        self.account_info_timeout_s = timeout_value
        self.account_info_api_key = _safe_str(account_info_cfg.get("api_key"))
        self.account_info_api_secret = _safe_str(account_info_cfg.get("api_secret"))

        share_payload: Dict[str, Any] = {}
        if isinstance(self.maker_taker_share, Mapping):
            share_payload.update(self.maker_taker_share)

        overrides = {
            "enabled": self.maker_taker_share_enabled,
            "mode": self.maker_taker_share_mode,
            "maker_share_default": self.maker_share_default,
            "spread_cost_maker_bps": self.spread_cost_maker_bps,
            "spread_cost_taker_bps": self.spread_cost_taker_bps,
            "taker_fee_override_bps": self.taker_fee_override_bps,
        }
        for key, value in overrides.items():
            if value is not None:
                share_payload.setdefault(key, value)

        share_cfg = MakerTakerShareSettings.parse(share_payload)
        self.maker_taker_share_cfg = share_cfg
        if share_cfg is not None:
            share_dict = share_cfg.as_dict()
            self.maker_taker_share = share_dict
            self.maker_taker_share_enabled = bool(share_cfg.enabled)
            self.maker_taker_share_mode = share_cfg.mode
            self.maker_share_default = float(share_cfg.maker_share_default)
            self.spread_cost_maker_bps = float(share_cfg.spread_cost_maker_bps)
            self.spread_cost_taker_bps = float(share_cfg.spread_cost_taker_bps)
            self.taker_fee_override_bps = (
                float(share_cfg.taker_fee_override_bps)
                if share_cfg.taker_fee_override_bps is not None
                else None
            )
        else:
            self.maker_taker_share = None
            if self.maker_taker_share_enabled is None:
                self.maker_taker_share_enabled = False
            if self.maker_taker_share_mode is None:
                self.maker_taker_share_mode = "fixed"
            if self.maker_share_default is None:
                self.maker_share_default = 0.5
            if self.spread_cost_maker_bps is None:
                self.spread_cost_maker_bps = 0.0
            if self.spread_cost_taker_bps is None:
                self.spread_cost_taker_bps = 0.0

        # keep an easily serialisable copy of overrides
        if self.maker_taker_share is None and share_payload:
            cleaned: Dict[str, Any] = {}
            for key, value in share_payload.items():
                if value is None:
                    continue
                cleaned[key] = value
            self.maker_taker_share = cleaned if cleaned else None


class FeesImpl:
    """Wrapper over :class:`fees.FeesModel` with simulator integration helpers."""

    def __init__(self, cfg: FeesConfig) -> None:
        self.cfg = cfg

        self.table_path: Optional[str] = None
        self.table_metadata: Dict[str, Any] = {}
        self.table_age_days: Optional[float] = None
        self.table_stale: bool = False
        self.table_error: Optional[str] = None
        self.symbol_fee_table_raw: Dict[str, Any] = {}
        self.inline_symbol_fee_table: Dict[str, Any] = dict(cfg.symbol_fee_table)
        self.symbol_fee_table: Dict[str, Any] = {}
        self._table_account_overrides: Dict[str, Any] = {}
        self._table_share_raw: Optional[Dict[str, Any]] = None
        self._table_rounding_normalised: Optional[Dict[str, Any]] = None
        self._table_rounding_enabled: Optional[bool] = None
        self._table_rounding_step: Optional[float] = None
        self._table_settlement_normalised: Optional[Dict[str, Any]] = None
        self.account_fee_info: AccountFeeInfo | None = None
        self.account_fee_overrides: Dict[str, Any] = {}
        self.account_fee_status: str = "disabled"
        self.account_fee_error: Optional[str] = None
        self.account_fee_endpoint: Optional[str] = None
        self.account_fee_recv_window: Optional[int] = None
        self.account_fee_timeout: Optional[float] = None
        self._public_fee_snapshot: PublicFeeSnapshot | None = None
        self._public_refresh_reason: Optional[str] = None
        self._public_refresh_error: Optional[str] = None
        self._public_refresh_attempted: bool = False
        self._account_fee_applied: Dict[str, bool] = {
            "vip_tier": False,
            "maker_bps": False,
            "taker_bps": False,
        }

        table_payload = self._load_symbol_fee_table()
        table_from_file = table_payload.get("table", {}) if table_payload else {}
        if isinstance(table_from_file, Mapping):
            self.symbol_fee_table_raw = {
                str(symbol).upper(): dict(payload)
                for symbol, payload in table_from_file.items()
                if isinstance(symbol, str) and isinstance(payload, Mapping)
            }
        if table_payload:
            account_payload = _plain_mapping(table_payload.get("account"))
            if account_payload:
                self._table_account_overrides = account_payload
                round_block = account_payload.get("rounding")
                fallback_step = _safe_float(account_payload.get("fee_rounding_step"))
                if isinstance(round_block, Mapping):
                    (
                        self._table_rounding_normalised,
                        self._table_rounding_enabled,
                        self._table_rounding_step,
                    ) = _normalise_rounding_options(
                        round_block, fallback_step=fallback_step
                    )
                settle_block = account_payload.get("settlement")
                if isinstance(settle_block, Mapping):
                    self._table_settlement_normalised = _normalise_settlement_options(
                        settle_block
                    )
            share_payload = table_payload.get("share")
            if isinstance(share_payload, Mapping):
                self._table_share_raw = dict(share_payload)

        meta_account = _plain_mapping(self.table_metadata.get("account_overrides"))
        if meta_account or self._table_account_overrides:
            combined_overrides: Dict[str, Any] = {}
            if meta_account:
                combined_overrides.update(meta_account)
            if self._table_account_overrides:
                combined_overrides.update(self._table_account_overrides)
            self._table_account_overrides = combined_overrides
        else:
            self._table_account_overrides = {}

        if self._table_account_overrides:
            use_bnb_meta = _safe_bool(self._table_account_overrides.get("use_bnb_discount"))
            if use_bnb_meta is None:
                use_bnb_meta = _safe_bool(self.table_metadata.get("use_bnb_discount"))
            if use_bnb_meta is not None:
                self.cfg.auto_use_bnb_discount = use_bnb_meta

            maker_mult_meta = _safe_non_negative_float(
                self._table_account_overrides.get("maker_discount_mult")
            )
            if maker_mult_meta is None:
                maker_mult_meta = _safe_non_negative_float(
                    self.table_metadata.get("maker_discount_mult")
                )
            if maker_mult_meta is not None:
                self.cfg.auto_maker_discount_mult = maker_mult_meta

            taker_mult_meta = _safe_non_negative_float(
                self._table_account_overrides.get("taker_discount_mult")
            )
            if taker_mult_meta is None:
                taker_mult_meta = _safe_non_negative_float(
                    self.table_metadata.get("taker_discount_mult")
                )
            if taker_mult_meta is not None:
                self.cfg.auto_taker_discount_mult = taker_mult_meta

            vip_meta = _safe_positive_int(self._table_account_overrides.get("vip_tier"))
            if vip_meta is None:
                vip_meta = _safe_positive_int(self.table_metadata.get("vip_tier_numeric"))
            if vip_meta is None and isinstance(self.table_metadata.get("vip_tier"), str):
                vip_meta = _safe_positive_int(
                    self.table_metadata.get("vip_tier").split(" ")[-1]
                )
            if vip_meta is not None:
                self.cfg.auto_vip_tier = vip_meta

        if not self.table_stale and self.table_error is None:
            self.symbol_fee_table.update(self.symbol_fee_table_raw)
        if self.inline_symbol_fee_table:
            for symbol, payload in self.inline_symbol_fee_table.items():
                if not isinstance(symbol, str):
                    continue
                if isinstance(payload, Mapping):
                    self.symbol_fee_table[symbol.upper()] = dict(payload)

        self.maker_taker_share_cfg: Optional[MakerTakerShareSettings]
        share_cfg = cfg.maker_taker_share_cfg
        share_raw: Optional[Dict[str, Any]] = None
        if share_cfg is not None:
            share_raw = share_cfg.as_dict()
        elif cfg.maker_taker_share is not None:
            share_raw = dict(cfg.maker_taker_share)
        if share_cfg is None and self._table_share_raw is not None:
            share_cfg = MakerTakerShareSettings.parse(self._table_share_raw)
            if share_cfg is None:
                share_raw = dict(self._table_share_raw)
        self.maker_taker_share_cfg = share_cfg
        self.maker_taker_share_raw = share_raw if share_cfg is None else share_cfg.as_dict()

        if cfg.account_info_enabled:
            info = self._load_account_fee_info()
            if info is not None:
                self.account_fee_info = info
                self.account_fee_overrides = info.to_fee_overrides()

        use_bnb_discount = cfg.use_bnb_discount
        if not cfg.use_bnb_discount_overridden and cfg.auto_use_bnb_discount is not None:
            use_bnb_discount = bool(cfg.auto_use_bnb_discount)

        maker_discount_mult = cfg.maker_discount_mult
        taker_discount_mult = cfg.taker_discount_mult
        if use_bnb_discount:
            if not cfg.maker_discount_overridden and cfg.auto_maker_discount_mult is not None:
                maker_discount_mult = float(cfg.auto_maker_discount_mult)
            if not cfg.taker_discount_overridden and cfg.auto_taker_discount_mult is not None:
                taker_discount_mult = float(cfg.auto_taker_discount_mult)
        else:
            if not cfg.maker_discount_overridden:
                maker_discount_mult = 1.0
            if not cfg.taker_discount_overridden:
                taker_discount_mult = 1.0

        self._maker_discount_mult = float(maker_discount_mult)
        self._taker_discount_mult = float(taker_discount_mult)
        self._use_bnb_discount = bool(use_bnb_discount)

        rounding_payload = (
            copy.deepcopy(cfg.rounding_options)
            if cfg.rounding_options is not None
            else None
        )
        settlement_payload = (
            copy.deepcopy(cfg.settlement_options)
            if cfg.settlement_options is not None
            else None
        )
        rounding_disabled_explicit = (
            rounding_payload is not None
            and rounding_payload.get("enabled") is False
        )

        fee_rounding_step = cfg.fee_rounding_step
        if fee_rounding_step is not None and fee_rounding_step <= 0.0:
            fee_rounding_step = None
        if fee_rounding_step is None and not rounding_disabled_explicit:
            if self._table_rounding_enabled is False:
                fee_rounding_step = None
            elif self._table_rounding_step is not None:
                fee_rounding_step = self._table_rounding_step
            else:
                candidate = _safe_float(
                    self._table_account_overrides.get("fee_rounding_step")
                )
                if candidate is not None and candidate > 0.0:
                    fee_rounding_step = candidate

        if rounding_payload is None:
            if self._table_rounding_normalised is not None:
                rounding_payload = copy.deepcopy(self._table_rounding_normalised)
            elif self._table_rounding_enabled is False:
                rounding_payload = {"enabled": False}
        if rounding_payload is None and fee_rounding_step is not None:
            rounding_payload = {"enabled": True, "step": float(fee_rounding_step)}

        if settlement_payload is None and self._table_settlement_normalised is not None:
            settlement_payload = copy.deepcopy(self._table_settlement_normalised)

        final_rounding_payload = (
            copy.deepcopy(rounding_payload) if rounding_payload is not None else None
        )
        final_settlement_payload = (
            copy.deepcopy(settlement_payload) if settlement_payload is not None else None
        )
        self.rounding_options = final_rounding_payload
        self.settlement_options = final_settlement_payload

        vip_tier = cfg.vip_tier
        if vip_tier is None and not cfg.vip_tier_overridden and cfg.auto_vip_tier is not None:
            vip_tier = int(cfg.auto_vip_tier)
        if vip_tier is None:
            vip_candidate = _safe_positive_int(
                self._table_account_overrides.get("vip_tier")
            )
            if vip_candidate is not None:
                vip_tier = vip_candidate
        if vip_tier is None:
            account_vip = _safe_positive_int(self.account_fee_overrides.get("vip_tier"))
            if account_vip is not None:
                vip_tier = account_vip
                self._account_fee_applied["vip_tier"] = True
        if vip_tier is None:
            vip_tier = 0

        maker_bps = float(cfg.maker_bps)
        if not cfg.maker_bps_overridden and cfg.auto_maker_bps is not None:
            maker_bps = float(cfg.auto_maker_bps)
        taker_bps = float(cfg.taker_bps)
        if not cfg.taker_bps_overridden and cfg.auto_taker_bps is not None:
            taker_bps = float(cfg.auto_taker_bps)
        account_maker_bps = _safe_float(self.account_fee_overrides.get("maker_bps"))
        account_maker_override = account_maker_bps is not None
        if account_maker_override:
            maker_bps = account_maker_bps
            self._account_fee_applied["maker_bps"] = True
        account_taker_bps = _safe_float(self.account_fee_overrides.get("taker_bps"))
        account_taker_override = account_taker_bps is not None
        if account_taker_override:
            taker_bps = account_taker_bps
            self._account_fee_applied["taker_bps"] = True

        if account_maker_override:
            self._maker_discount_mult = 1.0
        if account_taker_override:
            self._taker_discount_mult = 1.0

        self.base_fee_bps: Dict[str, float] = {
            "maker_fee_bps": maker_bps * self._maker_discount_mult,
            "taker_fee_bps": taker_bps * self._taker_discount_mult,
        }

        self.maker_taker_share_expected: Optional[Dict[str, float]] = None
        if self.maker_taker_share_cfg is not None:
            self.maker_taker_share_expected = (
                self.maker_taker_share_cfg.expected_fee_breakdown(
                    self.base_fee_bps["maker_fee_bps"],
                    self.base_fee_bps["taker_fee_bps"],
                )
            )

        self.expected_fee_bps: Dict[str, float] = dict(self.base_fee_bps)
        if self.maker_taker_share_expected is not None:
            self.expected_fee_bps.update(self.maker_taker_share_expected)

        symbol_table_payload = (
            {k: dict(v) for k, v in self.symbol_fee_table.items()}
            if self.symbol_fee_table
            else {}
        )

        self.model_payload: Dict[str, Any] = {
            "maker_bps": maker_bps,
            "taker_bps": taker_bps,
            "use_bnb_discount": self._use_bnb_discount,
            "maker_discount_mult": self._maker_discount_mult,
            "taker_discount_mult": self._taker_discount_mult,
            "vip_tier": int(vip_tier),
        }
        if fee_rounding_step is not None:
            self.model_payload["fee_rounding_step"] = float(fee_rounding_step)
        if final_rounding_payload is not None:
            self.model_payload["rounding"] = copy.deepcopy(final_rounding_payload)
        if final_settlement_payload is not None:
            self.model_payload["settlement"] = copy.deepcopy(final_settlement_payload)
        if symbol_table_payload:
            self.model_payload["symbol_fee_table"] = symbol_table_payload

        self._model = (
            FeesModel.from_dict(dict(self.model_payload))
            if FeesModel is not None and cfg.enabled
            else None
        )

        self.metadata = self._build_metadata(
            vip_tier=vip_tier,
            fee_rounding_step=fee_rounding_step,
            rounding=final_rounding_payload,
            settlement=final_settlement_payload,
        )

        self.expected_payload: Dict[str, Any] = self._build_expected_payload(
            vip_tier=vip_tier,
            rounding=final_rounding_payload,
            settlement=final_settlement_payload,
        )

    def _build_metadata(
        self,
        *,
        vip_tier: int,
        fee_rounding_step: Optional[float],
        rounding: Optional[Dict[str, Any]],
        settlement: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if self.cfg.metadata:
            meta.update(self.cfg.metadata)
        table_meta = dict(self.table_metadata)
        table_meta.setdefault("path", self.table_path)
        table_meta.setdefault("age_days", self.table_age_days)
        table_meta.setdefault("refresh_days", self.cfg.refresh_days)
        table_meta.setdefault("stale", self.table_stale)
        table_meta.setdefault("error", self.table_error)
        table_meta.setdefault("file_symbol_count", len(self.symbol_fee_table_raw))
        if self._table_account_overrides:
            table_meta.setdefault("account_overrides", self._table_account_overrides)
        if self._table_share_raw is not None:
            table_meta.setdefault("share_from_file", self._table_share_raw)
        if self._table_rounding_normalised is not None:
            table_meta.setdefault(
                "rounding_from_file", copy.deepcopy(self._table_rounding_normalised)
            )
        if self._table_rounding_enabled is not None:
            table_meta.setdefault("rounding_enabled", bool(self._table_rounding_enabled))
        if self._table_rounding_step is not None:
            table_meta.setdefault("rounding_step", float(self._table_rounding_step))
        if self._table_settlement_normalised is not None:
            table_meta.setdefault(
                "settlement_from_file",
                copy.deepcopy(self._table_settlement_normalised),
            )
        if self._public_refresh_attempted:
            table_meta.setdefault("auto_refresh_attempted", True)
        if self._public_refresh_reason is not None:
            table_meta.setdefault("auto_refresh_reason", self._public_refresh_reason)
        if self._public_refresh_error is not None:
            table_meta.setdefault("auto_refresh_error", self._public_refresh_error)
        if self.cfg.auto_refresh_metadata:
            table_meta.setdefault(
                "public_refresh", copy.deepcopy(self.cfg.auto_refresh_metadata)
            )
        meta["table"] = table_meta
        account_meta: Dict[str, Any] = {
            "enabled": bool(self.cfg.account_info_enabled),
            "status": self.account_fee_status,
        }
        if self.cfg.account_info_enabled:
            if self.account_fee_endpoint is not None:
                account_meta["endpoint"] = self.account_fee_endpoint
            if self.account_fee_recv_window is not None:
                account_meta["recv_window_ms"] = int(self.account_fee_recv_window)
            if self.account_fee_timeout is not None:
                account_meta["timeout_s"] = float(self.account_fee_timeout)
            account_meta["applied"] = dict(self._account_fee_applied)
            if self.account_fee_info is not None:
                if self.account_fee_info.vip_tier is not None:
                    account_meta.setdefault("vip_tier", int(self.account_fee_info.vip_tier))
                if self.account_fee_info.maker_bps is not None:
                    account_meta.setdefault("maker_bps", float(self.account_fee_info.maker_bps))
                if self.account_fee_info.taker_bps is not None:
                    account_meta.setdefault("taker_bps", float(self.account_fee_info.taker_bps))
                if self.account_fee_info.update_time_ms is not None:
                    account_meta.setdefault(
                        "update_time_ms", int(self.account_fee_info.update_time_ms)
                    )
            if self.account_fee_error is not None:
                account_meta["error"] = self.account_fee_error
        meta["account_fetch"] = account_meta
        meta["inline_symbol_count"] = len(self.inline_symbol_fee_table)
        meta["symbol_fee_table_used"] = len(self.symbol_fee_table)
        meta["maker_bps"] = float(self.cfg.maker_bps)
        meta["taker_bps"] = float(self.cfg.taker_bps)
        meta["maker_discount_mult"] = self._maker_discount_mult
        meta["taker_discount_mult"] = self._taker_discount_mult
        meta["vip_tier"] = int(vip_tier)
        if self.cfg.auto_maker_bps is not None:
            meta.setdefault("maker_bps_auto", float(self.cfg.auto_maker_bps))
        if self.cfg.auto_taker_bps is not None:
            meta.setdefault("taker_bps_auto", float(self.cfg.auto_taker_bps))
        if self.cfg.auto_use_bnb_discount is not None:
            meta.setdefault("use_bnb_discount_auto", bool(self.cfg.auto_use_bnb_discount))
        if fee_rounding_step is not None:
            meta["fee_rounding_step"] = float(fee_rounding_step)
        meta["rounding"] = copy.deepcopy(rounding) if rounding is not None else None
        meta["settlement"] = (
            copy.deepcopy(settlement) if settlement is not None else None
        )
        meta["maker_taker_share"] = (
            dict(self.maker_taker_share_raw)
            if isinstance(self.maker_taker_share_raw, Mapping)
            else None
        )
        meta["enabled"] = bool(self.cfg.enabled)
        meta["table_applied"] = bool(self.symbol_fee_table)
        return meta

    def _build_expected_payload(
        self,
        *,
        vip_tier: int,
        rounding: Optional[Dict[str, Any]],
        settlement: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "maker_fee_bps": self.base_fee_bps["maker_fee_bps"],
            "taker_fee_bps": self.base_fee_bps["taker_fee_bps"],
            "maker_discount_mult": self._maker_discount_mult,
            "taker_discount_mult": self._taker_discount_mult,
            "use_bnb_discount": self._use_bnb_discount,
            "vip_tier": int(vip_tier),
        }
        if self.maker_taker_share_expected is not None:
            payload.update(self.maker_taker_share_expected)
        else:
            payload.setdefault("expected_fee_bps", payload["taker_fee_bps"])
        if rounding is not None:
            payload["rounding"] = copy.deepcopy(rounding)
        if settlement is not None:
            payload["settlement"] = copy.deepcopy(settlement)
        return payload

    @staticmethod
    def _parse_fee_table(raw: Mapping[str, Any]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        table: Dict[str, Any] = {}
        account: Dict[str, Any] = {}
        share: Optional[Dict[str, Any]] = None

        meta_block = raw.get("meta") or raw.get("metadata")
        if isinstance(meta_block, Mapping):
            meta = {k: v for k, v in meta_block.items()}

        account_block = raw.get("account")
        if isinstance(account_block, Mapping):
            for key, value in account_block.items():
                account[key] = value
        account_keys = {
            "maker_bps",
            "taker_bps",
            "use_bnb_discount",
            "maker_discount_mult",
            "taker_discount_mult",
            "vip_tier",
            "fee_rounding_step",
            "rounding",
            "settlement",
        }
        for key in account_keys:
            if key in raw and raw[key] is not None and key not in account:
                account[key] = raw[key]

        share_block = raw.get("maker_taker_share")
        if share_block is None and isinstance(account_block, Mapping):
            share_block = account_block.get("maker_taker_share")
        if isinstance(share_block, Mapping):
            share = {k: v for k, v in share_block.items()}

        table_block: Any = None
        for key in ("symbol_fee_table", "symbols", "fees_by_symbol", "data"):
            candidate = raw.get(key)
            if isinstance(candidate, Mapping):
                table_block = candidate
                break
        if table_block is None:
            candidate_table: Dict[str, Any] = {}
            for key, value in raw.items():
                if isinstance(key, str) and isinstance(value, Mapping):
                    candidate_table[key] = value
            if candidate_table:
                table_block = candidate_table
        if isinstance(table_block, Mapping):
            for symbol, payload in table_block.items():
                if not isinstance(symbol, str) or not isinstance(payload, Mapping):
                    continue
                table[symbol.upper()] = dict(payload)

        return {"table": table, "meta": meta, "account": account, "share": share}

    @classmethod
    def _read_fee_table(
        cls, path: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        abspath = os.path.abspath(path)
        try:
            stat = os.stat(abspath)
        except OSError as exc:
            logger.warning("Fees table %s is not accessible: %s", abspath, exc)
            _FEE_TABLE_CACHE.pop(abspath, None)
            return None, None
        mtime = stat.st_mtime
        cached = _FEE_TABLE_CACHE.get(abspath)
        if cached and cached[0] == mtime:
            return cached[1], mtime
        try:
            with open(abspath, "r", encoding="utf-8") as f:
                raw_payload = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load fees table %s: %s", abspath, exc)
            _FEE_TABLE_CACHE.pop(abspath, None)
            return None, mtime
        if not isinstance(raw_payload, Mapping):
            logger.warning(
                "Fees table %s has invalid structure (%s); ignoring",
                abspath,
                type(raw_payload).__name__,
            )
            _FEE_TABLE_CACHE.pop(abspath, None)
            return None, mtime
        payload = cls._parse_fee_table(raw_payload)
        _FEE_TABLE_CACHE[abspath] = (mtime, payload)
        return payload, mtime

    def _load_symbol_fee_table(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        path_candidate = self.cfg.path
        default_path = _DEFAULT_FEE_TABLE_PATH
        if path_candidate is None and default_path.exists():
            path_candidate = str(default_path)

        abspath = (
            os.path.abspath(path_candidate)
            if path_candidate is not None
            else os.path.abspath(str(default_path))
        )
        self.table_path = abspath

        data: Optional[Dict[str, Any]] = None
        mtime: Optional[float] = None
        auto_reason: Optional[str] = None

        if path_candidate is not None:
            data, mtime = self._read_fee_table(abspath)

        if data is None:
            if path_candidate is not None and os.path.exists(abspath):
                logger.warning(
                    "Fees table %s is unusable; falling back to global fees", abspath
                )
                self.table_error = "invalid"
                auto_reason = "invalid"
            else:
                if self.cfg.path:
                    logger.warning(
                        "Fees table %s not found; falling back to global fees", abspath
                    )
                self.table_error = "missing"
                auto_reason = "missing"
            self.table_metadata = {
                "path": abspath,
                "age_days": None,
                "refresh_days": self.cfg.refresh_days,
                "stale": False,
                "error": self.table_error,
            }
        else:
            payload = data
            age_days: Optional[float] = None
            if mtime is not None:
                age_days = max((time.time() - mtime) / 86400.0, 0.0)
                self.table_age_days = age_days

            meta = dict(payload.get("meta", {}))
            built_at = parse_timestamp(meta.get("built_at"))
            if built_at is not None:
                now = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
                meta_age = max((now - built_at).total_seconds() / 86400.0, 0.0)
                if age_days is None or meta_age > age_days:
                    age_days = meta_age
                    self.table_age_days = meta_age

            refresh_days = self.cfg.refresh_days
            threshold: Optional[float] = None
            if refresh_days is not None and refresh_days >= 0:
                threshold = float(refresh_days)
            elif self._can_auto_refresh(abspath):
                threshold = float(DEFAULT_UPDATE_THRESHOLD_DAYS)

            if (
                threshold is not None
                and age_days is not None
                and age_days > threshold
            ):
                self.table_stale = True
                auto_reason = "stale"
                logger.warning(
                    "Fees table %s is stale (age %.1f days > %.1f); refreshing", abspath, age_days, threshold
                )

            if not payload.get("table"):
                auto_reason = auto_reason or "empty"

            meta.update(
                {
                    "path": abspath,
                    "age_days": self.table_age_days,
                    "refresh_days": self.cfg.refresh_days,
                    "stale": self.table_stale,
                    "error": self.table_error,
                }
            )
            self.table_metadata = meta

        if auto_reason and self._can_auto_refresh(abspath):
            refreshed = self._refresh_symbol_fee_table(reason=auto_reason, target_path=abspath)
            if refreshed is not None:
                return refreshed
        return payload

    def _can_auto_refresh(self, path: Optional[str]) -> bool:
        if path is None:
            return False
        disable = os.environ.get("BINANCE_PUBLIC_FEES_DISABLE_AUTO", "").strip().lower()
        if disable in {"1", "true", "yes", "on"}:
            return False
        default_abspath = os.path.abspath(str(_DEFAULT_FEE_TABLE_PATH))
        path_norm = os.path.abspath(path)
        if self.cfg.path:
            try:
                return os.path.abspath(self.cfg.path) == path_norm
            except Exception:
                return False
        return path_norm == default_abspath

    def _refresh_symbol_fee_table(
        self, *, reason: str, target_path: Optional[str]
    ) -> Dict[str, Any] | None:
        self._public_refresh_attempted = True
        csv_env = _safe_str(os.environ.get("BINANCE_FEE_SNAPSHOT_CSV"))
        csv_path: Path | None = None
        if csv_env:
            csv_candidate = Path(os.path.expanduser(csv_env)).expanduser()
            if csv_candidate.exists():
                csv_path = csv_candidate
            else:
                logger.warning(
                    "Configured BINANCE_FEE_SNAPSHOT_CSV %s not found; ignoring", csv_candidate
                )

        public_url = _safe_str(os.environ.get("BINANCE_PUBLIC_FEE_URL")) or PUBLIC_FEE_URL

        timeout_env = os.environ.get("BINANCE_FEE_TIMEOUT")
        timeout_value = _safe_positive_int(timeout_env)
        if timeout_value is None or timeout_value <= 0:
            timeout_value = 30

        discount_env = os.environ.get("BINANCE_BNB_DISCOUNT_RATE")
        try:
            discount_rate = (
                float(discount_env) if discount_env is not None else DEFAULT_BNB_DISCOUNT_RATE
            )
        except (TypeError, ValueError):
            discount_rate = DEFAULT_BNB_DISCOUNT_RATE
        if self.cfg.public_snapshot_discount_rate is not None:
            try:
                discount_rate = float(self.cfg.public_snapshot_discount_rate)
            except (TypeError, ValueError):
                pass

        api_key = _safe_str(os.environ.get("BINANCE_API_KEY"))
        api_secret = _safe_str(os.environ.get("BINANCE_API_SECRET"))
        if not api_key or not api_secret:
            api_key = None
            api_secret = None

        vip_label = self.cfg.public_snapshot_vip_label or DEFAULT_VIP_TIER_LABEL
        vip_numeric: Optional[int] = None
        if self.cfg.public_snapshot_vip_tier is not None:
            vip_numeric = int(self.cfg.public_snapshot_vip_tier)
            if not self.cfg.public_snapshot_vip_label:
                vip_label = f"VIP {vip_numeric}"
        else:
            vip_candidate: Optional[int] = None
            if self.cfg.vip_tier_overridden and self.cfg.vip_tier is not None:
                vip_candidate = int(self.cfg.vip_tier)
            elif self.cfg.auto_vip_tier is not None:
                vip_candidate = int(self.cfg.auto_vip_tier)
            if vip_candidate is not None:
                vip_numeric = vip_candidate
                if not self.cfg.public_snapshot_vip_label:
                    vip_label = f"VIP {vip_numeric}"
            elif isinstance(self.table_metadata.get("vip_tier"), str):
                vip_label = self.cfg.public_snapshot_vip_label or str(
                    self.table_metadata.get("vip_tier")
                )
            elif self.cfg.public_snapshot_vip_label:
                vip_label = self.cfg.public_snapshot_vip_label

        snapshot_use_bnb = self.cfg.public_snapshot_use_bnb_discount
        snapshot_maker_mult = self.cfg.public_snapshot_maker_discount_mult
        snapshot_taker_mult = self.cfg.public_snapshot_taker_discount_mult

        try:
            snapshot = load_public_fee_snapshot(
                vip_tier=vip_label,
                vip_tier_numeric=vip_numeric,
                use_bnb_discount=snapshot_use_bnb,
                maker_discount_mult=snapshot_maker_mult,
                taker_discount_mult=snapshot_taker_mult,
                timeout=timeout_value,
                public_url=public_url,
                csv_path=csv_path,
                api_key=api_key,
                api_secret=api_secret,
                bnb_discount_rate=discount_rate,
            )
        except Exception as exc:  # pragma: no cover - network/environment issues
            error_text = f"{exc.__class__.__name__}: {exc}"
            self._public_refresh_error = error_text
            meta = dict(self.table_metadata)
            meta.setdefault("auto_refresh_attempted", True)
            meta.setdefault("auto_refresh_reason", reason)
            meta["auto_refresh_error"] = error_text
            self.table_metadata = meta
            self.cfg.auto_refresh_metadata = {"reason": reason, "error": error_text}
            logger.warning("Failed to refresh public fee snapshot: %s", exc, exc_info=True)
            return None

        logger.info(
            "Auto-refreshed Binance fee table using %s (reason=%s)",
            snapshot.source,
            reason,
        )
        self._public_fee_snapshot = snapshot
        self._public_refresh_reason = reason
        self._public_refresh_error = None
        return self._apply_public_snapshot(snapshot, reason=reason, target_path=target_path)

    def _apply_public_snapshot(
        self, snapshot: PublicFeeSnapshot, *, reason: str, target_path: Optional[str]
    ) -> Dict[str, Any]:
        fees_payload = snapshot.payload.get("fees", {})
        table: Dict[str, Any] = {}
        if isinstance(fees_payload, Mapping):
            for symbol, entry in fees_payload.items():
                if not isinstance(symbol, str) or not isinstance(entry, Mapping):
                    continue
                table[symbol.upper()] = dict(entry)

        meta = dict(snapshot.payload.get("metadata") or {})
        meta.update(
            {
                "path": target_path,
                "age_days": 0.0,
                "refresh_days": self.cfg.refresh_days,
                "stale": False,
                "error": None,
                "auto_refreshed": True,
                "auto_reason": reason,
                "auto_source": snapshot.source,
                "auto_refresh_attempted": True,
            }
        )
        self.table_metadata = meta
        self.table_error = None
        self.table_stale = False
        self.table_age_days = 0.0

        self.cfg.auto_maker_bps = snapshot.maker_bps_default
        self.cfg.auto_taker_bps = snapshot.taker_bps_default
        self.cfg.auto_maker_discount_mult = snapshot.maker_discount_mult
        self.cfg.auto_taker_discount_mult = snapshot.taker_discount_mult
        self.cfg.auto_use_bnb_discount = snapshot.use_bnb_discount
        self.cfg.auto_vip_tier = snapshot.vip_tier
        self.cfg.auto_refresh_metadata = {
            "reason": reason,
            "source": snapshot.source,
            "built_at": meta.get("built_at"),
            "maker_bps": snapshot.maker_bps_default,
            "taker_bps": snapshot.taker_bps_default,
            "maker_discount_mult": snapshot.maker_discount_mult,
            "taker_discount_mult": snapshot.taker_discount_mult,
            "use_bnb_discount": snapshot.use_bnb_discount,
            "discount_rate": snapshot.discount_rate,
            "vip_label": snapshot.vip_label,
            "vip_tier": snapshot.vip_tier,
        }

        meta.setdefault("auto_refresh_metadata", dict(self.cfg.auto_refresh_metadata))
        return {"table": table, "meta": meta, "account": {}, "share": None}

    def _load_account_fee_info(self) -> AccountFeeInfo | None:
        self.account_fee_status = "pending"
        base_url = self.cfg.account_info_endpoint
        if base_url:
            self.account_fee_endpoint = base_url
        else:
            self.account_fee_endpoint = _DEFAULT_BINANCE_SAPI_BASE

        recv_window = self.cfg.account_info_recv_window_ms
        if recv_window is None or recv_window <= 0:
            recv_window = 5_000
        self.account_fee_recv_window = recv_window

        timeout_s = self.cfg.account_info_timeout_s
        if timeout_s is None or timeout_s <= 0.0:
            timeout_s = 10.0
        self.account_fee_timeout = timeout_s

        api_key = _safe_str(self.cfg.account_info_api_key) or _safe_str(
            os.environ.get("BINANCE_API_KEY")
        )
        api_secret = _safe_str(self.cfg.account_info_api_secret) or _safe_str(
            os.environ.get("BINANCE_API_SECRET")
        )

        if not api_key or not api_secret:
            self.account_fee_status = "missing_credentials"
            logger.warning(
                "Account info fetch enabled but API credentials are missing; skipping"
            )
            return None

        try:
            info = fetch_account_fee_info(
                api_key=api_key,
                api_secret=api_secret,
                base_url=base_url,
                recv_window_ms=recv_window,
                timeout=timeout_s,
            )
        except Exception as exc:  # pragma: no cover - network/auth failures
            self.account_fee_status = "error"
            self.account_fee_error = f"{exc.__class__.__name__}: {exc}"
            logger.warning("Failed to fetch Binance account info: %s", exc, exc_info=True)
            return None

        self.account_fee_status = "ok"
        self.account_fee_error = None
        logger.info(
            "Fetched Binance account fees: vip_tier=%s maker_bps=%s taker_bps=%s",
            info.vip_tier,
            info.maker_bps,
            info.taker_bps,
        )
        return info

    @property
    def model(self):
        return self._model

    def get_expected_info(self) -> Dict[str, Any]:
        return {
            "expected": dict(self.expected_payload),
            "metadata": dict(self.metadata),
            "symbol_fee_table": {
                "count": len(self.symbol_fee_table),
                "inline_count": len(self.inline_symbol_fee_table),
                "file_count": len(self.symbol_fee_table_raw),
            },
        }

    def attach_to(self, sim) -> None:
        config_payload = dict(self.model_payload)
        raw_table = config_payload.get("symbol_fee_table")
        symbol_table: Dict[str, Any] = {}
        if isinstance(raw_table, Mapping):
            for symbol, payload in raw_table.items():
                if not isinstance(symbol, str):
                    continue
                if isinstance(payload, Mapping):
                    symbol_table[symbol.upper()] = {k: v for k, v in payload.items()}
                else:
                    symbol_table[symbol.upper()] = payload

        quantizer = getattr(sim, "quantizer", None)
        getter = getattr(quantizer, "get_commission_step", None)
        if callable(getter):
            symbols_to_update = set(symbol_table.keys())
            sim_symbol = getattr(sim, "symbol", None)
            if isinstance(sim_symbol, str):
                symbols_to_update.add(sim_symbol.upper())
            for symbol_key in symbols_to_update:
                try:
                    step_raw = getter(symbol_key)
                except Exception:
                    continue
                step_val = _safe_non_negative_float(step_raw)
                if step_val is None or step_val <= 0.0:
                    continue
                entry_payload = symbol_table.get(symbol_key, {})
                if isinstance(entry_payload, Mapping):
                    entry = {k: v for k, v in entry_payload.items()}
                else:
                    entry = {}
                existing_step = _safe_non_negative_float(entry.get("commission_step"))
                if existing_step is None or existing_step <= 0.0:
                    entry["commission_step"] = float(step_val)
                quant_block = entry.get("quantizer")
                if isinstance(quant_block, Mapping):
                    quant_payload = {k: v for k, v in quant_block.items()}
                else:
                    quant_payload = {}
                quant_step = _safe_non_negative_float(quant_payload.get("commission_step"))
                if quant_step is None or quant_step <= 0.0:
                    quant_payload["commission_step"] = float(step_val)
                entry["quantizer"] = quant_payload
                symbol_table[symbol_key] = entry
        if symbol_table:
            config_payload["symbol_fee_table"] = symbol_table

        if FeesModel is not None and self.cfg.enabled:
            try:
                self._model = FeesModel.from_dict(copy.deepcopy(config_payload))
            except Exception:
                logger.debug(
                    "Failed to rebuild FeesModel with simulator context", exc_info=True
                )

        self.model_payload = config_payload
        if symbol_table:
            self.symbol_fee_table = {k: dict(v) for k, v in symbol_table.items() if isinstance(v, Mapping)}

        if self._model is not None:
            setattr(sim, "fees", self._model)
        share_payload = None
        if self.maker_taker_share_cfg is not None:
            share_payload = self.maker_taker_share_cfg.to_sim_payload(
                self.base_fee_bps["maker_fee_bps"],
                self.base_fee_bps["taker_fee_bps"],
            )
        elif isinstance(self.maker_taker_share_raw, Mapping):
            share_payload = dict(self.maker_taker_share_raw)
        setattr(sim, "_maker_taker_share_cfg", share_payload)
        try:
            setattr(sim, "fees_config_payload", dict(self.model_payload))
        except Exception:
            logger.debug("Failed to attach fees_config_payload to simulator", exc_info=True)
        try:
            setattr(sim, "fees_metadata", dict(self.metadata))
        except Exception:
            logger.debug("Failed to attach fees_metadata to simulator", exc_info=True)
        try:
            setattr(sim, "fees_expected_payload", dict(self.expected_payload))
        except Exception:
            logger.debug("Failed to attach fees_expected_payload to simulator", exc_info=True)
        setter = getattr(sim, "set_fees_config", None)
        if callable(setter):
            try:
                setter(
                    dict(self.model_payload),
                    share_payload,
                    dict(self.metadata),
                    dict(self.expected_payload),
                )
            except Exception:
                logger.debug("Simulator set_fees_config call failed", exc_info=True)
        try:
            setattr(sim, "_fees_get_expected_info", self.get_expected_info)
        except Exception:
            logger.debug("Failed to attach _fees_get_expected_info", exc_info=True)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FeesImpl":
        use_bnb = d.get("use_bnb_discount")
        maker_mult = d.get("maker_discount_mult")
        taker_mult = d.get("taker_discount_mult")

        share_block = d.get("maker_taker_share")
        share_payload: Optional[Dict[str, Any]] = None
        share_cfg = MakerTakerShareSettings.parse(share_block)
        if share_cfg is not None:
            share_payload = share_cfg.as_dict()
        elif isinstance(share_block, Mapping):
            share_payload = dict(share_block)

        share_enabled = d.get("maker_taker_share_enabled")
        share_mode = d.get("maker_taker_share_mode") or d.get("maker_share_mode")
        share_default = d.get("maker_share_default")
        spread_maker = d.get("spread_cost_maker_bps")
        spread_taker = d.get("spread_cost_taker_bps")
        taker_override = d.get("taker_fee_override_bps")

        symbol_table = None
        for key in ("symbol_fee_table", "symbols", "fees_by_symbol"):
            block = d.get(key)
            if isinstance(block, Mapping):
                symbol_table = dict(block)
                break

        metadata = None
        for key in ("metadata", "meta"):
            block = d.get(key)
            if isinstance(block, Mapping):
                metadata = dict(block)
                break

        account_info_cfg = None
        for key in ("account_info", "account_fetch"):
            block = d.get(key)
            if isinstance(block, Mapping):
                account_info_cfg = dict(block)
                break

        rounding_cfg = None
        for key in ("rounding", "rounding_options"):
            block = d.get(key)
            if isinstance(block, Mapping):
                rounding_cfg = dict(block)
                break

        settlement_cfg = None
        for key in ("settlement", "settlement_options"):
            block = d.get(key)
            if isinstance(block, Mapping):
                settlement_cfg = dict(block)
                break

        public_snapshot_cfg = None
        for key in ("public_snapshot", "public_fee_snapshot", "public_fee_overrides"):
            block = d.get(key)
            if isinstance(block, Mapping):
                public_snapshot_cfg = dict(block)
                break

        path = None
        for key in ("path", "fees_path", "symbol_fee_path"):
            candidate = d.get(key)
            if candidate:
                path = candidate
                break

        refresh_days = d.get("refresh_days")

        vip_tier = d.get("vip_tier")
        fee_rounding_step = d.get("fee_rounding_step")

        return FeesImpl(
            FeesConfig(
                enabled=d.get("enabled", True),
                path=path,
                refresh_days=refresh_days,
                maker_bps=d.get("maker_bps"),
                taker_bps=d.get("taker_bps"),
                use_bnb_discount=use_bnb,
                maker_discount_mult=maker_mult,
                taker_discount_mult=taker_mult,
                vip_tier=vip_tier,
                fee_rounding_step=fee_rounding_step,
                rounding=rounding_cfg or {},
                settlement=settlement_cfg or {},
                public_snapshot=public_snapshot_cfg or {},
                symbol_fee_table=symbol_table,
                metadata=metadata,
                account_info=account_info_cfg or {},
                maker_taker_share=share_payload,
                maker_taker_share_enabled=share_enabled,
                maker_taker_share_mode=share_mode,
                maker_share_default=share_default,
                spread_cost_maker_bps=spread_maker,
                spread_cost_taker_bps=spread_taker,
                taker_fee_override_bps=taker_override,
            )
        )
