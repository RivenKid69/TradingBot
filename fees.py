# sim/fees.py
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from decimal import (
    Decimal,
    DivisionByZero,
    InvalidOperation,
    ROUND_DOWN,
    ROUND_HALF_UP,
    ROUND_UP,
)
from typing import Optional, Dict, Any, Tuple, List, Mapping


def _sanitize_optional_non_negative(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < 0.0:
        return None
    return result


def _sanitize_non_negative(value: Any, default: float) -> float:
    sanitized = _sanitize_optional_non_negative(value)
    if sanitized is None:
        return float(default)
    return float(sanitized)


def _sanitize_probability(value: Any, default: float = 0.5) -> float:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(prob):
        return float(default)
    return float(min(max(prob, 0.0), 1.0))


def _sanitize_int(value: Any, default: int = 0, *, minimum: int = 0) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        ivalue = int(default)
    if ivalue < minimum:
        ivalue = minimum
    return int(ivalue)


def _sanitize_rounding_step(value: Any) -> float:
    sanitized = _sanitize_optional_non_negative(value)
    if sanitized is None or sanitized <= 0.0:
        return 0.0
    return float(sanitized)


def _sanitize_positive_float(value: Any) -> Optional[float]:
    sanitized = _sanitize_optional_non_negative(value)
    if sanitized is None:
        return None
    if sanitized <= 0.0:
        return None
    return float(sanitized)


def _precision_to_step(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        precision = int(value)
    except (TypeError, ValueError):
        return None
    if precision < 0:
        return None
    try:
        return float(Decimal(1).scaleb(-precision))
    except (InvalidOperation, ValueError):
        try:
            return float(10.0 ** (-precision))
        except Exception:
            return None


def _extract_commission_step(payload: Any) -> Optional[float]:
    if payload is None:
        return None

    if isinstance(payload, Mapping):
        direct = payload.get("commission_step")
        if direct is None and "commissionStep" in payload:
            direct = payload.get("commissionStep")
        step = _sanitize_positive_float(direct)
        if step is not None:
            return step

        precision_candidate = payload.get("commission_precision")
        if precision_candidate is None and "commissionPrecision" in payload:
            precision_candidate = payload.get("commissionPrecision")
        step = _precision_to_step(precision_candidate)
        if step is not None:
            return step

        quote_candidate = payload.get("quote_precision")
        if quote_candidate is None and "quotePrecision" in payload:
            quote_candidate = payload.get("quotePrecision")
        step = _precision_to_step(quote_candidate)
        if step is not None:
            return step

        for nested_key in ("quantizer", "filters", "symbol_filters"):
            nested = payload.get(nested_key)
            if nested is None:
                continue
            step = _extract_commission_step(nested)
            if step is not None:
                return step

        return None

    step = _sanitize_positive_float(getattr(payload, "commission_step", None))
    if step is not None:
        return step

    precision_candidate = getattr(payload, "commission_precision", None)
    step = _precision_to_step(precision_candidate)
    if step is not None:
        return step

    quote_candidate = getattr(payload, "quote_precision", None)
    return _precision_to_step(quote_candidate)


def _round_value_to_decimals(value: float, decimals: int) -> float:
    if decimals < 0:
        return float(value)
    try:
        quant = Decimal(1).scaleb(-decimals)
        rounded = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
        result = float(rounded)
        if math.isfinite(result):
            return result
    except (InvalidOperation, ValueError):
        pass

    try:
        factor = 10 ** decimals
        scaled = float(value) * factor
        if scaled >= 0.0:
            adjusted = math.floor(scaled + 0.5)
        else:
            adjusted = math.ceil(scaled - 0.5)
        result = adjusted / factor
        if math.isfinite(result):
            return float(result)
    except Exception:
        pass

    return float(value)


def _round_value_to_step(value: float, step: float, mode: str) -> float:
    if step <= 0.0:
        return float(value)

    mode_normalized = (mode or "").strip().lower()
    if mode_normalized in {"nearest", "round", "half", "half_up"}:
        rounding = ROUND_HALF_UP
    elif mode_normalized in {"down", "floor"}:
        rounding = ROUND_DOWN
    else:
        rounding = ROUND_UP

    try:
        dec_value = Decimal(str(value))
        dec_step = Decimal(str(step))
        if dec_step <= 0:
            return float(value)
        units = (dec_value / dec_step).to_integral_value(rounding=rounding)
        rounded = units * dec_step
        result = float(rounded)
        if math.isfinite(result):
            return result
    except (DivisionByZero, InvalidOperation, ValueError):
        pass

    ratio = float(value) / float(step)
    if mode_normalized in {"nearest", "round", "half", "half_up"}:
        units = round(ratio)
    elif mode_normalized in {"down", "floor"}:
        units = math.floor(ratio + 1e-15)
    else:
        units = math.ceil(ratio - 1e-15)
    return float(units * float(step))


def _apply_rounding_rules(
    value: float, *, step: Optional[float], options: Optional[Mapping[str, Any]]
) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 0.0

    opts = options or {}
    mode = str(opts.get("mode", "") or "").strip().lower()

    decimals = opts.get("decimals")
    precision = opts.get("precision")
    minimum_fee = opts.get("minimum_fee")
    maximum_fee = opts.get("maximum_fee")

    result = float(value)

    step_value = step if step is not None and step > 0.0 else None
    if step_value is not None:
        result = _round_value_to_step(result, step_value, mode)

    digits: Optional[int] = None
    if decimals is not None:
        try:
            digits = int(decimals)
        except (TypeError, ValueError):
            digits = None
    if digits is None and precision is not None:
        try:
            digits = int(precision)
        except (TypeError, ValueError):
            digits = None
    if digits is not None and digits >= 0:
        result = _round_value_to_decimals(result, digits)

    if minimum_fee is not None:
        try:
            min_value = float(minimum_fee)
        except (TypeError, ValueError):
            min_value = None
        else:
            if math.isfinite(min_value) and min_value > 0.0:
                result = max(result, min_value)

    if maximum_fee is not None:
        try:
            max_value = float(maximum_fee)
        except (TypeError, ValueError):
            max_value = None
        else:
            if math.isfinite(max_value) and max_value >= 0.0:
                result = min(result, max_value)

    if result < 0.0 or not math.isfinite(result):
        return 0.0
    return float(result)


def _sanitize_discount(value: Any, default: float) -> float:
    sanitized = _sanitize_optional_non_negative(value)
    if sanitized is None:
        return float(default)
    return float(sanitized)


def _sanitize_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def _sanitize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    return text or None


def _sanitize_rounding_config(data: Any) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        return {}

    mapping = dict(data)
    normalized: Dict[str, Any] = {}

    step_value = _sanitize_optional_non_negative(mapping.get("step"))
    if step_value is not None and step_value > 0.0:
        normalized["step"] = float(step_value)

    mode = _sanitize_text(mapping.get("mode"))
    if mode:
        normalized["mode"] = mode.lower()

    precision = mapping.get("precision")
    if precision is not None:
        try:
            precision_int = int(precision)
        except (TypeError, ValueError):
            precision_int = None
        else:
            if precision_int >= 0:
                normalized["precision"] = precision_int

    decimals = mapping.get("decimals")
    if decimals is not None:
        try:
            decimals_int = int(decimals)
        except (TypeError, ValueError):
            decimals_int = None
        else:
            if decimals_int >= 0:
                normalized["decimals"] = decimals_int

    minimum_fee = _sanitize_optional_non_negative(mapping.get("minimum_fee"))
    if minimum_fee is None:
        minimum_fee = _sanitize_optional_non_negative(mapping.get("min_fee"))
    if minimum_fee is None:
        minimum_fee = _sanitize_optional_non_negative(mapping.get("minimum"))
    if minimum_fee is not None:
        normalized["minimum_fee"] = float(minimum_fee)

    maximum_fee = _sanitize_optional_non_negative(mapping.get("maximum_fee"))
    if maximum_fee is None:
        maximum_fee = _sanitize_optional_non_negative(mapping.get("max_fee"))
    if maximum_fee is None:
        maximum_fee = _sanitize_optional_non_negative(mapping.get("maximum"))
    if maximum_fee is not None:
        normalized["maximum_fee"] = float(maximum_fee)

    per_symbol_raw = mapping.get("per_symbol") or mapping.get("symbols")
    if isinstance(per_symbol_raw, Mapping):
        per_symbol: Dict[str, Any] = {}
        for symbol, payload in per_symbol_raw.items():
            if not isinstance(symbol, str):
                continue
            nested = _sanitize_rounding_config(payload)
            if nested:
                per_symbol[symbol.upper()] = nested
        if per_symbol:
            normalized["per_symbol"] = per_symbol

    handled_keys = {
        "enabled",
        "step",
        "mode",
        "precision",
        "decimals",
        "minimum",
        "minimum_fee",
        "min_fee",
        "maximum",
        "maximum_fee",
        "max_fee",
        "per_symbol",
        "symbols",
    }
    for key, value in mapping.items():
        if key in handled_keys:
            continue
        if isinstance(value, Mapping):
            nested = _sanitize_rounding_config(value)
            if nested:
                normalized[key] = nested
        elif isinstance(value, (str, int, float, bool)):
            normalized[key] = value

    enabled = _sanitize_optional_bool(mapping.get("enabled"))
    if enabled is False:
        normalized.pop("step", None)
        normalized["enabled"] = False
    else:
        has_payload = bool(normalized)
        if "step" in normalized:
            normalized["enabled"] = True if enabled is None else bool(enabled)
        elif has_payload:
            normalized["enabled"] = True if enabled is None else bool(enabled)
        elif enabled is not None:
            normalized["enabled"] = bool(enabled)

    return normalized


def _sanitize_settlement_config(data: Any) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        return {}

    mapping = dict(data)
    normalized: Dict[str, Any] = {}

    enabled = _sanitize_optional_bool(mapping.get("enabled"))
    if enabled is not None:
        normalized["enabled"] = bool(enabled)

    mode = (
        _sanitize_text(mapping.get("mode"))
        or _sanitize_text(mapping.get("type"))
        or _sanitize_text(mapping.get("settle_mode"))
    )
    if mode:
        normalized["mode"] = mode.lower()

    currency = (
        _sanitize_text(mapping.get("currency"))
        or _sanitize_text(mapping.get("asset"))
        or _sanitize_text(mapping.get("symbol"))
    )
    if currency:
        normalized["currency"] = currency.upper()

    fallback_currency = (
        _sanitize_text(mapping.get("fallback_currency"))
        or _sanitize_text(mapping.get("fallback_asset"))
    )
    if fallback_currency:
        normalized["fallback_currency"] = fallback_currency.upper()

    priority = mapping.get("priority")
    if priority is not None:
        normalized["priority"] = _sanitize_int(priority, default=0, minimum=0)

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
        for symbol, payload in per_symbol_raw.items():
            if not isinstance(symbol, str):
                continue
            nested = _sanitize_settlement_config(payload)
            if nested:
                per_symbol[symbol.upper()] = nested
        if per_symbol:
            normalized["per_symbol"] = per_symbol

    handled_keys = {
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
    }
    for key, value in mapping.items():
        if key in handled_keys:
            continue
        if isinstance(value, Mapping):
            nested = _sanitize_settlement_config(value)
            if nested:
                normalized[key] = nested
        elif isinstance(value, (str, int, float, bool)):
            normalized[key] = value

    return normalized


@dataclass
class FeeRateSpec:
    maker_bps: Optional[float] = None
    taker_bps: Optional[float] = None

    def __post_init__(self) -> None:
        self.maker_bps = _sanitize_optional_non_negative(self.maker_bps)
        self.taker_bps = _sanitize_optional_non_negative(self.taker_bps)

    @property
    def is_empty(self) -> bool:
        return self.maker_bps is None and self.taker_bps is None

    def merge(self, fallback: "FeeRate") -> "FeeRate":
        return FeeRate(
            maker_bps=self.maker_bps if self.maker_bps is not None else fallback.maker_bps,
            taker_bps=self.taker_bps if self.taker_bps is not None else fallback.taker_bps,
        )


@dataclass
class FeeRate:
    maker_bps: float
    taker_bps: float

    def __post_init__(self) -> None:
        self.maker_bps = _sanitize_non_negative(self.maker_bps, 0.0)
        self.taker_bps = _sanitize_non_negative(self.taker_bps, 0.0)


@dataclass
class SymbolFeeConfig:
    base_rate: FeeRateSpec = field(default_factory=FeeRateSpec)
    vip_rates: Dict[int, FeeRateSpec] = field(default_factory=dict)
    maker_discount_mult: Optional[float] = None
    taker_discount_mult: Optional[float] = None
    fee_rounding_step: Optional[float] = None
    commission_step: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_rate, FeeRateSpec):
            if isinstance(self.base_rate, Mapping):
                self.base_rate = FeeRateSpec(**dict(self.base_rate))
            else:
                self.base_rate = FeeRateSpec()

        normalized_vip: Dict[int, FeeRateSpec] = {}
        for tier, spec in (self.vip_rates or {}).items():
            try:
                tier_int = int(tier)
            except (TypeError, ValueError):
                continue
            if isinstance(spec, FeeRateSpec):
                normalized_vip[tier_int] = FeeRateSpec(
                    maker_bps=spec.maker_bps, taker_bps=spec.taker_bps
                )
            elif isinstance(spec, Mapping):
                normalized_vip[tier_int] = FeeRateSpec(**dict(spec))
        self.vip_rates = normalized_vip

        self.maker_discount_mult = _sanitize_optional_non_negative(self.maker_discount_mult)
        self.taker_discount_mult = _sanitize_optional_non_negative(self.taker_discount_mult)

        if self.commission_step is not None:
            commission_step = _sanitize_positive_float(self.commission_step)
            self.commission_step = commission_step if commission_step is not None else None
        else:
            self.commission_step = None

        if self.fee_rounding_step is not None:
            step = _sanitize_rounding_step(self.fee_rounding_step)
            self.fee_rounding_step = step if step > 0.0 else None
        else:
            self.fee_rounding_step = None

        if self.fee_rounding_step is None and self.commission_step is not None:
            self.fee_rounding_step = self.commission_step

    @classmethod
    def from_dict(cls, data: Any) -> "SymbolFeeConfig":
        if not isinstance(data, Mapping):
            return cls()
        base = FeeRateSpec(
            maker_bps=data.get("maker_bps"),
            taker_bps=data.get("taker_bps"),
        )

        vip_raw = data.get("vip_levels") or data.get("vip_rates") or {}
        vip_rates: Dict[int, FeeRateSpec] = {}
        if isinstance(vip_raw, Mapping):
            for tier, payload in vip_raw.items():
                try:
                    tier_int = int(tier)
                except (TypeError, ValueError):
                    continue
                if isinstance(payload, Mapping):
                    vip_rates[tier_int] = FeeRateSpec(
                        maker_bps=payload.get("maker_bps"),
                        taker_bps=payload.get("taker_bps"),
                    )

        maker_mult = _sanitize_optional_non_negative(data.get("maker_discount_mult"))
        taker_mult = _sanitize_optional_non_negative(data.get("taker_discount_mult"))
        rounding_step_raw = data.get("fee_rounding_step")
        rounding_step = (
            _sanitize_positive_float(rounding_step_raw)
            if rounding_step_raw is not None
            else None
        )

        commission_step = _extract_commission_step(data)
        if commission_step is not None and commission_step <= 0.0:
            commission_step = None
        if rounding_step is None and commission_step is not None:
            rounding_step = commission_step

        return cls(
            base_rate=base,
            vip_rates=vip_rates,
            maker_discount_mult=maker_mult,
            taker_discount_mult=taker_mult,
            fee_rounding_step=rounding_step,
            commission_step=commission_step,
        )

    def resolve_rate(self, vip_tier: int, fallback: FeeRate) -> FeeRate:
        if vip_tier in self.vip_rates:
            rate_spec = self.vip_rates[vip_tier]
        else:
            rate_spec = self.base_rate
        if rate_spec.is_empty:
            return fallback
        return rate_spec.merge(fallback)


@dataclass
class FeeComputation:
    fee: float
    fee_before_rounding: float
    commission_step: Optional[float]
    rounding_step: Optional[float]
    rounding_enabled: bool
    settlement_mode: Optional[str]
    settlement_currency: Optional[str]
    use_bnb_settlement: bool
    bnb_conversion_rate: Optional[float]
    requires_bnb_conversion: bool


@dataclass
class FeesModel:
    """Расширенная модель комиссий Binance.

    Параметры по умолчанию описывают глобальные ставки в базисных пунктах (bps) и
    мультипликаторы скидки BNB. Для конкретных символов можно задать отдельные
    ставки и правила округления через :attr:`symbol_fee_table`.

    Attributes
    ----------
    maker_bps, taker_bps:
        Глобальные комиссии в bps для maker/taker сделок.
    maker_discount_mult, taker_discount_mult:
        Мультипликаторы скидки для расчёта итоговой комиссии. По умолчанию 1.0,
        но могут быть заданы, например, 0.75 при оплате в BNB.
    vip_tier:
        Текущий VIP уровень аккаунта Binance. Используется для выбора ставок из
        таблицы :attr:`symbol_fee_table`.
    symbol_fee_table:
        Словарь ``symbol -> SymbolFeeConfig`` c переопределениями ставок.
    fee_rounding_step:
        Глобальный шаг округления комиссии (например, 0.0001 USDT). Значение
        ``0`` отключает округление.
    rounding:
        Дополнительные параметры округления (включая вложенные настройки).
    settlement:
        Настройки валюты/режима списания комиссий.
    """

    maker_bps: float = 1.0
    taker_bps: float = 5.0
    use_bnb_discount: bool = False
    maker_discount_mult: float = 1.0
    taker_discount_mult: float = 1.0
    vip_tier: int = 0
    symbol_fee_table: Dict[str, SymbolFeeConfig] = field(default_factory=dict)
    fee_rounding_step: float = 0.0
    rounding: Dict[str, Any] = field(default_factory=dict)
    settlement: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.maker_bps = _sanitize_non_negative(self.maker_bps, 1.0)
        self.taker_bps = _sanitize_non_negative(self.taker_bps, 5.0)
        self.maker_discount_mult = _sanitize_discount(
            self.maker_discount_mult, 0.75 if self.use_bnb_discount else 1.0
        )
        self.taker_discount_mult = _sanitize_discount(
            self.taker_discount_mult, 0.75 if self.use_bnb_discount else 1.0
        )
        self.vip_tier = _sanitize_int(self.vip_tier, default=0, minimum=0)
        self.fee_rounding_step = _sanitize_rounding_step(self.fee_rounding_step)

        normalized: Dict[str, SymbolFeeConfig] = {}
        for symbol, cfg in (self.symbol_fee_table or {}).items():
            if not isinstance(symbol, str):
                continue
            key = symbol.upper()
            if isinstance(cfg, SymbolFeeConfig):
                normalized[key] = cfg
            elif isinstance(cfg, Mapping):
                normalized[key] = SymbolFeeConfig.from_dict(cfg)
        self.symbol_fee_table = normalized

        self.rounding = _sanitize_rounding_config(self.rounding)
        rounding_enabled = self.rounding.get("enabled")
        rounding_step = self.rounding.get("step")
        if rounding_enabled is False:
            self.rounding.pop("step", None)
            self.fee_rounding_step = 0.0
        elif rounding_step is not None:
            sanitized_step = _sanitize_rounding_step(rounding_step)
            if sanitized_step > 0.0:
                self.rounding["step"] = sanitized_step
                self.fee_rounding_step = sanitized_step
            else:
                self.rounding.pop("step", None)
        elif "step" in self.rounding:
            self.rounding.pop("step")
        if "enabled" not in self.rounding and self.rounding:
            self.rounding["enabled"] = True

        self.settlement = _sanitize_settlement_config(self.settlement)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeesModel":
        maker_bps = _sanitize_non_negative(d.get("maker_bps"), 1.0)
        taker_bps = _sanitize_non_negative(d.get("taker_bps"), 5.0)
        use_bnb = bool(d.get("use_bnb_discount", False))
        maker_mult = _sanitize_discount(
            d.get("maker_discount_mult"), 0.75 if use_bnb else 1.0
        )
        taker_mult = _sanitize_discount(
            d.get("taker_discount_mult"), 0.75 if use_bnb else 1.0
        )
        vip_tier = _sanitize_int(d.get("vip_tier", 0), default=0, minimum=0)
        raw_rounding = d.get("rounding") or d.get("rounding_options") or {}
        rounding_cfg = _sanitize_rounding_config(raw_rounding)
        raw_settlement = d.get("settlement") or d.get("settlement_options") or {}
        settlement_cfg = _sanitize_settlement_config(raw_settlement)

        rounding_step_candidate = d.get("fee_rounding_step")
        if rounding_step_candidate is None:
            rounding_step_candidate = rounding_cfg.get("step")
        fee_rounding_step = _sanitize_rounding_step(rounding_step_candidate)
        if rounding_cfg.get("enabled") is False:
            fee_rounding_step = 0.0

        symbol_fee_table: Dict[str, SymbolFeeConfig] = {}
        raw_table = d.get("symbol_fee_table") or {}
        if isinstance(raw_table, Mapping):
            for symbol, payload in raw_table.items():
                if not isinstance(symbol, str):
                    continue
                cfg = SymbolFeeConfig.from_dict(payload)
                symbol_fee_table[symbol.upper()] = cfg

        return cls(
            maker_bps=maker_bps,
            taker_bps=taker_bps,
            use_bnb_discount=use_bnb,
            maker_discount_mult=maker_mult,
            taker_discount_mult=taker_mult,
            vip_tier=vip_tier,
            symbol_fee_table=symbol_fee_table,
            fee_rounding_step=fee_rounding_step,
            rounding=rounding_cfg,
            settlement=settlement_cfg,
        )

    def _fallback_rate(self) -> FeeRate:
        return FeeRate(maker_bps=self.maker_bps, taker_bps=self.taker_bps)

    def _symbol_config(self, symbol: Optional[str]) -> Optional[SymbolFeeConfig]:
        if not symbol or not isinstance(symbol, str):
            return None
        return self.symbol_fee_table.get(symbol.upper())

    def _resolve_rounding_context(
        self, symbol: Optional[str]
    ) -> Tuple[bool, Optional[float], Dict[str, Any]]:
        rounding_cfg: Mapping[str, Any] = self.rounding if isinstance(self.rounding, Mapping) else {}

        symbol_key = symbol.upper() if isinstance(symbol, str) else None
        per_symbol_cfg: Optional[Mapping[str, Any]] = None
        if symbol_key and isinstance(rounding_cfg.get("per_symbol"), Mapping):
            candidate = rounding_cfg["per_symbol"].get(symbol_key)
            if isinstance(candidate, Mapping):
                per_symbol_cfg = candidate

        base_step = _sanitize_positive_float(rounding_cfg.get("step")) if isinstance(rounding_cfg, Mapping) else None
        symbol_step = _sanitize_positive_float(per_symbol_cfg.get("step")) if per_symbol_cfg else None

        cfg = self._symbol_config(symbol)
        cfg_step = _sanitize_positive_float(cfg.fee_rounding_step) if cfg and cfg.fee_rounding_step is not None else None

        fallback_step = _sanitize_positive_float(self.fee_rounding_step)

        step_candidates = [symbol_step, cfg_step, base_step, fallback_step]
        step: Optional[float] = None
        for candidate in step_candidates:
            if candidate is not None and candidate > 0.0:
                step = candidate
                break

        merged_options: Dict[str, Any] = {}
        for source in (rounding_cfg, per_symbol_cfg):
            if not isinstance(source, Mapping):
                continue
            for key, value in source.items():
                if key in {"enabled", "step", "per_symbol"}:
                    continue
                merged_options[key] = value

        enabled_override: Optional[bool] = None
        if per_symbol_cfg and "enabled" in per_symbol_cfg:
            enabled_override = bool(per_symbol_cfg.get("enabled"))
        elif isinstance(rounding_cfg, Mapping) and "enabled" in rounding_cfg:
            enabled_override = bool(rounding_cfg.get("enabled"))

        if enabled_override is not None:
            enabled = enabled_override
        else:
            enabled = bool(step is not None or merged_options)

        return enabled, step, merged_options

    def _resolve_settlement_context(
        self, symbol: Optional[str]
    ) -> Tuple[bool, Dict[str, Any]]:
        settlement_cfg: Mapping[str, Any] = (
            self.settlement if isinstance(self.settlement, Mapping) else {}
        )

        symbol_key = symbol.upper() if isinstance(symbol, str) else None
        per_symbol_cfg: Optional[Mapping[str, Any]] = None
        if symbol_key and isinstance(settlement_cfg.get("per_symbol"), Mapping):
            candidate = settlement_cfg["per_symbol"].get(symbol_key)
            if isinstance(candidate, Mapping):
                per_symbol_cfg = candidate

        merged: Dict[str, Any] = {}
        for source in (settlement_cfg, per_symbol_cfg):
            if not isinstance(source, Mapping):
                continue
            for key, value in source.items():
                if key == "per_symbol":
                    continue
                merged[key] = value

        enabled_override: Optional[bool] = None
        if per_symbol_cfg and "enabled" in per_symbol_cfg:
            enabled_override = bool(per_symbol_cfg.get("enabled"))
        elif isinstance(settlement_cfg, Mapping) and "enabled" in settlement_cfg:
            enabled_override = bool(settlement_cfg.get("enabled"))

        if enabled_override is not None:
            enabled = enabled_override
        else:
            enabled = bool(merged)

        return enabled, merged

    def _discount_multiplier(self, symbol: Optional[str], is_maker: bool) -> float:
        base = self.maker_discount_mult if is_maker else self.taker_discount_mult
        cfg = self._symbol_config(symbol)
        if cfg:
            override = (
                cfg.maker_discount_mult if is_maker else cfg.taker_discount_mult
            )
            if override is not None:
                base = _sanitize_discount(override, base)
        return _sanitize_discount(base, 1.0)

    def _round_fee(self, fee: float, symbol: Optional[str]) -> float:
        enabled, step, options = self._resolve_rounding_context(symbol)
        if not enabled:
            return float(_sanitize_non_negative(fee, 0.0))
        rounded = _apply_rounding_rules(float(fee), step=step, options=options)
        return float(_sanitize_non_negative(rounded, 0.0))

    def get_fee_bps(self, symbol: Optional[str], is_maker: bool) -> float:
        """Возвращает актуальную ставку комиссии в bps для заданного символа."""

        fallback = self._fallback_rate()
        cfg = self._symbol_config(symbol)
        if cfg:
            rate = cfg.resolve_rate(self.vip_tier, fallback)
        else:
            rate = fallback
        return float(rate.maker_bps if is_maker else rate.taker_bps)

    def expected_fee_bps(self, symbol: Optional[str], p_maker: float) -> float:
        """Возвращает ожидаемую ставку комиссии с учётом вероятности maker-сделки."""

        prob = _sanitize_probability(p_maker)
        maker_bps = self.get_fee_bps(symbol, True) * self._discount_multiplier(symbol, True)
        taker_bps = self.get_fee_bps(symbol, False) * self._discount_multiplier(symbol, False)
        expected = prob * maker_bps + (1.0 - prob) * taker_bps
        return float(_sanitize_non_negative(expected, 0.0))


    def compute(
        self,
        *,
        side: str,
        price: float,
        qty: float,
        liquidity: str,
        symbol: Optional[str] = None,
        bnb_conversion_rate: Optional[float] = None,
        return_details: bool = False,
    ) -> float | FeeComputation:
        """Расчитывает абсолютную комиссию в валюте котировки.

        Parameters
        ----------
        side:
            ``"BUY"`` или ``"SELL"`` — направление сделки (на комиссию не влияет).
        price:
            Цена сделки.
        qty:
            Количество базового актива (абсолютное значение).
        liquidity:
            ``"maker"`` или ``"taker"`` — тип исполнения.
        symbol:
            Торговый символ. Если не передан, используются глобальные ставки.
        bnb_conversion_rate:
            Цена BNB в валюте котировки. Используется, когда комиссии списываются
            в BNB (``settlement.mode == "bnb"`` или ``settlement.currency == "BNB"``).
        return_details:
            Если ``True``, возвращает :class:`FeeComputation` с подробным описанием
            расчёта комиссии. По умолчанию возвращает только числовое значение.

        Returns
        -------
        float | FeeComputation
            Абсолютная величина комиссии (>= 0). При некорректных данных возвращает ``0``.
            Если комиссии списываются в BNB и передан ``bnb_conversion_rate``,
            результат возвращается в BNB. При ``return_details=True`` возвращается
            экземпляр :class:`FeeComputation` с дополнительными полями.
        """

        def _zero_result() -> FeeComputation:
            return FeeComputation(
                fee=0.0,
                fee_before_rounding=0.0,
                commission_step=None,
                rounding_step=None,
                rounding_enabled=False,
                settlement_mode=None,
                settlement_currency=None,
                use_bnb_settlement=False,
                bnb_conversion_rate=None,
                requires_bnb_conversion=False,
            )

        try:
            price_f = float(price)
            qty_f = float(qty)
        except (TypeError, ValueError):
            return 0.0 if not return_details else _zero_result()
        if not (math.isfinite(price_f) and math.isfinite(qty_f)):
            return 0.0 if not return_details else _zero_result()

        notional = abs(price_f * qty_f)
        if notional <= 0.0:
            return 0.0 if not return_details else _zero_result()

        is_maker = str(liquidity).lower() == "maker"
        rate_bps = self.get_fee_bps(symbol, is_maker)
        rate_bps *= self._discount_multiplier(symbol, is_maker)

        fee = notional * (rate_bps / 1e4)
        if not math.isfinite(fee) or fee <= 0.0:
            return 0.0 if not return_details else FeeComputation(
                fee=0.0,
                fee_before_rounding=0.0,
                commission_step=None,
                rounding_step=None,
                rounding_enabled=False,
                settlement_mode=None,
                settlement_currency=None,
                use_bnb_settlement=False,
                bnb_conversion_rate=None,
                requires_bnb_conversion=False,
            )

        settlement_enabled, settlement_cfg = self._resolve_settlement_context(symbol)
        settlement_mode = str(settlement_cfg.get("mode") or "").lower()
        settlement_currency = str(settlement_cfg.get("currency") or "").upper()

        use_bnb_settlement = settlement_enabled and (
            settlement_mode == "bnb" or settlement_currency == "BNB"
        )

        effective_fee = float(fee)
        conversion_used: Optional[float] = None
        requires_conversion = False
        if use_bnb_settlement:
            conversion = _sanitize_positive_float(bnb_conversion_rate)
            if conversion is not None and conversion > 0.0:
                conversion_used = float(conversion)
                effective_fee = effective_fee / conversion_used
            else:
                requires_conversion = True

        rounding_enabled, rounding_step, rounding_options = self._resolve_rounding_context(
            symbol
        )
        adjusted_rounding_step = rounding_step
        adjusted_rounding_options = rounding_options

        if (
            rounding_enabled
            and use_bnb_settlement
            and conversion_used is not None
            and conversion_used > 0.0
        ):
            if rounding_step is not None and rounding_step > 0.0:
                adjusted_rounding_step = float(rounding_step) / conversion_used

            if rounding_options:
                adjusted_rounding_options = copy.deepcopy(rounding_options)

                def _scale_rounding_payload(payload: Dict[str, Any]) -> None:
                    for key, value in list(payload.items()):
                        if isinstance(value, Mapping):
                            _scale_rounding_payload(value)
                        elif isinstance(value, list):
                            scaled_items = []
                            for item in value:
                                if isinstance(item, Mapping):
                                    item_copy = copy.deepcopy(item)
                                    _scale_rounding_payload(item_copy)
                                    scaled_items.append(item_copy)
                                elif isinstance(item, (int, float)) and not isinstance(item, bool):
                                    scaled_items.append(float(item) / conversion_used)
                                else:
                                    scaled_items.append(item)
                            payload[key] = scaled_items
                        elif isinstance(value, (int, float)) and not isinstance(value, bool):
                            if key not in {"precision", "decimals"}:
                                payload[key] = float(value) / conversion_used

                _scale_rounding_payload(adjusted_rounding_options)

        fee_before_rounding = float(effective_fee)
        if rounding_enabled:
            effective_fee = _apply_rounding_rules(
                float(effective_fee),
                step=adjusted_rounding_step,
                options=adjusted_rounding_options,
            )
        final_fee = float(_sanitize_non_negative(effective_fee, 0.0))

        if not return_details:
            return final_fee

        commission_step: Optional[float] = None
        cfg = self._symbol_config(symbol)
        if cfg and cfg.commission_step is not None:
            step_candidate = _sanitize_positive_float(cfg.commission_step)
            if step_candidate is not None and step_candidate > 0.0:
                commission_step = float(step_candidate)
                if conversion_used is not None and conversion_used > 0.0:
                    commission_step = commission_step / conversion_used
        if (
            commission_step is None
            and adjusted_rounding_step is not None
            and adjusted_rounding_step > 0.0
        ):
            commission_step = float(adjusted_rounding_step)

        rounded_step = None
        if adjusted_rounding_step is not None and adjusted_rounding_step > 0.0:
            rounded_step = float(adjusted_rounding_step)

        return FeeComputation(
            fee=final_fee,
            fee_before_rounding=float(_sanitize_non_negative(fee_before_rounding, 0.0)),
            commission_step=commission_step,
            rounding_step=rounded_step,
            rounding_enabled=bool(rounding_enabled),
            settlement_mode=settlement_mode or None,
            settlement_currency=settlement_currency or None,
            use_bnb_settlement=bool(use_bnb_settlement),
            bnb_conversion_rate=conversion_used,
            requires_bnb_conversion=bool(requires_conversion),
        )


@dataclass
class FundingEvent:
    ts_ms: int
    rate: float
    position_qty: float
    mark_price: float
    cashflow: float  # положительно — получили, отрицательно — заплатили


class FundingCalculator:
    """
    Упрощённый калькулятор funding для перпетуалов.
    Модель: дискретные события каждые interval_seconds (по умолчанию 8 часов).
    Ставка фиксированная (const) на каждое событие. Для гибкости допускаем таблицу ставок.

    Знак cashflow:
      - Для long (qty>0) при rate>0 — платёж (cashflow < 0)
      - Для short (qty<0) при rate>0 — получение (cashflow > 0)
      - При отрицательной ставке — наоборот.
    """
    def __init__(
        self,
        *,
        enabled: bool = False,
        rate_source: str = "const",  # "const" | "curve"
        const_rate_per_interval: float = 0.0,  # например 0.0001 = 1 б.п. за интервал
        interval_seconds: int = 8 * 60 * 60,
        curve: Optional[Dict[int, float]] = None,  # {timestamp_ms->rate}, если rate_source="curve"
        align_to_epoch: bool = True,  # привязка к кратным интервала Epoch (даёт 00:00/08:00/16:00 UTC для 8h)
    ):
        self.enabled = bool(enabled)
        self.rate_source = str(rate_source)
        self.const_rate_per_interval = float(const_rate_per_interval)
        self.interval_seconds = int(interval_seconds)
        self.curve = dict(curve or {})
        self.align_to_epoch = bool(align_to_epoch)
        self._next_ts_ms: Optional[int] = None

    def _next_boundary(self, ts_ms: int) -> int:
        if not self.align_to_epoch:
            return int(ts_ms + self.interval_seconds * 1000)
        sec = int(ts_ms // 1000)
        next_sec = ((sec // self.interval_seconds) + 1) * self.interval_seconds
        return int(next_sec * 1000)

    def _rate_for_ts(self, ts_ms: int) -> float:
        if self.rate_source == "curve":
            # Берём точную ставку на этот момент; если нет — 0
            return float(self.curve.get(int(ts_ms), 0.0))
        # const
        return float(self.const_rate_per_interval)

    def reset(self) -> None:
        self._next_ts_ms = None

    def accrue(self, *, position_qty: float, mark_price: Optional[float], now_ts_ms: int) -> Tuple[float, List[FundingEvent]]:
        """
        Начисляет funding за все прошедшие дискретные моменты с предыдущего вызова.
        :param position_qty: текущая чистая позиция (штук)
        :param mark_price: текущая справедливая цена (для оценки notional)
        :param now_ts_ms: текущее время (мс)
        :return: (total_cashflow, [events...])
        """
        if not self.enabled:
            return 0.0, []
        if mark_price is None or not math.isfinite(float(mark_price)) or abs(position_qty) <= 0.0:
            # Нет цены или позиции — funding не начисляем
            self._next_ts_ms = None
            return 0.0, []

        total = 0.0
        events: List[FundingEvent] = []

        now_ts_ms = int(now_ts_ms)
        if self._next_ts_ms is None:
            self._next_ts_ms = self._next_boundary(now_ts_ms)

        # Если успели пройти сразу несколько интервалов — начислим несколько событий
        while now_ts_ms >= int(self._next_ts_ms):
            rate = self._rate_for_ts(int(self._next_ts_ms))
            notional = abs(float(position_qty)) * float(mark_price)
            # cashflow = - sign(position) * rate * notional
            sign = 1.0 if position_qty > 0 else -1.0
            cf = float(-sign * rate * notional)
            total += cf
            events.append(FundingEvent(
                ts_ms=int(self._next_ts_ms),
                rate=float(rate),
                position_qty=float(position_qty),
                mark_price=float(mark_price),
                cashflow=float(cf),
            ))
            # следующий интервал
            self._next_ts_ms = int(self._next_ts_ms + self.interval_seconds * 1000)

        return float(total), events
