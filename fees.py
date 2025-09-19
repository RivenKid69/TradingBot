# sim/fees.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
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


def _sanitize_discount(value: Any, default: float) -> float:
    sanitized = _sanitize_optional_non_negative(value)
    if sanitized is None:
        return float(default)
    return float(sanitized)


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
        self.fee_rounding_step = (
            None
            if self.fee_rounding_step is None
            else _sanitize_rounding_step(self.fee_rounding_step)
        )

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
        rounding_step = data.get("fee_rounding_step")
        rounding_step = (
            None
            if rounding_step is None
            else _sanitize_rounding_step(rounding_step)
        )

        return cls(
            base_rate=base,
            vip_rates=vip_rates,
            maker_discount_mult=maker_mult,
            taker_discount_mult=taker_mult,
            fee_rounding_step=rounding_step,
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
    """

    maker_bps: float = 1.0
    taker_bps: float = 5.0
    use_bnb_discount: bool = False
    maker_discount_mult: float = 1.0
    taker_discount_mult: float = 1.0
    vip_tier: int = 0
    symbol_fee_table: Dict[str, SymbolFeeConfig] = field(default_factory=dict)
    fee_rounding_step: float = 0.0

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
        fee_rounding_step = _sanitize_rounding_step(d.get("fee_rounding_step"))

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
        )

    def _fallback_rate(self) -> FeeRate:
        return FeeRate(maker_bps=self.maker_bps, taker_bps=self.taker_bps)

    def _symbol_config(self, symbol: Optional[str]) -> Optional[SymbolFeeConfig]:
        if not symbol or not isinstance(symbol, str):
            return None
        return self.symbol_fee_table.get(symbol.upper())

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
        step = self.fee_rounding_step
        cfg = self._symbol_config(symbol)
        if cfg and cfg.fee_rounding_step:
            step = cfg.fee_rounding_step
        if step <= 0.0:
            return float(fee)
        return round(float(fee) / step) * step

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
    ) -> float:
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

        Returns
        -------
        float
            Абсолютная величина комиссии (>= 0). При некорректных данных возвращает ``0``.
        """

        try:
            price_f = float(price)
            qty_f = float(qty)
        except (TypeError, ValueError):
            return 0.0
        if not (math.isfinite(price_f) and math.isfinite(qty_f)):
            return 0.0

        notional = abs(price_f * qty_f)
        if notional <= 0.0:
            return 0.0

        is_maker = str(liquidity).lower() == "maker"
        rate_bps = self.get_fee_bps(symbol, is_maker)
        rate_bps *= self._discount_multiplier(symbol, is_maker)

        fee = notional * (rate_bps / 1e4)
        if not math.isfinite(fee) or fee <= 0.0:
            return 0.0
        fee = self._round_fee(fee, symbol)
        return float(_sanitize_non_negative(fee, 0.0))


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
            self._next_ts_ms = None if self._next_ts_ms is None else self._next_ts_ms
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
