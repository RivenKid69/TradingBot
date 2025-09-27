# sim/quantizer.py
from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Optional, Any, Tuple

Number = float


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        if isinstance(v, (int, float)):
            return float(v)
        # Binance JSON usually stores numbers as strings
        return float(str(v))
    except Exception:
        return float(default)


def _to_optional_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return float(str(v))
    except Exception:
        return None


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if math.isfinite(v):
            return int(v)
        return None
    try:
        text = str(v).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _extract_precision_fields(filters: Dict[str, Any]) -> Tuple[Optional[int], float]:
    quote_precision = _to_int(filters.get("quotePrecision"))

    direct_step = _to_optional_float(filters.get("commission_step"))
    if direct_step is not None and direct_step > 0:
        return quote_precision, float(direct_step)

    precision_candidates = (
        filters.get("quoteCommissionPrecision"),
        filters.get("commissionPrecision"),
        filters.get("baseCommissionPrecision"),
        filters.get("quoteAssetPrecision"),
        filters.get("baseAssetPrecision"),
        filters.get("quotePrecision"),
        quote_precision,
    )
    for candidate in precision_candidates:
        precision = _to_int(candidate)
        if precision is None or precision < 0:
            continue
        step = float(10.0 ** (-precision))
        if step > 0:
            return quote_precision, step

    return quote_precision, 0.0


@dataclass
class OrderCheckResult:
    """Result of :meth:`Quantizer.quantize_order` processing."""

    price: float
    qty: float
    reason_code: Optional[str] = None
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    @property
    def accepted(self) -> bool:
        """Whether the order passed all checks."""

        return self.reason_code is None


@dataclass
class SymbolFilters:
    price_tick: float = 0.0
    price_min: float = 0.0
    price_max: float = float("inf")
    qty_step: float = 0.0
    qty_min: float = 0.0
    qty_max: float = float("inf")
    min_notional: float = 0.0
    # PERCENT_PRICE_BY_SIDE (spot) / PERCENT_PRICE (futures)
    multiplier_up: Optional[float] = None
    multiplier_down: Optional[float] = None
    quote_precision: Optional[int] = None
    commission_step: float = 0.0

    @classmethod
    def from_exchange_filters(cls, filters: Dict[str, Any]) -> "SymbolFilters":
        pf = filters.get("PRICE_FILTER", {})
        ls = filters.get("LOT_SIZE", {})
        mn = filters.get("MIN_NOTIONAL", {})
        ppbs = filters.get("PERCENT_PRICE_BY_SIDE", {}) or filters.get("PERCENT_PRICE", {})
        quote_precision, commission_step = _extract_precision_fields(filters)
        return cls(
            price_tick=_to_float(pf.get("tickSize"), 0.0),
            price_min=_to_float(pf.get("minPrice"), 0.0),
            price_max=_to_float(pf.get("maxPrice"), float("inf")),
            qty_step=_to_float(ls.get("stepSize"), 0.0),
            qty_min=_to_float(ls.get("minQty"), 0.0),
            qty_max=_to_float(ls.get("maxQty"), float("inf")),
            min_notional=_to_float(mn.get("minNotional"), 0.0),
            multiplier_up=_to_optional_float(ppbs.get("multiplierUp")) if ppbs else None,
            multiplier_down=_to_optional_float(ppbs.get("multiplierDown")) if ppbs else None,
            quote_precision=quote_precision,
            commission_step=commission_step,
        )


class Quantizer:
    """
    Единый квантайзер цен/количеств и проверок для Binance-символов.
    Используется и в симуляторе, и в live-адаптере. Для унификации
    последовательности проверок при подготовке ордеров следует
    использовать :meth:`quantize_order`.
    """

    def __init__(self, filters: Dict[str, Dict[str, Any]], strict: bool = True):
        """
        :param filters: словарь вида:
            {
              "BTCUSDT": {
                "PRICE_FILTER": {...},
                "LOT_SIZE": {...},
                "MIN_NOTIONAL": {...},
                "PERCENT_PRICE_BY_SIDE": {...}  # если есть
              },
              ...
            }
        :param strict: если True — нарушения фильтров приводят к исключениям;
                       если False — нарушения приводят к «обнулению» объёма.
        """
        self.strict = bool(strict)
        self._filters: Dict[str, SymbolFilters] = {}
        for sym, f in (filters or {}).items():
            key = self._normalize_symbol_key(sym)
            if not key:
                continue
            self._filters[key] = SymbolFilters.from_exchange_filters(f or {})

    # ------------ Вспомогательные методы ------------
    @staticmethod
    def _snap(value: Number, step: Number) -> Number:
        if step <= 0:
            return float(value)
        # Binance требует округление вниз к ближайшему валидному шагу.
        # В арифметике с плавающей точкой деление ``value / step`` может
        # давать результат на несколько ULP ниже ожидаемого целого числа,
        # что в сочетании с ``floor`` приводит к «срезанию» дополнительного
        # шага (например, 2.0 -> 1.99999 при step=1e-5). Добавляем малый
        # положительный допуск перед ``floor`` для компенсации накопленной
        # погрешности.
        ratio = float(value) / step
        snapped_units = math.floor(ratio + 1e-9)
        return snapped_units * step

    # ------------ Публичные методы ------------
    @staticmethod
    def _normalize_symbol_key(symbol: Any) -> str:
        try:
            text = str(symbol).strip()
        except Exception:
            return ""
        return text.upper()

    def _get_filters(self, symbol: str) -> Optional[SymbolFilters]:
        key = self._normalize_symbol_key(symbol)
        if not key:
            return None
        return self._filters.get(key)

    def has_symbol(self, symbol: str) -> bool:
        key = self._normalize_symbol_key(symbol)
        if not key:
            return False
        return key in self._filters

    def get_commission_step(self, symbol: str) -> float:
        f = self._get_filters(symbol)
        if not f:
            return 0.0
        step = float(f.commission_step or 0.0)
        if step > 0:
            return step
        qp = f.quote_precision
        if qp is not None and qp >= 0:
            try:
                return float(10.0 ** (-int(qp)))
            except Exception:
                return 0.0
        return 0.0

    def quantize_price(self, symbol: str, price: Number) -> Number:
        f = self._get_filters(symbol)
        if not f:
            return float(price)
        p = float(price)
        if f.price_tick > 0:
            p = self._snap(p, f.price_tick)
        if p < f.price_min:
            p = f.price_min
        if p > f.price_max:
            p = f.price_max
        return p

    def quantize_qty(self, symbol: str, qty: Number) -> Number:
        f = self._get_filters(symbol)
        if not f:
            return float(qty)
        q = abs(float(qty))
        if f.qty_step > 0:
            q = self._snap(q, f.qty_step)
        if q < f.qty_min:
            q = 0.0 if not self.strict else f.qty_min
        if q > f.qty_max:
            q = f.qty_max
        return q

    def clamp_notional(self, symbol: str, price: Number, qty: Number) -> Number:
        f = self._get_filters(symbol)
        if not f:
            return float(qty)
        notional = abs(float(price) * float(qty))
        if f.min_notional <= 0:
            return float(qty)
        if notional >= f.min_notional:
            return float(qty)
        # Увеличим qty до минимума, соблюдая qty_step
        required = f.min_notional / max(1e-12, float(price))
        if f.qty_step > 0:
            required = math.ceil(required / f.qty_step) * f.qty_step
        if required < f.qty_min:
            required = f.qty_min
        if required > f.qty_max:
            # невозможно удовлетворить — вернём 0 (или исключение в strict)
            if self.strict:
                raise ValueError(f"MIN_NOTIONAL cannot be met for {symbol}: price={price}, "
                                 f"qty_max={f.qty_max}, min_notional={f.min_notional}")
            return 0.0
        return float(required)

    def check_percent_price_by_side(self, symbol: str, side: str, price: Number, ref_price: Number) -> bool:
        """
        Проверка PERCENT_PRICE_BY_SIDE (для spot) или PERCENT_PRICE (для futures).
        :param side: "BUY" или "SELL"
        """
        f = self._get_filters(symbol)
        if not f or f.multiplier_up is None or f.multiplier_down is None:
            return True
        p = float(price)
        r = max(1e-12, float(ref_price))
        if str(side).upper() == "BUY":
            # BUY price <= ref * multiplierUp
            return p <= r * float(f.multiplier_up)
        else:
            # SELL price >= ref * multiplierDown
            return p >= r * float(f.multiplier_down)

    def quantize_order(
        self,
        symbol: str,
        side: str,
        price: Number,
        qty: Number,
        ref_price: Number,
        *,
        enforce_ppbs: bool = True,
    ) -> OrderCheckResult:
        """Quantize ``price``/``qty`` and evaluate exchange filters in order.

        The helper consolidates the ``quantize_price`` → ``quantize_qty`` →
        ``clamp_notional`` → ``check_percent_price_by_side`` pipeline so the
        call sites do not need to repeat the sequencing.  On success the
        returned :class:`OrderCheckResult` contains the adjusted values and
        ``reason_code`` is ``None``.  When a filter rejects the order the
        quantized values are still returned along with a normalized reason code
        (``MIN_NOTIONAL``, ``PPBS`` and so on) plus optional details for
        logging.
        """

        quantized_price = float(self.quantize_price(symbol, price))
        quantized_qty = float(self.quantize_qty(symbol, qty))
        filters = self._get_filters(symbol)

        # Detect quantity being clipped to zero by LOT_SIZE
        if filters and quantized_qty <= 0.0 and float(qty) > 0.0 and filters.qty_min > 0.0:
            return OrderCheckResult(
                price=quantized_price,
                qty=quantized_qty,
                reason_code="MIN_QTY",
                reason=f"Quantity {qty} is below LOT_SIZE.minQty={filters.qty_min}",
                details={"min_qty": filters.qty_min, "original_qty": float(qty)},
            )

        try:
            clamped_qty = float(self.clamp_notional(symbol, quantized_price, quantized_qty))
        except ValueError as exc:
            return OrderCheckResult(
                price=quantized_price,
                qty=0.0,
                reason_code="MIN_NOTIONAL",
                reason=str(exc),
                details={
                    "min_notional": filters.min_notional if filters else None,
                    "price": quantized_price,
                    "qty": quantized_qty,
                },
            )

        if filters and quantized_qty > 0.0 and clamped_qty <= 0.0 and filters.min_notional > 0.0:
            return OrderCheckResult(
                price=quantized_price,
                qty=clamped_qty,
                reason_code="MIN_NOTIONAL",
                reason=(
                    f"Order notional {quantized_price * quantized_qty} below MIN_NOTIONAL="
                    f"{filters.min_notional}"
                ),
                details={
                    "min_notional": filters.min_notional,
                    "price": quantized_price,
                    "qty": quantized_qty,
                },
            )

        quantized_qty = clamped_qty

        if enforce_ppbs and not self.check_percent_price_by_side(symbol, side, quantized_price, ref_price):
            return OrderCheckResult(
                price=quantized_price,
                qty=0.0,
                reason_code="PPBS",
                reason="PERCENT_PRICE_BY_SIDE filter rejected the order",
                details={
                    "side": str(side).upper(),
                    "price": quantized_price,
                    "ref_price": float(ref_price),
                    "multiplier_up": filters.multiplier_up if filters else None,
                    "multiplier_down": filters.multiplier_down if filters else None,
                },
            )

        return OrderCheckResult(price=quantized_price, qty=quantized_qty)

    # ------------ Фабрики загрузки ------------
    def raw_filters(self) -> Dict[str, Dict[str, Any]]:
        exported: Dict[str, Dict[str, Any]] = {}
        for key, value in self._filters.items():
            try:
                exported[key] = asdict(value)
            except Exception:
                try:
                    exported[key] = dict(vars(value))
                except Exception:
                    exported[key] = {}
        return exported

    @classmethod
    def from_json_file(cls, path: str, strict: bool = True) -> "Quantizer":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f) or {}
        data = raw.get("filters", raw)
        return cls(data, strict=strict)

    @staticmethod
    def load_filters(
        path: str,
        *,
        max_age_days: int = 30,
        fatal: bool = False,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Загружает словарь фильтров и метаданные из JSON.

        Помимо чтения файла проверяет свежесть поля ``metadata.built_at`` (если есть)
        или ``metadata.generated_at`` и предупреждает, если данные устарели. При
        ``fatal=True`` вместо предупреждения выбрасывается :class:`RuntimeError`.

        Возвращает пару ``(filters, metadata)`` или ``({}, {})`` если файл
        отсутствует. Совместимо с прежним форматом без метаданных.
        """
        if not path or not os.path.exists(path):
            return {}, {}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f) or {}
        if "filters" in raw:
            filters = raw.get("filters", {})
            meta_raw = raw.get("metadata", {})
        else:
            filters, meta_raw = raw, {}

        meta: Dict[str, Any] = dict(meta_raw) if isinstance(meta_raw, dict) else {}

        timestamp_field: Optional[str] = None
        timestamp_value: Optional[str] = None
        for field in ("built_at", "generated_at"):
            value = meta.get(field)
            if isinstance(value, str):
                timestamp_field = field
                timestamp_value = value
                break

        if timestamp_value:
            if timestamp_field == "built_at" and "generated_at" not in meta:
                meta["generated_at"] = timestamp_value
            try:
                ts = datetime.fromisoformat(timestamp_value.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - ts).days
                if age_days > int(max_age_days):
                    field_label = f"metadata.{timestamp_field}" if timestamp_field else "metadata timestamp"
                    msg = (
                        f"{path} {field_label} is {age_days} days old (>={max_age_days}d); "
                        f"refresh via python scripts/fetch_binance_filters.py"
                    )
                    if fatal:
                        raise RuntimeError(msg)
                    warnings.warn(msg)
            except Exception:
                pass

        return filters, meta


# для обратной совместимости
def load_filters(
    path: str,
    *,
    max_age_days: int = 30,
    fatal: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    return Quantizer.load_filters(path, max_age_days=max_age_days, fatal=fatal)
