# sim/quantizer.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Any

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

    @classmethod
    def from_exchange_filters(cls, filters: Dict[str, Any]) -> "SymbolFilters":
        pf = filters.get("PRICE_FILTER", {})
        ls = filters.get("LOT_SIZE", {})
        mn = filters.get("MIN_NOTIONAL", {})
        ppbs = filters.get("PERCENT_PRICE_BY_SIDE", {}) or filters.get("PERCENT_PRICE", {})
        return cls(
            price_tick=_to_float(pf.get("tickSize"), 0.0),
            price_min=_to_float(pf.get("minPrice"), 0.0),
            price_max=_to_float(pf.get("maxPrice"), float("inf")),
            qty_step=_to_float(ls.get("stepSize"), 0.0),
            qty_min=_to_float(ls.get("minQty"), 0.0),
            qty_max=_to_float(ls.get("maxQty"), float("inf")),
            min_notional=_to_float(mn.get("minNotional"), 0.0),
            multiplier_up=_to_float(ppbs.get("multiplierUp"), None) if ppbs else None,
            multiplier_down=_to_float(ppbs.get("multiplierDown"), None) if ppbs else None,
        )


class Quantizer:
    """
    Единый квантайзер цен/количеств и проверок для Binance-символов.
    Используется и в симуляторе, и в live-адаптере.
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
            self._filters[sym] = SymbolFilters.from_exchange_filters(f or {})

    # ------------ Вспомогательные методы ------------
    @staticmethod
    def _snap(value: Number, step: Number) -> Number:
        if step <= 0:
            return float(value)
        # Binance требует округление вниз к ближайшему валидному шагу
        return math.floor(float(value) / step) * step

    # ------------ Публичные методы ------------
    def has_symbol(self, symbol: str) -> bool:
        return symbol in self._filters

    def quantize_price(self, symbol: str, price: Number) -> Number:
        f = self._filters.get(symbol)
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
        f = self._filters.get(symbol)
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
        f = self._filters.get(symbol)
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
        f = self._filters.get(symbol)
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

    # ------------ Фабрики загрузки ------------
    @classmethod
    def from_json_file(cls, path: str, strict: bool = True) -> "Quantizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, strict=strict)

    @staticmethod
    def load_filters(path: str) -> Dict[str, Dict[str, Any]]:
        """Загружает словарь фильтров из JSON. Возвращает {} если файл отсутствует."""
        if not path:
            return {}
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# для обратной совместимости
def load_filters(path: str) -> Dict[str, Dict[str, Any]]:
    return Quantizer.load_filters(path)
