# exchange/specs.py
from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Any, Tuple


@dataclass
class ExchangeRule:
    symbol: str
    tick_size: float
    step_size: float
    min_notional: float

    @classmethod
    def from_dict(cls, sym: str, d: dict) -> "ExchangeRule":
        return cls(
            symbol=sym.upper(),
            tick_size=float(d.get("tickSize", d.get("tick_size", 0.0)) or 0.0),
            step_size=float(d.get("stepSize", d.get("step_size", 0.0)) or 0.0),
            min_notional=float(d.get("minNotional", d.get("min_notional", 0.0)) or 0.0),
        )


class ExchangeSpecs:
    def __init__(self, rules: Dict[str, ExchangeRule]):
        self._rules = {k.upper(): v for k, v in rules.items()}

    def get(self, symbol: str) -> Optional[ExchangeRule]:
        return self._rules.get(str(symbol).upper())


def load_specs(path: str) -> Tuple[ExchangeSpecs, Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return ExchangeSpecs({}), {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f) or {}
    meta = raw.get("metadata", {})
    body = raw.get("specs", raw)
    rules: Dict[str, ExchangeRule] = {}
    for k, v in (body or {}).items():
        try:
            rules[str(k).upper()] = ExchangeRule.from_dict(str(k), v or {})
        except Exception:
            continue
    ga = meta.get("generated_at")
    if isinstance(ga, str):
        try:
            ts = datetime.fromisoformat(ga.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - ts).days
            if age_days > 30:
                warnings.warn(
                    f"{path} is {age_days} days old (>=30d); refresh via script_fetch_exchange_specs.py"
                )
        except Exception:
            pass
    return ExchangeSpecs(rules), meta


def _round_to_step(x: float, step: float, *, mode: str = "nearest") -> float:
    if step <= 0:
        return float(x)
    q = x / step
    if mode == "down":
        return math.floor(q) * step
    if mode == "up":
        return math.ceil(q) * step
    return round(q) * step


def round_price_to_tick(symbol: str, price: float, specs: ExchangeSpecs, *, side: Optional[str] = None) -> float:
    rule = specs.get(symbol)
    if rule is None or rule.tick_size <= 0:
        return float(price)
    # стандарт: bid округляем вниз, ask — вверх, иначе ближайший тик
    if side is not None:
        side_u = str(side).upper()
        if side_u == "BID":
            return _round_to_step(float(price), rule.tick_size, mode="down")
        if side_u == "ASK":
            return _round_to_step(float(price), rule.tick_size, mode="up")
    return _round_to_step(float(price), rule.tick_size, mode="nearest")


def round_qty_to_step(symbol: str, qty: float, specs: ExchangeSpecs, *, mode: str = "down") -> float:
    rule = specs.get(symbol)
    if rule is None or rule.step_size <= 0:
        return float(qty)
    # обычно объём округляют вниз к шагу
    return _round_to_step(float(qty), rule.step_size, mode=mode)


def notional_ok(symbol: str, price: float, qty: float, specs: ExchangeSpecs) -> bool:
    rule = specs.get(symbol)
    if rule is None:
        return True
    notional = abs(float(price) * float(qty))
    return notional + 1e-12 >= float(rule.min_notional or 0.0)
