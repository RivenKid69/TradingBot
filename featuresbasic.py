from __future__ import annotations
from typing import Any
from .featuresregistry import register

@register("units")
def feat_units(state: Any) -> float:
    return float(getattr(state, "units", 0.0))

@register("cash")
def feat_cash(state: Any) -> float:
    return float(getattr(state, "cash", 0.0))

@register("net_worth")
def feat_nw(state: Any) -> float:
    return float(getattr(state, "net_worth", 0.0))
