"""Adapters between Gym-style dict actions and :class:`ActionProto`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

import math
from numbers import Number

try:  # NumPy is optional in some minimal setups
    import numpy as _np
except Exception:  # pragma: no cover - fallback without NumPy
    _np = None  # type: ignore

from action_proto import ActionProto, ActionType

__all__ = ["gym_to_action_v1", "action_v1_to_proto", "normalize_volume"]


@dataclass(frozen=True)
class _ActionV1:
    """Minimal container mirroring the historical ActionV1 payload."""

    type: int
    price_offset_ticks: int
    volume_frac: float
    ttl_steps: int
    abs_price: float | None = None
    tif: str = "GTC"
    client_tag: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "price_offset_ticks": self.price_offset_ticks,
            "volume_frac": self.volume_frac,
            "ttl_steps": self.ttl_steps,
            "abs_price": self.abs_price,
            "tif": self.tif,
            "client_tag": self.client_tag,
        }


def _first_element(value: Any) -> Any:
    """Extract a scalar from containers like list/tuple/ndarray."""

    if isinstance(value, (list, tuple)):
        return value[0] if value else 0.0
    if _np is not None and isinstance(value, _np.ndarray):
        if value.size == 0:
            return 0.0
        return float(value.reshape(-1)[0])
    return value


def normalize_volume(value: Any, *, clip: bool = True) -> float:
    """Normalise ``volume_frac`` to a float; optionally clamp to [-1, 1]."""

    scalar = _first_element(value)
    if isinstance(scalar, Number):
        vol = float(scalar)
    else:
        try:
            vol = float(scalar)
        except Exception as exc:  # pragma: no cover - defensive path
            raise TypeError(f"volume_frac must be numeric, got {value!r}") from exc

    if clip:
        vol = max(-1.0, min(1.0, vol))
    if not math.isfinite(vol):
        raise ValueError("volume_frac must be finite")
    return vol


def _coerce_int(value: Any, *, name: str, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        ivalue = int(value)
    except Exception as exc:  # pragma: no cover - defensive path
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc

    if minimum is not None and ivalue < minimum:
        ivalue = minimum
    if maximum is not None and ivalue > maximum:
        ivalue = maximum
    return ivalue


def gym_to_action_v1(action: Mapping[str, Any] | MutableMapping[str, Any]) -> Mapping[str, Any]:
    """Convert a Gym Dict action into the legacy ``ActionV1`` mapping."""

    if not isinstance(action, Mapping):
        raise TypeError(f"Expected a mapping action, received {type(action)!r}")

    action_type = _coerce_int(action.get("type", 0), name="type", minimum=0, maximum=3)
    price_offset = _coerce_int(
        action.get("price_offset_ticks", 0),
        name="price_offset_ticks",
        minimum=-10_000,
        maximum=10_000,
    )
    ttl_steps = _coerce_int(action.get("ttl_steps", 0), name="ttl_steps", minimum=0)
    volume = normalize_volume(action.get("volume_frac", 0.0))

    abs_price = action.get("abs_price")
    if abs_price is not None:
        try:
            abs_price = float(abs_price)
        except Exception as exc:  # pragma: no cover - defensive path
            raise TypeError(f"abs_price must be numeric, got {abs_price!r}") from exc

    tif = str(action.get("tif", "GTC"))
    client_tag = action.get("client_tag")
    if client_tag is not None:
        client_tag = str(client_tag)

    payload = _ActionV1(
        type=action_type,
        price_offset_ticks=price_offset,
        volume_frac=volume,
        ttl_steps=ttl_steps,
        abs_price=abs_price,
        tif=tif,
        client_tag=client_tag,
    )
    return payload.as_dict()


def action_v1_to_proto(action: Mapping[str, Any] | _ActionV1 | ActionProto) -> ActionProto:
    """Convert an ``ActionV1`` payload (or :class:`ActionProto`) into ``ActionProto``."""

    if isinstance(action, ActionProto):
        return action

    if isinstance(action, _ActionV1):
        payload = action.as_dict()
    elif isinstance(action, Mapping):
        payload = dict(action)
    else:  # pragma: no cover - defensive path
        raise TypeError(f"Unsupported action payload: {type(action)!r}")

    try:
        action_type = ActionType(int(payload.get("type", payload.get("action_type", 0))))
    except ValueError as exc:
        raise ValueError(f"Unknown action type: {payload.get('type')}") from exc

    volume = normalize_volume(payload.get("volume_frac", 0.0), clip=False)
    price_offset = _coerce_int(payload.get("price_offset_ticks", 0), name="price_offset_ticks")
    ttl_steps = _coerce_int(payload.get("ttl_steps", 0), name="ttl_steps", minimum=0)

    abs_price = payload.get("abs_price")
    if abs_price is not None:
        try:
            abs_price = float(abs_price)
        except Exception as exc:  # pragma: no cover - defensive path
            raise TypeError(f"abs_price must be numeric, got {abs_price!r}") from exc

    tif = str(payload.get("tif", "GTC"))
    client_tag = payload.get("client_tag")
    if client_tag is not None:
        client_tag = str(client_tag)

    return ActionProto(
        action_type=action_type,
        volume_frac=volume,
        price_offset_ticks=price_offset,
        ttl_steps=ttl_steps,
        abs_price=abs_price,
        tif=tif,
        client_tag=client_tag,
    )

