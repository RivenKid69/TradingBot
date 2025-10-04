"""Utilities for converting Gym-style actions into :class:`ActionProto`.

The reinforcement-learning helpers in :mod:`trading_patchnew` expect to import
``domain.adapters`` which historically lived in an internal monorepo.  The
original module normalised the Gym ``Dict`` actions produced by wrappers and
translated them into :class:`~action_proto.ActionProto` objects.  During the
open-source migration that package was not brought over which resulted in
``ImportError: No module named 'domain'`` when vectorised environments spawned
new worker processes.

This file re-implements the tiny subset of helpers that we rely on:

``gym_to_action_v1``
    Accepts a Gym ``Dict`` action (or a close variant) and normalises data
    types, applying sensible defaults.  The function returns a lightweight
    :class:`ActionV1` dataclass so we can re-use the validation logic in a single
    place.

``action_v1_to_proto``
    Converts an :class:`ActionV1` (or compatible mapping) into an
    :class:`~action_proto.ActionProto` instance.

Both helpers are deliberately strict and raise descriptive ``TypeError``/
``ValueError`` exceptions when the payload cannot be interpreted.  This keeps
behaviour close to the original implementation and surfaces agent bugs early.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from action_proto import ActionProto, ActionType


def _numpy_scalar(value: Any) -> Any:
    """Extract a Python scalar from numpy arrays/scalars.

    Gymnasium commonly passes ``np.ndarray`` with shape ``(1,)`` for continuous
    boxes.  The helper mirrors the behaviour of the original adapters which
    silently squeezed those arrays.
    """

    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError("volume_frac array must contain at least one element")
        return value.reshape(-1)[0]
    return value


def _coerce_float(value: Any, *, field: str) -> float:
    try:
        return float(_numpy_scalar(value))
    except Exception as exc:  # pragma: no cover - defensive branch
        raise TypeError(f"{field} must be convertible to float, got {value!r}") from exc


def _coerce_int(value: Any, *, field: str) -> int:
    try:
        return int(_numpy_scalar(value))
    except Exception as exc:  # pragma: no cover - defensive branch
        raise TypeError(f"{field} must be convertible to int, got {value!r}") from exc


def _coerce_action_type(value: Any) -> ActionType:
    if isinstance(value, ActionType):
        return value
    if isinstance(value, str):
        key = value.upper()
        try:
            return ActionType[key]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown action type name: {value!r}") from exc
    try:
        return ActionType(int(value))
    except Exception as exc:
        raise ValueError(f"Unknown action type code: {value!r}") from exc


@dataclass(frozen=True)
class ActionV1:
    """Normalised representation of Gym Dict action payload."""

    action_type: ActionType
    volume_frac: float
    price_offset_ticks: int = 0
    ttl_steps: int = 0
    abs_price: float | None = None
    tif: str = "GTC"
    client_tag: str | None = None

    def to_proto(self) -> ActionProto:
        return ActionProto(
            action_type=self.action_type,
            volume_frac=self.volume_frac,
            price_offset_ticks=self.price_offset_ticks,
            ttl_steps=self.ttl_steps,
            abs_price=self.abs_price,
            tif=self.tif,
            client_tag=self.client_tag,
        )


def _normalise_mapping(action: Mapping[str, Any]) -> Mapping[str, Any]:
    if "type" not in action or "volume_frac" not in action:
        raise ValueError(
            "Dict action is missing required keys; expected at least 'type' and 'volume_frac'"
        )
    return action


def gym_to_action_v1(action: Any) -> ActionV1:
    """Convert an incoming Gym action into :class:`ActionV1`.

    Parameters
    ----------
    action:
        Expected to be a mapping produced by ``DictToMultiDiscreteActionWrapper``
        or a compatible payload.  ``ActionProto`` instances are passed through
        unchanged by converting them to :class:`ActionV1`.
    """

    if isinstance(action, ActionProto):
        return ActionV1(
            action_type=action.action_type,
            volume_frac=float(action.volume_frac),
            price_offset_ticks=int(action.price_offset_ticks),
            ttl_steps=int(action.ttl_steps),
            abs_price=float(action.abs_price) if action.abs_price is not None else None,
            tif=str(action.tif),
            client_tag=action.client_tag,
        )

    if isinstance(action, Mapping):
        payload = dict(_normalise_mapping(action))
    else:
        raise TypeError(
            f"Gym action must be a mapping or ActionProto; received {type(action)!r}"
        )

    action_type = _coerce_action_type(payload.get("type"))
    volume = _coerce_float(payload.get("volume_frac", 0.0), field="volume_frac")
    price_offset = _coerce_int(payload.get("price_offset_ticks", 0), field="price_offset_ticks")
    ttl_steps = _coerce_int(payload.get("ttl_steps", 0), field="ttl_steps")

    abs_price_val = payload.get("abs_price")
    abs_price: float | None
    if abs_price_val is None:
        abs_price = None
    else:
        abs_price = _coerce_float(abs_price_val, field="abs_price")

    tif = payload.get("tif", "GTC")
    if tif is not None:
        tif = str(tif)

    client_tag = payload.get("client_tag")
    if client_tag is not None:
        client_tag = str(client_tag)

    return ActionV1(
        action_type=action_type,
        volume_frac=volume,
        price_offset_ticks=price_offset,
        ttl_steps=ttl_steps,
        abs_price=abs_price,
        tif=tif or "GTC",
        client_tag=client_tag,
    )


def action_v1_to_proto(action: Any) -> ActionProto:
    """Convert an :class:`ActionV1` (or compatible object) into ``ActionProto``."""

    if isinstance(action, ActionProto):
        return action
    if isinstance(action, ActionV1):
        return action.to_proto()
    if isinstance(action, Mapping):
        return gym_to_action_v1(action).to_proto()
    raise TypeError(
        f"Action must be ActionV1, ActionProto or Mapping; received {type(action)!r}"
    )


__all__ = ["ActionV1", "gym_to_action_v1", "action_v1_to_proto"]
