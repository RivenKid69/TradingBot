"""Core strategy contract and protocol."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence, runtime_checkable

from action_proto import ActionProto, ActionType


@runtime_checkable
class Strategy(Protocol):
    """Trading strategy interface.

    Deprecated:
        Use :class:`SignalPolicy` instead.
    """

    def setup(self, config: Dict[str, Any]) -> None:
        """Initialize strategy with configuration."""
        ...

    def on_features(self, row: Dict[str, Any]) -> None:
        """Receive new feature row from pipeline."""
        ...

    def decide(self, ctx: Dict[str, Any]) -> Sequence[Any]:
        """Make trading decision given context."""
        ...


@dataclass(frozen=True)
class Decision:
    """High level strategy decision.

    Deprecated:
        Use :class:`OrderIntent` instead.

    Parameters
    ----------
    side:
        "BUY" or "SELL" direction of the order.
    volume_frac:
        Target order size as a fraction of the allowed position in
        range ``[-1.0; 1.0]``.  Sign defines the side for market orders.
    price_offset_ticks:
        Price offset in ticks for limit orders.  Ignored for market
        orders.
    tif:
        Time in force of the order: ``GTC``, ``IOC`` or ``FOK``.
    client_tag:
        Optional custom tag attached to resulting orders.
    """

    side: str
    volume_frac: float
    price_offset_ticks: int = 0
    tif: str = "GTC"
    client_tag: Optional[str] = None

    def to_action_proto(self) -> ActionProto:
        """Convert decision to :class:`ActionProto` without information loss."""
        v = float(self.volume_frac)
        if str(self.side).upper() == "SELL":
            v = -abs(v)
        else:
            v = abs(v)
        return ActionProto(
            action_type=(
                ActionType.MARKET if self.price_offset_ticks == 0 else ActionType.LIMIT
            ),
            volume_frac=v,
            price_offset_ticks=int(self.price_offset_ticks),
            tif=str(self.tif),
            client_tag=self.client_tag,
        )

__all__ = ["Strategy", "Decision"]

warnings.warn(
    "core_strategy.Strategy and .Decision are deprecated; use SignalPolicy and OrderIntent instead",
    DeprecationWarning,
    stacklevel=2,
)
