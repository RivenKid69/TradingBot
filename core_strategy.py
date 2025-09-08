"""Core strategy contract and protocol."""
from __future__ import annotations

from typing import Protocol, Dict, Any, Sequence, runtime_checkable


@runtime_checkable
class Strategy(Protocol):
    """Trading strategy interface."""

    def setup(self, config: Dict[str, Any]) -> None:
        """Initialize strategy with configuration."""
        ...

    def on_features(self, row: Dict[str, Any]) -> None:
        """Receive new feature row from pipeline."""
        ...

    def decide(self, ctx: Dict[str, Any]) -> Sequence[Any]:
        """Make trading decision given context."""
        ...


__all__ = ["Strategy"]
