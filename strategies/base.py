# strategies/base.py
from __future__ import annotations

from typing import Any, Dict, List

from core_strategy import Decision, Strategy


class BaseStrategy(Strategy):
    """
    Базовый класс стратегии.

    Метод decide получает контекст и отдаёт список Decision.
    Контекст:
      {
        "ts_ms": int,
        "symbol": str,
        "ref_price": float | None,
        "bid": float | None,
        "ask": float | None,
        "features": Dict[str, Any]
      }
    """

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = dict(params or {})

    # --- Strategy interface -------------------------------------------------

    def setup(self, config: Dict[str, Any]) -> None:  # pragma: no cover - trivial
        """Configure strategy with ``config`` parameters."""
        self.params.update(dict(config or {}))

    def on_features(self, row: Dict[str, Any]) -> None:  # pragma: no cover - trivial
        """Receive feature row from pipeline. Base implementation does nothing."""
        return None

    def decide(self, ctx: Dict[str, Any]) -> List[Decision]:
        """
        По умолчанию — нет торгового действия.
        Реализации должны вернуть список Decision.
        """
        return []


__all__ = ["BaseStrategy", "Decision"]
