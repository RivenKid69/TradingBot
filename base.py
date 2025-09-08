# strategies/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core_strategy import Strategy

from action_proto import ActionProto, ActionType


@dataclass(frozen=True)
class Decision:
    """
    Результат решения стратегии на шаге.

    Поля:
      - side: "BUY" | "SELL"
      - volume_frac: целевая величина заявки в долях позиции ([-1.0; 1.0]).
        Для MARKET это величина, знак задаёт сторону.
      - price_offset_ticks: сдвиг цены в тиках для LIMIT (по умолчанию 0; для MARKET игнорируется)
      - tif: GTC|IOC|FOK (строка, совместимо с ActionProto.tif)
      - client_tag: произвольная метка стратегии
    """
    side: str
    volume_frac: float
    price_offset_ticks: int = 0
    tif: str = "GTC"
    client_tag: Optional[str] = None

    def to_action_proto(self) -> ActionProto:
        """
        Преобразование решения в ActionProto без потери информации.
        Для side="BUY" volume_frac >= 0; для "SELL" — volume_frac <= 0.
        """
        v = float(self.volume_frac)
        if str(self.side).upper() == "SELL":
            v = -abs(v)
        else:
            v = abs(v)
        return ActionProto(
            action_type=(ActionType.MARKET if self.price_offset_ticks == 0 else ActionType.LIMIT),
            volume_frac=v,
            price_offset_ticks=int(self.price_offset_ticks),
            tif=str(self.tif),
            client_tag=self.client_tag,
        )


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
