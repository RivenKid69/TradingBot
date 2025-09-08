# strategies/momentum.py
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List

from core_strategy import Decision
from .base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Простейшая стратегия импульса по цене ref_price:
      - считаем среднее за lookback
      - если ref_price > avg + threshold → BUY с фиксированным qty
      - если ref_price < avg - threshold → SELL с фиксированным qty
    В проде ты заменишь логику на свою ML/сигналы — контракт тот же.
    """
    def __init__(self) -> None:
        super().__init__()
        self.lookback = 5
        self.threshold = 0.0
        self.order_qty = 0.001  # доля позиции
        self.price_offset_ticks = 0
        self.tif = "GTC"
        self.client_tag: str | None = None
        self._window: deque[float] = deque(maxlen=5)

    def setup(self, config: Dict[str, Any]) -> None:
        super().setup(config)
        self.lookback = int(config.get("lookback", self.lookback))
        self.threshold = float(config.get("threshold", self.threshold))
        self.order_qty = float(config.get("order_qty", self.order_qty))
        self.price_offset_ticks = int(
            config.get("price_offset_ticks", self.price_offset_ticks)
        )
        self.tif = str(config.get("tif", self.tif))
        self.client_tag = config.get("client_tag", self.client_tag)
        self._window = deque(maxlen=self.lookback)

    def on_features(self, row: Dict[str, Any]) -> None:
        super().on_features(row)
        # ожидаем, что ref_price присутствует в features (или придёт в ctx)
        price = row.get("ref_price")
        if price is not None:
            try:
                self._window.append(float(price))
            except Exception:
                pass

    def decide(self, ctx: Dict[str, Any]) -> List[Decision]:
        ref = ctx.get("ref_price")
        if ref is None or len(self._window) < self._window.maxlen:  # не торгуем, пока не набрали окно
            return []
        avg = sum(self._window) / float(len(self._window))
        out: List[Decision] = []
        if float(ref) > avg + self.threshold:
            out.append(
                Decision(
                    side="BUY",
                    volume_frac=self.order_qty,
                    price_offset_ticks=self.price_offset_ticks,
                    tif=self.tif,
                    client_tag=self.client_tag,
                )
            )
        elif float(ref) < avg - self.threshold:
            out.append(
                Decision(
                    side="SELL",
                    volume_frac=self.order_qty,
                    price_offset_ticks=self.price_offset_ticks,
                    tif=self.tif,
                    client_tag=self.client_tag,
                )
            )
        return out
