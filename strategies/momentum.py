# strategies/momentum.py
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Mapping

from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from .base import BaseSignalPolicy


class MomentumStrategy(BaseSignalPolicy):
    """
    Простейшая стратегия импульса по цене ref_price:
      - считаем среднее за lookback
      - если ref_price > avg + threshold → BUY с фиксированным qty
      - если ref_price < avg - threshold → SELL с фиксированным qty
    В проде ты заменишь логику на свою ML/сигналы — контракт тот же.
    """
    required_features = ("ref_price",)

    def __init__(self) -> None:
        super().__init__()
        self.lookback = 5
        self.threshold = 0.0
        self.order_qty = 0.001  # абсолютное количество
        self.tif = TimeInForce.GTC
        self.client_tag: str | None = None
        self._window: deque[float] = deque(maxlen=5)

    def setup(self, config: Dict[str, Any]) -> None:
        self.lookback = int(config.get("lookback", self.lookback))
        self.threshold = float(config.get("threshold", self.threshold))
        self.order_qty = float(config.get("order_qty", self.order_qty))
        self.tif = TimeInForce(str(config.get("tif", self.tif.value)))
        self.client_tag = config.get("client_tag", self.client_tag)
        self._window = deque(maxlen=self.lookback)

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        ref = float(features["ref_price"])
        self._window.append(ref)
        maxlen = self._window.maxlen or 0
        if len(self._window) < maxlen:
            return []
        avg = sum(self._window) / float(len(self._window))
        if ref > avg + self.threshold:
            return [
                self.market_order(
                    side=Side.BUY,
                    qty=self.order_qty,
                    ctx=ctx,
                    tif=self.tif,
                    client_tag=self.client_tag,
                )
            ]
        if ref < avg - self.threshold:
            return [
                self.market_order(
                    side=Side.SELL,
                    qty=self.order_qty,
                    ctx=ctx,
                    tif=self.tif,
                    client_tag=self.client_tag,
                )
            ]
        return []
