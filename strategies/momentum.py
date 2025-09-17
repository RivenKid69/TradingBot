# strategies/momentum.py
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Mapping

from core_contracts import PolicyCtx
from core_models import Order, Side, TimeInForce
from .base import BaseSignalPolicy, SignalPosition


class MomentumStrategy(BaseSignalPolicy):
    """
    Простейшая стратегия импульса по цене ``ref_price`` с гистерезисом.

    - считаем среднее за ``lookback`` и сигнал ``ref - avg``;
    - держим состояние по каждому символу (:class:`SignalPosition`) в
      :class:`BaseSignalPolicy` и обновляем его только при переходах;
    - если сигнал ≥ ``enter_threshold`` → входим в LONG (покупка);
    - если сигнал ≤ ``-enter_threshold`` → входим в SHORT (продажа);
    - из LONG выходим при сигнале ≤ ``exit_threshold``, из SHORT — при
      сигнале ≥ ``-exit_threshold``;
    - при развороте LONG↔SHORT отправляем *два* рыночных ордера фиксированного
      размера: первый закрывает предыдущую сторону, второй открывает новую.

    По умолчанию ``enter_threshold`` и ``exit_threshold`` наследуют старый
    ``threshold`` для обратной совместимости.
    """
    required_features = ("ref_price",)

    def __init__(self) -> None:
        super().__init__()
        self.lookback = 5
        self.threshold = 0.0
        self.enter_threshold = self.threshold
        self.exit_threshold = self.threshold
        self.order_qty = 0.001  # абсолютное количество
        self.tif = TimeInForce.GTC
        self.client_tag: str | None = None
        self._window: deque[float] = deque(maxlen=5)

    def setup(self, config: Dict[str, Any]) -> None:
        self.lookback = int(config.get("lookback", self.lookback))
        fallback_threshold = float(config.get("threshold", self.threshold))
        enter_threshold = float(config.get("enter_threshold", fallback_threshold))
        exit_threshold = float(config.get("exit_threshold", fallback_threshold))
        if enter_threshold < exit_threshold:
            raise ValueError(
                "enter_threshold must be greater than or equal to exit_threshold"
            )
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        # ``threshold`` сохраняем для обратной совместимости со старыми конфигами
        self.threshold = exit_threshold
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
        signal = ref - avg
        symbol = ctx.symbol
        state = self.get_signal_state(symbol)
        new_state = state

        if state is SignalPosition.FLAT:
            if signal >= self.enter_threshold:
                new_state = SignalPosition.LONG
            elif signal <= -self.enter_threshold:
                new_state = SignalPosition.SHORT
        elif state is SignalPosition.LONG:
            if signal <= -self.enter_threshold:
                new_state = SignalPosition.SHORT
            elif signal <= self.exit_threshold:
                new_state = SignalPosition.FLAT
        else:  # SHORT
            if signal >= self.enter_threshold:
                new_state = SignalPosition.LONG
            elif signal >= -self.exit_threshold:
                new_state = SignalPosition.FLAT

        if new_state is state:
            return []

        self.update_signal_state(symbol, new_state)

        orders: List[Order] = []
        if state is SignalPosition.FLAT and new_state is SignalPosition.LONG:
            orders.append(
                self.market_order(
                    side=Side.BUY,
                    qty=self.order_qty,
                    ctx=ctx,
                    tif=self.tif,
                    client_tag=self.client_tag,
                )
            )
        elif state is SignalPosition.FLAT and new_state is SignalPosition.SHORT:
            orders.append(
                self.market_order(
                    side=Side.SELL,
                    qty=self.order_qty,
                    ctx=ctx,
                    tif=self.tif,
                    client_tag=self.client_tag,
                )
            )
        elif state is SignalPosition.LONG and new_state is SignalPosition.FLAT:
            orders.append(
                self.market_order(
                    side=Side.SELL,
                    qty=self.order_qty,
                    ctx=ctx,
                    tif=self.tif,
                    client_tag=self.client_tag,
                )
            )
        elif state is SignalPosition.SHORT and new_state is SignalPosition.FLAT:
            orders.append(
                self.market_order(
                    side=Side.BUY,
                    qty=self.order_qty,
                    ctx=ctx,
                    tif=self.tif,
                    client_tag=self.client_tag,
                )
            )
        elif state is SignalPosition.LONG and new_state is SignalPosition.SHORT:
            orders.extend(
                [
                    self.market_order(
                        side=Side.SELL,
                        qty=self.order_qty,
                        ctx=ctx,
                        tif=self.tif,
                        client_tag=self.client_tag,
                    ),
                    self.market_order(
                        side=Side.SELL,
                        qty=self.order_qty,
                        ctx=ctx,
                        tif=self.tif,
                        client_tag=self.client_tag,
                    ),
                ]
            )
        elif state is SignalPosition.SHORT and new_state is SignalPosition.LONG:
            orders.extend(
                [
                    self.market_order(
                        side=Side.BUY,
                        qty=self.order_qty,
                        ctx=ctx,
                        tif=self.tif,
                        client_tag=self.client_tag,
                    ),
                    self.market_order(
                        side=Side.BUY,
                        qty=self.order_qty,
                        ctx=ctx,
                        tif=self.tif,
                        client_tag=self.client_tag,
                    ),
                ]
            )

        return orders
