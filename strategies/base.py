# strategies/base.py
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Mapping, Sequence

from core_contracts import PolicyCtx, SignalPolicy
from core_models import Order, OrderType, Side, TimeInForce, to_decimal
from core_strategy import Strategy


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


class BaseSignalPolicy(SignalPolicy):
    """Convenience base class for :class:`SignalPolicy` implementations."""

    required_features: Sequence[str] = ()

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = dict(params)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _validate_inputs(self, features: Mapping[str, Any], ctx: PolicyCtx) -> None:
        if not isinstance(features, Mapping):
            raise TypeError("features must be a mapping")
        if not isinstance(ctx, PolicyCtx):
            raise TypeError("ctx must be PolicyCtx")
        if ctx.ts is None or ctx.symbol is None:
            raise ValueError("ctx.ts and ctx.symbol must be provided")
        for name in self.required_features:
            if name not in features:
                raise ValueError(f"missing feature '{name}'")

    def decide(self, features: Mapping[str, Any], ctx: PolicyCtx) -> List[Order]:
        self._validate_inputs(features, ctx)
        return []

    # Helper methods to construct orders
    def market_order(
        self,
        *,
        side: Side,
        qty: Decimal | float | int,
        ctx: PolicyCtx,
        tif: TimeInForce = TimeInForce.GTC,
        client_tag: str | None = None,
    ) -> Order:
        quantity = to_decimal(qty)
        return Order(
            ts=ctx.ts,
            symbol=ctx.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=None,
            time_in_force=tif,
            client_order_id=client_tag or "",
        )

    def limit_order(
        self,
        *,
        side: Side,
        qty: Decimal | float | int,
        price: Decimal | float | int,
        ctx: PolicyCtx,
        tif: TimeInForce = TimeInForce.GTC,
        client_tag: str | None = None,
    ) -> Order:
        quantity = to_decimal(qty)
        price_dec = to_decimal(price)
        return Order(
            ts=ctx.ts,
            symbol=ctx.symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price_dec,
            time_in_force=tif,
            client_order_id=client_tag or "",
        )


__all__ = ["BaseStrategy", "BaseSignalPolicy"]
