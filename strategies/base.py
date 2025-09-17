# strategies/base.py
from __future__ import annotations

from collections import deque
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Mapping, Sequence

from core_contracts import PolicyCtx, SignalPolicy
from core_models import Order, OrderType, Side, TimeInForce, to_decimal
from core_strategy import Strategy


class SignalPosition(str, Enum):
    """Enumerates the direction of a trading signal."""

    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


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
        self._signal_state: dict[str, SignalPosition] = {}
        self._dirty_signal_state: set[str] = set()
        self._pending_transitions: dict[str, Dict[str, Any]] = {}

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

    # ------------------------------------------------------------------
    # Signal state management
    # ------------------------------------------------------------------
    def get_signal_state(self, symbol: str) -> SignalPosition:
        return self._signal_state.get(symbol, SignalPosition.FLAT)

    def update_signal_state(
        self, symbol: str, state: SignalPosition | str | int
    ) -> SignalPosition:
        normalized = self._normalize_signal_state(state)
        previous = self.get_signal_state(symbol)
        if previous == normalized:
            return normalized

        if normalized is SignalPosition.FLAT:
            if symbol in self._signal_state:
                del self._signal_state[symbol]
        else:
            self._signal_state[symbol] = normalized
        self._dirty_signal_state.add(symbol)

        steps = self._build_transition_steps(previous, normalized)
        entry_steps = steps.count("entry")
        self._pending_transitions[symbol] = {
            "prev": previous,
            "new": normalized,
            "steps": deque(steps),
            "entry_steps": entry_steps,
            "complete": not steps,
        }
        return normalized

    def load_signal_state(self, states: Mapping[str, Any]) -> None:
        self._signal_state.clear()
        self._dirty_signal_state.clear()
        self._pending_transitions.clear()
        for symbol, state in states.items():
            normalized = self._normalize_signal_state(state)
            if normalized is SignalPosition.FLAT:
                continue
            self._signal_state[str(symbol)] = normalized

    def export_signal_state(self) -> dict[str, str]:
        return {symbol: state.value for symbol, state in self._signal_state.items()}

    def consume_dirty_signal_state(self) -> dict[str, str]:
        dirty_symbols = self._dirty_signal_state.copy()
        self._dirty_signal_state.clear()
        return {
            symbol: self.get_signal_state(symbol).value
            for symbol in dirty_symbols
        }

    def consume_signal_transitions(self) -> List[Dict[str, Any]]:
        ready: List[Dict[str, Any]] = []
        consumed: List[str] = []
        for symbol, data in self._pending_transitions.items():
            if not data.get("complete"):
                continue
            ready.append(
                {
                    "symbol": symbol,
                    "prev": data["prev"],
                    "new": data["new"],
                    "entry_steps": data["entry_steps"],
                }
            )
            consumed.append(symbol)
        for symbol in consumed:
            self._pending_transitions.pop(symbol, None)
        return ready

    def _normalize_signal_state(self, state: SignalPosition | str | int) -> SignalPosition:
        if isinstance(state, SignalPosition):
            return state
        if isinstance(state, str):
            token = state.strip()
            if not token:
                raise ValueError("signal state string cannot be empty")
            upper_token = token.upper()
            try:
                return SignalPosition[upper_token]
            except KeyError:
                lowered = token.lower()
                for member in SignalPosition:
                    if member.value == lowered:
                        return member
            raise ValueError(f"unknown signal state string: {state!r}")
        if isinstance(state, int):
            if state == 0:
                return SignalPosition.FLAT
            if state > 0:
                return SignalPosition.LONG
            return SignalPosition.SHORT
        raise TypeError(f"unsupported signal state type: {type(state)!r}")

    def revert_signal_state(
        self, symbol: str, previous: SignalPosition | str | int
    ) -> SignalPosition:
        normalized = self._normalize_signal_state(previous)
        current = self.get_signal_state(symbol)
        if normalized is SignalPosition.FLAT:
            self._signal_state.pop(symbol, None)
        else:
            self._signal_state[symbol] = normalized
        if current != normalized:
            self._dirty_signal_state.add(symbol)
        self._pending_transitions.pop(symbol, None)
        return normalized

    def _build_transition_steps(
        self, previous: SignalPosition, new: SignalPosition
    ) -> List[str]:
        steps: List[str] = []
        if previous is not SignalPosition.FLAT:
            steps.append("exit")
        if new is not SignalPosition.FLAT and new is not previous:
            steps.append("entry")
        return steps

    def _pull_signal_leg(self, symbol: str | None) -> str:
        if not symbol:
            return "unknown"
        data = self._pending_transitions.get(symbol)
        if not data:
            return "unknown"
        steps = data.get("steps")
        if not steps:
            data["complete"] = True
            return "unknown"
        leg = steps.popleft()
        if not steps:
            data["complete"] = True
        return leg

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
        order = Order(
            ts=ctx.ts,
            symbol=ctx.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=None,
            time_in_force=tif,
            client_order_id=client_tag or "",
        )
        order.meta["signal_leg"] = self._pull_signal_leg(ctx.symbol)
        return order

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
        order = Order(
            ts=ctx.ts,
            symbol=ctx.symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price_dec,
            time_in_force=tif,
            client_order_id=client_tag or "",
        )
        order.meta["signal_leg"] = self._pull_signal_leg(ctx.symbol)
        return order


__all__ = ["SignalPosition", "BaseStrategy", "BaseSignalPolicy"]
