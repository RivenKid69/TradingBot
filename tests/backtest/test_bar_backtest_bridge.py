"""Unit tests for :class:`service_backtest.BarBacktestSimBridge`."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import importlib
import math
import sys
from types import ModuleType
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import pytest

from core_models import Order, OrderType, Side


@dataclass
class _StubOrder:
    desired_weight: float
    meta: MutableMapping[str, Any] | None = None


@dataclass
class _StubReport:
    meta: Mapping[str, Any]


@dataclass
class _StubPosition:
    meta: Mapping[str, Any]
    qty: float = 0.0


class _StubBarExecutor:
    """Deterministic :class:`BarExecutor` stub for bridge unit tests."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.run_id = "stub"
        self.seen_orders: List[_StubOrder] = []
        self._execute_meta_queue: List[Mapping[str, Any]] = []
        self._position_queue: List[_StubPosition] = []

    def queue_execute_meta(self, meta: Mapping[str, Any]) -> None:
        self._execute_meta_queue.append(meta)

    def queue_position(self, position: _StubPosition) -> None:
        self._position_queue.append(position)

    def execute(self, order: _StubOrder) -> _StubReport:
        self.seen_orders.append(order)
        meta = self._execute_meta_queue.pop(0) if self._execute_meta_queue else {}
        return _StubReport(meta=meta)

    def get_open_positions(self, symbols: Sequence[str]) -> Dict[str, _StubPosition]:
        _ = symbols  # bridge always queries a single symbol, ignore exact list
        if self._position_queue:
            position = self._position_queue.pop(0)
        else:
            position = _StubPosition(meta={}, qty=0.0)
        return {self.symbol: position}


@pytest.fixture(name="bar_bridge_cls")
def _bar_bridge_cls(monkeypatch: pytest.MonkeyPatch) -> type[Any]:
    """Import ``service_backtest`` with exchange dependencies patched out."""

    exchange_mod = ModuleType("exchange")
    specs_mod = ModuleType("exchange.specs")

    def _load_specs(*_args: Any, **_kwargs: Any) -> tuple[dict, dict]:  # pragma: no cover - stub
        return {}, {}

    def _round_price_to_tick(price: float, _tick: float, *_args: Any, **_kwargs: Any) -> float:  # pragma: no cover - stub
        return float(price)

    specs_mod.load_specs = _load_specs  # type: ignore[attr-defined]
    specs_mod.round_price_to_tick = _round_price_to_tick  # type: ignore[attr-defined]
    exchange_mod.specs = specs_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "exchange", exchange_mod)
    monkeypatch.setitem(sys.modules, "exchange.specs", specs_mod)
    for name in ("service_backtest", "sandbox.backtest_adapter"):
        sys.modules.pop(name, None)

    module = importlib.import_module("service_backtest")
    return module.BarBacktestSimBridge


def _run_step(
    bridge: Any,
    *,
    ts_ms: int,
    price: float | None,
    orders: Iterable[_StubOrder],
    timeframe_ms: int = 60_000,
) -> Mapping[str, Any]:
    """Helper to execute a bridge step with symmetric OHLC prices."""

    return bridge.step(
        ts_ms=ts_ms,
        ref_price=price,
        bid=price,
        ask=price,
        vol_factor=1.0,
        liquidity=None,
        orders=list(orders),
        bar_open=price,
        bar_high=price,
        bar_low=price,
        bar_close=price,
        bar_timeframe_ms=timeframe_ms,
    )


def test_orders_attach_bar_payload_and_track_costs(bar_bridge_cls: type[Any]) -> None:
    symbol = "ETHUSDT"
    executor = _StubBarExecutor(symbol)
    bridge = bar_bridge_cls(
        executor,
        symbol=symbol,
        timeframe_ms=60_000,
        initial_equity=500.0,
    )

    executor.queue_execute_meta(
        {
            "decision": {"turnover_usd": 200.0, "cost_bps": 12.5},
            "instructions": [{"kind": "rebalance", "slices_total": 1}],
        }
    )
    executor.queue_position(_StubPosition(meta={"weight": 0.4}, qty=2.0))
    order_one = _StubOrder(desired_weight=0.4, meta={"payload": {"target_weight": 0.4}})

    first_report = _run_step(
        bridge,
        ts_ms=1,
        price=100.0,
        orders=[order_one],
    )

    assert executor.seen_orders == [order_one]
    assert order_one.meta is not None
    assert "bar" in order_one.meta
    bar_payload = order_one.meta["bar"]
    assert getattr(bar_payload, "ts", None) == 1
    assert float(getattr(bar_payload, "close")) == pytest.approx(100.0)
    assert order_one.meta["equity_usd"] == pytest.approx(500.0)

    cost_usd = 200.0 * 12.5 / 10_000.0
    assert first_report["bar_return"] == pytest.approx(0.0)
    assert first_report["bar_pnl"] == pytest.approx(-cost_usd)
    assert first_report["equity_before_costs"] == pytest.approx(500.0)
    assert first_report["equity"] == pytest.approx(500.0 - cost_usd)
    assert first_report["equity_after_costs"] == pytest.approx(500.0 - cost_usd)
    assert first_report["turnover_usd"] == pytest.approx(200.0)
    assert first_report["bar_cost_usd"] == pytest.approx(cost_usd)
    assert first_report["fee_total"] == pytest.approx(cost_usd)
    assert bridge._cum_cost_usd == pytest.approx(cost_usd)  # type: ignore[attr-defined]
    assert first_report["instructions"] == [
        {"kind": "rebalance", "slices_total": 1}
    ]

    executor.queue_execute_meta(
        {
            "decision": {"turnover_usd": 150.0, "cost_bps": 10.0},
            "instructions": [{"kind": "twap", "slices_total": 2}],
        }
    )
    executor.queue_position(_StubPosition(meta={"weight": 0.6}, qty=3.0))
    order_two = _StubOrder(desired_weight=0.6, meta={"payload": {"target_weight": 0.6}})

    second_report = _run_step(
        bridge,
        ts_ms=2,
        price=100.0,
        orders=[order_two],
    )

    assert executor.seen_orders == [order_one, order_two]
    second_cost = 150.0 * 10.0 / 10_000.0
    assert second_report["bar_pnl"] == pytest.approx(-second_cost)
    assert second_report["equity_before_costs"] == pytest.approx(
        first_report["equity"]
    )
    assert second_report["equity"] == pytest.approx(500.0 - cost_usd - second_cost)
    assert bridge._cum_cost_usd == pytest.approx(cost_usd + second_cost)  # type: ignore[attr-defined]
    assert second_report["cumulative_pnl"] == pytest.approx(
        -cost_usd - second_cost
    )
    assert second_report["instructions"] == [
        {"kind": "twap", "slices_total": 2}
    ]


def test_apply_exchange_rules_preserves_side_enum(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("exchange", "exchange.specs", "sandbox.backtest_adapter"):
        sys.modules.pop(name, None)

    exchange_mod = ModuleType("exchange")
    specs_mod = ModuleType("exchange.specs")

    def _load_specs(*_args: Any, **_kwargs: Any) -> tuple[dict, dict]:
        return {}, {}

    def _round_price_to_tick(price: float, *_args: Any, **_kwargs: Any) -> float:
        return float(price)

    specs_mod.load_specs = _load_specs  # type: ignore[attr-defined]
    specs_mod.round_price_to_tick = _round_price_to_tick  # type: ignore[attr-defined]
    exchange_mod.specs = specs_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "exchange", exchange_mod)
    monkeypatch.setitem(sys.modules, "exchange.specs", specs_mod)

    from sandbox.backtest_adapter import BacktestAdapter

    class _StubPolicy:
        def decide(self, *_args: Any, **_kwargs: Any) -> List[Order]:
            return []

    class _StubSim:
        symbol = "BTCUSDT"
        interval_ms = 60_000
        sim = None

    adapter = BacktestAdapter(policy=_StubPolicy(), sim_bridge=_StubSim())

    order = Order(
        ts=0,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1"),
        price=None,
        meta={"price_offset_ticks": 5},
    )

    normalised = adapter._apply_exchange_rules_to_orders("BTCUSDT", 100.0, [order])

    assert len(normalised) == 1
    assert normalised[0].side == Side.BUY


def test_bar_mode_keeps_zero_quantity_order(bar_bridge_cls: type[Any]) -> None:
    from sandbox.backtest_adapter import BacktestAdapter

    symbol = "BTCUSDT"
    executor = _StubBarExecutor(symbol)
    bridge = bar_bridge_cls(
        executor,
        symbol=symbol,
        timeframe_ms=60_000,
        initial_equity=1_000.0,
    )

    class _StubPolicy:
        def decide(self, *_args: Any, **_kwargs: Any) -> List[Order]:
            return []

    adapter = BacktestAdapter(policy=_StubPolicy(), sim_bridge=bridge)

    order = Order(
        ts=0,
        symbol=symbol,
        side=Side.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={},
    )

    normalised = adapter._apply_exchange_rules_to_orders(symbol, 100.0, [order])

    assert len(normalised) == 1
    assert normalised[0].quantity == Decimal("0")
    assert normalised[0].side == Side.SELL


def test_missing_price_does_not_move_equity(bar_bridge_cls: type[Any]) -> None:
    symbol = "BTCUSDT"
    executor = _StubBarExecutor(symbol)
    bridge = bar_bridge_cls(
        executor,
        symbol=symbol,
        timeframe_ms=60_000,
        initial_equity=1_000.0,
    )

    executor.queue_position(_StubPosition(meta={"weight": 0.5}, qty=1.0))
    _run_step(
        bridge,
        ts_ms=1,
        price=100.0,
        orders=[],
    )

    executor.queue_position(_StubPosition(meta={"weight": 0.5}, qty=1.0))
    missing_report = bridge.step(
        ts_ms=2,
        ref_price=None,
        bid=None,
        ask=None,
        vol_factor=None,
        liquidity=None,
        orders=[],
        bar_open=None,
        bar_high=None,
        bar_low=None,
        bar_close=None,
        bar_timeframe_ms=60_000,
    )

    assert missing_report["bar_return"] == pytest.approx(0.0)
    assert missing_report["bar_pnl"] == pytest.approx(0.0)
    assert missing_report["equity"] == pytest.approx(1_000.0)
    assert missing_report["instructions"] == []
    assert missing_report["ref_price"] == pytest.approx(100.0)


def test_missing_price_skips_execution(bar_bridge_cls: type[Any]) -> None:
    symbol = "SOLUSDT"
    executor = _StubBarExecutor(symbol)
    bridge = bar_bridge_cls(
        executor,
        symbol=symbol,
        timeframe_ms=60_000,
        initial_equity=2_500.0,
    )

    # Seed the previous price so the bridge has a reference value.
    executor.queue_position(_StubPosition(meta={"weight": 0.0}, qty=0.0))
    _run_step(
        bridge,
        ts_ms=1,
        price=50.0,
        orders=[],
    )

    order = _StubOrder(desired_weight=0.2, meta={"payload": {"target_weight": 0.2}})

    executor.queue_position(_StubPosition(meta={"weight": 0.0}, qty=0.0))
    report = bridge.step(
        ts_ms=2,
        ref_price=None,
        bid=None,
        ask=None,
        vol_factor=None,
        liquidity=None,
        orders=[order],
        bar_open=None,
        bar_high=None,
        bar_low=None,
        bar_close=None,
        bar_timeframe_ms=60_000,
    )

    assert executor.seen_orders == []
    assert order.meta == {"payload": {"target_weight": 0.2}}
    assert report["instructions"] == []
    assert report["turnover_usd"] == pytest.approx(0.0)
    assert report["bar_cost_usd"] == pytest.approx(0.0)
    assert report["equity"] == pytest.approx(2_500.0)
    assert report["equity_before_costs"] == pytest.approx(2_500.0)
    assert report.get("bar_skipped") is True
    assert report.get("skip_reason") == "missing_bar_price"


@pytest.mark.parametrize(
    "qty_sequence",
    [
        [0.0, 3.0, 0.0],
    ],
)
def test_position_quantity_fallback_updates(bar_bridge_cls: type[Any], qty_sequence: Sequence[float]) -> None:
    symbol = "BNBUSDT"
    executor = _StubBarExecutor(symbol)
    bridge = bar_bridge_cls(
        executor,
        symbol=symbol,
        timeframe_ms=60_000,
        initial_equity=1_000.0,
    )

    first_qty, second_qty, third_qty = qty_sequence

    executor.queue_position(_StubPosition(meta={"weight": 0.5}, qty=first_qty))
    first_report = _run_step(
        bridge,
        ts_ms=1,
        price=100.0,
        orders=[],
    )

    expected_first_qty = (
        0.5 * first_report["equity"] / 100.0 if math.isclose(first_qty, 0.0) else first_qty
    )
    assert bridge._position_qtys[symbol] == pytest.approx(expected_first_qty)  # type: ignore[attr-defined]
    assert first_report["bar_pnl"] == pytest.approx(0.0)

    executor.queue_position(_StubPosition(meta={"weight": 0.3}, qty=second_qty))
    second_report = _run_step(
        bridge,
        ts_ms=2,
        price=110.0,
        orders=[],
    )

    assert second_report["bar_return"] == pytest.approx(0.1)
    assert bridge._position_qtys[symbol] == pytest.approx(second_qty)  # type: ignore[attr-defined]

    executor.queue_position(_StubPosition(meta={"weight": 0.0}, qty=third_qty))
    third_report = _run_step(
        bridge,
        ts_ms=3,
        price=120.0,
        orders=[],
    )

    assert third_report["bar_return"] == pytest.approx(120.0 / 110.0 - 1.0)
    assert symbol not in bridge._position_qtys  # type: ignore[attr-defined]
    assert bridge._weights[symbol] == pytest.approx(0.0)  # type: ignore[attr-defined]
