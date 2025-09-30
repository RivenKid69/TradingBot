"""Timing semantics for :class:`service_backtest.BarBacktestSimBridge`."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from types import ModuleType
from typing import Any, Dict, Iterable, Mapping, Sequence

import pytest


@dataclass
class _StubOrder:
    desired_weight: float
    meta: Mapping[str, Any] | None = None


@dataclass
class _StubReport:
    meta: Mapping[str, Any]


@dataclass
class _StubPosition:
    meta: Mapping[str, Any]
    qty: float = 0.0


class _StubBarExecutor:
    """Minimal executor that records weights and trades instantly."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.run_id = "stub"
        self._weight = 0.0
        self._qty = 0.0
        self.executed_orders: list[_StubOrder] = []

    def execute(self, order: _StubOrder) -> _StubReport:
        desired = getattr(order, "desired_weight", self._weight)
        prev_weight = self._weight
        self._weight = float(desired)
        self.executed_orders.append(order)
        equity_usd = float(getattr(order, "meta", {}).get("equity_usd", 0.0))
        turnover_usd = abs(self._weight - prev_weight) * equity_usd
        decision = {"turnover_usd": turnover_usd, "cost_bps": 0.0}
        bar = getattr(order, "meta", {}).get("bar")
        close_price = None
        if bar is not None:
            close_raw = getattr(bar, "close", None)
            try:
                close_price = float(close_raw)
            except (TypeError, ValueError):
                close_price = None
        if close_price and close_price > 0.0:
            self._qty = self._weight * equity_usd / close_price
        return _StubReport(meta={"decision": decision})

    def get_open_positions(self, symbols: Sequence[str]) -> Dict[str, _StubPosition]:
        symbol = next(iter(symbols), self.symbol)
        return {symbol: _StubPosition(meta={"weight": self._weight}, qty=self._qty)}


def _run_step(
    bridge: Any,
    *,
    ts_ms: int,
    price: float | None,
    orders: Iterable[_StubOrder],
) -> Dict[str, Any]:
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
        bar_timeframe_ms=60_000,
    )


@pytest.fixture(name="bar_bridge_cls")
def _bar_bridge_cls(monkeypatch: pytest.MonkeyPatch) -> type[Any]:
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


def test_step_overwrites_stale_bar_metadata(bar_bridge_cls: type[Any]) -> None:
    symbol = "BTCUSDT"
    executor = _StubBarExecutor(symbol)
    bridge = bar_bridge_cls(
        executor,
        symbol=symbol,
        timeframe_ms=60_000,
        initial_equity=1_000.0,
    )

    stale_bar = bridge._build_bar(  # type: ignore[attr-defined]
        ts_ms=1,
        symbol=symbol,
        open_price=90.0,
        high_price=95.0,
        low_price=85.0,
        close_price=90.0,
    )

    order = _StubOrder(desired_weight=1.0, meta={"bar": stale_bar})

    _run_step(
        bridge,
        ts_ms=2,
        price=100.0,
        orders=[order],
    )

    assert len(executor.executed_orders) == 1
    executed_meta = getattr(executor.executed_orders[0], "meta", {})
    new_bar = executed_meta.get("bar")
    assert new_bar is not None, "Bridge should attach a bar payload to the order"
    assert getattr(new_bar, "ts", None) == 2
    close_price = getattr(new_bar, "close", None)
    assert close_price is not None
    assert float(close_price) == pytest.approx(100.0)


def test_trade_pnl_alignment(bar_bridge_cls: type[Any]) -> None:
    symbol = "BTCUSDT"
    executor = _StubBarExecutor(symbol)
    bridge = bar_bridge_cls(
        executor,
        symbol=symbol,
        timeframe_ms=60_000,
        initial_equity=1_000.0,
    )

    first_report = _run_step(
        bridge,
        ts_ms=1,
        price=100.0,
        orders=[_StubOrder(desired_weight=1.0)],
    )

    assert len(executor.executed_orders) == 1
    assert pytest.approx(first_report["bar_pnl"], rel=1e-9) == 0.0
    assert pytest.approx(first_report["equity"], rel=1e-9) == 1_000.0

    second_report = _run_step(
        bridge,
        ts_ms=2,
        price=110.0,
        orders=[],
    )

    assert len(executor.executed_orders) == 1, "No additional trades expected on second bar"
    assert pytest.approx(second_report["bar_return"], rel=1e-9) == 0.1
    assert pytest.approx(second_report["bar_pnl"], rel=1e-9) == 100.0
    assert pytest.approx(second_report["equity"], rel=1e-9) == 1_100.0


def test_missing_price_does_not_book_pnl(bar_bridge_cls: type[Any]) -> None:
    symbol = "BTCUSDT"
    executor = _StubBarExecutor(symbol)
    bridge = bar_bridge_cls(
        executor,
        symbol=symbol,
        timeframe_ms=60_000,
        initial_equity=1_000.0,
    )

    first_report = _run_step(
        bridge,
        ts_ms=1,
        price=100.0,
        orders=[_StubOrder(desired_weight=1.0)],
    )

    missing_price_report = _run_step(
        bridge,
        ts_ms=2,
        price=None,
        orders=[],
    )

    assert len(executor.executed_orders) == 1
    assert pytest.approx(missing_price_report["bar_pnl"], rel=1e-9) == 0.0
    assert missing_price_report["equity"] == pytest.approx(
        first_report["equity"],
        rel=1e-9,
    )

    resumed_report = _run_step(
        bridge,
        ts_ms=3,
        price=110.0,
        orders=[],
    )

    assert pytest.approx(resumed_report["bar_return"], rel=1e-9) == 0.1
    assert pytest.approx(resumed_report["bar_pnl"], rel=1e-9) == 100.0
    assert pytest.approx(resumed_report["equity"], rel=1e-9) == 1_100.0


def test_fallback_qty_from_weight(bar_bridge_cls: type[Any]) -> None:
    symbol = "BTCUSDT"
    executor = _StubBarExecutor(symbol)
    executor._weight = 1.0
    bridge = bar_bridge_cls(
        executor,
        symbol=symbol,
        timeframe_ms=60_000,
        initial_equity=1_000.0,
    )

    quiet_report = _run_step(
        bridge,
        ts_ms=1,
        price=100.0,
        orders=[],
    )

    assert not executor.executed_orders
    assert pytest.approx(quiet_report["bar_pnl"], rel=1e-9) == 0.0
    assert pytest.approx(quiet_report["equity"], rel=1e-9) == 1_000.0

    mark_report = _run_step(
        bridge,
        ts_ms=2,
        price=110.0,
        orders=[],
    )

    assert not executor.executed_orders
    assert pytest.approx(mark_report["bar_return"], rel=1e-9) == 0.1
    assert pytest.approx(mark_report["bar_pnl"], rel=1e-9) == 100.0
    assert pytest.approx(mark_report["equity"], rel=1e-9) == 1_100.0
