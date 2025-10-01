"""Integration tests for :class:`service_backtest.BarBacktestSimBridge`."""

from __future__ import annotations

from decimal import Decimal
import importlib
import sys
from types import ModuleType
from typing import Any

import pytest

from api.spot_signals import (
    SpotSignalEconomics,
    SpotSignalEnvelope,
    SpotSignalTargetWeightPayload,
)
from core_config import SpotCostConfig
from core_models import Order, OrderType, Side
from impl_bar_executor import BarExecutor


class _MutableOrder:
    def __init__(
        self,
        *,
        ts: int,
        symbol: str,
        side: Side,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None,
        meta: Any,
    ) -> None:
        self.ts = ts
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.client_order_id = f"test-{symbol}-{ts}"
        self.reduce_only = False
        self.meta = meta


@pytest.fixture(name="bar_bridge_cls")
def _bar_bridge_cls(monkeypatch: pytest.MonkeyPatch) -> type[Any]:
    """Return ``BarBacktestSimBridge`` with light-weight exchange stubs."""

    exchange_mod = ModuleType("exchange")
    specs_mod = ModuleType("exchange.specs")

    def _load_specs(*_args: Any, **_kwargs: Any) -> tuple[dict, dict]:  # pragma: no cover - stub
        return {}, {}

    def _round_price_to_tick(price: float, *_args: Any, **_kwargs: Any) -> float:  # pragma: no cover - stub
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


def _make_order(symbol: str, meta: Any) -> _MutableOrder:
    return _MutableOrder(
        ts=1,
        symbol=symbol,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta=meta,
    )


def _run_step(bridge: Any, *, ts_ms: int, price: float, order: _MutableOrder) -> dict[str, Any]:
    return bridge.step(
        ts_ms=ts_ms,
        ref_price=price,
        bid=price,
        ask=price,
        vol_factor=1.0,
        liquidity=None,
        orders=[order],
        bar_open=price,
        bar_high=price,
        bar_low=price,
        bar_close=price,
        bar_timeframe_ms=60_000,
    )


def test_spot_signal_envelope_payload_passthrough(bar_bridge_cls: type[Any]) -> None:
    symbol = "BTCUSDT"
    executor = BarExecutor(
        run_id="test",
        cost_config=SpotCostConfig(),
        default_equity_usd=1_000.0,
    )
    bridge = bar_bridge_cls(
        executor,
        symbol=symbol,
        timeframe_ms=60_000,
        initial_equity=1_000.0,
    )

    economics = SpotSignalEconomics(
        edge_bps=50.0,
        cost_bps=0.0,
        net_bps=50.0,
        turnover_usd=100.0,
        act_now=True,
        impact=0.0,
        impact_mode="none",
    )
    payload = SpotSignalTargetWeightPayload(target_weight=0.5, economics=economics)
    envelope = SpotSignalEnvelope(
        symbol=symbol,
        bar_close_ms=60_000,
        expires_at_ms=120_000,
        payload=payload,
    )
    order = _make_order(symbol, envelope)

    report = _run_step(bridge, ts_ms=60_000, price=100.0, order=order)

    decisions = report.get("decisions") or []
    assert decisions, "Expected BarExecutor to emit a decision payload"
    assert decisions[0]["target_weight"] == pytest.approx(0.5)
    assert report["bar_weight"] == pytest.approx(0.5)

    positions = executor.get_open_positions([symbol])
    assert positions[symbol].meta["weight"] == pytest.approx(0.5)
    assert positions[symbol].qty == Decimal("5")


def test_frozen_order_meta_enriched_in_place(bar_bridge_cls: type[Any]) -> None:
    symbol = "SOLUSDT"
    executor = BarExecutor(
        run_id="test",
        cost_config=SpotCostConfig(),
        default_equity_usd=1_000.0,
    )
    bridge = bar_bridge_cls(
        executor,
        symbol=symbol,
        timeframe_ms=60_000,
        initial_equity=1_000.0,
    )

    order = Order(
        ts=1,
        symbol=symbol,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={"custom": "value"},
    )

    _run_step(bridge, ts_ms=60_000, price=100.0, order=order)

    assert order.meta["custom"] == "value"
    assert order.meta["equity_usd"] == pytest.approx(1_000.0)
    assert isinstance(order.meta.get("payload"), dict)
    bar_payload = order.meta.get("bar")
    assert bar_payload is not None and bar_payload.close == Decimal("100")


def test_rebalance_only_payload_preserved(bar_bridge_cls: type[Any]) -> None:
    symbol = "ETHUSDT"
    executor = BarExecutor(
        run_id="test",
        cost_config=SpotCostConfig(),
        default_equity_usd=2_000.0,
    )
    bridge = bar_bridge_cls(
        executor,
        symbol=symbol,
        timeframe_ms=60_000,
        initial_equity=2_000.0,
    )

    rebalance_meta = {
        "rebalance": {
            "target_weight": 0.3,
            "edge_bps": 25.0,
            "turnover_usd": 600.0,
            "act_now": True,
        }
    }
    order = _make_order(symbol, rebalance_meta)

    report = _run_step(bridge, ts_ms=120_000, price=200.0, order=order)

    decisions = report.get("decisions") or []
    assert decisions, "Expected BarExecutor to emit a decision payload"
    assert decisions[0]["target_weight"] == pytest.approx(0.3)
    assert report["bar_weight"] == pytest.approx(0.3)

    positions = executor.get_open_positions([symbol])
    assert positions[symbol].meta["weight"] == pytest.approx(0.3)
    assert positions[symbol].qty == Decimal("3")


def test_open_price_field_updates_equity(bar_bridge_cls: type[Any]) -> None:
    symbol = "BNBUSDT"
    executor = BarExecutor(
        run_id="test",
        cost_config=SpotCostConfig(),
        default_equity_usd=1_000.0,
    )
    bridge = bar_bridge_cls(
        executor,
        symbol=symbol,
        timeframe_ms=60_000,
        initial_equity=1_000.0,
        bar_price_field="open",
    )

    economics = SpotSignalEconomics(
        edge_bps=50.0,
        cost_bps=0.0,
        net_bps=50.0,
        turnover_usd=100.0,
        act_now=True,
        impact=0.0,
        impact_mode="none",
    )
    payload = SpotSignalTargetWeightPayload(target_weight=0.5, economics=economics)
    envelope = SpotSignalEnvelope(
        symbol=symbol,
        bar_close_ms=60_000,
        expires_at_ms=120_000,
        payload=payload,
    )
    order = _make_order(symbol, envelope)

    first_report = bridge.step(
        ts_ms=60_000,
        ref_price=110.0,
        bid=110.0,
        ask=110.0,
        vol_factor=1.0,
        liquidity=None,
        orders=[order],
        bar_open=100.0,
        bar_high=115.0,
        bar_low=95.0,
        bar_close=110.0,
        bar_timeframe_ms=60_000,
    )

    assert first_report["ref_price"] == pytest.approx(100.0)
    assert float(order.meta["bar"].close) == pytest.approx(100.0)
    assert bridge._last_prices[symbol] == pytest.approx(100.0)

    second_report = bridge.step(
        ts_ms=120_000,
        ref_price=130.0,
        bid=130.0,
        ask=130.0,
        vol_factor=1.0,
        liquidity=None,
        orders=[],
        bar_open=120.0,
        bar_high=135.0,
        bar_low=115.0,
        bar_close=130.0,
        bar_timeframe_ms=60_000,
    )

    assert second_report["bar_return"] == pytest.approx(0.2)
    assert second_report["bar_pnl"] == pytest.approx(100.0)
    assert second_report["equity"] == pytest.approx(1_100.0)
    assert bridge._last_prices[symbol] == pytest.approx(120.0)
    assert second_report["ref_price"] == pytest.approx(120.0)
