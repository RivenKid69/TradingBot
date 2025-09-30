from __future__ import annotations

from decimal import Decimal
import importlib
import sys
import types

import pytest

from core_models import Bar, Order, OrderType, Side
from services import state_storage


def test_bar_executor_restores_weights_on_restart(monkeypatch, tmp_path):
    requests_stub = sys.modules.get("requests")
    if requests_stub is None:
        requests_stub = types.ModuleType("requests")
        sys.modules["requests"] = requests_stub
    for name in ("get", "post", "put", "delete", "request"):
        monkeypatch.setattr(requests_stub, name, lambda *args, **kwargs: None)

    import service_signal_runner

    service_signal_runner = importlib.reload(service_signal_runner)

    initial_state = state_storage.TradingState(
        exposure_state={"weights": {"BTCUSDT": 0.25}},
        total_notional=0.0,
    )

    weights = service_signal_runner._extract_bar_initial_weights_from_state(initial_state)
    assert weights == {"BTCUSDT": pytest.approx(0.25)}

    executor = service_signal_runner.BarExecutor(
        run_id="bar",
        default_equity_usd=1000.0,
        initial_weights=weights,
    )

    assert executor._states.get("BTCUSDT") is not None
    assert executor._states["BTCUSDT"].weight == pytest.approx(0.25)

    bar = Bar(
        ts=1,
        symbol="BTCUSDT",
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("95"),
        close=Decimal("102"),
    )
    order = Order(
        ts=1,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        meta={
            "payload": {"target_weight": 0.25},
            "bar": bar.to_dict(),
            "equity_usd": 1000.0,
        },
    )

    executor.execute(order)
    snapshot = executor.monitoring_snapshot()
    instructions = snapshot.get("instructions") or []
    if instructions:
        first_delta = instructions[0]["delta_weight"]
    else:
        first_delta = snapshot.get("delta_weight")
    assert first_delta is not None
    assert first_delta == pytest.approx(0.0)
