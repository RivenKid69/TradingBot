from __future__ import annotations

from decimal import Decimal
import logging
from types import SimpleNamespace

import pytest

from service_signal_runner import _Worker
from core_models import Side
from strategies.base import SignalPosition
from services import state_storage


@pytest.fixture
def worker_with_state(monkeypatch: pytest.MonkeyPatch):
    """Create a worker with state persistence hooks patched for inspection."""

    original_update = state_storage.update_state
    monkeypatch.setattr(state_storage, "_state", state_storage.TradingState())

    updates: list[dict[str, object]] = []

    def _record_update(**kwargs):
        updates.append(kwargs)
        original_update(**kwargs)

    monkeypatch.setattr(state_storage, "update_state", _record_update)

    policy = SimpleNamespace(
        revert_signal_state=lambda *args, **kwargs: None,
        consume_signal_transitions=lambda: [],
    )
    worker = _Worker(
        fp=SimpleNamespace(),
        policy=policy,
        logger=logging.getLogger("test"),
        executor=SimpleNamespace(submit=lambda order: None),
        guards=None,
        enforce_closed_bars=False,
        state_enabled=True,
    )
    worker._last_prices.clear()
    return worker, updates


def test_stage_and_rollback_exposure(worker_with_state):
    worker, updates = worker_with_state
    worker._last_prices["BTC"] = 100.0
    order = SimpleNamespace(
        symbol="BTC",
        side=Side.BUY,
        quantity=Decimal("2"),
        meta={"signal_leg": "entry"},
    )

    worker._stage_exposure_adjustments([order], ts_ms=1234)

    assert worker._positions["BTC"] == pytest.approx(2.0)
    assert id(order) in worker._pending_exposure
    assert updates
    assert updates[-1]["total_notional"] == pytest.approx(200.0)

    worker._rollback_exposure(order)

    assert worker._positions.get("BTC", 0.0) == pytest.approx(0.0)
    assert id(order) not in worker._pending_exposure
    assert updates[-1]["total_notional"] == pytest.approx(0.0)
    assert worker._exposure_state["positions"] == {}


def test_entry_limit_refusal_rolls_back_pending(worker_with_state):
    worker, updates = worker_with_state
    worker._last_prices["ETH"] = 50.0
    order = SimpleNamespace(
        symbol="ETH",
        side=Side.BUY,
        quantity=Decimal("1"),
        meta={"signal_leg": "entry"},
    )

    worker._stage_exposure_adjustments([order], ts_ms=1111)
    assert worker._positions["ETH"] == pytest.approx(1.0)

    transition = {"prev": SignalPosition.FLAT, "new": SignalPosition.LONG}
    worker._handle_entry_limit_refusal(
        "ETH",
        1111,
        transition,
        {},
        entry_steps=1,
        removed_count=1,
        removed_orders=[order],
    )

    assert worker._positions.get("ETH", 0.0) == pytest.approx(0.0)
    assert worker._pending_exposure == {}
    assert updates[-1]["total_notional"] == pytest.approx(0.0)


def test_bar_mode_total_notional_uses_equity_override():
    policy = SimpleNamespace(
        revert_signal_state=lambda *args, **kwargs: None,
        consume_signal_transitions=lambda: [],
    )
    worker = _Worker(
        fp=SimpleNamespace(),
        policy=policy,
        logger=logging.getLogger("test_equity_override"),
        executor=SimpleNamespace(submit=lambda order: None),
        guards=None,
        enforce_closed_bars=True,
        execution_mode="bar",
        portfolio_equity=None,
        state_enabled=False,
    )
    worker._last_prices["FOO"] = 20.0

    order = SimpleNamespace(
        symbol="FOO",
        meta={"payload": {"target_weight": 0.5, "equity_usd": 200.0}},
    )
    order.meta["_bar_execution"] = {
        "filled": True,
        "target_weight": 0.5,
        "delta_weight": 0.5,
        "turnover_usd": 100.0,
    }

    worker._commit_exposure(order)

    assert worker._symbol_equity["FOO"] == pytest.approx(200.0)
    assert worker._positions["FOO"] == pytest.approx((0.5 * 200.0) / 20.0)
    assert worker._exposure_state["total_notional"] == pytest.approx(0.5 * 200.0)
