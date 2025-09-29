import logging
from decimal import Decimal
from types import SimpleNamespace

import pytest

from core_config import SpotTurnoverCaps, SpotTurnoverLimit
from core_models import Order, OrderType, Side
from service_signal_runner import _Worker


def _make_worker_with_daily_cap(limit_usd: float) -> _Worker:
    class DummyFeaturePipe:
        def __init__(self) -> None:
            self.spread_ttl_ms = 0
            self._spread_ttl_ms = 0
            self.signal_quality = {}
            self.metrics = SimpleNamespace(reset_symbol=lambda *_: None)

    class DummyPolicy:
        def consume_signal_transitions(self):  # pragma: no cover - stub
            return []

    turnover_caps = SpotTurnoverCaps(
        per_symbol=SpotTurnoverLimit(daily_usd=limit_usd),
        portfolio=SpotTurnoverLimit(daily_usd=limit_usd),
    )
    executor = SimpleNamespace(turnover_caps=turnover_caps)
    worker = _Worker(
        DummyFeaturePipe(),
        DummyPolicy(),
        logging.getLogger("test_daily_turnover"),
        executor,
        None,
        enforce_closed_bars=True,
        close_lag_ms=0,
        ws_dedup_enabled=False,
        ws_dedup_log_skips=False,
        ws_dedup_timeframe_ms=0,
        throttle_cfg=None,
        no_trade_cfg=None,
        pipeline_cfg=None,
        signal_quality_cfg=None,
        zero_signal_alert=0,
        state_enabled=False,
        rest_candidates=None,
        monitoring=None,
        monitoring_agg=None,
        worker_id="worker-test",
        status_callback=None,
        execution_mode="bar",
        portfolio_equity=1_000.0,
        max_total_weight=None,
    )
    return worker


def _make_order(symbol: str, target_weight: float, turnover_usd: float) -> Order:
    return Order(
        ts=0,
        symbol=symbol,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {"target_weight": target_weight},
            "decision": {"turnover_usd": turnover_usd},
        },
    )


def test_daily_turnover_cap_clamps_and_defers() -> None:
    worker = _make_worker_with_daily_cap(500.0)

    first = _make_order("BTCUSDT", 0.4, 400.0)
    adjusted_first = worker._apply_daily_turnover_limits([first], "BTCUSDT", 1)
    assert len(adjusted_first) == 1
    assert adjusted_first[0].meta.get("_daily_turnover_usd") == pytest.approx(400.0)
    worker._commit_exposure(adjusted_first[0])
    assert worker._daily_symbol_turnover["BTCUSDT"]["total"] == pytest.approx(400.0)

    second = _make_order("BTCUSDT", 0.9, 500.0)
    adjusted_second = worker._apply_daily_turnover_limits([second], "BTCUSDT", 1)
    assert len(adjusted_second) == 1
    payload = adjusted_second[0].meta["payload"]
    assert payload["target_weight"] == pytest.approx(0.5)
    assert adjusted_second[0].meta.get("_daily_turnover_usd") == pytest.approx(100.0)
    worker._commit_exposure(adjusted_second[0])
    assert worker._daily_symbol_turnover["BTCUSDT"]["total"] == pytest.approx(500.0)

    third = _make_order("BTCUSDT", 0.6, 100.0)
    adjusted_third = worker._apply_daily_turnover_limits([third], "BTCUSDT", 1)
    assert adjusted_third == []
    snapshot = worker._daily_turnover_snapshot()
    assert snapshot["portfolio"]["remaining_usd"] == pytest.approx(0.0)
