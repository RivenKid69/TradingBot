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


def _make_economics_order(
    symbol: str, target_weight: float, turnover_usd: float
) -> Order:
    return Order(
        ts=0,
        symbol=symbol,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {
                "target_weight": target_weight,
                "economics": {"turnover_usd": turnover_usd},
            }
        },
    )


def _make_turnover_rich_order(
    symbol: str, target_weight: float, turnover_usd: float
) -> Order:
    return Order(
        ts=0,
        symbol=symbol,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={
            "payload": {
                "target_weight": target_weight,
                "turnover_usd": turnover_usd,
                "turnover": turnover_usd,
                "economics": {
                    "turnover_usd": turnover_usd,
                    "notional_usd": turnover_usd,
                },
                "decision": {
                    "turnover_usd": turnover_usd,
                    "economics": {"turnover_usd": turnover_usd},
                },
            },
            "economics": {"turnover_usd": turnover_usd},
            "decision": {
                "turnover_usd": turnover_usd,
                "economics": {"turnover_usd": turnover_usd},
            },
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


def test_daily_turnover_cap_uses_economics_payload() -> None:
    worker = _make_worker_with_daily_cap(200.0)

    first = _make_economics_order("ETHUSDT", 0.5, 150.0)
    adjusted_first = worker._apply_daily_turnover_limits([first], "ETHUSDT", 1)
    assert len(adjusted_first) == 1
    assert adjusted_first[0].meta.get("_daily_turnover_usd") == pytest.approx(150.0)
    worker._commit_exposure(adjusted_first[0])
    assert worker._daily_symbol_turnover["ETHUSDT"]["total"] == pytest.approx(150.0)

    second = _make_economics_order("ETHUSDT", 0.9, 100.0)
    adjusted_second = worker._apply_daily_turnover_limits([second], "ETHUSDT", 1)
    assert len(adjusted_second) == 1
    payload = adjusted_second[0].meta["payload"]
    assert payload["target_weight"] == pytest.approx(0.7)
    assert adjusted_second[0].meta.get("_daily_turnover_usd") == pytest.approx(50.0)
    worker._commit_exposure(adjusted_second[0])
    assert worker._daily_symbol_turnover["ETHUSDT"]["total"] == pytest.approx(200.0)

    third = _make_economics_order("ETHUSDT", 0.8, 10.0)
    adjusted_third = worker._apply_daily_turnover_limits([third], "ETHUSDT", 1)
    assert adjusted_third == []
    snapshot = worker._daily_turnover_snapshot()
    assert snapshot["portfolio"]["remaining_usd"] == pytest.approx(0.0)


def test_daily_turnover_limits_use_sequential_weights_for_orders() -> None:
    worker = _make_worker_with_daily_cap(600.0)

    orders = [
        _make_order("BTCUSDT", 0.4, 0.0),
        _make_order("BTCUSDT", 0.6, 0.0),
    ]

    adjusted = worker._apply_daily_turnover_limits(orders, "BTCUSDT", 1)
    assert len(adjusted) == 2

    first_meta = adjusted[0].meta
    assert first_meta.get("_daily_turnover_usd") == pytest.approx(400.0)
    assert first_meta["payload"]["target_weight"] == pytest.approx(0.4)
    assert first_meta["daily_turnover"]["clamped"] is False

    second_meta = adjusted[1].meta
    assert second_meta.get("_daily_turnover_usd") == pytest.approx(200.0)
    assert second_meta["payload"]["target_weight"] == pytest.approx(0.6)
    assert second_meta["daily_turnover"]["clamped"] is False


def test_daily_turnover_scaling_updates_turnover_fields() -> None:
    worker = _make_worker_with_daily_cap(100.0)

    order = _make_turnover_rich_order("BTCUSDT", 0.5, 250.0)
    normalization = {
        "factor": 0.5,
        "available_delta": 0.5,
        "delta_positive_total": 1.0,
        "delta_negative_total": 0.0,
    }
    order.meta["payload"]["normalized"] = True
    order.meta["payload"]["normalization"] = dict(normalization)
    order.meta["payload"]["decision"]["normalization"] = dict(normalization)
    order.meta["normalization"] = dict(normalization)
    order.meta["normalized"] = True
    order.meta["decision"]["normalization"] = dict(normalization)
    worker._pending_weight[id(order)] = {
        "symbol": "BTCUSDT",
        "target_weight": 0.5,
        "delta_weight": 0.5,
        "normalized": True,
        "factor": normalization["factor"],
        "normalization": dict(normalization),
    }

    adjusted = worker._apply_daily_turnover_limits([order], "BTCUSDT", 1)
    assert len(adjusted) == 1
    result = adjusted[0]
    factor = 100.0 / 250.0

    meta = result.meta
    assert meta.get("_daily_turnover_usd") == pytest.approx(100.0)

    payload = meta["payload"]
    assert payload["turnover_usd"] == pytest.approx(250.0 * factor)
    assert payload["turnover"] == pytest.approx(250.0 * factor)
    assert payload["economics"]["turnover_usd"] == pytest.approx(250.0 * factor)
    assert payload["economics"]["notional_usd"] == pytest.approx(250.0 * factor)
    assert payload["decision"]["turnover_usd"] == pytest.approx(250.0 * factor)
    assert payload["decision"]["economics"]["turnover_usd"] == pytest.approx(
        250.0 * factor
    )

    expected_norm_factor = normalization["factor"] * factor
    expected_available_delta = normalization["available_delta"] * factor
    payload_norm = payload["normalization"]
    assert payload_norm["factor"] == pytest.approx(expected_norm_factor)
    assert payload_norm["available_delta"] == pytest.approx(expected_available_delta)
    payload_decision_norm = payload["decision"]["normalization"]
    assert payload_decision_norm["factor"] == pytest.approx(expected_norm_factor)
    assert payload_decision_norm["available_delta"] == pytest.approx(
        expected_available_delta
    )

    assert meta["economics"]["turnover_usd"] == pytest.approx(250.0 * factor)
    assert meta["decision"]["turnover_usd"] == pytest.approx(250.0 * factor)
    assert meta["decision"]["economics"]["turnover_usd"] == pytest.approx(
        250.0 * factor
    )
    meta_norm = meta["normalization"]
    assert meta_norm["factor"] == pytest.approx(expected_norm_factor)
    assert meta_norm["available_delta"] == pytest.approx(expected_available_delta)
    meta_decision_norm = meta["decision"]["normalization"]
    assert meta_decision_norm["factor"] == pytest.approx(expected_norm_factor)
    assert meta_decision_norm["available_delta"] == pytest.approx(
        expected_available_delta
    )

    pending = worker._pending_weight[id(result)]
    assert pending["factor"] == pytest.approx(expected_norm_factor)
    assert pending["normalization"]["available_delta"] == pytest.approx(
        expected_available_delta
    )
