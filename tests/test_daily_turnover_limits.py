import logging
import math
from decimal import Decimal
from types import MethodType, SimpleNamespace
from typing import Any, Mapping

import pytest

from core_config import SpotTurnoverCaps, SpotTurnoverLimit
from core_models import Order, OrderType, Side
from service_signal_runner import _Worker
from api.spot_signals import (
    SpotSignalEnvelope,
    SpotSignalTargetWeightPayload,
    SpotSignalEconomics,
)


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


def _make_enveloped_order(
    symbol: str, target_weight: float, turnover_usd: float
) -> Order:
    envelope = SpotSignalEnvelope(
        symbol=symbol,
        bar_close_ms=0,
        expires_at_ms=60_000,
        payload=SpotSignalTargetWeightPayload(
            target_weight=target_weight,
            economics=SpotSignalEconomics(
                edge_bps=0.0,
                cost_bps=0.0,
                net_bps=0.0,
                turnover_usd=turnover_usd,
                act_now=True,
            ),
        ),
    )
    return Order(
        ts=0,
        symbol=symbol,
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta=envelope,
    )


def test_daily_turnover_cap_clamps_and_defers() -> None:
    worker = _make_worker_with_daily_cap(500.0)

    first = _make_order("BTCUSDT", 0.4, 400.0)
    adjusted_first = worker._apply_daily_turnover_limits([first], "BTCUSDT", 1)
    assert len(adjusted_first) == 1
    assert adjusted_first[0].meta.get("_daily_turnover_usd") == pytest.approx(400.0)
    payload_first = worker._extract_signal_payload(adjusted_first[0])
    adjusted_first[0].meta["_bar_execution"] = {
        "filled": True,
        "target_weight": payload_first["target_weight"],
        "delta_weight": payload_first["target_weight"],
        "turnover_usd": adjusted_first[0].meta.get("_daily_turnover_usd"),
    }
    worker._commit_exposure(adjusted_first[0])
    assert worker._daily_symbol_turnover["BTCUSDT"]["total"] == pytest.approx(400.0)

    second = _make_order("BTCUSDT", 0.9, 500.0)
    adjusted_second = worker._apply_daily_turnover_limits([second], "BTCUSDT", 1)
    assert len(adjusted_second) == 1
    payload = adjusted_second[0].meta["payload"]
    assert payload["target_weight"] == pytest.approx(0.5)
    assert adjusted_second[0].meta.get("_daily_turnover_usd") == pytest.approx(100.0)
    adjusted_second[0].meta["_bar_execution"] = {
        "filled": True,
        "target_weight": payload["target_weight"],
        "delta_weight": payload["target_weight"] - 0.4,
        "turnover_usd": adjusted_second[0].meta.get("_daily_turnover_usd"),
    }
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
    first_meta = worker._ensure_order_meta(adjusted_first[0])
    assert first_meta.get("_daily_turnover_usd") == pytest.approx(150.0)
    payload_first = worker._extract_signal_payload(adjusted_first[0])
    first_meta["_bar_execution"] = {
        "filled": True,
        "target_weight": payload_first["target_weight"],
        "delta_weight": payload_first["target_weight"],
        "turnover_usd": first_meta.get("_daily_turnover_usd"),
    }
    worker._commit_exposure(adjusted_first[0])
    assert worker._daily_symbol_turnover["ETHUSDT"]["total"] == pytest.approx(150.0)

    second = _make_economics_order("ETHUSDT", 0.9, 100.0)
    adjusted_second = worker._apply_daily_turnover_limits([second], "ETHUSDT", 1)
    assert len(adjusted_second) == 1
    payload = adjusted_second[0].meta["payload"]
    assert payload["target_weight"] == pytest.approx(0.7)
    assert adjusted_second[0].meta.get("_daily_turnover_usd") == pytest.approx(50.0)
    adjusted_second[0].meta["_bar_execution"] = {
        "filled": True,
        "target_weight": payload["target_weight"],
        "delta_weight": payload["target_weight"] - 0.5,
        "turnover_usd": adjusted_second[0].meta.get("_daily_turnover_usd"),
    }
    worker._commit_exposure(adjusted_second[0])
    assert worker._daily_symbol_turnover["ETHUSDT"]["total"] == pytest.approx(200.0)

    third = _make_economics_order("ETHUSDT", 0.8, 10.0)
    adjusted_third = worker._apply_daily_turnover_limits([third], "ETHUSDT", 1)
    assert adjusted_third == []
    snapshot = worker._daily_turnover_snapshot()
    assert snapshot["portfolio"]["remaining_usd"] == pytest.approx(0.0)


def test_daily_turnover_cap_handles_enveloped_meta() -> None:
    worker = _make_worker_with_daily_cap(200.0)

    def _target_from_payload(payload: Mapping[str, Any]) -> float:
        if "target_weight" in payload:
            return float(payload["target_weight"])
        if "target" in payload:
            return float(payload["target"])
        raise AssertionError("target weight not present in payload")

    first = _make_enveloped_order("SOLUSDT", 0.5, 150.0)
    adjusted_first = worker._apply_daily_turnover_limits([first], "SOLUSDT", 1)
    assert len(adjusted_first) == 1
    payload_first = worker._extract_signal_payload(adjusted_first[0])
    first_meta = worker._ensure_order_meta(adjusted_first[0])
    object.__setattr__(adjusted_first[0], "meta", first_meta)
    first_meta.setdefault("payload", payload_first)
    assert first_meta.get("_daily_turnover_usd") == pytest.approx(150.0)
    assert _target_from_payload(payload_first) == pytest.approx(0.5)
    economics_first = payload_first.get("economics", {})
    assert economics_first.get("turnover_usd") == pytest.approx(150.0)
    first_meta["_bar_execution"] = {
        "filled": True,
        "target_weight": payload_first["target_weight"],
        "delta_weight": payload_first["target_weight"],
        "turnover_usd": first_meta.get("_daily_turnover_usd"),
    }
    worker._commit_exposure(adjusted_first[0])

    second = _make_enveloped_order("SOLUSDT", 0.9, 100.0)
    adjusted_second = worker._apply_daily_turnover_limits([second], "SOLUSDT", 1)
    assert len(adjusted_second) == 1
    payload_second = worker._extract_signal_payload(adjusted_second[0])
    second_meta = worker._ensure_order_meta(adjusted_second[0])
    object.__setattr__(adjusted_second[0], "meta", second_meta)
    second_meta.setdefault("payload", payload_second)
    assert second_meta.get("_daily_turnover_usd") == pytest.approx(50.0)
    assert _target_from_payload(payload_second) == pytest.approx(0.7)
    second_meta["_bar_execution"] = {
        "filled": True,
        "target_weight": _target_from_payload(payload_second),
        "delta_weight": _target_from_payload(payload_second) - 0.5,
        "turnover_usd": second_meta.get("_daily_turnover_usd"),
    }
    worker._commit_exposure(adjusted_second[0])
    assert worker._daily_symbol_turnover["SOLUSDT"]["total"] == pytest.approx(200.0)


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


def test_daily_turnover_non_finite_request_is_sanitized() -> None:
    worker = _make_worker_with_daily_cap(1000.0)

    order = _make_turnover_rich_order("BTCUSDT", 0.2, float("nan"))
    payload = order.meta["payload"]
    payload["turnover_usd"] = "NaN"
    payload.setdefault("decision", {})["turnover_usd"] = "NaN"
    order.meta.setdefault("decision", {})["turnover_usd"] = "NaN"
    payload.setdefault("economics", {})["turnover_usd"] = "NaN"
    order.meta.setdefault("economics", {})["turnover_usd"] = "NaN"
    order.meta.setdefault("decision", {}).setdefault("economics", {})["turnover_usd"] = "NaN"

    original_extract = worker._extract_order_turnover

    def _extract_with_nan(self: _Worker, candidate: Order) -> float:
        payload_map = self._extract_signal_payload(candidate)
        if isinstance(payload_map, dict):
            turnover_val = payload_map.get("turnover_usd")
            if isinstance(turnover_val, str) and turnover_val.lower() == "nan":
                return float("nan")
        return original_extract(candidate)

    worker._extract_order_turnover = MethodType(_extract_with_nan, worker)

    adjusted = worker._apply_daily_turnover_limits([order], "BTCUSDT", 1)

    assert len(adjusted) == 1
    meta = adjusted[0].meta
    assert meta.get("_daily_turnover_usd") == pytest.approx(0.0)

    daily_info = meta.get("daily_turnover")
    assert isinstance(daily_info, dict)
    assert daily_info.get("requested_usd") == pytest.approx(0.0)
    assert daily_info.get("executed_usd") == pytest.approx(0.0)

    symbol_tracker = worker._daily_symbol_turnover.get("BTCUSDT")
    assert symbol_tracker is not None
    assert math.isfinite(symbol_tracker.get("total", float("nan")))
    assert symbol_tracker.get("total") == pytest.approx(0.0)

    assert math.isfinite(worker._daily_portfolio_turnover.get("total", float("nan")))
    assert worker._daily_portfolio_turnover.get("total") == pytest.approx(0.0)


def test_daily_turnover_dropped_orders_clear_pending_weight() -> None:
    worker = _make_worker_with_daily_cap(100.0)
    symbol = "BTCUSDT"
    order = _make_turnover_rich_order(symbol, 0.5, 50.0)
    order_id = id(order)
    worker._pending_weight[order_id] = {"symbol": symbol}
    worker._pending_weight_refs[order_id] = order

    day_key = worker._current_day_key(1)
    worker._daily_symbol_turnover[symbol] = {"day": day_key, "total": 100.0}
    worker._daily_portfolio_turnover = {"day": day_key, "total": 100.0}

    adjusted = worker._apply_daily_turnover_limits([order], symbol, 1)

    assert adjusted == []
    assert order_id not in worker._pending_weight
    assert order_id not in worker._pending_weight_refs
    assert worker._replay_pending_weight_entries(order) is False

    worker_fail = _make_worker_with_daily_cap(200.0)
    fail_symbol = "ETHUSDT"
    fail_order = _make_turnover_rich_order(fail_symbol, 0.6, 150.0)
    fail_order_id = id(fail_order)
    worker_fail._pending_weight[fail_order_id] = {"symbol": fail_symbol}
    worker_fail._pending_weight_refs[fail_order_id] = fail_order

    fail_day_key = worker_fail._current_day_key(1)
    worker_fail._daily_symbol_turnover[fail_symbol] = {
        "day": fail_day_key,
        "total": 150.0,
    }
    worker_fail._daily_portfolio_turnover = {"day": fail_day_key, "total": 150.0}

    def _fail_scale(
        self: _Worker, *_: object, **__: object
    ) -> bool:  # pragma: no cover - simple stub
        return False

    worker_fail._scale_order_for_turnover = MethodType(_fail_scale, worker_fail)

    adjusted_fail = worker_fail._apply_daily_turnover_limits([fail_order], fail_symbol, 1)

    assert adjusted_fail == []
    assert fail_order_id not in worker_fail._pending_weight
    assert fail_order_id not in worker_fail._pending_weight_refs
    assert worker_fail._replay_pending_weight_entries(fail_order) is False
