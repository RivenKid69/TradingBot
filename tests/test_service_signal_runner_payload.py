from datetime import datetime, timezone
from types import SimpleNamespace
import logging

import pytest

from service_signal_runner import _Worker, CooldownSettings


def _make_worker() -> _Worker:
    worker = _Worker.__new__(_Worker)
    worker._weights = {}
    worker._logger = logging.getLogger("test_worker")
    worker._cooldown_settings = CooldownSettings()
    worker._symbol_cooldowns = {}
    worker._symbol_cooldown_set_ts = {}
    return worker


def test_build_envelope_payload_captures_valid_until_from_meta():
    worker = _make_worker()
    valid_until_iso = "2024-01-01T00:00:01Z"
    order_payload = {"target_weight": 0.2}
    order = SimpleNamespace(meta={"payload": order_payload, "valid_until": valid_until_iso})

    payload, valid_until_ms, adv_quote = worker._build_envelope_payload(order, "BTCUSDT")

    expected = int(datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc).timestamp() * 1000)
    assert payload["valid_until_ms"] == expected
    assert valid_until_ms == expected
    assert payload["kind"] == "target_weight"
    assert payload["target_weight"] == 0.2
    assert adv_quote is None


def test_resolve_weight_targets_rejects_out_of_range_target() -> None:
    worker = _make_worker()
    worker._weights["BTCUSDT"] = 0.3

    target, delta, reason = worker._resolve_weight_targets(
        "BTCUSDT", {"target_weight": 1.2}
    )

    assert target == pytest.approx(0.3)
    assert delta == pytest.approx(0.0)
    assert reason == "target_weight_out_of_bounds"


def test_resolve_weight_targets_rejects_delta_out_of_range() -> None:
    worker = _make_worker()
    worker._weights["BTCUSDT"] = 0.8

    target, delta, reason = worker._resolve_weight_targets(
        "BTCUSDT", {"delta_weight": 0.5}
    )

    assert target == pytest.approx(0.8)
    assert delta == pytest.approx(0.0)
    assert reason == "delta_weight_out_of_bounds"


def test_build_envelope_payload_flags_rejected_target() -> None:
    worker = _make_worker()
    worker._weights["BTCUSDT"] = 0.1
    order = SimpleNamespace(meta={"payload": {"target_weight": 1.5}})

    payload, _, adv_quote = worker._build_envelope_payload(order, "BTCUSDT")

    assert payload["target_weight"] == pytest.approx(0.1)
    assert payload["reject_reason"] == "target_weight_out_of_bounds"
    assert payload["requested_target_weight"] == pytest.approx(1.5)
    assert adv_quote is None


def test_build_envelope_payload_flags_rejected_delta() -> None:
    worker = _make_worker()
    worker._weights["BTCUSDT"] = 0.2
    order = SimpleNamespace(meta={"payload": {"delta_weight": -0.5}})

    payload, _, adv_quote = worker._build_envelope_payload(order, "BTCUSDT")

    assert payload["delta_weight"] == pytest.approx(0.0)
    assert payload["reject_reason"] == "delta_weight_out_of_bounds"
    assert payload["requested_delta_weight"] == pytest.approx(-0.5)
    assert adv_quote is None


def test_build_envelope_payload_preserves_nested_economics() -> None:
    worker = _make_worker()
    economics = {
        "edge_bps": 42.0,
        "cost_bps": 10.0,
        "net_bps": 32.0,
        "turnover_usd": 123.0,
        "act_now": True,
        "impact": 0.5,
        "impact_mode": "model",
    }
    order_payload = {"target_weight": 0.25, "economics": economics}
    order = SimpleNamespace(meta={"payload": order_payload})

    payload, _, adv_quote = worker._build_envelope_payload(order, "BTCUSDT")

    assert "edge_bps" not in payload
    assert payload["economics"]["edge_bps"] == pytest.approx(economics["edge_bps"])
    assert payload["economics"]["turnover_usd"] == pytest.approx(
        economics["turnover_usd"]
    )
    assert adv_quote is None


def test_build_envelope_payload_extracts_adv_quote() -> None:
    worker = _make_worker()
    adv_quote = 25_000.0
    order_payload = {
        "target_weight": 0.3,
        "economics": {
            "edge_bps": 15.0,
            "cost_bps": 5.0,
            "net_bps": 10.0,
            "turnover_usd": 300.0,
            "act_now": True,
            "impact": 0.0,
            "impact_mode": "none",
            "adv_quote": str(adv_quote),
        },
    }
    order = SimpleNamespace(meta={"payload": order_payload})

    payload, _, extracted_adv = worker._build_envelope_payload(order, "BTCUSDT")

    assert extracted_adv == pytest.approx(adv_quote)
    assert payload["economics"]["adv_quote"] == pytest.approx(adv_quote)


def test_build_envelope_payload_meta_cap_only_leaves_adv_none() -> None:
    worker = _make_worker()
    order_payload = {"target_weight": 0.1}
    order = SimpleNamespace(meta={"cap_usd": 1_000.0}, payload=order_payload)

    payload, _, adv_quote = worker._build_envelope_payload(order, "BTCUSDT")

    assert adv_quote is None
    assert payload["economics"].get("adv_quote") is None


def test_normalize_weight_targets_aggregates_symbol_totals() -> None:
    worker = _make_worker()
    worker._execution_mode = "bar"
    worker._max_total_weight = 0.5
    worker._portfolio_equity = None
    worker._pending_weight = {}
    worker._symbol_equity = {}
    worker._pending_weight_refs = {}
    worker._weights = {"BTCUSDT": 0.3}

    base_payload = {"target_weight": 0.6}
    order1 = SimpleNamespace(symbol="BTCUSDT", meta={"payload": dict(base_payload)})
    order2 = SimpleNamespace(symbol="BTCUSDT", meta={"payload": dict(base_payload)})

    normalized_orders, applied = worker._normalize_weight_targets([order1, order2])

    assert applied is True
    assert normalized_orders == [order1, order2]
    payload1 = normalized_orders[0].meta["payload"]
    payload2 = normalized_orders[1].meta["payload"]
    normalization = payload1["normalization"]
    assert normalization["delta_positive_total"] == pytest.approx(0.3)
    assert normalization["delta_negative_total"] == pytest.approx(0.0)
    assert normalization["delta_total"] == pytest.approx(0.3)
    assert normalization["requested_total"] == pytest.approx(0.6)
    assert normalization["current_total"] == pytest.approx(0.3)
    assert normalization["available_delta"] == pytest.approx(0.2)
    assert normalization["factor"] == pytest.approx(2.0 / 3.0)
    assert payload2["normalization"]["factor"] == pytest.approx(normalization["factor"])
    for order in normalized_orders:
        payload = order.meta["payload"]
        assert payload["normalized"] is True
        assert payload["target_weight"] >= 0.0
        assert payload.get("delta_weight") is not None
    assert payload1["target_weight"] == pytest.approx(0.5)
    assert payload2["target_weight"] == pytest.approx(0.5)
    assert payload1["delta_weight"] == pytest.approx(0.2)
    assert payload2["delta_weight"] == pytest.approx(0.0)
    assert len(worker._pending_weight) == 2
