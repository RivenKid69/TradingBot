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

    payload, valid_until_ms = worker._build_envelope_payload(order, "BTCUSDT")

    expected = int(datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc).timestamp() * 1000)
    assert payload["valid_until_ms"] == expected
    assert valid_until_ms == expected
    assert payload["kind"] == "target_weight"
    assert payload["target_weight"] == 0.2


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

    payload, _ = worker._build_envelope_payload(order, "BTCUSDT")

    assert payload["target_weight"] == pytest.approx(0.1)
    assert payload["reject_reason"] == "target_weight_out_of_bounds"
    assert payload["requested_target_weight"] == pytest.approx(1.5)


def test_build_envelope_payload_flags_rejected_delta() -> None:
    worker = _make_worker()
    worker._weights["BTCUSDT"] = 0.2
    order = SimpleNamespace(meta={"payload": {"delta_weight": -0.5}})

    payload, _ = worker._build_envelope_payload(order, "BTCUSDT")

    assert payload["delta_weight"] == pytest.approx(0.0)
    assert payload["reject_reason"] == "delta_weight_out_of_bounds"
    assert payload["requested_delta_weight"] == pytest.approx(-0.5)
