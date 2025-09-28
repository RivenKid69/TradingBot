from datetime import datetime, timezone
from types import SimpleNamespace

from service_signal_runner import _Worker


def _make_worker() -> _Worker:
    worker = _Worker.__new__(_Worker)
    worker._weights = {}
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
