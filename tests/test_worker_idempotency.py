from __future__ import annotations

import types

import clock
import service_signal_runner
from service_signal_runner import _Worker  # type: ignore


class DummyMetric:
    def __init__(self) -> None:
        self.label_calls: list[tuple[str, ...]] = []
        self.count = 0
        self.observations: list[float] = []

    def labels(self, *labels: str) -> "DummyMetric":
        self.label_calls.append(tuple(str(label) for label in labels))
        return self

    def inc(self, *args, **kwargs) -> None:
        self.count += 1

    def observe(self, value: float) -> None:
        self.observations.append(float(value))


class DummyLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, tuple, dict]] = []

    def info(self, msg, *args, **kwargs):
        self.messages.append((msg, args, kwargs))

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


def _make_order(signal_id: str, created_ts_ms: int) -> types.SimpleNamespace:
    payload = {"kind": "target_weight", "target_weight": 0.25}
    meta = {"signal_id": signal_id, "payload": payload}
    return types.SimpleNamespace(
        created_ts_ms=created_ts_ms,
        meta=meta,
        score=1.0,
        features_hash="abc123",
        side="buy",
    )


def _make_worker(monkeypatch):
    fp = types.SimpleNamespace(spread_ttl_ms=0)
    policy = types.SimpleNamespace()
    logger = DummyLogger()
    executor_calls: list[types.SimpleNamespace] = []

    def _submit(order):
        executor_calls.append(order)

    executor = types.SimpleNamespace(submit=_submit)

    published_metric = DummyMetric()
    age_metric = DummyMetric()
    skipped_metric = DummyMetric()

    monkeypatch.setattr(clock, "now_ms", lambda: 5000)
    monkeypatch.setattr(
        service_signal_runner.monitoring, "signal_published_count", published_metric
    )
    monkeypatch.setattr(service_signal_runner.monitoring, "age_at_publish_ms", age_metric)
    monkeypatch.setattr(
        service_signal_runner.monitoring,
        "signal_idempotency_skipped_count",
        skipped_metric,
    )

    publish_calls: list[tuple] = []

    def _publish(symbol, bar_close_ms, payload, callback, **kwargs):
        publish_calls.append((symbol, bar_close_ms, payload, kwargs))
        return True

    monkeypatch.setattr(service_signal_runner, "publish_signal_envelope", _publish)

    worker = _Worker(
        fp,
        policy,
        logger,
        executor,
        enforce_closed_bars=False,
        ws_dedup_timeframe_ms=60_000,
        idempotency_cache_size=4,
    )

    return worker, logger, publish_calls, executor_calls, {
        "published": published_metric,
        "age": age_metric,
        "skipped": skipped_metric,
    }


def test_emit_skips_duplicate_idempotency(monkeypatch) -> None:
    worker, logger, publish_calls, _executor_calls, metrics = _make_worker(monkeypatch)

    first = _make_order("sig-1", created_ts_ms=4000)
    assert worker._emit(first, "BTCUSDT", 3500) is True
    assert len(publish_calls) == 1
    assert metrics["published"].count == 1
    assert metrics["skipped"].count == 0

    duplicate = _make_order("sig-1", created_ts_ms=4500)
    assert worker._emit(duplicate, "BTCUSDT", 3500) is False
    assert len(publish_calls) == 1
    assert metrics["skipped"].count == 1
    assert logger.messages
    last_msg, last_args, _ = logger.messages[-1]
    assert "SKIP_DUPLICATE" in last_msg
    assert last_args
    assert last_args[0]["idempotency_key"] == "sig-1"


def test_emit_accepts_new_idempotency_key(monkeypatch) -> None:
    worker, _logger, publish_calls, _executor_calls, metrics = _make_worker(monkeypatch)

    first = _make_order("sig-1", created_ts_ms=4000)
    second = _make_order("sig-2", created_ts_ms=4500)

    assert worker._emit(first, "BTCUSDT", 3500) is True
    assert worker._emit(second, "BTCUSDT", 3500) is True

    assert len(publish_calls) == 2
    assert metrics["published"].count == 2
    assert metrics["skipped"].count == 0
