from __future__ import annotations

import types
from typing import Callable

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


def _make_worker(
    monkeypatch,
    *,
    execution_mode: str = "order",
    now_ms: int | Callable[[], int] = 5000,
    throttle_cfg: types.SimpleNamespace | None = None,
    ws_dedup_timeframe_ms: int = 60_000,
):
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

    if callable(now_ms):
        monkeypatch.setattr(clock, "now_ms", now_ms)
    else:
        monkeypatch.setattr(clock, "now_ms", lambda: now_ms)
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
        ws_dedup_timeframe_ms=ws_dedup_timeframe_ms,
        idempotency_cache_size=4,
        execution_mode=execution_mode,
        throttle_cfg=throttle_cfg,
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


def test_emit_bypasses_ttl_when_timeframe_zero(monkeypatch) -> None:
    worker, logger, publish_calls, _executor_calls, _metrics = _make_worker(
        monkeypatch, execution_mode="bar", ws_dedup_timeframe_ms=0
    )

    ttl_stage_calls: list[tuple] = []
    monkeypatch.setattr(
        service_signal_runner.monitoring,
        "inc_stage",
        lambda *args, **kwargs: ttl_stage_calls.append(args),
    )

    order = _make_order("sig-zero", created_ts_ms=4000)
    assert worker._emit(order, "BTCUSDT", 3500) is True

    assert len(publish_calls) == 1
    assert any("TTL_BYPASSED" in msg for msg, *_ in logger.messages)
    assert ttl_stage_calls == []


def test_bar_queue_order_expires_after_bar_ttl(monkeypatch) -> None:
    bar_close_ms = 1_000_000
    late_created_ms = bar_close_ms + 50_000
    now_value = {"ms": late_created_ms}

    def _now_ms() -> int:
        return now_value["ms"]

    queue_cfg = types.SimpleNamespace(ttl_ms=3_600_000, max_items=10)
    throttle_cfg = types.SimpleNamespace(
        enabled=True,
        global_=types.SimpleNamespace(rps=0.0, burst=1.0),
        symbol=types.SimpleNamespace(rps=0.0, burst=1.0),
        mode="queue",
        queue=queue_cfg,
    )

    worker, logger, publish_calls, _executor_calls, metrics = _make_worker(
        monkeypatch, execution_mode="bar", now_ms=_now_ms, throttle_cfg=throttle_cfg
    )

    absolute_metric = DummyMetric()
    drop_metric = DummyMetric()

    monkeypatch.setattr(service_signal_runner.monitoring, "inc_stage", lambda *a, **k: None)
    monkeypatch.setattr(
        service_signal_runner.monitoring, "signal_absolute_count", absolute_metric
    )
    monkeypatch.setattr(service_signal_runner, "pipeline_stage_drop_count", drop_metric)
    monkeypatch.setattr(service_signal_runner, "log_drop", lambda *a, **k: None)

    order = _make_order("sig-ttl", created_ts_ms=late_created_ms)
    result = worker.publish_decision(order, "BTCUSDT", bar_close_ms)
    assert result.action == "queue"
    assert worker._queue is not None and len(worker._queue) == 1
    assert len(publish_calls) == 0
    assert metrics["published"].count == 0

    now_value["ms"] = bar_close_ms + 70_000

    worker._global_bucket.tokens = worker._global_bucket.burst
    symbol_bucket = worker._symbol_buckets["BTCUSDT"]
    symbol_bucket.tokens = symbol_bucket.burst

    emitted = worker._drain_queue()
    assert emitted == []
    assert worker._queue is not None and len(worker._queue) == 0
    assert len(publish_calls) == 0
    assert absolute_metric.count == 1
    assert drop_metric.count == 1
    assert any("TTL_EXPIRED_PUBLISH" in msg for msg, *_ in logger.messages)
