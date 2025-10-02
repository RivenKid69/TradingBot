import types
from types import SimpleNamespace

import clock
import service_signal_runner
from pipeline import Reason, Stage
from service_signal_runner import _Worker


class DummyMetric:
    def __init__(self) -> None:
        self.label_calls: list[tuple[str, ...]] = []
        self.count = 0

    def labels(self, *labels: str) -> "DummyMetric":
        self.label_calls.append(tuple(str(label) for label in labels))
        return self

    def inc(self, *args, **kwargs) -> None:
        self.count += 1


class DropRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str]] = []

    def __call__(self, envelope, reason: str) -> None:  # pragma: no cover - signature helper
        self.calls.append((envelope, reason))


class DummyLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, tuple, dict]] = []

    def info(self, *args, **kwargs):  # pragma: no cover - logging helper
        self.messages.append((args[0] if args else "", args[1:], kwargs))

    def warning(self, *args, **kwargs):  # pragma: no cover
        return None

    def error(self, *args, **kwargs):  # pragma: no cover
        return None


def _make_worker(
    monkeypatch,
    *,
    execution_mode: str = "bar",
    throttle_cfg: types.SimpleNamespace | None = None,
    ws_timeframe_ms: int = 60_000,
) -> _Worker:
    fp = SimpleNamespace(timeframe_ms=ws_timeframe_ms, spread_ttl_ms=0)
    policy = SimpleNamespace(timeframe_ms=ws_timeframe_ms)
    logger = DummyLogger()

    executor_calls: list[SimpleNamespace] = []

    def _submit(order):
        executor_calls.append(order)

    executor = SimpleNamespace(submit=_submit, execute=lambda order: None)
    monkeypatch.setattr(clock, "now_ms", lambda: 1_000_000)

    worker = _Worker(
        fp,
        policy,
        logger,
        executor,
        guards=None,
        enforce_closed_bars=False,
        ws_dedup_timeframe_ms=ws_timeframe_ms,
        bar_timeframe_ms=ws_timeframe_ms,
        throttle_cfg=throttle_cfg,
        execution_mode=execution_mode,
    )
    return worker


def _make_order(signal_id: str = "sig", created_ts_ms: int = 999_000) -> SimpleNamespace:
    payload = {"kind": "target_weight", "target_weight": 0.25}
    meta = {"signal_id": signal_id, "payload": payload}
    return SimpleNamespace(
        created_ts_ms=created_ts_ms,
        meta=meta,
        score=1.0,
        features_hash="abc123",
        side="buy",
    )


def test_bar_mode_risk_drop_logs_and_counts(monkeypatch) -> None:
    worker = _make_worker(monkeypatch, execution_mode="bar")
    drop_metric = DummyMetric()
    drop_recorder = DropRecorder()

    monkeypatch.setattr(service_signal_runner, "pipeline_stage_drop_count", drop_metric)
    monkeypatch.setattr(service_signal_runner, "log_drop", drop_recorder)

    worker._guards = SimpleNamespace(apply=lambda ts, sym, decisions: ([], "POSITION_LIMIT"))

    order = _make_order()
    dropped = worker._emit(order, "BTCUSDT", 1_001_000)

    assert dropped is False
    assert len(drop_recorder.calls) == 1
    envelope, reason = drop_recorder.calls[0]
    assert reason == "POSITION_LIMIT"
    assert envelope.symbol == "BTCUSDT"
    assert drop_metric.count == 1
    assert drop_metric.label_calls[0] == (
        "BTCUSDT",
        Stage.PUBLISH.name,
        Reason.RISK_POSITION.name,
    )


def test_throttle_drop_logs_and_counts(monkeypatch) -> None:
    throttle_cfg = SimpleNamespace(
        enabled=True,
        global_=SimpleNamespace(rps=0.0, burst=1.0),
        symbol=SimpleNamespace(rps=0.0, burst=1.0),
        mode="drop",
        queue=SimpleNamespace(ttl_ms=1_000, max_items=10),
    )
    worker = _make_worker(monkeypatch, execution_mode="bar", throttle_cfg=throttle_cfg)

    drop_metric = DummyMetric()
    throttle_dropped = DummyMetric()
    drop_recorder = DropRecorder()

    monkeypatch.setattr(service_signal_runner.monitoring, "inc_stage", lambda *a, **k: None)
    monkeypatch.setattr(service_signal_runner, "pipeline_stage_drop_count", drop_metric)
    monkeypatch.setattr(service_signal_runner.monitoring, "throttle_dropped_count", throttle_dropped)
    monkeypatch.setattr(service_signal_runner, "log_drop", drop_recorder)

    worker._acquire_tokens = lambda symbol: (False, "GLOBAL_LIMIT")  # type: ignore[method-assign]

    order = _make_order()
    result = worker.publish_decision(order, "BTCUSDT", 1_000_000, bar_close_ms=1_001_000)

    assert result.action == "drop"
    assert len(drop_recorder.calls) == 1
    envelope, reason = drop_recorder.calls[0]
    assert reason == "GLOBAL_LIMIT"
    assert envelope.symbol == "BTCUSDT"
    assert throttle_dropped.count == 1
    assert throttle_dropped.label_calls[0] == ("BTCUSDT", "GLOBAL_LIMIT")
    assert drop_metric.count == 1
    assert drop_metric.label_calls[0] == (
        "BTCUSDT",
        Stage.THROTTLE.name,
        "GLOBAL_LIMIT",
    )


def test_queue_expiry_logs_and_counts(monkeypatch) -> None:
    throttle_cfg = SimpleNamespace(
        enabled=True,
        global_=SimpleNamespace(rps=0.0, burst=1.0),
        symbol=SimpleNamespace(rps=0.0, burst=1.0),
        mode="queue",
        queue=SimpleNamespace(ttl_ms=1_000, max_items=10),
    )
    worker = _make_worker(monkeypatch, execution_mode="bar", throttle_cfg=throttle_cfg)

    drop_recorder = DropRecorder()
    queue_expired_metric = DummyMetric()
    throttle_dropped_metric = DummyMetric()
    enqueued_metric = DummyMetric()

    monkeypatch.setattr(service_signal_runner.monitoring, "inc_stage", lambda *a, **k: None)
    monkeypatch.setattr(service_signal_runner.monitoring, "throttle_enqueued_count", enqueued_metric)
    monkeypatch.setattr(
        service_signal_runner.monitoring, "throttle_queue_expired_count", queue_expired_metric
    )
    monkeypatch.setattr(
        service_signal_runner.monitoring, "throttle_dropped_count", throttle_dropped_metric
    )
    monkeypatch.setattr(service_signal_runner, "log_drop", drop_recorder)

    monotonic_time = {"value": 0.0}
    monkeypatch.setattr(
        service_signal_runner.time, "monotonic", lambda: monotonic_time["value"]
    )

    worker._acquire_tokens = lambda symbol: (False, "GLOBAL_LIMIT")  # type: ignore[method-assign]

    order = _make_order()
    queued = worker.publish_decision(order, "BTCUSDT", 1_000_000, bar_close_ms=1_001_000)
    assert queued.action == "queue"
    assert worker._queue is not None and len(worker._queue) == 1
    assert enqueued_metric.count == 1

    monotonic_time["value"] = 2.0
    emitted = worker._drain_queue()

    assert emitted == []
    assert len(drop_recorder.calls) == 1
    envelope, reason = drop_recorder.calls[0]
    assert reason == "QUEUE_EXPIRED"
    assert envelope.symbol == "BTCUSDT"
    assert queue_expired_metric.count == 1
    assert queue_expired_metric.label_calls[0] == ("BTCUSDT",)
    assert throttle_dropped_metric.count == 1
    assert throttle_dropped_metric.label_calls[0] == ("BTCUSDT", "QUEUE_EXPIRED")
