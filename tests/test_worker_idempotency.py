from __future__ import annotations

import types
from typing import Any, Callable
from decimal import Decimal

import pytest

from core_config import SpotCostConfig
from core_models import Bar, Order, OrderType, Side
from pipeline import PipelineConfig, PipelineStageConfig

import clock
import service_signal_runner
from impl_bar_executor import BarExecutor
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

    def set(self, value: float) -> None:
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
    execution_mode: str = "bar",
    now_ms: int | Callable[[], int] = 5000,
    throttle_cfg: types.SimpleNamespace | None = None,
    ws_dedup_timeframe_ms: int = 60_000,
    bar_timeframe_ms: int | None = None,
    executor: Any | None = None,
):
    fp = types.SimpleNamespace(spread_ttl_ms=0)
    policy = types.SimpleNamespace()
    logger = DummyLogger()
    executor_calls: list[types.SimpleNamespace] = []

    def _submit(order):
        executor_calls.append(order)

    if executor is None:
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
    monkeypatch.setattr(
        service_signal_runner.monitoring,
        "record_signals",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        service_signal_runner.monitoring,
        "alert_zero_signals",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        service_signal_runner.monitoring,
        "signal_error_rate",
        DummyMetric(),
        raising=False,
    )

    publish_calls: list[tuple] = []
    dispatch_calls: list[Any] = []

    def _publish(symbol, bar_close_ms, payload, callback, **kwargs):
        publish_calls.append((symbol, bar_close_ms, payload, kwargs))
        callback({"symbol": symbol, "bar_close_ms": bar_close_ms, "payload": payload})
        return True

    def _dispatch(payload):
        dispatch_calls.append(payload)

    monkeypatch.setattr(service_signal_runner, "publish_signal_envelope", _publish)

    worker = _Worker(
        fp,
        policy,
        logger,
        executor,
        enforce_closed_bars=False,
        ws_dedup_timeframe_ms=ws_dedup_timeframe_ms,
        bar_timeframe_ms=bar_timeframe_ms,
        idempotency_cache_size=4,
        execution_mode=execution_mode,
        throttle_cfg=throttle_cfg,
        signal_dispatcher=_dispatch,
    )

    return worker, logger, publish_calls, executor_calls, {
        "published": published_metric,
        "age": age_metric,
        "skipped": skipped_metric,
    }, dispatch_calls


def _make_bar_executor_policy_worker(monkeypatch, executor: BarExecutor) -> _Worker:
    worker, _logger, _publish_calls, _executor_calls, _metrics, _dispatch_calls = _make_worker(
        monkeypatch,
        execution_mode="bar",
        ws_dedup_timeframe_ms=0,
        bar_timeframe_ms=0,
        executor=executor,
    )
    return worker


def test_emit_skips_duplicate_idempotency(monkeypatch) -> None:
    (
        worker,
        logger,
        publish_calls,
        _executor_calls,
        metrics,
        dispatch_calls,
    ) = _make_worker(monkeypatch)

    first = _make_order("sig-1", created_ts_ms=4000)
    bar_open_ms = 3500
    bar_close_ms = bar_open_ms + worker._ws_dedup_timeframe_ms
    assert worker._emit(first, "BTCUSDT", bar_close_ms, bar_open_ms=bar_open_ms) is True
    assert len(publish_calls) == 1
    assert metrics["published"].count == 1
    assert metrics["skipped"].count == 0
    assert len(dispatch_calls) == 1

    duplicate = _make_order("sig-1", created_ts_ms=4500)
    assert (
        worker._emit(duplicate, "BTCUSDT", bar_close_ms, bar_open_ms=bar_open_ms) is False
    )
    assert len(publish_calls) == 1
    assert metrics["skipped"].count == 1
    assert len(dispatch_calls) == 1
    assert logger.messages
    last_msg, last_args, _ = logger.messages[-1]
    assert "SKIP_DUPLICATE" in last_msg
    assert last_args
    assert last_args[0]["idempotency_key"] == "sig-1"


def test_emit_accepts_new_idempotency_key(monkeypatch) -> None:
    (
        worker,
        _logger,
        publish_calls,
        _executor_calls,
        metrics,
        dispatch_calls,
    ) = _make_worker(monkeypatch)

    first = _make_order("sig-1", created_ts_ms=4000)
    second = _make_order("sig-2", created_ts_ms=4500)

    bar_open_ms = 3500
    bar_close_ms = bar_open_ms + worker._ws_dedup_timeframe_ms

    assert worker._emit(first, "BTCUSDT", bar_close_ms, bar_open_ms=bar_open_ms) is True
    assert worker._emit(second, "BTCUSDT", bar_close_ms, bar_open_ms=bar_open_ms) is True

    assert len(publish_calls) == 2
    assert metrics["published"].count == 2
    assert metrics["skipped"].count == 0
    assert len(dispatch_calls) == 2


def test_emit_bypasses_ttl_when_no_timeframe_available(monkeypatch) -> None:
    (
        worker,
        logger,
        publish_calls,
        _executor_calls,
        _metrics,
        dispatch_calls,
    ) = _make_worker(
        monkeypatch,
        execution_mode="bar",
        ws_dedup_timeframe_ms=0,
        bar_timeframe_ms=0,
    )

    ttl_stage_calls: list[tuple] = []
    monkeypatch.setattr(
        service_signal_runner.monitoring,
        "inc_stage",
        lambda *args, **kwargs: ttl_stage_calls.append(args),
    )

    order = _make_order("sig-zero", created_ts_ms=4000)
    assert worker._emit(order, "BTCUSDT", 3_500, bar_open_ms=3_500) is True

    assert len(publish_calls) == 1
    assert len(dispatch_calls) == 1
    assert any("TTL_BYPASSED" in msg for msg, *_ in logger.messages)
    assert ttl_stage_calls == []


def test_emit_uses_fallback_timeframe_when_ws_timeframe_zero(monkeypatch) -> None:
    fallback_timeframe = 60_000
    (
        worker,
        logger,
        publish_calls,
        _executor_calls,
        _metrics,
        dispatch_calls,
    ) = _make_worker(
        monkeypatch,
        execution_mode="bar",
        ws_dedup_timeframe_ms=0,
        bar_timeframe_ms=fallback_timeframe,
    )

    ttl_stage_calls: list[tuple] = []
    monkeypatch.setattr(
        service_signal_runner.monitoring,
        "inc_stage",
        lambda *args, **kwargs: ttl_stage_calls.append(args),
    )

    order = _make_order("sig-zero", created_ts_ms=4000)
    bar_open_ms = 3_500
    bar_close_ms = bar_open_ms + fallback_timeframe

    assert worker._resolve_ttl_timeframe_ms(log_if_invalid=True) == fallback_timeframe
    assert worker._emit(order, "BTCUSDT", bar_close_ms, bar_open_ms=bar_open_ms) is True

    assert len(publish_calls) == 1
    assert len(dispatch_calls) == 1
    assert not any("TTL_BYPASSED" in msg for msg, *_ in logger.messages)
    assert ttl_stage_calls


def test_emit_publishes_with_ttl_disabled(monkeypatch) -> None:
    actual_timeframe = 60_000
    ttl_timeframe = 1_000
    bar_open_ms = 1_000_000
    now_state = {"ms": bar_open_ms + actual_timeframe - 1}

    (
        worker,
        _logger,
        _publish_calls,
        _executor_calls,
        _metrics,
        dispatch_calls,
    ) = _make_worker(
        monkeypatch,
        execution_mode="bar",
        ws_dedup_timeframe_ms=ttl_timeframe,
        bar_timeframe_ms=actual_timeframe,
        now_ms=lambda: now_state["ms"],
    )

    worker._pipeline_cfg = PipelineConfig(
        stages={"ttl": PipelineStageConfig(enabled=False)}
    )

    assert worker._resolve_ttl_timeframe_ms(log_if_invalid=False) == ttl_timeframe
    assert worker._bar_timeframe_ms == actual_timeframe

    publish_events: list[dict[str, int]] = []

    def _publish(symbol, bar_close_ms, payload, callback, *, expires_at_ms, **kwargs):
        current = clock.now_ms()
        publish_events.append(
            {
                "symbol": symbol,
                "bar_close_ms": bar_close_ms,
                "expires_at_ms": expires_at_ms,
                "now": current,
            }
        )
        if current >= expires_at_ms:
            return False
        callback({"symbol": symbol, "payload": payload})
        return True

    monkeypatch.setattr(service_signal_runner, "publish_signal_envelope", _publish)
    monkeypatch.setattr(service_signal_runner.signal_bus, "ENABLED", True, raising=False)

    order = _make_order("sig-ttl-disabled", created_ts_ms=now_state["ms"])
    bar_close_ms = bar_open_ms + actual_timeframe

    assert worker._emit(order, "BTCUSDT", bar_close_ms, bar_open_ms=bar_open_ms) is True
    assert publish_events
    event = publish_events[0]
    assert event["bar_close_ms"] == bar_close_ms
    assert event["expires_at_ms"] >= bar_close_ms
    assert event["now"] < event["expires_at_ms"]
    assert dispatch_calls


def test_process_uses_executor_bar_price(monkeypatch) -> None:
    executor = BarExecutor(bar_price="open")
    worker = _make_bar_executor_policy_worker(monkeypatch, executor)
    worker._portfolio_equity = 100.0
    worker._weights["BTCUSDT"] = 0.5
    worker._pipeline_cfg = PipelineConfig(enabled=False)

    bar = Bar(
        ts=1_000,
        symbol="BTCUSDT",
        open=Decimal("10"),
        high=Decimal("11"),
        low=Decimal("9"),
        close=Decimal("12"),
    )

    worker.process(bar)

    assert worker._bar_price_field == "open"
    assert worker._last_prices["BTCUSDT"] == pytest.approx(10.0)
    assert worker._positions["BTCUSDT"] == pytest.approx(5.0)


def test_publish_decision_uses_fallback_timeframe_for_close(monkeypatch) -> None:
    fallback_timeframe = 120_000
    (
        worker,
        _logger,
        publish_calls,
        _executor_calls,
        _metrics,
        _dispatch_calls,
    ) = _make_worker(
        monkeypatch,
        execution_mode="bar",
        ws_dedup_timeframe_ms=0,
        bar_timeframe_ms=fallback_timeframe,
    )

    ttl_stage_calls: list[tuple] = []
    monkeypatch.setattr(
        service_signal_runner.monitoring,
        "inc_stage",
        lambda *args, **kwargs: ttl_stage_calls.append(args),
    )

    order = _make_order("sig-close", created_ts_ms=4000)
    bar_open_ms = 5_000

    result = worker.publish_decision(order, "BTCUSDT", bar_open_ms)

    assert result.action == "pass"
    assert publish_calls
    _, recorded_close_ms, *_ = publish_calls[0]
    assert recorded_close_ms == bar_open_ms + fallback_timeframe
    assert ttl_stage_calls


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

    (
        worker,
        logger,
        publish_calls,
        _executor_calls,
        metrics,
        _dispatch_calls,
    ) = _make_worker(
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
    result = worker.publish_decision(
        order,
        "BTCUSDT",
        bar_close_ms - worker._ws_dedup_timeframe_ms,
        bar_close_ms=bar_close_ms,
    )
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


def test_emit_allows_until_next_open(monkeypatch) -> None:
    timeframe_ms = 60_000
    bar_open_ms = timeframe_ms * 20
    bar_close_ms = bar_open_ms + timeframe_ms
    now_state = {"ms": bar_close_ms}

    def _now_ms() -> int:
        return now_state["ms"]

    (
        worker,
        logger,
        publish_calls,
        _executor_calls,
        metrics,
        dispatch_calls,
    ) = _make_worker(
        monkeypatch,
        execution_mode="bar",
        now_ms=_now_ms,
        ws_dedup_timeframe_ms=timeframe_ms,
    )
    assert worker._ws_dedup_timeframe_ms == timeframe_ms
    assert worker._resolve_ttl_timeframe_ms(log_if_invalid=True) == timeframe_ms

    order = _make_order("sig-fresh", created_ts_ms=now_state["ms"])
    result = worker._emit(
        order,
        "BTCUSDT",
        bar_close_ms,
        bar_open_ms=bar_open_ms,
    )
    assert result is True, logger.messages
    assert len(publish_calls) == 1
    assert len(dispatch_calls) == 1
    assert metrics["published"].count == 1
    assert not any("TTL_EXPIRED" in msg for msg, *_ in logger.messages)


def test_dispatch_signal_envelope_executes_bar_order(monkeypatch) -> None:
    (
        worker,
        _logger,
        _publish_calls,
        _executor_calls,
        _metrics,
        dispatch_calls,
    ) = _make_worker(monkeypatch, execution_mode="bar")

    class RecordingExecutor:
        def __init__(self) -> None:
            self.executed: list[Any] = []

        def execute(self, order: Any) -> None:
            self.executed.append(order)

    executor = RecordingExecutor()
    worker._executor = executor

    monkeypatch.setattr(service_signal_runner.signal_bus, "ENABLED", True, raising=False)

    economics = {
        "edge_bps": 10.0,
        "cost_bps": 1.0,
        "net_bps": 9.0,
        "turnover_usd": 100.0,
        "act_now": True,
        "impact": 0.0,
        "impact_mode": "model",
    }
    order_payload = {"target_weight": 0.5, "economics": economics}
    bar = Bar(
        ts=1,
        symbol="BTCUSDT",
        open=Decimal("100"),
        high=Decimal("100"),
        low=Decimal("100"),
        close=Decimal("100"),
        volume_base=Decimal("0"),
        volume_quote=Decimal("0"),
    )
    order = types.SimpleNamespace(
        ts=bar.ts,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={"payload": order_payload},
        client_order_id="test-order",
    )
    order.created_ts_ms = bar.ts

    def _closed_guard(**_kwargs: Any) -> service_signal_runner.PipelineResult:
        return service_signal_runner.PipelineResult(
            action="pass", stage=service_signal_runner.Stage.CLOSED_BAR
        )

    def _policy_decide(*_args: Any, **_kwargs: Any) -> service_signal_runner.PipelineResult:
        return service_signal_runner.PipelineResult(
            action="pass",
            stage=service_signal_runner.Stage.POLICY,
            decision=[order],
        )

    def _apply_risk(
        *_args: Any, **_kwargs: Any
    ) -> service_signal_runner.PipelineResult:
        return service_signal_runner.PipelineResult(
            action="pass",
            stage=service_signal_runner.Stage.RISK,
            decision=[order],
        )

    monkeypatch.setattr(service_signal_runner, "closed_bar_guard", _closed_guard)
    monkeypatch.setattr(service_signal_runner, "policy_decide", _policy_decide)
    monkeypatch.setattr(service_signal_runner, "apply_risk", _apply_risk)

    def _allow_signal_quality(
        self: service_signal_runner._Worker,
        bar: Any,
        *,
        skip_metrics: bool = False,
    ) -> tuple[bool, dict[str, Any]]:
        return True, {}

    def _allow_windows(
        self: service_signal_runner._Worker,
        ts_ms: int,
        symbol: str,
        *,
        stage_cfg: Any = None,
    ) -> tuple[service_signal_runner.PipelineResult, str | None]:
        return (
            service_signal_runner.PipelineResult(
                action="pass", stage=service_signal_runner.Stage.WINDOWS
            ),
            None,
        )

    worker._apply_signal_quality_filter = types.MethodType(
        _allow_signal_quality, worker
    )
    worker._extract_features = types.MethodType(
        lambda self, bar, *, skip_metrics=False: {}, worker
    )
    worker._evaluate_no_trade_windows = types.MethodType(
        _allow_windows,
        worker,
    )

    bar = Bar(
        ts=1,
        symbol="BTCUSDT",
        open=Decimal("100"),
        high=Decimal("100"),
        low=Decimal("100"),
        close=Decimal("100"),
        volume_base=Decimal("0"),
        volume_quote=Decimal("0"),
    )

    emitted = worker.process(bar)

    assert emitted == [order]
    assert executor.executed == [order]
    assert getattr(order, "_bar_dispatched", False) is True
    assert order.meta["bar"] is bar
    assert dispatch_calls, "Expected signal dispatcher to receive the envelope"


def test_worker_propagates_adv_to_executor(monkeypatch) -> None:
    (
        worker,
        _logger,
        publish_calls,
        _executor_calls,
        _metrics,
        dispatch_calls,
    ) = _make_worker(monkeypatch, execution_mode="bar")

    class RecordingBarExecutor(BarExecutor):
        def __init__(self) -> None:
            super().__init__(
                run_id="test",
                bar_price="close",
                cost_config=SpotCostConfig(),
                default_equity_usd=1_000.0,
            )
            self.last_report = None
            self.last_error: Any | None = None

        def execute(self, order: Any):  # type: ignore[override]
            try:
                report = super().execute(order)
            except Exception as exc:  # pragma: no cover - propagate for visibility
                self.last_error = exc
                raise
            self.last_report = report
            return report

    executor = RecordingBarExecutor()
    worker._executor = executor

    monkeypatch.setattr(service_signal_runner.signal_bus, "ENABLED", True, raising=False)

    adv_quote = 2_000.0
    economics = {
        "edge_bps": 20.0,
        "cost_bps": 5.0,
        "net_bps": 15.0,
        "turnover_usd": 400.0,
        "act_now": True,
        "impact": 0.0,
        "impact_mode": "none",
        "adv_quote": adv_quote,
    }
    order_payload = {
        "delta_weight": 0.4,
        "economics": economics,
        "max_participation": 0.05,
    }
    bar = Bar(
        ts=1,
        symbol="BTCUSDT",
        open=Decimal("100"),
        high=Decimal("100"),
        low=Decimal("100"),
        close=Decimal("100"),
        volume_base=Decimal("0"),
        volume_quote=Decimal("0"),
    )
    order = types.SimpleNamespace(
        ts=bar.ts,
        symbol="BTCUSDT",
        side=Side.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        price=None,
        meta={"payload": order_payload},
        client_order_id="adv-test-order",
    )
    order.created_ts_ms = bar.ts

    def _closed_guard(**_kwargs: Any) -> service_signal_runner.PipelineResult:
        return service_signal_runner.PipelineResult(
            action="pass", stage=service_signal_runner.Stage.CLOSED_BAR
        )

    def _policy_decide(*_args: Any, **_kwargs: Any) -> service_signal_runner.PipelineResult:
        return service_signal_runner.PipelineResult(
            action="pass",
            stage=service_signal_runner.Stage.POLICY,
            decision=[order],
        )

    def _apply_risk(*_args: Any, **_kwargs: Any) -> service_signal_runner.PipelineResult:
        return service_signal_runner.PipelineResult(
            action="pass",
            stage=service_signal_runner.Stage.RISK,
            decision=[order],
        )

    monkeypatch.setattr(service_signal_runner, "closed_bar_guard", _closed_guard)
    monkeypatch.setattr(service_signal_runner, "policy_decide", _policy_decide)
    monkeypatch.setattr(service_signal_runner, "apply_risk", _apply_risk)

    def _allow_signal_quality(
        self: service_signal_runner._Worker,
        bar: Any,
        *,
        skip_metrics: bool = False,
    ) -> tuple[bool, dict[str, Any]]:
        return True, {}

    def _allow_windows(
        self: service_signal_runner._Worker,
        ts_ms: int,
        symbol: str,
        *,
        stage_cfg: Any = None,
    ) -> tuple[service_signal_runner.PipelineResult, str | None]:
        return (
            service_signal_runner.PipelineResult(
                action="pass", stage=service_signal_runner.Stage.WINDOWS
            ),
            None,
        )

    worker._apply_signal_quality_filter = types.MethodType(
        _allow_signal_quality, worker
    )
    worker._extract_features = types.MethodType(
        lambda self, bar, *, skip_metrics=False: {}, worker
    )
    worker._evaluate_no_trade_windows = types.MethodType(
        _allow_windows,
        worker,
    )

    bar = Bar(
        ts=1,
        symbol="BTCUSDT",
        open=Decimal("100"),
        high=Decimal("100"),
        low=Decimal("100"),
        close=Decimal("100"),
        volume_base=Decimal("0"),
        volume_quote=Decimal("0"),
    )

    emitted = worker.process(bar)

    assert emitted == [order]
    assert order.meta["bar"] is bar
    assert order.meta["adv_quote"] == pytest.approx(adv_quote)
    assert dispatch_calls, "Expected signal dispatcher to receive the envelope"
    dispatched_payload = dispatch_calls[0]["payload"]
    assert dispatched_payload["economics"]["adv_quote"] == pytest.approx(adv_quote)

    report = executor.last_report
    assert report is not None, getattr(executor, "last_error", None)
    assert report.meta["adv_quote"] == pytest.approx(adv_quote)
    decision = report.meta["decision"]
    assert decision["impact_mode"] == "model"
    instructions = report.meta["instructions"]
    assert len(instructions) == 4


def test_process_uses_true_bar_boundaries(monkeypatch) -> None:
    timeframe_ms = 120_000
    dedup_timeframe_ms = 90_000
    (
        worker,
        _logger,
        _publish_calls,
        _executor_calls,
        _metrics,
        _dispatch_calls,
    ) = _make_worker(
        monkeypatch,
        ws_dedup_timeframe_ms=dedup_timeframe_ms,
        bar_timeframe_ms=timeframe_ms,
    )

    worker._ws_dedup_enabled = True
    worker._pipeline_cfg = PipelineConfig()

    monkeypatch.setattr(service_signal_runner.monitoring, "inc_stage", lambda *a, **k: None)
    monkeypatch.setattr(service_signal_runner.monitoring, "inc_reason", lambda *a, **k: None)
    monkeypatch.setattr(service_signal_runner.monitoring, "record_signals", lambda *a, **k: None)
    monkeypatch.setattr(service_signal_runner.monitoring, "signal_error_rate", DummyMetric())
    monkeypatch.setattr(service_signal_runner.monitoring, "ws_dup_skipped_count", DummyMetric())
    monkeypatch.setattr(service_signal_runner.monitoring, "signal_boundary_count", DummyMetric())
    monkeypatch.setattr(service_signal_runner.monitoring, "ttl_expired_boundary_count", DummyMetric())
    monkeypatch.setattr(service_signal_runner, "pipeline_stage_drop_count", DummyMetric())
    monkeypatch.setattr(service_signal_runner, "skipped_incomplete_bars", DummyMetric())

    monkeypatch.setattr(service_signal_runner.signal_bus, "ENABLED", True, raising=False)

    bar_ts = 1_700_000
    bar = Bar(
        ts=bar_ts,
        symbol="BTCUSDT",
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("95"),
        close=Decimal("102"),
    )

    order = types.SimpleNamespace(meta={}, symbol=bar.symbol)

    def _closed_guard(*args, **kwargs):
        return service_signal_runner.PipelineResult(
            action="pass", stage=service_signal_runner.Stage.CLOSED_BAR
        )

    def _policy_decide(*_args, **_kwargs):
        return service_signal_runner.PipelineResult(
            action="pass",
            stage=service_signal_runner.Stage.POLICY,
            decision=[order],
        )

    def _apply_risk(*_args, **_kwargs):
        return service_signal_runner.PipelineResult(
            action="pass",
            stage=service_signal_runner.Stage.RISK,
            decision=[order],
        )

    worker._apply_signal_quality_filter = types.MethodType(
        lambda self, bar, *, skip_metrics=False: (True, {}),
        worker,
    )
    worker._extract_features = types.MethodType(
        lambda self, bar, *, skip_metrics=False: {},
        worker,
    )
    worker._evaluate_no_trade_windows = types.MethodType(
        lambda self, ts_ms, symbol, *, stage_cfg=None: (
            service_signal_runner.PipelineResult(
                action="pass", stage=service_signal_runner.Stage.WINDOWS
            ),
            None,
        ),
        worker,
    )

    monkeypatch.setattr(service_signal_runner, "closed_bar_guard", _closed_guard)
    monkeypatch.setattr(service_signal_runner, "policy_decide", _policy_decide)
    monkeypatch.setattr(service_signal_runner, "apply_risk", _apply_risk)

    ttl_calls: list[dict[str, int]] = []

    def _check_ttl(*, bar_close_ms: int, now_ms: int, timeframe_ms: int):
        ttl_calls.append(
            {
                "bar_close_ms": bar_close_ms,
                "now_ms": now_ms,
                "timeframe_ms": timeframe_ms,
            }
        )
        return True, bar_close_ms, None

    monkeypatch.setattr(service_signal_runner, "check_ttl", _check_ttl)

    skip_calls: list[tuple[str, int]] = []
    update_calls: list[tuple[str, int]] = []

    def _should_skip(symbol: str, bar_close_ms: int) -> bool:
        skip_calls.append((symbol, bar_close_ms))
        return False

    def _update(symbol: str, bar_close_ms: int) -> None:
        update_calls.append((symbol, bar_close_ms))

    monkeypatch.setattr(service_signal_runner.signal_bus, "should_skip", _should_skip)
    monkeypatch.setattr(service_signal_runner.signal_bus, "update", _update)

    now_ms = bar_ts + 1_000
    monkeypatch.setattr(clock, "now_ms", lambda: now_ms)

    publish_calls: list[dict[str, int]] = []

    def _publish_decision(
        self: service_signal_runner._Worker,
        o: Any,
        symbol: str,
        bar_open_ms: int,
        *,
        bar_close_ms: int,
        stage_cfg: PipelineStageConfig | None = None,
    ) -> service_signal_runner.PipelineResult:
        publish_calls.append(
            {
                "symbol": symbol,
                "bar_open_ms": bar_open_ms,
                "bar_close_ms": bar_close_ms,
            }
        )
        return service_signal_runner.PipelineResult(
            action="pass", stage=service_signal_runner.Stage.PUBLISH, decision=o
        )

    worker.publish_decision = types.MethodType(_publish_decision, worker)

    emitted = worker.process(bar)

    assert emitted == [order]
    expected_open_ms = bar_ts - timeframe_ms
    assert publish_calls == [
        {
            "symbol": "BTCUSDT",
            "bar_open_ms": expected_open_ms,
            "bar_close_ms": bar_ts,
        }
    ]
    assert ttl_calls == [
        {
            "bar_close_ms": bar_ts,
            "now_ms": now_ms,
            "timeframe_ms": dedup_timeframe_ms,
        }
    ]
    assert skip_calls == [("BTCUSDT", bar_ts)]
    assert update_calls == [("BTCUSDT", bar_ts)]

