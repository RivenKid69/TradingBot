from __future__ import annotations

import logging
import types
from collections.abc import Mapping as MappingABC
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

import clock
import service_signal_runner
from core_models import Bar
from pipeline import PipelineResult, Stage
from api.spot_signals import (
    SpotSignalEconomics,
    SpotSignalEnvelope,
    SpotSignalTargetWeightPayload,
)


class _DummyMetric:
    def __init__(self) -> None:
        self.label_calls: list[tuple[str, ...]] = []
        self.observe_calls: list[tuple[tuple[str, ...], tuple[Any, ...]]] = []

    def labels(self, *labels: str) -> "_DummyMetric":
        self.label_calls.append(tuple(str(label) for label in labels))
        return self

    def inc(self, *args, **kwargs) -> None:  # pragma: no cover - metric helper
        return None

    def set(self, *args, **kwargs) -> None:  # pragma: no cover - metric helper
        return None

    def observe(self, *args, **kwargs) -> None:  # pragma: no cover - metric helper
        labels = self.label_calls[-1] if self.label_calls else tuple()
        self.observe_calls.append((labels, args))
        return None


def _make_monitoring_stub() -> SimpleNamespace:
    stub = SimpleNamespace()
    stub.inc_stage = lambda *args, **kwargs: None
    stub.record_signals = lambda *args, **kwargs: None
    stub.record_fill = lambda *args, **kwargs: None
    stub.record_pnl = lambda *args, **kwargs: None
    stub.inc_reason = lambda *args, **kwargs: None
    stub.alert_zero_signals = lambda *args, **kwargs: None
    stub.signal_error_rate = _DummyMetric()
    stub.ws_dup_skipped_count = _DummyMetric()
    stub.ttl_expired_boundary_count = _DummyMetric()
    stub.signal_boundary_count = _DummyMetric()
    stub.signal_published_count = _DummyMetric()
    stub.age_at_publish_ms = _DummyMetric()
    stub.signal_idempotency_skipped_count = _DummyMetric()
    stub.signal_absolute_count = _DummyMetric()
    stub.throttle_dropped_count = _DummyMetric()
    stub.throttle_enqueued_count = _DummyMetric()
    stub.throttle_queue_expired_count = _DummyMetric()
    return stub


def _make_simple_order(symbol: str, *, created_ts_ms: int) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        side="buy",
        score=0.0,
        features_hash="",
        volume_frac=0.1,
        meta={"created_ts_ms": created_ts_ms, "payload": {}},
        created_ts_ms=created_ts_ms,
    )


def test_process_propagates_open_and_close(monkeypatch) -> None:
    timeframe_ms = 60_000
    bar_close_ms = 1_700_000_120_000
    expected_open_ms = bar_close_ms - timeframe_ms

    monitoring_stub = _make_monitoring_stub()
    monkeypatch.setattr(service_signal_runner, "monitoring", monitoring_stub)
    monkeypatch.setattr(
        service_signal_runner, "pipeline_stage_drop_count", _DummyMetric()
    )
    monkeypatch.setattr(
        service_signal_runner, "skipped_incomplete_bars", _DummyMetric()
    )

    dedup_should_calls: list[tuple[str, int]] = []
    dedup_update_calls: list[tuple[str, int]] = []

    def _should_skip(symbol: str, close_ms: int) -> bool:
        dedup_should_calls.append((symbol, close_ms))
        return False

    def _update(symbol: str, close_ms: int) -> None:
        dedup_update_calls.append((symbol, close_ms))

    monkeypatch.setattr(service_signal_runner.signal_bus, "should_skip", _should_skip)
    monkeypatch.setattr(service_signal_runner.signal_bus, "update", _update)

    ttl_calls: list[tuple[int, int, int]] = []

    def _check_ttl(*, bar_close_ms: int, now_ms: int, timeframe_ms: int):
        ttl_calls.append((bar_close_ms, now_ms, timeframe_ms))
        return True, bar_close_ms, None

    monkeypatch.setattr(service_signal_runner, "check_ttl", _check_ttl)

    class _StubFeaturePipe:
        def __init__(self) -> None:
            self.signal_quality: dict[str, object] = {}
            self.timeframe_ms = timeframe_ms
            self.spread_ttl_ms = 0

        def update(
            self, bar: Bar, *, skip_metrics: bool | None = None
        ) -> dict[str, float]:
            return {"close": float(bar.close)}

    class _StubPolicy:
        def __init__(self) -> None:
            self.timeframe_ms = timeframe_ms

        def decide(self, features, ctx):
            return [SimpleNamespace(symbol=ctx.symbol, meta={}, side="buy")]

        def consume_signal_transitions(self):
            return []

    executor = SimpleNamespace(submit=lambda order: None, execute=lambda order: None)

    worker = service_signal_runner._Worker(
        _StubFeaturePipe(),
        _StubPolicy(),
        logging.getLogger("test-worker"),
        executor,
        enforce_closed_bars=False,
        ws_dedup_enabled=True,
        ws_dedup_timeframe_ms=timeframe_ms,
        bar_timeframe_ms=timeframe_ms,
    )

    monkeypatch.setattr(clock, "now_ms", lambda: bar_close_ms + 1_000)

    publish_calls: list[tuple[int, int]] = []

    def _publish(self, order, symbol, bar_open_ms, *, bar_close_ms=None, stage_cfg=None):
        publish_calls.append((bar_open_ms, int(bar_close_ms)))
        return PipelineResult(action="pass", stage=Stage.PUBLISH, decision=order)

    worker.publish_decision = types.MethodType(_publish, worker)

    bar = Bar(
        ts=bar_close_ms,
        symbol="BTCUSDT",
        open=Decimal("1"),
        high=Decimal("1"),
        low=Decimal("1"),
        close=Decimal("1"),
    )

    worker.process(bar)

    assert publish_calls == [(expected_open_ms, bar_close_ms)]
    assert ttl_calls == [(bar_close_ms, bar_close_ms + 1_000, timeframe_ms)]
    assert dedup_should_calls == [("BTCUSDT", bar_close_ms)]
    assert dedup_update_calls == [("BTCUSDT", bar_close_ms)]


def test_publish_decision_respects_bar_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    timeframe_ms = 60_000
    bar_close_ms = 1_700_000_000_000
    bar_open_ms = bar_close_ms - timeframe_ms
    symbol = "BTCUSDT"

    monitoring_stub = _make_monitoring_stub()
    monkeypatch.setattr(service_signal_runner, "monitoring", monitoring_stub)
    monkeypatch.setattr(service_signal_runner, "pipeline_stage_drop_count", _DummyMetric())
    monkeypatch.setattr(service_signal_runner, "skipped_incomplete_bars", _DummyMetric())
    monkeypatch.setattr(service_signal_runner.signal_bus, "ENABLED", False, raising=False)

    class _StubFeaturePipe:
        spread_ttl_ms = 0

        def __init__(self, timeframe_ms: int) -> None:
            self.timeframe_ms = timeframe_ms

    class _StubPolicy:
        def __init__(self, timeframe_ms: int) -> None:
            self.timeframe_ms = timeframe_ms

        def consume_signal_transitions(self):
            return []

    executor = SimpleNamespace(execute=lambda order: None, submit=lambda order: None)

    worker = service_signal_runner._Worker(
        _StubFeaturePipe(timeframe_ms),
        _StubPolicy(timeframe_ms),
        logging.getLogger("test-worker"),
        executor,
        enforce_closed_bars=False,
        ws_dedup_enabled=True,
        ws_dedup_timeframe_ms=timeframe_ms,
        bar_timeframe_ms=timeframe_ms,
        execution_mode="bar",
    )

    monkeypatch.setattr(worker, "_build_envelope_payload", lambda order, sym: ({}, None, None))
    monkeypatch.setattr(worker, "_should_skip_idempotent", lambda *args, **kwargs: False)
    monkeypatch.setattr(worker, "_remember_idempotency_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_commit_exposure", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_rollback_exposure", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_refund_tokens", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_log_drop_envelope", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_update_queue_metrics", lambda: None)
    monkeypatch.setattr(worker, "_maybe_drop_due_to_cooldown", lambda *args, **kwargs: False)

    within_order = _make_simple_order(symbol, created_ts_ms=bar_close_ms)
    monkeypatch.setattr(clock, "now_ms", lambda: bar_close_ms + timeframe_ms - 1)

    within_result = worker.publish_decision(
        within_order,
        symbol,
        bar_open_ms,
        bar_close_ms=bar_close_ms,
    )

    assert within_result.action == "pass"

    expired_order = _make_simple_order(symbol, created_ts_ms=bar_close_ms)
    monkeypatch.setattr(clock, "now_ms", lambda: bar_close_ms + timeframe_ms + 1)

    expired_result = worker.publish_decision(
        expired_order,
        symbol,
        bar_open_ms,
        bar_close_ms=bar_close_ms,
    )

    assert expired_result.action == "drop"


def test_process_spot_envelope_records_created_ts(monkeypatch) -> None:
    timeframe_ms = 60_000
    bar_close_ms = 1_700_000_000_000
    now_ms = bar_close_ms + 500

    monitoring_stub = _make_monitoring_stub()
    monkeypatch.setattr(service_signal_runner, "monitoring", monitoring_stub)
    monkeypatch.setattr(
        service_signal_runner, "pipeline_stage_drop_count", _DummyMetric()
    )
    monkeypatch.setattr(
        service_signal_runner, "skipped_incomplete_bars", _DummyMetric()
    )

    monkeypatch.setattr(service_signal_runner.signal_bus, "ENABLED", True, raising=False)

    skip_calls: list[tuple[str, int]] = []
    update_calls: list[tuple[str, int]] = []

    def _should_skip(symbol: str, close_ms: int) -> bool:
        skip_calls.append((symbol, close_ms))
        return False

    def _update(symbol: str, close_ms: int) -> None:
        update_calls.append((symbol, close_ms))

    monkeypatch.setattr(service_signal_runner.signal_bus, "should_skip", _should_skip)
    monkeypatch.setattr(service_signal_runner.signal_bus, "update", _update)

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

    publish_calls: list[dict[str, Any]] = []

    def _publish_signal(
        symbol: str,
        close_ms: int,
        payload: Any,
        dispatch: Any,
        *,
        expires_at_ms: int | None = None,
        **kwargs: Any,
    ) -> bool:
        publish_calls.append(
            {
                "symbol": symbol,
                "bar_close_ms": close_ms,
                "payload": payload,
                "expires_at_ms": expires_at_ms,
            }
        )
        return True

    monkeypatch.setattr(
        service_signal_runner, "publish_signal_envelope", _publish_signal
    )

    class _StubFeaturePipe:
        def __init__(self) -> None:
            self.signal_quality: dict[str, object] = {}
            self.timeframe_ms = timeframe_ms
            self.spread_ttl_ms = 0

        def update(
            self, bar: Bar, *, skip_metrics: bool | None = None
        ) -> dict[str, float]:
            return {"close": float(bar.close)}

    class _StubPolicy:
        def __init__(self) -> None:
            self.timeframe_ms = timeframe_ms

        def decide(self, features, ctx):
            return [ctx.symbol]

        def consume_signal_transitions(self):
            return []

    executor = SimpleNamespace(submit=lambda order: None, execute=lambda order: None)

    worker = service_signal_runner._Worker(
        _StubFeaturePipe(),
        _StubPolicy(),
        logging.getLogger("test-worker"),
        executor,
        enforce_closed_bars=False,
        ws_dedup_enabled=True,
        ws_dedup_timeframe_ms=timeframe_ms,
        bar_timeframe_ms=timeframe_ms,
        execution_mode="bar",
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

    def _closed_guard(*args, **kwargs):
        return service_signal_runner.PipelineResult(
            action="pass", stage=service_signal_runner.Stage.CLOSED_BAR
        )

    def _policy_decide(*_args, **_kwargs):
        return service_signal_runner.PipelineResult(
            action="pass",
            stage=service_signal_runner.Stage.POLICY,
            decision=[envelope],
        )

    def _apply_risk(*_args, **_kwargs):
        return service_signal_runner.PipelineResult(
            action="pass",
            stage=service_signal_runner.Stage.RISK,
            decision=[envelope],
        )

    monkeypatch.setattr(service_signal_runner, "closed_bar_guard", _closed_guard)
    monkeypatch.setattr(service_signal_runner, "policy_decide", _policy_decide)
    monkeypatch.setattr(service_signal_runner, "apply_risk", _apply_risk)

    economics = SpotSignalEconomics(
        edge_bps=10.0,
        cost_bps=1.0,
        net_bps=9.0,
        turnover_usd=1_000.0,
        act_now=True,
        impact=0.0,
        impact_mode="model",
        adv_quote=50_000.0,
    )
    payload = SpotSignalTargetWeightPayload(
        target_weight=0.5,
        economics=economics,
    )
    envelope = SpotSignalEnvelope(
        symbol="BTCUSDT",
        bar_close_ms=bar_close_ms,
        expires_at_ms=bar_close_ms,
        payload=payload,
    )

    monkeypatch.setattr(clock, "now_ms", lambda: now_ms)

    bar = Bar(
        ts=bar_close_ms,
        symbol="BTCUSDT",
        open=Decimal("1"),
        high=Decimal("1"),
        low=Decimal("1"),
        close=Decimal("1"),
    )

    emitted = worker.process(bar)

    assert emitted == [envelope]
    assert skip_calls == [("BTCUSDT", bar_close_ms)]
    assert update_calls == [("BTCUSDT", bar_close_ms)]
    assert publish_calls, "expected the envelope to be published"
    expires_vals = {call["expires_at_ms"] for call in publish_calls}
    assert expires_vals == {bar_close_ms}
    assert any(call["now_ms"] == now_ms for call in ttl_calls)


def test_emit_clamps_expires_at_to_bar_close(monkeypatch) -> None:
    timeframe_ms = 60_000
    bar_close_ms = 1_700_000_200_000
    bar_open_ms = bar_close_ms - timeframe_ms
    now_ms = bar_close_ms + 500

    monitoring_stub = _make_monitoring_stub()
    monkeypatch.setattr(service_signal_runner, "monitoring", monitoring_stub)
    monkeypatch.setattr(
        service_signal_runner, "pipeline_stage_drop_count", _DummyMetric()
    )
    monkeypatch.setattr(
        service_signal_runner, "skipped_incomplete_bars", _DummyMetric()
    )

    monkeypatch.setattr(service_signal_runner.signal_bus, "ENABLED", True, raising=False)
    monkeypatch.setattr(service_signal_runner.signal_bus, "OUT_WRITER", None, raising=False)

    def _check_ttl(*, bar_close_ms: int, now_ms: int, timeframe_ms: int):
        return True, bar_close_ms - 10_000, None

    monkeypatch.setattr(service_signal_runner, "check_ttl", _check_ttl)
    monkeypatch.setattr(clock, "now_ms", lambda: now_ms)

    publish_calls: list[dict[str, Any]] = []

    def _publish(
        symbol: str,
        close_ms: int,
        payload: Any,
        dispatcher: Any,
        *,
        expires_at_ms: int,
        dedup_key: str | None = None,
        valid_until_ms: int | None = None,
    ) -> bool:
        publish_calls.append(
            {
                "symbol": symbol,
                "bar_close_ms": close_ms,
                "expires_at_ms": expires_at_ms,
                "valid_until_ms": valid_until_ms,
                "payload": payload,
            }
        )
        return True

    monkeypatch.setattr(service_signal_runner, "publish_signal_envelope", _publish)

    class _StubFeaturePipe:
        def __init__(self) -> None:
            self.signal_quality: dict[str, object] = {}
            self.timeframe_ms = timeframe_ms
            self.spread_ttl_ms = 0

        def update(
            self, bar: Bar, *, skip_metrics: bool | None = None
        ) -> dict[str, float]:
            return {"close": float(bar.close)}

    class _StubPolicy:
        def __init__(self) -> None:
            self.timeframe_ms = timeframe_ms

        def decide(self, features, ctx):
            return [SimpleNamespace(symbol=ctx.symbol, meta={}, side="buy")]

        def consume_signal_transitions(self):
            return []

    executor = SimpleNamespace(submit=lambda order: None, execute=lambda order: None)

    worker = service_signal_runner._Worker(
        _StubFeaturePipe(),
        _StubPolicy(),
        logging.getLogger("emit-test"),
        executor,
        enforce_closed_bars=False,
        ws_dedup_enabled=True,
        ws_dedup_timeframe_ms=timeframe_ms,
        bar_timeframe_ms=timeframe_ms,
    )

    order = SimpleNamespace(
        meta={
            "payload": {
                "kind": "target_weight",
                "target_weight": 0.5,
                "valid_until_ms": bar_close_ms - 5_000,
            },
            "dedup_key": "test-key",
        },
        created_ts_ms=bar_close_ms - 15_000,
        score=1.0,
        side="buy",
        features_hash="abc",
        volume_frac=0.1,
    )

    result = worker._emit(order, "BTCUSDT", bar_close_ms, bar_open_ms=bar_open_ms)

    assert result is True
    assert len(publish_calls) == 1
    call = publish_calls[0]
    assert call["expires_at_ms"] == bar_close_ms
    assert call["valid_until_ms"] == bar_close_ms - 5_000


def test_emit_populates_equity_before_bar_execute(monkeypatch) -> None:
    timeframe_ms = 60_000
    bar_close_ms = 1_700_000_250_000

    monitoring_stub = _make_monitoring_stub()
    monkeypatch.setattr(service_signal_runner, "monitoring", monitoring_stub)
    monkeypatch.setattr(
        service_signal_runner, "pipeline_stage_drop_count", _DummyMetric()
    )
    monkeypatch.setattr(
        service_signal_runner, "skipped_incomplete_bars", _DummyMetric()
    )

    monkeypatch.setattr(service_signal_runner.signal_bus, "ENABLED", True, raising=False)
    monkeypatch.setattr(service_signal_runner.signal_bus, "OUT_WRITER", None, raising=False)

    def _should_skip(_symbol: str, _close_ms: int) -> bool:
        return False

    def _update(_symbol: str, _close_ms: int) -> None:
        return None

    monkeypatch.setattr(service_signal_runner.signal_bus, "should_skip", _should_skip)
    monkeypatch.setattr(service_signal_runner.signal_bus, "update", _update)

    monkeypatch.setattr(clock, "now_ms", lambda: bar_close_ms + 1_000)

    publish_calls: list[dict[str, Any]] = []

    def _publish_signal(
        symbol: str,
        close_ms: int,
        payload: Any,
        dispatcher: Any,
        *,
        expires_at_ms: int,
        **kwargs: Any,
    ) -> bool:
        publish_calls.append(
            {
                "symbol": symbol,
                "bar_close_ms": close_ms,
                "payload": payload,
                "expires_at_ms": expires_at_ms,
            }
        )
        dispatcher({"symbol": symbol, "payload": payload})
        return True

    monkeypatch.setattr(
        service_signal_runner, "publish_signal_envelope", _publish_signal
    )

    class _StubFeaturePipe:
        def __init__(self) -> None:
            self.signal_quality: dict[str, object] = {}
            self.timeframe_ms = timeframe_ms
            self.spread_ttl_ms = 0

        def update(
            self, bar: Bar, *, skip_metrics: bool | None = None
        ) -> dict[str, float]:
            return {"close": float(bar.close)}

    class _StubPolicy:
        def __init__(self) -> None:
            self.timeframe_ms = timeframe_ms

        def decide(self, features, ctx):
            return [SimpleNamespace(symbol=ctx.symbol, meta={}, side="buy")]

        def consume_signal_transitions(self):
            return []

    class _RecordingExecutor:
        def __init__(self) -> None:
            self.executed_orders: list[Any] = []
            self.equity_values: list[Any] = []

        def submit(self, order: Any) -> None:  # pragma: no cover - interface stub
            return None

        def execute(self, order: Any) -> None:
            self.executed_orders.append(order)
            meta = getattr(order, "meta", {})
            if isinstance(meta, MappingABC):
                equity_val = meta.get("equity_usd")
            else:
                equity_val = getattr(meta, "get", lambda *_: None)("equity_usd")
            self.equity_values.append(equity_val)

    executor = _RecordingExecutor()

    worker = service_signal_runner._Worker(
        _StubFeaturePipe(),
        _StubPolicy(),
        logging.getLogger("equity-test"),
        executor,
        enforce_closed_bars=False,
        ws_dedup_enabled=True,
        ws_dedup_timeframe_ms=timeframe_ms,
        bar_timeframe_ms=timeframe_ms,
        execution_mode="bar",
    )

    worker._symbol_equity["BTCUSDT"] = 10_000.0

    order = SimpleNamespace(
        meta={
            "payload": {
                "kind": "target_weight",
                "target_weight": 0.25,
            }
        },
        created_ts_ms=bar_close_ms - 5_000,
        score=0.0,
        side="buy",
        features_hash="hash",
    )

    result = worker._emit(order, "BTCUSDT", bar_close_ms, bar_open_ms=bar_close_ms - timeframe_ms)

    assert result is True
    assert publish_calls, "expected the signal to be published"
    assert executor.executed_orders, "expected the executor to be invoked"
    equity_vals = executor.equity_values
    assert len(equity_vals) == 1
    assert equity_vals[0] is not None and float(equity_vals[0]) > 0.0
    assert order.meta.get("equity_usd") == equity_vals[0]

def test_build_drop_envelope_clamps_expires(monkeypatch) -> None:
    timeframe_ms = 60_000
    bar_close_ms = 1_700_000_300_000

    monitoring_stub = _make_monitoring_stub()
    monkeypatch.setattr(service_signal_runner, "monitoring", monitoring_stub)
    monkeypatch.setattr(
        service_signal_runner, "pipeline_stage_drop_count", _DummyMetric()
    )
    monkeypatch.setattr(
        service_signal_runner, "skipped_incomplete_bars", _DummyMetric()
    )

    def _check_ttl(*, bar_close_ms: int, now_ms: int, timeframe_ms: int):
        return True, bar_close_ms - 20_000, None

    monkeypatch.setattr(service_signal_runner, "check_ttl", _check_ttl)
    monkeypatch.setattr(clock, "now_ms", lambda: bar_close_ms + 1_000)

    class _StubFeaturePipe:
        def __init__(self) -> None:
            self.signal_quality: dict[str, object] = {}
            self.timeframe_ms = timeframe_ms
            self.spread_ttl_ms = 0

        def update(
            self, bar: Bar, *, skip_metrics: bool | None = None
        ) -> dict[str, float]:
            return {"close": float(bar.close)}

    class _StubPolicy:
        def __init__(self) -> None:
            self.timeframe_ms = timeframe_ms

        def decide(self, features, ctx):
            return [SimpleNamespace(symbol=ctx.symbol, meta={}, side="buy")]

        def consume_signal_transitions(self):
            return []

    executor = SimpleNamespace(submit=lambda order: None, execute=lambda order: None)

    worker = service_signal_runner._Worker(
        _StubFeaturePipe(),
        _StubPolicy(),
        logging.getLogger("drop-test"),
        executor,
        enforce_closed_bars=False,
        ws_dedup_enabled=True,
        ws_dedup_timeframe_ms=timeframe_ms,
        bar_timeframe_ms=timeframe_ms,
    )

    order = SimpleNamespace(
        meta={
            "payload": {
                "kind": "target_weight",
                "target_weight": 0.25,
                "valid_until_ms": bar_close_ms - 30_000,
            }
        },
        created_ts_ms=bar_close_ms - 40_000,
        score=1.0,
        side="buy",
        features_hash="xyz",
        volume_frac=0.1,
    )

    envelope = worker._build_drop_envelope(order, "BTCUSDT", bar_close_ms)

    assert isinstance(envelope, SpotSignalEnvelope)
    assert envelope.bar_close_ms == bar_close_ms
    assert envelope.expires_at_ms == bar_close_ms
