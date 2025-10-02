from __future__ import annotations

import logging
import types
from decimal import Decimal
from types import SimpleNamespace

import clock
import service_signal_runner
from core_models import Bar
from pipeline import PipelineResult, Stage


class _DummyMetric:
    def __init__(self) -> None:
        self.label_calls: list[tuple[str, ...]] = []

    def labels(self, *labels: str) -> "_DummyMetric":
        self.label_calls.append(tuple(str(label) for label in labels))
        return self

    def inc(self, *args, **kwargs) -> None:  # pragma: no cover - metric helper
        return None

    def set(self, *args, **kwargs) -> None:  # pragma: no cover - metric helper
        return None


def test_process_propagates_open_and_close(monkeypatch) -> None:
    timeframe_ms = 60_000
    bar_close_ms = 1_700_000_120_000
    expected_open_ms = bar_close_ms - timeframe_ms

    monitoring_stub = SimpleNamespace()
    monitoring_stub.inc_stage = lambda *args, **kwargs: None
    monitoring_stub.record_signals = lambda *args, **kwargs: None
    monitoring_stub.record_fill = lambda *args, **kwargs: None
    monitoring_stub.record_pnl = lambda *args, **kwargs: None
    monitoring_stub.inc_reason = lambda *args, **kwargs: None
    monitoring_stub.alert_zero_signals = lambda *args, **kwargs: None
    monitoring_stub.signal_error_rate = _DummyMetric()
    monitoring_stub.ws_dup_skipped_count = _DummyMetric()
    monitoring_stub.ttl_expired_boundary_count = _DummyMetric()
    monitoring_stub.signal_boundary_count = _DummyMetric()
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
        return True, bar_close_ms + timeframe_ms, None

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
