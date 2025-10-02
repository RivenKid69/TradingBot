import json
from types import SimpleNamespace

import clock
import impl_binance_public
import service_signal_runner
from pipeline import PipelineConfig


class _DummyMetric:
    def labels(self, *args, **kwargs):  # pragma: no cover - testing helper
        return self

    def inc(self, *args, **kwargs):  # pragma: no cover - testing helper
        return None

    def observe(self, *args, **kwargs):  # pragma: no cover - testing helper
        return None


def test_binance_ws_bar_uses_close_timestamp_for_ttl(monkeypatch):
    monkeypatch.setattr(impl_binance_public, "websockets", object())

    source = impl_binance_public.BinancePublicBarSource("1m")

    open_ts = 1_700_000
    close_ts = open_ts + 60_000

    message = json.dumps(
        {
            "data": {
                "k": {
                    "t": open_ts,
                    "T": close_ts,
                    "s": "BTCUSDT",
                    "o": "10",
                    "h": "11",
                    "l": "9",
                    "c": "10.5",
                    "v": "100",
                    "q": "1000",
                    "n": 7,
                    "x": True,
                }
            }
        }
    )

    source._handle_message(message)

    bar = source._q.get_nowait()
    assert bar.ts == close_ts
    assert source._last_open_ts[bar.symbol] == open_ts

    captured: dict[str, int] = {}

    def fake_check_ttl(*, bar_close_ms: int, now_ms: int, timeframe_ms: int):
        captured["bar_close_ms"] = bar_close_ms
        captured["now_ms"] = now_ms
        captured["timeframe_ms"] = timeframe_ms
        expires_at = bar_close_ms
        captured["expires_at_ms"] = expires_at
        return True, expires_at, ""

    monkeypatch.setattr(service_signal_runner, "check_ttl", fake_check_ttl)

    dummy_monitoring = SimpleNamespace(
        inc_stage=lambda *a, **k: None,
        signal_published_count=_DummyMetric(),
        age_at_publish_ms=_DummyMetric(),
        signal_absolute_count=_DummyMetric(),
        throttle_enqueued_count=_DummyMetric(),
        throttle_dropped_count=_DummyMetric(),
        throttle_queue_expired_count=_DummyMetric(),
        record_signals=lambda *a, **k: None,
        inc_reason=lambda *a, **k: None,
        alert_zero_signals=lambda *a, **k: None,
        signal_error_rate=_DummyMetric(),
    )
    monkeypatch.setattr(service_signal_runner, "monitoring", dummy_monitoring)

    dummy_signal_bus = SimpleNamespace(
        ENABLED=False,
        OUT_WRITER=None,
        should_skip=lambda *a, **k: False,
        update=lambda *a, **k: None,
    )
    monkeypatch.setattr(service_signal_runner, "signal_bus", dummy_signal_bus)
    monkeypatch.setattr(
        service_signal_runner, "publish_signal_envelope", lambda *a, **k: True
    )

    monkeypatch.setattr(clock, "now_ms", lambda: close_ts + 1_000)

    fp = SimpleNamespace(
        timeframe_ms=60_000,
        spread_ttl_ms=0,
        metrics=SimpleNamespace(reset_symbol=lambda *a, **k: None),
        signal_quality={},
    )
    policy = SimpleNamespace(timeframe_ms=60_000)
    logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )
    executor = SimpleNamespace(submit=lambda order: None, execute=lambda order: None)

    worker = service_signal_runner._Worker(
        fp,
        policy,
        logger,
        executor,
        guards=None,
        enforce_closed_bars=False,
        ws_dedup_timeframe_ms=60_000,
        bar_timeframe_ms=60_000,
        pipeline_cfg=PipelineConfig(),
    )

    order = SimpleNamespace(
        created_ts_ms=open_ts,
        side="buy",
        volume_frac=0.1,
        features_hash="hash",
        score=1.0,
    )

    result = worker._emit(order, bar.symbol, bar_close_ms=close_ts, bar_open_ms=open_ts)

    assert result is True
    assert captured["bar_close_ms"] == close_ts
    assert captured["timeframe_ms"] == 60_000
    assert captured["expires_at_ms"] == close_ts + 60_000
    assert captured["now_ms"] == close_ts + 1_000
