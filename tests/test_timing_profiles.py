import logging
from decimal import Decimal

import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from core_config import (
    load_timing_profiles,
    resolve_execution_timing,
    ExecutionProfile,
)
from leakguard import LeakGuard, LeakConfig
from pipeline import PipelineConfig, PipelineResult, Stage
from core_models import Bar
import service_signal_runner
from trading_patchnew import TradingEnv, DecisionTiming


def _make_minimal_df(rows: int = 5, timeframe_ms: int = 60_000) -> pd.DataFrame:
    data = []
    for idx in range(rows):
        base = 100.0 + idx
        data.append(
            {
                "ts_ms": idx * timeframe_ms,
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.5,
                "price": base + 0.5,
                "quote_asset_volume": 1_000.0 + idx,
            }
        )
    return pd.DataFrame(data)


def test_execution_profile_switch_changes_env_behavior():
    timing_defaults, timing_profiles = load_timing_profiles()
    mkt_timing = resolve_execution_timing(
        ExecutionProfile.MKT_OPEN_NEXT_H1, timing_defaults, timing_profiles
    )
    vwap_timing = resolve_execution_timing(
        ExecutionProfile.VWAP_CURRENT_H1, timing_defaults, timing_profiles
    )

    df = _make_minimal_df()

    env_mkt = TradingEnv(
        df,
        decision_mode=DecisionTiming[mkt_timing.decision_mode],
        decision_delay_ms=mkt_timing.decision_delay_ms,
        latency_steps=mkt_timing.latency_steps,
        leak_guard=LeakGuard(
            LeakConfig(
                decision_delay_ms=mkt_timing.decision_delay_ms,
                min_lookback_ms=mkt_timing.min_lookback_ms,
            )
        ),
    )
    env_vwap = TradingEnv(
        df,
        decision_mode=DecisionTiming[vwap_timing.decision_mode],
        decision_delay_ms=vwap_timing.decision_delay_ms,
        latency_steps=vwap_timing.latency_steps,
        leak_guard=LeakGuard(
            LeakConfig(
                decision_delay_ms=vwap_timing.decision_delay_ms,
                min_lookback_ms=vwap_timing.min_lookback_ms,
            )
        ),
    )

    try:
        assert env_mkt.decision_mode == DecisionTiming.CLOSE_TO_OPEN
        assert env_vwap.decision_mode == DecisionTiming.INTRA_HOUR_WITH_LATENCY
        assert env_vwap.latency_steps >= 1
    finally:
        env_mkt.close()
        env_vwap.close()


def test_mkt_open_profile_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    timing_defaults, timing_profiles = load_timing_profiles()
    resolved = resolve_execution_timing(
        ExecutionProfile.MKT_OPEN_NEXT_H1, timing_defaults, timing_profiles
    )
    timeframe_ms = timing_defaults.timeframe_ms or 60_000

    df = _make_minimal_df(rows=3, timeframe_ms=timeframe_ms)
    env = TradingEnv(
        df,
        decision_mode=DecisionTiming[resolved.decision_mode],
        decision_delay_ms=resolved.decision_delay_ms,
        latency_steps=resolved.latency_steps,
        leak_guard=LeakGuard(
            LeakConfig(
                decision_delay_ms=resolved.decision_delay_ms,
                min_lookback_ms=resolved.min_lookback_ms,
            )
        ),
    )
    try:
        env.reset()
        decision_ts = int(env.df.loc[0, "decision_ts"])
    finally:
        env.close()

    class _DummyMetric:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs) -> None:
            return None

        def set(self, *args, **kwargs) -> None:
            return None

        def observe(self, *args, **kwargs) -> None:
            return None

    class DummyFeaturePipe:
        def __init__(self) -> None:
            self.signal_quality = {}

        def update(self, bar, skip_metrics: bool = False):
            return {}

    class DummyPolicy:
        def __init__(self, symbol: str) -> None:
            self._symbol = symbol

        def decide(self, features, ctx):
            return [DummyOrder(self._symbol)]

        def consume_signal_transitions(self):
            return []

    class DummyOrder:
        def __init__(self, symbol: str, quantity: float = 1.0, side: str = "BUY") -> None:
            self.symbol = symbol
            self.quantity = quantity
            self.side = side
            self.meta = {}

    class DummyExecutor:
        def submit(self, order) -> None:
            return None

    dummy_metric = _DummyMetric()
    for attr in (
        "skipped_incomplete_bars",
        "pipeline_stage_drop_count",
    ):
        monkeypatch.setattr(service_signal_runner, attr, dummy_metric)
    monkeypatch.setattr(
        service_signal_runner.monitoring, "inc_stage", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        service_signal_runner.monitoring, "inc_reason", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        service_signal_runner.monitoring, "record_signals", lambda *args, **kwargs: None
    )
    for attr in (
        "signal_error_rate",
        "throttle_enqueued_count",
        "throttle_dropped_count",
        "ws_dup_skipped_count",
        "ttl_expired_boundary_count",
        "signal_boundary_count",
        "queue_len",
        "age_at_publish_ms",
        "signal_published_count",
        "signal_absolute_count",
        "throttle_queue_expired_count",
    ):
        monkeypatch.setattr(service_signal_runner.monitoring, attr, dummy_metric)
    monkeypatch.setattr(
        service_signal_runner.monitoring, "kill_switch_triggered", lambda: False
    )
    monkeypatch.setattr(
        service_signal_runner.monitoring, "alert_zero_signals", lambda *a, **k: None
    )

    guard_call: dict[str, int] = {}

    def _guard(bar, now_ms, enforce, lag_ms, *, stage_cfg=None):
        guard_call["now_ms"] = int(now_ms)
        guard_call["lag_ms"] = int(lag_ms)
        return PipelineResult(action="pass", stage=Stage.CLOSED_BAR)

    monkeypatch.setattr(service_signal_runner, "closed_bar_guard", _guard)
    monkeypatch.setattr(service_signal_runner.clock, "now_ms", lambda: decision_ts)

    recorded_dedup: list[int] = []
    monkeypatch.setattr(
        service_signal_runner.signal_bus, "should_skip", lambda symbol, close_ms: False
    )
    monkeypatch.setattr(
        service_signal_runner.signal_bus,
        "update",
        lambda symbol, close_ms: recorded_dedup.append(int(close_ms)),
    )

    captured_publish: list[int] = []

    def _publish(self, order, symbol: str, bar_close_ms: int, *, stage_cfg=None):
        captured_publish.append(int(bar_close_ms))
        return PipelineResult(action="pass", stage=Stage.PUBLISH, decision=order)

    monkeypatch.setattr(service_signal_runner._Worker, "publish_decision", _publish)

    worker = service_signal_runner._Worker(
        fp=DummyFeaturePipe(),
        policy=DummyPolicy("BTCUSDT"),
        logger=logging.getLogger("timing-test"),
        executor=DummyExecutor(),
        guards=None,
        enforce_closed_bars=True,
        close_lag_ms=resolved.decision_delay_ms,
        ws_dedup_enabled=True,
        ws_dedup_timeframe_ms=timeframe_ms,
        throttle_cfg=None,
        pipeline_cfg=PipelineConfig(),
    )

    bar = Bar(
        ts=int(df.loc[0, "ts_ms"]),
        symbol="BTCUSDT",
        open=Decimal(str(df.loc[0, "open"])),
        high=Decimal(str(df.loc[0, "high"])),
        low=Decimal(str(df.loc[0, "low"])),
        close=Decimal(str(df.loc[0, "close"])),
        volume_base=None,
        volume_quote=Decimal("0"),
        is_final=True,
    )

    worker.process(bar)

    assert guard_call["lag_ms"] == resolved.decision_delay_ms
    assert guard_call["now_ms"] == decision_ts
    assert captured_publish == [int(bar.ts)]
    assert recorded_dedup == [int(bar.ts) + timeframe_ms]
    assert int(bar.ts) + resolved.decision_delay_ms == decision_ts
