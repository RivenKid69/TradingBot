import logging
import time
from decimal import Decimal

import pytest

from core_models import Bar
from services.monitoring import MonitoringAggregator
from core_config import MonitoringConfig, MonitoringThresholdsConfig
from pipeline import PipelineConfig
from service_signal_runner import _Worker


class DummyAlerts:
    def __init__(self) -> None:
        self.notifications: list[tuple[str, str]] = []

    def notify(self, key: str, message: str) -> None:
        self.notifications.append((key, message))


def _record_bar_metrics(
    agg: MonitoringAggregator, symbol: str, metrics: dict[str, object]
) -> None:
    agg.record_bar_execution(
        symbol,
        decisions=int(metrics.get("decisions", 0) or 0),
        act_now=int(metrics.get("act_now", 0) or 0),
        turnover_usd=float(metrics.get("turnover_usd", 0.0) or 0.0),
        cap_usd=metrics.get("cap_usd"),
        impact_mode=metrics.get("impact_mode"),
        modeled_cost_bps=metrics.get("modeled_cost_bps"),
        realized_slippage_bps=metrics.get("realized_slippage_bps"),
        cost_bias_bps=metrics.get("cost_bias_bps"),
        bar_ts=metrics.get("bar_ts"),
    )


def test_monitoring_cost_bias_alerts() -> None:
    thresholds = MonitoringThresholdsConfig(cost_bias_bps=5.0)
    cfg = MonitoringConfig(enabled=True, thresholds=thresholds)
    alerts = DummyAlerts()
    agg = MonitoringAggregator(cfg, alerts)

    agg.set_execution_mode("bar")
    agg.record_bar_execution(
        "BTCUSDT",
        decisions=1,
        act_now=1,
        turnover_usd=1_000.0,
        modeled_cost_bps=90.0,
        realized_slippage_bps=100.0,
    )
    agg.tick(int(time.time() * 1000))
    assert any(key.startswith("cost_bias_") for key, _ in alerts.notifications)

    alerts.notifications.clear()
    agg.record_bar_execution(
        "BTCUSDT",
        decisions=1,
        act_now=1,
        turnover_usd=1_000.0,
        modeled_cost_bps=100.0,
        realized_slippage_bps=100.0,
    )
    agg.tick(int(time.time() * 1000))
    assert not alerts.notifications
    assert not agg._cost_bias_alerted


def test_worker_forwards_cost_metrics_to_monitoring() -> None:
    thresholds = MonitoringThresholdsConfig()
    cfg = MonitoringConfig(enabled=True, thresholds=thresholds)
    alerts = DummyAlerts()
    agg = MonitoringAggregator(cfg, alerts)

    class DummyMetrics:
        def reset_symbol(self, symbol: str) -> None:  # pragma: no cover - noop
            pass

    class DummyFeaturePipe:
        def __init__(self) -> None:
            self.metrics = DummyMetrics()
            self.signal_quality: dict[str, object] = {}

    class DummyExecutor:
        def __init__(self, snapshot: dict[str, object]) -> None:
            self.monitoring_snapshot = snapshot

    modeled = 123.45
    realized = 150.0
    bias = 7.89
    snapshot = {
        "decision": {
            "turnover_usd": 1_000.0,
            "act_now": True,
            "modeled_cost_bps": modeled,
            "realized_slippage_bps": realized,
            "cost_bias_bps": bias,
        },
        "turnover_usd": 1_000.0,
        "cap_usd": 5_000.0,
        "bar_ts": 1_000_000,
    }

    worker = _Worker(
        fp=DummyFeaturePipe(),
        policy=object(),
        logger=logging.getLogger("test"),
        executor=DummyExecutor(snapshot),
        enforce_closed_bars=False,
        pipeline_cfg=PipelineConfig(enabled=False),
        monitoring=agg,
        execution_mode="bar",
        rest_candidates=[],
    )

    runtime_snapshot = worker._extract_monitoring_snapshot(worker._executor)
    extracted_metrics = worker._extract_bar_execution_metrics(
        runtime_snapshot,
        expected_bar_ts=snapshot["bar_ts"],
        total_signals=1,
    )
    assert extracted_metrics is not None
    assert extracted_metrics.get("bar_ts") == snapshot["bar_ts"]

    bar = Bar(
        ts=1_000_000,
        symbol="BTCUSDT",
        open=Decimal("1"),
        high=Decimal("1"),
        low=Decimal("1"),
        close=Decimal("1"),
    )

    _record_bar_metrics(agg, bar.symbol, extracted_metrics)
    initial_events = len(agg._bar_events["1m"])
    worker.process(bar)

    assert len(agg._bar_events["1m"]) == initial_events
    entries = agg._bar_events["1m"]
    assert entries, "expected bar execution metrics to be recorded"
    entry = entries[-1]
    assert entry["modeled_cost_bps"] == pytest.approx(modeled)
    assert entry["realized_slippage_bps"] == pytest.approx(realized)
    assert entry["cost_bias_bps"] == pytest.approx(bias)
    assert entry["ts"] == bar.ts

    window_snapshot = agg._bar_window_snapshot("1m")
    assert window_snapshot["modeled_cost_bps"] == pytest.approx(modeled)
    assert window_snapshot["realized_slippage_bps"] == pytest.approx(realized)
    assert window_snapshot["cost_bias_bps"] == pytest.approx(bias)


def test_worker_does_not_substitute_adv_for_cap() -> None:
    thresholds = MonitoringThresholdsConfig()
    cfg = MonitoringConfig(enabled=True, thresholds=thresholds)
    alerts = DummyAlerts()
    agg = MonitoringAggregator(cfg, alerts)

    class DummyMetrics:
        def reset_symbol(self, symbol: str) -> None:  # pragma: no cover - noop
            pass

    class DummyFeaturePipe:
        def __init__(self) -> None:
            self.metrics = DummyMetrics()
            self.signal_quality: dict[str, object] = {}

    class DummyExecutor:
        def __init__(self, snapshot: dict[str, object]) -> None:
            self.monitoring_snapshot = snapshot

    adv_quote = 25_000.0
    turnover = 1_000.0
    snapshot = {
        "decision": {
            "turnover_usd": turnover,
            "act_now": False,
        },
        "turnover_usd": turnover,
        "adv_quote": adv_quote,
        "bar_ts": 2_000_000,
    }

    worker = _Worker(
        fp=DummyFeaturePipe(),
        policy=object(),
        logger=logging.getLogger("test"),
        executor=DummyExecutor(snapshot),
        enforce_closed_bars=False,
        pipeline_cfg=PipelineConfig(enabled=False),
        monitoring=agg,
        execution_mode="bar",
        rest_candidates=[],
    )

    runtime_snapshot = worker._extract_monitoring_snapshot(worker._executor)
    metrics = worker._extract_bar_execution_metrics(
        runtime_snapshot,
        expected_bar_ts=snapshot["bar_ts"],
        total_signals=1,
    )
    assert metrics is not None
    assert metrics.get("cap_usd") is None
    assert metrics.get("adv_quote") == pytest.approx(adv_quote)

    bar = Bar(
        ts=2_000_000,
        symbol="BTCUSDT",
        open=Decimal("1"),
        high=Decimal("1"),
        low=Decimal("1"),
        close=Decimal("1"),
    )

    _record_bar_metrics(agg, bar.symbol, metrics)
    initial_events = len(agg._bar_events["1m"])
    worker.process(bar)

    assert len(agg._bar_events["1m"]) == initial_events
    snapshot = agg._bar_execution_snapshot()
    for window_key in ("window_1m", "window_5m"):
        window_snapshot = snapshot[window_key]
        assert window_snapshot["cap_usd"] is None
        assert window_snapshot["turnover_vs_cap"] is None
    cumulative = snapshot["cumulative"]
    assert cumulative["cap_usd"] is None
    assert cumulative["turnover_vs_cap"] is None


def test_bar_execution_pruning_uses_bar_timestamp() -> None:
    thresholds = MonitoringThresholdsConfig()
    cfg = MonitoringConfig(enabled=True, thresholds=thresholds)
    agg = MonitoringAggregator(cfg, DummyAlerts())

    agg.set_execution_mode("bar")
    agg.record_bar_execution(
        "BTCUSDT",
        decisions=1,
        act_now=1,
        turnover_usd=1_000.0,
        bar_ts=0,
    )

    assert len(agg._bar_events["1m"]) == 1
    agg.tick(180_000)
    assert not agg._bar_events["1m"]
    assert agg._bar_events["5m"], "expected longer window to retain event"
    assert agg._bar_events["5m"][0]["ts"] == 0


def test_worker_ignores_empty_bar_snapshot() -> None:
    thresholds = MonitoringThresholdsConfig()
    cfg = MonitoringConfig(enabled=True, thresholds=thresholds)
    alerts = DummyAlerts()
    agg = MonitoringAggregator(cfg, alerts)

    class DummyMetrics:
        def reset_symbol(self, symbol: str) -> None:  # pragma: no cover - noop
            pass

    class DummyFeaturePipe:
        def __init__(self) -> None:
            self.metrics = DummyMetrics()
            self.signal_quality: dict[str, object] = {}

    class DummyExecutor:
        def __init__(self, snapshot: dict[str, object]) -> None:
            self.monitoring_snapshot = snapshot

    worker = _Worker(
        fp=DummyFeaturePipe(),
        policy=object(),
        logger=logging.getLogger("test"),
        executor=DummyExecutor({}),
        enforce_closed_bars=False,
        pipeline_cfg=PipelineConfig(enabled=False),
        monitoring=agg,
        execution_mode="bar",
        rest_candidates=[],
    )

    runtime_snapshot = worker._extract_monitoring_snapshot(worker._executor)
    assert worker._extract_bar_execution_metrics(runtime_snapshot) is None

    bar = Bar(
        ts=1_000_000,
        symbol="BTCUSDT",
        open=Decimal("1"),
        high=Decimal("1"),
        low=Decimal("1"),
        close=Decimal("1"),
    )

    worker.process(bar)

    assert not agg._bar_events["1m"], "expected empty snapshot to be ignored"


def test_worker_skips_bar_execution_without_activity() -> None:
    thresholds = MonitoringThresholdsConfig()
    cfg = MonitoringConfig(enabled=True, thresholds=thresholds)
    alerts = DummyAlerts()
    agg = MonitoringAggregator(cfg, alerts)

    class DummyMetrics:
        def reset_symbol(self, symbol: str) -> None:  # pragma: no cover - noop
            pass

    class DummyFeaturePipe:
        def __init__(self) -> None:
            self.metrics = DummyMetrics()
            self.signal_quality: dict[str, object] = {}

    class DummyExecutor:
        def __init__(self, snapshot: dict[str, object]) -> None:
            self.monitoring_snapshot = snapshot

    first_bar_ts = 3_000_000
    snapshot = {
        "decision": {
            "turnover_usd": 500.0,
            "act_now": True,
        },
        "turnover_usd": 500.0,
        "bar_ts": first_bar_ts,
    }

    worker = _Worker(
        fp=DummyFeaturePipe(),
        policy=object(),
        logger=logging.getLogger("test"),
        executor=DummyExecutor(snapshot),
        enforce_closed_bars=False,
        pipeline_cfg=PipelineConfig(enabled=False),
        monitoring=agg,
        execution_mode="bar",
        rest_candidates=[],
    )

    runtime_snapshot = worker._extract_monitoring_snapshot(worker._executor)
    active_metrics = worker._extract_bar_execution_metrics(
        runtime_snapshot,
        expected_bar_ts=first_bar_ts,
        total_signals=1,
    )
    assert active_metrics is not None
    _record_bar_metrics(agg, "BTCUSDT", active_metrics)
    initial_events = len(agg._bar_events["1m"])
    assert initial_events == 1

    mismatched_metrics = worker._extract_bar_execution_metrics(
        runtime_snapshot,
        expected_bar_ts=first_bar_ts + 60_000,
        total_signals=1,
    )
    assert mismatched_metrics is None

    next_bar = Bar(
        ts=first_bar_ts + 60_000,
        symbol="BTCUSDT",
        open=Decimal("1"),
        high=Decimal("1"),
        low=Decimal("1"),
        close=Decimal("1"),
    )

    worker.process(next_bar)

    assert len(agg._bar_events["1m"]) == initial_events
