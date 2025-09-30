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

    bar = Bar(
        ts=1_000_000,
        symbol="BTCUSDT",
        open=Decimal("1"),
        high=Decimal("1"),
        low=Decimal("1"),
        close=Decimal("1"),
    )

    worker.process(bar)

    entries = agg._bar_events["1m"]
    assert entries, "expected bar execution metrics to be recorded"
    entry = entries[-1]
    assert entry["modeled_cost_bps"] == pytest.approx(modeled)
    assert entry["realized_slippage_bps"] == pytest.approx(realized)
    assert entry["cost_bias_bps"] == pytest.approx(bias)

    window_snapshot = agg._bar_window_snapshot("1m")
    assert window_snapshot["modeled_cost_bps"] == pytest.approx(modeled)
    assert window_snapshot["realized_slippage_bps"] == pytest.approx(realized)
    assert window_snapshot["cost_bias_bps"] == pytest.approx(bias)
