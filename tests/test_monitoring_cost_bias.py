import time

from services.monitoring import MonitoringAggregator
from core_config import MonitoringConfig, MonitoringThresholdsConfig


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
