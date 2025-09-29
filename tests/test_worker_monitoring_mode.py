import logging

from core_config import MonitoringConfig
from service_signal_runner import _Worker
from services.monitoring import MonitoringAggregator


class DummyAlerts:
    def notify(self, key: str, message: str) -> None:  # pragma: no cover - test helper
        pass


class DummyFeaturePipe:
    spread_ttl_ms = 0


class DummyPolicy:
    pass


def test_worker_sets_monitoring_execution_mode_to_bar() -> None:
    monitoring_cfg = MonitoringConfig(enabled=True)
    aggregator = MonitoringAggregator(monitoring_cfg, DummyAlerts())

    _Worker(
        DummyFeaturePipe(),
        DummyPolicy(),
        logging.getLogger(__name__),
        object(),
        enforce_closed_bars=True,
        monitoring=aggregator,
        execution_mode="bar",
    )

    metrics = aggregator._build_metrics(0, {}, [])
    assert metrics["execution_mode"] == "bar"
