import math

import pytest

from core_config import MonitoringConfig
from services.alerts import AlertManager
from services.monitoring import MonitoringAggregator


def _make_enabled_monitoring_aggregator() -> MonitoringAggregator:
    cfg = MonitoringConfig()
    cfg.enabled = True
    cfg.snapshot_metrics_sec = 10
    alerts = AlertManager({"channel": "noop"})
    return MonitoringAggregator(cfg, alerts)


def test_bar_execution_snapshot_filters_modes_and_sanitises_metrics() -> None:
    aggregator = _make_enabled_monitoring_aggregator()

    aggregator._bar_totals = {
        "decisions": 10.0,
        "act_now": 4.0,
        "turnover_usd": 200.0,
        "cap_usd": 0.0,
        "realized_cost_weight": 5.0,
        "realized_cost_wsum": math.nan,
        "modeled_cost_weight": 2.0,
        "modeled_cost_wsum": 6.0,
    }

    aggregator._bar_mode_totals.clear()
    aggregator._bar_mode_totals.update(
        {
            "aggressive": 3.8,
            "passive": 0.0,
            "unknown": -1.0,
            "nan_mode": math.nan,
        }
    )

    snapshot = aggregator._bar_execution_snapshot()

    assert set(snapshot.keys()) == {"window_1m", "window_5m", "cumulative"}

    for window_key in ("window_1m", "window_5m"):
        window_snapshot = snapshot[window_key]
        assert window_snapshot["decisions"] == 0
        assert window_snapshot["act_now"] == 0
        assert window_snapshot["turnover_usd"] == 0.0
        assert window_snapshot["impact_mode_counts"] == {}
        assert window_snapshot["realized_slippage_bps"] is None
        assert window_snapshot["modeled_cost_bps"] is None
        assert window_snapshot["cost_bias_bps"] is None

    cumulative = snapshot["cumulative"]
    assert cumulative["decisions"] == 10
    assert cumulative["act_now"] == 4
    assert cumulative["act_now_rate"] == pytest.approx(0.4)
    assert cumulative["turnover_usd"] == 200.0
    assert cumulative["cap_usd"] is None
    assert cumulative["turnover_vs_cap"] is None
    assert cumulative["realized_slippage_bps"] is None
    assert cumulative["modeled_cost_bps"] == pytest.approx(3.0)
    assert cumulative["cost_bias_bps"] is None
    assert cumulative["impact_mode_counts"] == {"aggressive": 3}
