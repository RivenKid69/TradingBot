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

    aggregator._bar_totals.update(
        {
            "decisions": 10.0,
            "act_now": 4.0,
            "turnover_usd": 200.0,
            "realized_cost_weight": 5.0,
            "realized_cost_wsum": math.nan,
            "modeled_cost_weight": 2.0,
            "modeled_cost_wsum": 6.0,
        }
    )
    aggregator._bar_caps_by_symbol.clear()

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


def test_bar_execution_snapshot_aggregates_unique_caps() -> None:
    aggregator = _make_enabled_monitoring_aggregator()

    aggregator.record_bar_execution(
        "BTCUSDT",
        decisions=10,
        act_now=5,
        turnover_usd=1_000.0,
        cap_usd=10_000.0,
    )
    aggregator.record_bar_execution(
        "BTCUSDT",
        decisions=5,
        act_now=2,
        turnover_usd=500.0,
        cap_usd=10_000.0,
    )
    aggregator.record_bar_execution(
        "ETHUSDT",
        decisions=8,
        act_now=4,
        turnover_usd=800.0,
        cap_usd=5_000.0,
    )

    snapshot = aggregator._bar_execution_snapshot()

    expected_cap = 15_000.0
    expected_turnover = 2_300.0
    expected_ratio = expected_turnover / expected_cap

    for window_key in ("window_1m", "window_5m"):
        window_snapshot = snapshot[window_key]
        assert window_snapshot["cap_usd"] == pytest.approx(expected_cap)
        assert window_snapshot["turnover_vs_cap"] == pytest.approx(expected_ratio)

    cumulative = snapshot["cumulative"]
    assert cumulative["cap_usd"] == pytest.approx(expected_cap)
    assert cumulative["turnover_vs_cap"] == pytest.approx(expected_ratio)


def test_bar_execution_snapshot_removes_zero_caps() -> None:
    aggregator = _make_enabled_monitoring_aggregator()

    aggregator.record_bar_execution(
        "BTCUSDT",
        decisions=10,
        act_now=5,
        turnover_usd=1_000.0,
        cap_usd=10_000.0,
    )
    aggregator.record_bar_execution(
        "ETHUSDT",
        decisions=6,
        act_now=3,
        turnover_usd=600.0,
        cap_usd=5_000.0,
    )

    aggregator.record_bar_execution(
        "BTCUSDT",
        decisions=4,
        act_now=2,
        turnover_usd=400.0,
        cap_usd=0.0,
    )

    snapshot = aggregator._bar_execution_snapshot()
    cumulative = snapshot["cumulative"]

    expected_cap = 5_000.0
    expected_turnover = 1_000.0 + 600.0 + 400.0

    assert cumulative["cap_usd"] == pytest.approx(expected_cap)
    assert cumulative["turnover_vs_cap"] == pytest.approx(
        expected_turnover / expected_cap
    )


def test_record_bar_execution_sanitises_turnover_values() -> None:
    aggregator = _make_enabled_monitoring_aggregator()

    aggregator.record_bar_execution(
        "BTCUSDT",
        decisions=4,
        act_now=2,
        turnover_usd=float("nan"),
        modeled_cost_bps=10.0,
        realized_slippage_bps=12.0,
    )
    aggregator.record_bar_execution(
        "ETHUSDT",
        decisions=6,
        act_now=3,
        turnover_usd=float("inf"),
        modeled_cost_bps=6.0,
        realized_slippage_bps=3.0,
    )
    aggregator.record_bar_execution(
        "XRPUSDT",
        decisions=5,
        act_now=1,
        turnover_usd=-50.0,
        modeled_cost_bps=5.0,
        realized_slippage_bps=7.0,
    )

    snapshot = aggregator._bar_execution_snapshot()
    cumulative = snapshot["cumulative"]

    assert cumulative["decisions"] == 15
    assert cumulative["act_now"] == 6
    assert cumulative["act_now_rate"] == pytest.approx(6 / 15)
    assert math.isfinite(cumulative["turnover_usd"])
    assert cumulative["turnover_usd"] == 0.0
    assert cumulative["modeled_cost_bps"] == pytest.approx(101 / 15)
    assert cumulative["realized_slippage_bps"] == pytest.approx(101 / 15)
