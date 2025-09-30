import math
from decimal import Decimal

import pandas as pd
import pytest

from transformers import FeatureSpec

from core_models import Bar
from core_config import ExecutionRuntimeConfig, SpotCostConfig
from feature_pipe import FeaturePipe


def _make_bar(ts: int, close: str) -> Bar:
    price = Decimal(close)
    return Bar(
        ts=ts,
        symbol="BTCUSDT",
        open=price,
        high=price,
        low=price,
        close=price,
        volume_base=Decimal("1"),
        is_final=True,
    )


def test_feature_pipe_tracks_returns_sigma_and_warmup():
    pipe = FeaturePipe(FeatureSpec(lookbacks_prices=[1]), sigma_window=3, min_sigma_periods=2)
    bars = [
        _make_bar(0, "100"),
        _make_bar(60_000, "101"),
        _make_bar(120_000, "102"),
    ]
    for bar in bars:
        pipe.update(bar)

    snapshot = pipe.get_market_metrics("BTCUSDT")
    assert snapshot is not None
    assert snapshot.bar_count == 3
    assert snapshot.window_ready is True

    expected_returns = [0.01, 102.0 / 101.0 - 1.0]
    expected_mean = sum(expected_returns) / len(expected_returns)
    expected_var = sum((r - expected_mean) ** 2 for r in expected_returns) / (len(expected_returns) - 1)
    expected_sigma = math.sqrt(expected_var)
    expected_tr = [
        max(101.0 - 101.0, abs(101.0 - 100.0), abs(101.0 - 100.0)) / 100.0,
        max(102.0 - 102.0, abs(102.0 - 101.0), abs(102.0 - 101.0)) / 101.0,
    ]
    expected_atr = sum(expected_tr) / len(expected_tr)

    assert snapshot.ret_last is not None
    assert math.isclose(snapshot.ret_last, expected_returns[-1], rel_tol=1e-12)
    assert snapshot.sigma is not None
    assert math.isclose(snapshot.sigma, expected_sigma, rel_tol=1e-12)
    assert snapshot.atr_pct is not None
    assert snapshot.atr_pct == pytest.approx(expected_atr)


def test_feature_pipe_records_spread_with_ttl_expiry():
    pipe = FeaturePipe(FeatureSpec(lookbacks_prices=[1]), sigma_window=2, spread_ttl_ms=500)
    first = _make_bar(0, "100")
    pipe.update(first)
    pipe.record_spread("BTCUSDT", bid=100.0, ask=100.1, ts_ms=0)

    snap_initial = pipe.get_market_metrics("BTCUSDT")
    assert snap_initial is not None and snap_initial.spread_bps is not None
    assert snap_initial.spread_bps > 0

    second = _make_bar(2_000, "100")
    pipe.update(second)
    snap_after = pipe.get_market_metrics("BTCUSDT")
    assert snap_after is not None
    assert snap_after.last_bar_ts == 2_000
    assert snap_after.spread_bps is None


def test_make_targets_no_turnover_data_leaves_returns_unchanged():
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "ts_ms": [0, 60_000, 120_000],
            "price": [100.0, 101.0, 102.0],
        }
    )

    pipe = FeaturePipe(
        FeatureSpec(lookbacks_prices=[1]),
        execution=ExecutionRuntimeConfig(mode="bar"),
        costs=SpotCostConfig(taker_fee_bps=5.0),
    )

    expected = (
        df.groupby("symbol")["price"].shift(-1).div(df["price"]) - 1.0
    ).rename("target")

    result = pipe.make_targets(df)

    assert result is not None
    pd.testing.assert_series_equal(result, expected)


def test_make_targets_scales_costs_with_turnover_fraction():
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "ts_ms": [0, 60_000, 120_000, 180_000],
            "price": [100.0, 105.0, 109.0, 120.0],
            "turnover_usd": [0.0, 1_000.0, 500.0, 0.0],
            "equity_usd": [10_000.0, 10_000.0, 10_000.0, 10_000.0],
            "adv_usd": [50_000.0, 50_000.0, 50_000.0, 50_000.0],
        }
    )

    pipe = FeaturePipe(
        FeatureSpec(lookbacks_prices=[1]),
        execution=ExecutionRuntimeConfig(mode="bar"),
        costs=SpotCostConfig(
            taker_fee_bps=10.0,
            impact={"linear_coeff": 25.0},
        ),
    )

    raw = (
        df.groupby("symbol")["price"].shift(-1).div(df["price"]) - 1.0
    ).rename("target")
    result = pipe.make_targets(df)

    assert result is not None

    turnover_fraction = df["turnover_usd"] / df["equity_usd"]
    base_component = turnover_fraction * 10.0 * 1e-4
    participation = df["turnover_usd"] / df["adv_usd"]
    impact_component = turnover_fraction * participation * 25.0 * 1e-4
    expected = raw - (base_component + impact_component)
    expected.name = "target"

    pd.testing.assert_series_equal(result, expected)

    costs = raw - result
    assert costs.iloc[1] > costs.iloc[2] > 0.0
