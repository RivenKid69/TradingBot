import numpy as np
import pandas as pd
import pytest

from transformers import FeatureSpec

from core_config import ExecutionRuntimeConfig, SpotCostConfig
from feature_pipe import FeaturePipe


@pytest.fixture
def base_pipe() -> FeaturePipe:
    return FeaturePipe(
        FeatureSpec(lookbacks_prices=[1]),
        execution=ExecutionRuntimeConfig(mode="bar"),
        costs=SpotCostConfig(taker_fee_bps=5.0, half_spread_bps=10.0),
    )


def test_compute_bar_mode_costs_returns_none_for_empty_dataframe(base_pipe: FeaturePipe) -> None:
    empty = pd.DataFrame(columns=["symbol", "ts_ms", "turnover_usd"])

    result = base_pipe._compute_bar_mode_costs(empty)

    assert result is None


def test_compute_bar_mode_costs_applies_base_and_impact_with_sanitisation() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 5,
            "ts_ms": [0, 1, 2, 3, 4],
            "turnover_usd": [np.nan, 1_000.0, -500.0, np.inf, 200.0],
            "equity_usd": [np.nan, 10_000.0, np.inf, 5_000.0, -100.0],
            "adv_usd": [np.nan, 1_000_000.0, 0.0, 20_000.0, np.inf],
        }
    )

    pipe = FeaturePipe(
        FeatureSpec(lookbacks_prices=[1]),
        execution=ExecutionRuntimeConfig(mode="bar"),
        costs=SpotCostConfig(
            taker_fee_bps=5.0,
            half_spread_bps=10.0,
            impact={"linear_coeff": 25.0},
        ),
    )

    result = pipe._compute_bar_mode_costs(df)

    assert result is not None
    assert list(result.index) == list(df.index)

    turnover_values = np.array([0.0, 1_000.0, 500.0, 0.0, 200.0])
    turnover_fraction = np.array([0.0, 0.1, 0.0, 0.0, 0.0])
    base_cost_fraction = (5.0 + 10.0) * 1e-4
    base_component = turnover_fraction * base_cost_fraction

    participation = np.array([0.0, 0.001, 0.0, 0.0, 0.0])
    impact_bps = pipe._impact_bps(participation)
    assert impact_bps is not None
    impact_component = turnover_fraction * (impact_bps * 1e-4)

    expected = base_component + impact_component

    np.testing.assert_allclose(result.to_numpy(), expected)
    assert np.count_nonzero(result.to_numpy()) == 1
    assert result.iloc[1] > base_component[1]


@pytest.mark.parametrize(
    "impact_config, participation, expected",
    [
        (
            {"sqrt_coeff": 30.0},
            np.array([-0.5, 0.0, 0.25, 1.0]),
            np.array([0.0, 0.0, 30.0 * np.sqrt(0.25), 30.0 * np.sqrt(1.0)]),
        ),
        (
            {"linear_coeff": 20.0},
            np.array([0.0, 0.1, 2.0]),
            np.array([0.0, 2.0, 40.0]),
        ),
        (
            {"power_coefficient": 50.0, "power_exponent": 0.5},
            np.array([0.0, 1.0, 4.0]),
            np.array([0.0, 50.0, 100.0]),
        ),
        (
            {
                "sqrt_coeff": 10.0,
                "linear_coeff": 20.0,
                "power_coefficient": 5.0,
                "power_exponent": 2.0,
            },
            np.array([0.0, 0.25, 2.0]),
            np.array(
                [
                    0.0,
                    10.0 * np.sqrt(0.25)
                    + 20.0 * 0.25
                    + 5.0 * np.power(0.25, 2.0),
                    10.0 * np.sqrt(2.0)
                    + 20.0 * 2.0
                    + 5.0 * np.power(2.0, 2.0),
                ]
            ),
        ),
    ],
)
def test_impact_bps_variants(impact_config, participation, expected) -> None:
    pipe = FeaturePipe(
        FeatureSpec(lookbacks_prices=[1]),
        execution=ExecutionRuntimeConfig(mode="bar"),
        costs=SpotCostConfig(impact=impact_config),
    )

    result = pipe._impact_bps(participation)

    assert result is not None
    np.testing.assert_allclose(result, expected)
