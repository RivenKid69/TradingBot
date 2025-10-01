from typing import Any, Mapping, Optional

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


def test_select_numeric_series_falls_back_to_first_numeric_candidate(
    base_pipe: FeaturePipe,
) -> None:
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "bad_turnover": ["n/a", "missing"],
            "turnover": ["1000", "oops"],
            "turnover_usd": [np.nan, np.nan],
        }
    )

    selected = base_pipe._select_numeric_series(
        df, ("bad_turnover", "turnover", "turnover_usd")
    )

    assert selected is not None
    assert selected.dtype == float
    np.testing.assert_array_equal(selected.to_numpy(), np.array([1000.0, np.nan]))


def test_compute_bar_mode_costs_returns_none_for_empty_dataframe(base_pipe: FeaturePipe) -> None:
    empty = pd.DataFrame(columns=["symbol", "ts_ms", "turnover_usd"])

    result = base_pipe._compute_bar_mode_costs(empty)

    assert result is None


@pytest.mark.parametrize(
    "df, expected",
    [
        (
            pd.DataFrame(
                {
                    "symbol": ["BTCUSDT"] * 3,
                    "ts_ms": [0, 1, 2],
                    "turnover_usd": [0.0, 500.0, 100.0],
                    "equity_usd": [5_000.0, 20_000.0, np.inf],
                    "adv_usd": [50_000.0, 200_000.0, 50_000.0],
                }
            ),
            np.array([0.0, 3.78125e-05, 0.0]),
        ),
        (
            pd.DataFrame(
                {
                    "symbol": ["BTCUSDT"] * 2,
                    "ts_ms": [0, 1],
                    "turnover_usd": [1_000.0, np.nan],
                    "equity_usd": [5_000.0, -1.0],
                }
            ),
            np.array([3.0e-04, 0.0]),
        ),
        (
            pd.DataFrame(
                {
                    "symbol": ["BTCUSDT"],
                    "ts_ms": [0],
                    "turnover_usd": ["missing"],
                    "equity_usd": [10_000.0],
                }
            ),
            None,
        ),
    ],
)
def test_compute_bar_mode_costs_various_inputs(
    df: pd.DataFrame, expected: Optional[np.ndarray]
) -> None:
    pipe = FeaturePipe(
        FeatureSpec(lookbacks_prices=[1]),
        execution=ExecutionRuntimeConfig(mode="bar"),
        costs=SpotCostConfig(
            taker_fee_bps=5.0,
            half_spread_bps=10.0,
            impact={"linear_coeff": 50.0},
        ),
    )

    result = pipe._compute_bar_mode_costs(df)

    if expected is None:
        assert result is None
        return

    assert result is not None
    np.testing.assert_allclose(result.to_numpy(), expected)


@pytest.mark.parametrize(
    "impact_config, participation, expected",
    [
        (
            {"sqrt_coeff": 30.0},
            np.array([-0.5, 0.0, 0.25, 1.0]),
            np.array([0.0, 0.0, 30.0 * np.sqrt(0.25), 30.0]),
        ),
        (
            {"linear_coeff": 20.0},
            np.array([-1.0, 0.1, 2.0]),
            np.array([0.0, 2.0, 40.0]),
        ),
        (
            {"power_coefficient": 50.0, "power_exponent": 0.5},
            np.array([-1.0, 1.0, 4.0]),
            np.array([0.0, 50.0, 100.0]),
        ),
        (
            {
                "sqrt_coeff": 10.0,
                "linear_coeff": 20.0,
                "power_coefficient": 5.0,
                "power_exponent": 2.0,
            },
            np.array([-1.0, 0.25, 2.0]),
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
def test_impact_bps_terms(impact_config, participation, expected) -> None:
    pipe = FeaturePipe(
        FeatureSpec(lookbacks_prices=[1]),
        execution=ExecutionRuntimeConfig(mode="bar"),
        costs=SpotCostConfig(impact=impact_config),
    )

    result = pipe._impact_bps(participation)

    assert result is not None
    np.testing.assert_allclose(result, expected)


def test_impact_bps_clips_to_non_negative_outputs(base_pipe: FeaturePipe) -> None:
    negative_only = np.array([-5.0, -0.1])

    result = base_pipe._impact_bps(negative_only)

    if result is not None:
        assert np.all(result == 0.0)

    pipe = FeaturePipe(
        FeatureSpec(lookbacks_prices=[1]),
        execution=ExecutionRuntimeConfig(mode="bar"),
        costs=SpotCostConfig(impact={"linear_coeff": 5.0}),
    )

    mixed = np.array([-2.0, -0.5, 1.0])
    result = pipe._impact_bps(mixed)

    assert result is not None
    assert np.all(result >= 0.0)
    np.testing.assert_array_equal(result[:2], np.array([0.0, 0.0]))


@pytest.mark.parametrize(
    "func_name, telemetry",
    [
        ("_estimate_intrabar_latency", {"intrabar_latency_ms": None, "latency_ms": 12.5}),
        (
            "_bar_latency_hint",
            {"latency_ms": None, "latency": {"p50_ms": 7.0, "p95_ms": 12.0}},
        ),
    ],
)
def test_latency_hint_helpers_emit_scalar_when_available(
    base_pipe: FeaturePipe, func_name: str, telemetry: Mapping[str, Any]
) -> None:
    func = getattr(base_pipe, func_name, None)
    if func is None:
        pytest.skip(f"{func_name} not implemented on FeaturePipe")

    result = func(telemetry)

    if result is None:
        pytest.skip(f"{func_name} unavailable for provided telemetry")

    assert np.isscalar(result)
    assert float(result) >= 0.0
