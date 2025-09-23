import numpy as np
import pandas as pd
import pytest

from leakguard import LeakConfig, LeakGuard


def test_validate_ffill_gaps_enforces_span_limits_and_preserves_decision_delay():
    guard = LeakGuard(LeakConfig(decision_delay_ms=500))
    df = pd.DataFrame(
        {
            "symbol": [
                "ETH",
                "ETH",
                "ETH",
                "ETH",
                "ETH",
                "ETH",
                "BTC",
                "BTC",
                "BTC",
            ],
            "ts_ms": [3200, 0, 1500, 4000, 2200, 1000, 0, 500, 4000],
            "feature": [1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 5.0, 5.0, 5.0],
            "other": [20.0, 10.0, 10.0, 30.0, 10.0, 10.0, 1.0, 1.0, 1.0],
        }
    )

    df_with_decision = guard.attach_decision_time(df, ts_col="ts_ms")
    validated = guard.validate_ffill_gaps(
        df_with_decision,
        ts_col="ts_ms",
        group_keys=["symbol"],
        value_cols=["feature", "other"],
        max_gap_ms=1_500,
    )

    # Decision timestamps should remain consistent with the configured delay.
    expected_decision = df_with_decision["ts_ms"].astype("int64") + 500
    pd.testing.assert_series_equal(
        validated["decision_ts"].astype("int64"), expected_decision.astype("int64"), check_names=False
    )

    # Original ordering must be preserved to avoid altering downstream expectations.
    assert list(validated.index) == list(df_with_decision.index)

    expected_feature = [np.nan, 1.0, 1.0, 2.0, np.nan, 1.0, 5.0, 5.0, np.nan]
    expected_other = [20.0, 10.0, 10.0, 30.0, np.nan, 10.0, 1.0, 1.0, np.nan]

    np.testing.assert_allclose(
        validated["feature"].to_numpy(),
        np.array(expected_feature, dtype=float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        validated["other"].to_numpy(),
        np.array(expected_other, dtype=float),
        equal_nan=True,
    )


def test_validate_ffill_gaps_detects_min_lookback_violations():
    guard = LeakGuard(LeakConfig(min_lookback_ms=1_500))
    df = pd.DataFrame(
        {
            "symbol": ["ETH", "ETH", "ETH", "ETH"],
            "ts_ms": [0, 500, 1_000, 3_500],
            "feature": [np.nan, np.nan, np.nan, 1.0],
        }
    )

    with pytest.raises(ValueError, match="min_lookback_ms"):
        guard.validate_ffill_gaps(
            df,
            ts_col="ts_ms",
            group_keys=["symbol"],
            value_cols=["feature"],
            max_gap_ms=10_000,
        )


def test_validate_ffill_gaps_requires_timestamp_values():
    guard = LeakGuard()
    df = pd.DataFrame(
        {
            "symbol": ["ETH", "ETH"],
            "ts_ms": [0, np.nan],
            "feature": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="таймштампами"):
        guard.validate_ffill_gaps(
            df,
            ts_col="ts_ms",
            group_keys=["symbol"],
            value_cols=["feature"],
            max_gap_ms=1_000,
        )
