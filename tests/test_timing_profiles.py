import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from core_config import (
    load_timing_profiles,
    resolve_execution_timing,
    ExecutionProfile,
)
from leakguard import LeakGuard, LeakConfig
from trading_patchnew import TradingEnv, DecisionTiming


def _make_minimal_df(rows: int = 5, timeframe_ms: int = 60_000) -> pd.DataFrame:
    data = []
    for idx in range(rows):
        base = 100.0 + idx
        data.append(
            {
                "ts_ms": idx * timeframe_ms,
                "open": base,
                "high": base + 1.0,
                "low": base - 1.0,
                "close": base + 0.5,
                "price": base + 0.5,
                "quote_asset_volume": 1_000.0 + idx,
            }
        )
    return pd.DataFrame(data)


def test_execution_profile_switch_changes_env_behavior():
    timing_defaults, timing_profiles = load_timing_profiles()
    mkt_timing = resolve_execution_timing(
        ExecutionProfile.MKT_OPEN_NEXT_H1, timing_defaults, timing_profiles
    )
    vwap_timing = resolve_execution_timing(
        ExecutionProfile.VWAP_CURRENT_H1, timing_defaults, timing_profiles
    )

    df = _make_minimal_df()

    env_mkt = TradingEnv(
        df,
        decision_mode=DecisionTiming[mkt_timing.decision_mode],
        decision_delay_ms=mkt_timing.decision_delay_ms,
        latency_steps=mkt_timing.latency_steps,
        leak_guard=LeakGuard(
            LeakConfig(
                decision_delay_ms=mkt_timing.decision_delay_ms,
                min_lookback_ms=mkt_timing.min_lookback_ms,
            )
        ),
    )
    env_vwap = TradingEnv(
        df,
        decision_mode=DecisionTiming[vwap_timing.decision_mode],
        decision_delay_ms=vwap_timing.decision_delay_ms,
        latency_steps=vwap_timing.latency_steps,
        leak_guard=LeakGuard(
            LeakConfig(
                decision_delay_ms=vwap_timing.decision_delay_ms,
                min_lookback_ms=vwap_timing.min_lookback_ms,
            )
        ),
    )

    try:
        assert env_mkt.decision_mode == DecisionTiming.CLOSE_TO_OPEN
        assert env_vwap.decision_mode == DecisionTiming.INTRA_HOUR_WITH_LATENCY
        assert env_vwap.latency_steps >= 1
    finally:
        env_mkt.close()
        env_vwap.close()
