import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

# load compute_multipliers from script
spec_builder = importlib.util.spec_from_file_location(
    "build_hourly_seasonality", BASE / "scripts" / "build_hourly_seasonality.py"
)
builder_mod = importlib.util.module_from_spec(spec_builder)
sys.modules["build_hourly_seasonality"] = builder_mod
spec_builder.loader.exec_module(builder_mod)
compute_multipliers = builder_mod.compute_multipliers

# load ExecutionSimulator
spec_exec = importlib.util.spec_from_file_location("execution_sim", BASE / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec_exec)
sys.modules["execution_sim"] = exec_mod
spec_exec.loader.exec_module(exec_mod)
ExecutionSimulator = exec_mod.ExecutionSimulator

MS = 3_600_000  # one hour in milliseconds


def _data_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "data" / "hourly_pattern_trades.csv"


@pytest.fixture(scope="module")
def sample_multipliers() -> dict[str, np.ndarray]:
    df = pd.read_csv(_data_path())
    multipliers, _ = compute_multipliers(df, min_samples=1)
    return multipliers


def test_compute_multipliers_known_pattern(sample_multipliers):
    liq_expected = [0.88888889, 1.77777778, 0.88888889, 0.44444444]
    lat_expected = [0.95238095, 1.14285714, 1.04761905, 0.85714286]
    spr_expected = [1.0, 1.2, 1.0, 0.8]
    np.testing.assert_allclose(sample_multipliers["liquidity"][:4], liq_expected)
    np.testing.assert_allclose(sample_multipliers["latency"][:4], lat_expected)
    np.testing.assert_allclose(sample_multipliers["spread"][:4], spr_expected)


def test_simulator_applies_multipliers(sample_multipliers):
    sim = ExecutionSimulator(
        liquidity_seasonality=sample_multipliers["liquidity"],
        spread_seasonality=sample_multipliers["spread"],
    )
    base_ts = 1704067200000  # Monday 00:00 UTC
    ts_hour1 = base_ts + MS  # second hour (index 1)
    sim.set_market_snapshot(bid=100.0, ask=101.0, liquidity=5.0, spread_bps=1.0, ts_ms=ts_hour1)
    assert sim._last_liquidity == pytest.approx(5.0 * 1.77777778)
    assert sim._last_spread_bps == pytest.approx(1.0 * 1.2)


def test_compute_multipliers_trims_outliers():
    df = pd.DataFrame(
        {
            "ts_ms": [0, 0, MS, MS],
            "liquidity": [1.0, 1000.0, 1.0, 1.0],
        }
    )
    multipliers, _ = compute_multipliers(df, min_samples=1, trim_top_pct=25.0)
    assert multipliers["liquidity"][0] == pytest.approx(1.0)
    assert multipliers["liquidity"][1] == pytest.approx(1.0)


def test_compute_multipliers_by_day():
    df = pd.read_csv(_data_path())
    multipliers, _ = compute_multipliers(df, min_samples=1, by_day=True)
    arr = multipliers["liquidity"]
    assert len(arr) == 168
    assert np.allclose(arr, 1.0)
