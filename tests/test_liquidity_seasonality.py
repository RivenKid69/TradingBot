import importlib.util
import pathlib
import sys
import datetime
import pytest
import logging

BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

spec_exec = importlib.util.spec_from_file_location("execution_sim", BASE / "execution_sim.py")
exec_mod = importlib.util.module_from_spec(spec_exec)
sys.modules["execution_sim"] = exec_mod
spec_exec.loader.exec_module(exec_mod)

ExecutionSimulator = exec_mod.ExecutionSimulator


def test_liquidity_and_spread_seasonality_multiplier():
    liq_mult = [1.0] * 168
    spr_mult = [1.0] * 168
    hour_idx = 10
    liq_mult[hour_idx] = 2.0
    spr_mult[hour_idx] = 3.0
    sim = ExecutionSimulator(
        liquidity_seasonality=liq_mult,
        spread_seasonality=spr_mult,
    )
    base_dt = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    ts_ms = int(base_dt.timestamp() * 1000 + hour_idx * 3_600_000)
    sim.set_market_snapshot(bid=100.0, ask=101.0, liquidity=5.0, spread_bps=1.0, ts_ms=ts_ms)
    assert sim._last_liquidity == 10.0
    assert sim._last_spread_bps == 3.0


def test_seasonality_toggle_off():
    liq_mult = [1.0] * 168
    spr_mult = [1.0] * 168
    hour_idx = 8
    liq_mult[hour_idx] = 2.0
    spr_mult[hour_idx] = 3.0
    sim = ExecutionSimulator(
        liquidity_seasonality=liq_mult,
        spread_seasonality=spr_mult,
        use_seasonality=False,
    )
    base_dt = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    ts_ms = int(base_dt.timestamp() * 1000 + hour_idx * 3_600_000)
    sim.set_market_snapshot(bid=100.0, ask=101.0, liquidity=5.0, spread_bps=1.0, ts_ms=ts_ms)
    assert sim._last_liquidity == 5.0
    assert sim._last_spread_bps == 1.0


def test_env_flag_disables_seasonality(monkeypatch):
    liq_mult = [1.0] * 168
    spr_mult = [1.0] * 168
    hour_idx = 4
    liq_mult[hour_idx] = 2.0
    spr_mult[hour_idx] = 3.0
    monkeypatch.setenv("ENABLE_SEASONALITY", "0")
    sim = ExecutionSimulator(
        liquidity_seasonality=liq_mult,
        spread_seasonality=spr_mult,
        use_seasonality=True,
    )
    base_dt = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    ts_ms = int(base_dt.timestamp() * 1000 + hour_idx * 3_600_000)
    sim.set_market_snapshot(bid=100.0, ask=101.0, liquidity=5.0, spread_bps=1.0, ts_ms=ts_ms)
    assert sim._last_liquidity == 5.0
    assert sim._last_spread_bps == 1.0


def test_seasonality_edge_hours_wraparound():
    liq_mult = [1.0] * 168
    spr_mult = [1.0] * 168
    liq_mult[0] = 1.5
    liq_mult[1] = 2.0
    liq_mult[167] = 0.5
    spr_mult[0] = 1.1
    spr_mult[1] = 1.2
    spr_mult[167] = 1.3
    sim = ExecutionSimulator(
        liquidity_seasonality=liq_mult,
        spread_seasonality=spr_mult,
    )
    base_dt = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)

    def check(hour: int, liq_expected: float, spr_expected: float) -> None:
        ts_ms = int(base_dt.timestamp() * 1000 + hour * 3_600_000)
        sim.set_market_snapshot(
            bid=100.0,
            ask=101.0,
            liquidity=10.0,
            spread_bps=1.0,
            ts_ms=ts_ms,
        )
        assert sim._last_liquidity == pytest.approx(10.0 * liq_expected)
        assert sim._last_spread_bps == pytest.approx(1.0 * spr_expected)

    check(0, 1.5, 1.1)
    check(1, 2.0, 1.2)
    check(167, 0.5, 1.3)
    # Wrap-around to the start of the week
    check(168, 1.5, 1.1)


def test_day_only_multipliers():
    liq = [1.0] * 7
    spr = [1.0] * 7
    idx = 2  # Wednesday
    liq[idx] = 2.0
    spr[idx] = 3.0
    sim = ExecutionSimulator(
        liquidity_seasonality=liq,
        spread_seasonality=spr,
        seasonality_day_only=True,
    )
    base_dt = datetime.datetime(2024, 1, 3, 0, 0, tzinfo=datetime.timezone.utc)
    ts_ms = int(base_dt.timestamp() * 1000 + 5 * 3_600_000)
    sim.set_market_snapshot(bid=100.0, ask=101.0, liquidity=5.0, spread_bps=1.0, ts_ms=ts_ms)
    assert sim._last_liquidity == pytest.approx(10.0)
    assert sim._last_spread_bps == pytest.approx(3.0)


def test_seasonality_override_applied():
    liq_mult = [1.0] * 168
    spr_mult = [1.0] * 168
    hour_idx = 12
    liq_mult[hour_idx] = 2.0
    spr_mult[hour_idx] = 3.0
    liq_override = [1.0] * 168
    spr_override = [1.0] * 168
    liq_override[hour_idx] = 0.75
    spr_override[hour_idx] = 0.5
    sim = ExecutionSimulator(
        liquidity_seasonality=liq_mult,
        spread_seasonality=spr_mult,
        liquidity_seasonality_override=liq_override,
        spread_seasonality_override=spr_override,
    )
    base_dt = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    ts_ms = int(base_dt.timestamp() * 1000 + hour_idx * 3_600_000)
    sim.set_market_snapshot(bid=100.0, ask=101.0, liquidity=4.0, spread_bps=1.0, ts_ms=ts_ms)
    assert sim._last_liquidity == pytest.approx(4.0 * 2.0 * 0.75)
    assert sim._last_spread_bps == pytest.approx(1.0 * 3.0 * 0.5)


def test_seasonality_file_missing(tmp_path):
    missing = tmp_path / "missing.json"
    sim = ExecutionSimulator(liquidity_seasonality_path=str(missing))
    base_dt = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    ts_ms = int(base_dt.timestamp() * 1000 + 5 * 3_600_000)
    sim.set_market_snapshot(bid=100.0, ask=101.0, liquidity=5.0, spread_bps=1.0, ts_ms=ts_ms)
    assert sim._last_liquidity == 5.0
    assert sim._last_spread_bps == 1.0


def test_ts_ms_none_skips_multipliers_without_logging(caplog):
    liq_mult = [2.0] * 168
    spr_mult = [3.0] * 168
    sim = ExecutionSimulator(
        liquidity_seasonality=liq_mult,
        spread_seasonality=spr_mult,
    )
    with caplog.at_level(logging.WARNING, logger="execution_sim"):
        sim.set_market_snapshot(
            bid=100.0, ask=101.0, liquidity=5.0, spread_bps=1.0, ts_ms=None
        )
    assert sim._last_liquidity == 5.0
    assert sim._last_spread_bps == 1.0
    assert not caplog.records


def test_seasonality_linear_interpolation():
    liq_mult = [1.0] * 168
    spr_mult = [1.0] * 168
    hour = 5
    liq_mult[hour] = 1.0
    liq_mult[hour + 1] = 2.0
    spr_mult[hour] = 1.0
    spr_mult[hour + 1] = 3.0
    sim = ExecutionSimulator(
        liquidity_seasonality=liq_mult,
        spread_seasonality=spr_mult,
        seasonality_interpolate=True,
    )
    base_dt = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    # halfway between hour and next hour
    ts_ms = int(base_dt.timestamp() * 1000 + hour * 3_600_000 + 30 * 60_000)
    sim.set_market_snapshot(bid=100.0, ask=101.0, liquidity=10.0, spread_bps=1.0, ts_ms=ts_ms)
    assert sim._last_liquidity == pytest.approx(15.0)
    assert sim._last_spread_bps == pytest.approx(2.0)
