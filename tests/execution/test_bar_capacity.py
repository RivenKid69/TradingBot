import json
import logging
import math
import os
import time
from types import SimpleNamespace

import pytest

from execution_sim import ExecutionSimulator


class _StubADVStore:
    def __init__(self, quotes, *, default=None, floor=None):
        self._quotes = dict(quotes)
        self.default_quote = default
        self.floor_quote = floor
        self.reset_calls = 0

    def reset_runtime_state(self):
        self.reset_calls += 1

    def get_bar_capacity_quote(self, symbol):
        return self._quotes.get(symbol)


def _write_dataset(path, payload):
    path.write_text(json.dumps(payload))
    stat_time = getattr(_write_dataset, "_mtime", time.time()) + 1.0
    _write_dataset._mtime = stat_time  # type: ignore[attr-defined]
    os.utime(path, (stat_time, stat_time))


def test_bar_capacity_base_config_updates_nested_run_config(tmp_path):
    dataset_path = tmp_path / "adv_base.json"
    _write_dataset(dataset_path, {})

    run_config = SimpleNamespace(
        execution={
            "bar_capacity_base": {
                "enabled": True,
                "capacity_frac_of_ADV_base": 0.25,
                "floor_base": 123.0,
                "adv_base_path": str(dataset_path),
                "timeframe_ms": 120_000,
            }
        }
    )

    sim = ExecutionSimulator(symbol="BTCUSDT", run_config=run_config)

    assert sim._bar_cap_base_enabled is True
    assert sim._bar_cap_base_frac == pytest.approx(0.25)
    assert sim._bar_cap_base_floor == pytest.approx(123.0)
    assert sim._bar_cap_base_path == str(dataset_path)
    assert sim._bar_cap_base_timeframe_ms == 120_000


def test_resolve_cap_base_per_bar_dataset_and_floor(tmp_path, caplog):
    dataset_path = tmp_path / "adv_base.json"
    _write_dataset(
        dataset_path,
        {
            "data": {
                "BTCUSDT": {"adv": 1_440.0, "floor": 5_760.0},
            },
            "floors": {"ETHUSDT": 2_880.0, "LTCUSDT": 8_640.0},
        },
    )

    sim = ExecutionSimulator(symbol="BTCUSDT")
    sim.set_bar_capacity_base_config(
        enabled=True,
        adv_base_path=str(dataset_path),
        timeframe_ms=60_000,
        capacity_frac_of_ADV_base=0.25,
        floor_base=4_320.0,
    )

    bars_per_day = 86_400_000 / 60_000
    caplog.set_level(logging.WARNING)

    btc_capacity = sim._resolve_cap_base_per_bar("BTCUSDT", 60_000)
    expected_btc = (5_760.0 * 0.25) / bars_per_day
    assert math.isclose(btc_capacity, expected_btc, rel_tol=1e-9)

    caplog.clear()
    eth_capacity = sim._resolve_cap_base_per_bar("ETHUSDT", 60_000)
    expected_eth = (4_320.0 * 0.25) / bars_per_day
    assert math.isclose(eth_capacity, expected_eth, rel_tol=1e-9)
    assert len(caplog.records) == 1
    assert "using floor" in caplog.records[0].message

    caplog.clear()
    repeat_eth = sim._resolve_cap_base_per_bar("ETHUSDT", 60_000)
    assert math.isclose(repeat_eth, expected_eth, rel_tol=1e-9)
    assert len(caplog.records) == 0

    caplog.clear()
    ltc_capacity = sim._resolve_cap_base_per_bar("LTCUSDT", 60_000)
    expected_ltc = (8_640.0 * 0.25) / bars_per_day
    assert math.isclose(ltc_capacity, expected_ltc, rel_tol=1e-9)
    assert len(caplog.records) == 1
    assert "dataset floor" in caplog.records[0].message

    caplog.clear()
    repeat_ltc = sim._resolve_cap_base_per_bar("LTCUSDT", 60_000)
    assert math.isclose(repeat_ltc, expected_ltc, rel_tol=1e-9)
    assert len(caplog.records) == 0

    _write_dataset(
        dataset_path,
        {
            "data": {
                "BTCUSDT": {"adv": 2_880.0},
                "ETHUSDT": {"adv": 5_760.0},
            },
            "floors": {"LTCUSDT": 8_640.0},
        },
    )

    caplog.clear()
    updated_eth = sim._resolve_cap_base_per_bar("ETHUSDT", 60_000)
    expected_updated_eth = (5_760.0 * 0.25) / bars_per_day
    assert math.isclose(updated_eth, expected_updated_eth, rel_tol=1e-9)
    assert "ETHUSDT" not in sim._bar_cap_base_warned_symbols
    assert len(caplog.records) == 0

    updated_btc = sim._resolve_cap_base_per_bar("BTCUSDT", 60_000)
    expected_updated_btc = (4_320.0 * 0.25) / bars_per_day
    assert math.isclose(updated_btc, expected_updated_btc, rel_tol=1e-9)


def test_adv_bar_capacity_combines_sources_and_quote_fallback(tmp_path):
    dataset_path = tmp_path / "adv_base.json"
    _write_dataset(
        dataset_path,
        {
            "data": {
                "BTCUSDT": {"adv": 7_200.0},
                "ETHUSDT": {"adv": 2_880.0},
            }
        },
    )

    sim = ExecutionSimulator(symbol="BTCUSDT")
    sim.set_bar_capacity_base_config(
        enabled=True,
        adv_base_path=str(dataset_path),
        timeframe_ms=60_000,
        capacity_frac_of_ADV_base=0.25,
    )

    bars_per_day = 86_400_000 / 60_000

    adv_store = _StubADVStore({"BTCUSDT": 28_800.0}, default=None, floor=21_600.0)
    sim.set_adv_store(adv_store, enabled=True, capacity_fraction=0.5)

    base_btc = sim._resolve_cap_base_per_bar("BTCUSDT", 60_000)
    adv_capacity = sim._adv_bar_capacity("BTCUSDT", 60_000)
    expected_adv_capacity = max(
        base_btc,
        max((28_800.0 / bars_per_day) * 0.5, 21_600.0 / bars_per_day),
    )
    assert math.isclose(adv_capacity, expected_adv_capacity, rel_tol=1e-9)
    assert adv_store.reset_calls == 1

    default_store = _StubADVStore({}, default=14_400.0, floor=None)
    sim.set_adv_store(default_store, enabled=True, capacity_fraction=1.0)
    default_quote = sim.get_bar_capacity_quote("ETHUSDT")
    assert default_quote == pytest.approx(14_400.0)
    base_eth = sim._resolve_cap_base_per_bar("ETHUSDT", 60_000)
    default_capacity = sim._adv_bar_capacity("ETHUSDT", 60_000)
    expected_default_capacity = max(base_eth, 14_400.0 / bars_per_day)
    assert math.isclose(default_capacity, expected_default_capacity, rel_tol=1e-9)

    none_store = _StubADVStore({}, default=None, floor=None)
    sim.set_adv_store(none_store, enabled=True, capacity_fraction=1.0)
    assert sim.get_bar_capacity_quote("ETHUSDT") is None
    none_capacity = sim._adv_bar_capacity("ETHUSDT", 60_000)
    assert math.isclose(none_capacity, base_eth, rel_tol=1e-9)
