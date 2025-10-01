import sys
import types
from typing import Any

import pytest


if "exchange" not in sys.modules:
    exchange_pkg = types.ModuleType("exchange")
    exchange_pkg.__path__ = []  # mark as package
    specs_mod = types.ModuleType("exchange.specs")
    specs_mod.load_specs = lambda *args, **kwargs: {}
    specs_mod.round_price_to_tick = lambda price, symbol=None: price
    sys.modules["exchange"] = exchange_pkg
    sys.modules["exchange.specs"] = specs_mod

from service_backtest import ADVStore, _configure_adv_runtime


class DummyRunCfg(types.SimpleNamespace):
    pass


class DummySimulator:
    def __init__(self, *, adv_store: ADVStore | None = None, has_store: bool = False):
        self._adv_store = adv_store
        self._has_store = has_store
        self.calls: list[dict[str, Any]] = []

    def has_adv_store(self) -> bool:
        return self._has_store

    def set_adv_store(
        self,
        store: ADVStore,
        *,
        enabled: bool,
        capacity_fraction: float | None,
        bars_per_day_override: int | None,
    ) -> None:
        self.calls.append(
            {
                "store": store,
                "enabled": enabled,
                "capacity_fraction": capacity_fraction,
                "bars_per_day_override": bars_per_day_override,
            }
        )


@pytest.fixture
def base_exec_cfg() -> dict[str, Any]:
    return {"bar_capacity_base": {"enabled": True}}


def test_reuses_existing_store_and_forwards_primary_config(base_exec_cfg):
    existing_store = ADVStore({})
    adv_cfg = types.SimpleNamespace(
        enabled=True,
        capacity_fraction=0.25,
        bars_per_day_override=180,
    )
    run_cfg = DummyRunCfg(adv=adv_cfg, execution=base_exec_cfg)
    sim = DummySimulator(adv_store=existing_store, has_store=True)

    store, bar_capacity = _configure_adv_runtime(sim, run_cfg, context="test-run")

    assert store is existing_store
    assert bar_capacity == {"enabled": True}
    assert sim.calls == [
        {
            "store": existing_store,
            "enabled": True,
            "capacity_fraction": 0.25,
            "bars_per_day_override": 180,
        }
    ]


def test_creates_fresh_store_and_uses_extra_overrides(base_exec_cfg):
    adv_cfg = types.SimpleNamespace(
        enabled=True,
        extra={
            "capacity_fraction": 0.5,
            "bars_per_day": 96,
        },
    )
    run_cfg = DummyRunCfg(adv=adv_cfg, execution=base_exec_cfg)
    sim = DummySimulator()

    store, bar_capacity = _configure_adv_runtime(sim, run_cfg, context="fresh")

    assert isinstance(store, ADVStore)
    assert bar_capacity == {"enabled": True}
    assert sim.calls == [
        {
            "store": store,
            "enabled": True,
            "capacity_fraction": 0.5,
            "bars_per_day_override": 96,
        }
    ]


def test_missing_set_adv_store_logs_warning_and_disables(caplog, base_exec_cfg):
    caplog.set_level("WARNING")
    adv_cfg = types.SimpleNamespace(enabled=True)
    run_cfg = DummyRunCfg(adv=adv_cfg, execution=base_exec_cfg)
    sim = types.SimpleNamespace()

    store, bar_capacity = _configure_adv_runtime(sim, run_cfg, context="no-api")

    assert store is None
    assert bar_capacity == {"enabled": True}
    assert "lacks set_adv_store" in caplog.text


def test_advstore_initialisation_failure_returns_none(monkeypatch, base_exec_cfg, caplog):
    caplog.set_level("ERROR")
    adv_cfg = types.SimpleNamespace(enabled=True)
    run_cfg = DummyRunCfg(adv=adv_cfg, execution=base_exec_cfg)
    sim = DummySimulator()

    class ExplodingStore:
        def __init__(self, _cfg):
            raise RuntimeError("fail")

    monkeypatch.setattr("service_backtest.ADVStore", ExplodingStore)

    store, bar_capacity = _configure_adv_runtime(sim, run_cfg, context="boom-init")

    assert store is None
    assert bar_capacity == {"enabled": True}
    assert "failed to initialise ADV store" in caplog.text


def test_set_adv_store_failure_logs_and_returns_none(base_exec_cfg, caplog):
    caplog.set_level("ERROR")
    adv_cfg = types.SimpleNamespace(enabled=True)
    run_cfg = DummyRunCfg(adv=adv_cfg, execution=base_exec_cfg)

    class FailingSimulator(DummySimulator):
        def set_adv_store(self, *args, **kwargs):
            raise RuntimeError("attach fail")

    sim = FailingSimulator(adv_store=None)

    store, bar_capacity = _configure_adv_runtime(sim, run_cfg, context="boom-set")

    assert store is None
    assert bar_capacity == {"enabled": True}
    assert "failed to attach ADV store" in caplog.text
