import datetime
import importlib.util
import json
import pathlib
import sys

from typing import Optional

import pytest


BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

spec_lat = importlib.util.spec_from_file_location("latency", BASE / "latency.py")
lat_module = importlib.util.module_from_spec(spec_lat)
sys.modules["latency"] = lat_module
spec_lat.loader.exec_module(lat_module)

spec_impl = importlib.util.spec_from_file_location("impl_latency", BASE / "impl_latency.py")
impl_module = importlib.util.module_from_spec(spec_impl)
sys.modules["impl_latency"] = impl_module
spec_impl.loader.exec_module(impl_module)

LatencyImpl = impl_module.LatencyImpl
ExecutionSimulator = importlib.import_module("execution_sim").ExecutionSimulator
LatencyVolatilityCache = importlib.import_module(
    "latency_volatility_cache"
).LatencyVolatilityCache


class DummyCache:
    def __init__(self, multiplier: float, *, ready: bool = True):
        self.multiplier = float(multiplier)
        self.ready = ready
        self.calls = []
        self.updates = []

    def latency_multiplier(
        self,
        *,
        symbol: str,
        ts_ms: int,
        metric: str,
        window: int,
        gamma: float,
        clip: float,
    ):
        self.calls.append(
            {
                "symbol": symbol,
                "ts_ms": ts_ms,
                "metric": metric,
                "window": window,
                "gamma": gamma,
                "clip": clip,
            }
        )
        return self.multiplier, {"source": "dummy"}

    def update_latency_factor(
        self, *, symbol: str, ts_ms: int, value: float
    ) -> None:
        self.updates.append({"symbol": symbol, "ts_ms": ts_ms, "value": value})


class DummySim:
    def __init__(self, *, cache: Optional[DummyCache] = None, symbol: str = "BTCUSDT"):
        self.volatility_cache = cache
        self.symbol = symbol


def _make_impl(tmp_path, extra_cfg=None):
    multipliers = [1.0] * 168
    path = tmp_path / "latency.json"
    path.write_text(json.dumps({"latency": multipliers}))
    cfg = {
        "base_ms": 100,
        "jitter_ms": 0,
        "spike_p": 0.0,
        "timeout_ms": 1000,
        "seasonality_path": str(path),
        "symbol": "BTCUSDT",
    }
    if extra_cfg:
        cfg.update(extra_cfg)
    return LatencyImpl.from_dict(cfg)


def _sample_timestamp() -> int:
    base_dt = datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    return int(base_dt.timestamp() * 1000)


def test_latency_volatility_multiplier(tmp_path):
    cache = DummyCache(1.5)
    impl = _make_impl(tmp_path, {"volatility_gamma": 1.0, "debug_log": True})
    sim = DummySim(cache=cache, symbol="ETHUSDT")
    impl.attach_to(sim)
    lat = sim.latency

    result = lat.sample(_sample_timestamp())
    assert result["total_ms"] == 150
    assert cache.calls and cache.calls[0]["symbol"] == "ETHUSDT"
    debug = result.get("debug", {}).get("latency", {})
    assert debug.get("volatility_multiplier") == pytest.approx(1.5)
    vol_debug = debug.get("volatility_debug", {})
    assert vol_debug.get("source") == "dummy"
    assert vol_debug.get("symbol") == "ETHUSDT"
    expected_keys = {"value", "mean", "std", "zscore", "clip", "gamma", "window", "symbol", "ts"}
    assert expected_keys.issubset(vol_debug.keys())
    assert debug.get("jitter_component") == pytest.approx(0.0)
    assert debug.get("vol_adjust") == pytest.approx(50.0)


def test_latency_volatility_gamma_zero(tmp_path):
    cache = DummyCache(2.0)
    impl = _make_impl(tmp_path, {"volatility_gamma": 0.0})
    sim = DummySim(cache=cache)
    impl.attach_to(sim)
    lat = sim.latency

    result = lat.sample(_sample_timestamp())
    assert result["total_ms"] == 100
    assert cache.calls == []


def test_latency_volatility_cache_not_ready(tmp_path):
    cache = DummyCache(2.0, ready=False)
    impl = _make_impl(tmp_path, {"volatility_gamma": 1.0})
    sim = DummySim(cache=cache)
    impl.attach_to(sim)
    lat = sim.latency

    result = lat.sample(_sample_timestamp())
    assert result["total_ms"] == 100
    assert cache.calls == []


def test_latency_clamps_to_bounds(tmp_path):
    cache = DummyCache(3.0)
    impl = _make_impl(
        tmp_path,
        {
            "volatility_gamma": 1.0,
            "min_ms": 120,
            "max_ms": 180,
        },
    )
    sim = DummySim(cache=cache)
    impl.attach_to(sim)
    lat = sim.latency

    result = lat.sample(_sample_timestamp())
    assert result["total_ms"] == 180
    assert not result["timeout"]


def test_latency_update_forwards_to_cache(tmp_path):
    cache = DummyCache(1.0)
    impl = _make_impl(tmp_path, {"volatility_gamma": 1.0})
    sim = DummySim(cache=cache, symbol="ethusdt")
    impl.attach_to(sim)

    lat = sim.latency
    lat.update_volatility(None, 1234567890, 0.25)
    assert cache.updates
    entry = cache.updates[-1]
    assert entry["symbol"] == "ETHUSDT"
    assert entry["ts_ms"] == 1234567890
    assert entry["value"] == pytest.approx(0.25)


def test_latency_volatility_cache_ready_and_multiplier():
    cache = LatencyVolatilityCache(window=3)
    symbol = "btcusdt"
    cache.update_latency_factor(symbol=symbol, ts_ms=1, value=1.0)
    cache.update_latency_factor(symbol=symbol, ts_ms=2, value=2.0)
    cache.update_latency_factor(symbol=symbol, ts_ms=3, value=3.0)

    assert cache.is_ready(symbol)

    mult, debug = cache.latency_multiplier(
        symbol=symbol,
        ts_ms=3,
        metric="sigma",
        window=3,
        gamma=0.5,
        clip=3.0,
    )

    assert mult == pytest.approx(1.0 + 0.5 * debug["zscore"])
    assert debug["count"] == 3
    assert debug["symbol"] == "BTCUSDT"
    assert debug["vol_mult"] == pytest.approx(mult)


def test_latency_volatility_cache_not_ready_reason():
    cache = LatencyVolatilityCache(window=5, min_ready=4)
    symbol = "ethusdt"
    cache.update_latency_factor(symbol=symbol, ts_ms=1, value=1.0)
    cache.update_latency_factor(symbol=symbol, ts_ms=2, value=2.0)

    assert not cache.is_ready(symbol)

    mult, debug = cache.latency_multiplier(
        symbol=symbol,
        ts_ms=2,
        metric="sigma",
        window=5,
        gamma=1.0,
        clip=2.0,
    )

    assert mult == 1.0
    assert debug["reason"] == "not_ready"
    assert debug["count"] == 2
    assert debug["required"] == 4


def test_execution_simulator_updates_latency():
    sim = ExecutionSimulator(symbol="adausdt")

    class _Lat:
        def __init__(self) -> None:
            self.calls = []

        def update_volatility(self, symbol, ts_ms, value):
            self.calls.append((symbol, ts_ms, value))

    lat = _Lat()
    sim.latency = lat
    cache = DummyCache(1.0)
    sim.volatility_cache = cache
    sim._latency_symbol = "ADAUSDT"

    sim.set_market_snapshot(bid=1.0, ask=1.1, vol_factor=0.3, ts_ms=1_000)
    assert lat.calls
    assert lat.calls[-1] == ("ADAUSDT", 1000, 0.3)
    assert cache.updates
    cache_entry = cache.updates[-1]
    assert cache_entry["symbol"] == "ADAUSDT"
    assert cache_entry["ts_ms"] == 1000
    assert cache_entry["value"] == pytest.approx(0.3)

    lat.calls.clear()
    cache.updates.clear()
    sim.run_step(
        ts=2_000,
        ref_price=1.05,
        vol_factor=0.5,
        vol_raw={"sigma": 0.25},
        liquidity=1.0,
    )
    assert lat.calls
    assert lat.calls[-1] == ("ADAUSDT", 2000, 0.5)
    assert cache.updates
    cache_entry = cache.updates[-1]
    assert cache_entry["symbol"] == "ADAUSDT"
    assert cache_entry["ts_ms"] == 2000
    assert cache_entry["value"] == pytest.approx(0.25)

