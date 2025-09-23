from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace

import yaml

# Provide a lightweight ``requests`` stub if the dependency is absent.
if "requests" not in sys.modules:  # pragma: no cover - test environment helper
    class _DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

        def raise_for_status(self):
            return None

    class _DummyRequests(ModuleType):
        def get(self, url, params=None, timeout=0):  # pragma: no cover - simple stub
            if "ticker/24hr" in url:
                return _DummyResponse([
                    {"symbol": "BTCUSDT", "quoteVolume": 1_000_000}
                ])
            return _DummyResponse({
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "status": "TRADING",
                        "quoteAsset": "USDT",
                        "permissions": ["SPOT"],
                    }
                ]
            })

    sys.modules["requests"] = _DummyRequests("requests")

from core_config import Components, load_config
from core_config import RetryConfig
import di_registry
from impl_quantizer import QuantizerImpl, QuantizerConfig
from quantizer import Quantizer


def _components_stub() -> Components:
    data = {
        "market_data": {"target": "tests.di_stubs:DummyMarketData", "params": {}},
        "executor": {"target": "tests.di_stubs:DummyExecutor", "params": {}},
        "feature_pipe": {"target": "tests.di_stubs:DummyFeaturePipe", "params": {}},
        "policy": {"target": "tests.di_stubs:DummyPolicy", "params": {}},
        "risk_guards": {"target": "tests.di_stubs:DummyRiskGuards", "params": {}},
    }
    return Components.parse_obj(data)


def test_load_config_preserves_quantizer_section(tmp_path):
    config_path = tmp_path / "live.yaml"
    filters_path = tmp_path / "filters.json"
    cfg_dict = {
        "mode": "live",
        "api": {"api_key": "k", "api_secret": "s", "testnet": True},
        "data": {"symbols": ["BTCUSDT"], "timeframe": "1m"},
        "components": {
            "market_data": {"target": "tests.di_stubs:DummyMarketData"},
            "executor": {"target": "tests.di_stubs:DummyExecutor"},
            "feature_pipe": {"target": "tests.di_stubs:DummyFeaturePipe"},
            "policy": {"target": "tests.di_stubs:DummyPolicy"},
            "risk_guards": {"target": "tests.di_stubs:DummyRiskGuards"},
        },
        "quantizer": {
            "path": str(filters_path),
            "filters_path": str(filters_path),
            "auto_refresh_days": 5,
            "refresh_on_start": False,
        },
    }
    config_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    cfg = load_config(str(config_path))
    assert hasattr(cfg, "quantizer")
    assert cfg.quantizer == cfg_dict["quantizer"]


def test_build_graph_provides_quantizer_instance(tmp_path):
    filters_path = tmp_path / "filters.json"
    filters_path.write_text("{\"filters\": {}}", encoding="utf-8")
    components = _components_stub()
    run_cfg = SimpleNamespace(quantizer={"path": str(filters_path)}, retry=RetryConfig())

    container = di_registry.build_graph(components, run_cfg)

    assert "quantizer" in container
    quantizer = container["quantizer"]
    assert isinstance(quantizer, QuantizerImpl)
    executor = container["executor"]
    assert getattr(executor, "quantizer") is quantizer


def test_quantizer_warnings_are_logged(monkeypatch, caplog, tmp_path):
    import impl_quantizer

    filters_path = tmp_path / "filters.json"
    filters_path.write_text("{\"filters\": {}}", encoding="utf-8")

    def _fake_load_filters(path, max_age_days=0, fatal=False):
        import warnings

        warnings.warn("refresh recommended soon")
        return {"BTCUSDT": {"PRICE_FILTER": {}}}, {"generated_at": "2023-01-01T00:00:00Z"}

    monkeypatch.setattr(
        impl_quantizer.Quantizer,
        "load_filters",
        staticmethod(_fake_load_filters),
    )

    caplog.set_level(logging.WARNING, logger="impl_quantizer")
    QuantizerImpl(QuantizerConfig(path=str(filters_path), filters_path=str(filters_path)))

    assert any("refresh recommended soon" in record.message for record in caplog.records)


def test_quantizer_refresh_is_debounced(monkeypatch, tmp_path):
    import impl_quantizer

    QuantizerImpl._REFRESH_GUARD.clear()

    filters_path = tmp_path / "filters.json"
    filters_path.write_text("{\"filters\": {}}", encoding="utf-8")

    def _fake_load_filters(path, max_age_days=0, fatal=False):
        return {}, {}

    run_calls: list[list[str]] = []

    class _DummyCompletedProcess:
        def __init__(self):
            self.returncode = 0
            self.stdout = ""
            self.stderr = ""

    def _fake_run(cmd, capture_output=True, text=True):  # pragma: no cover - simple stub
        run_calls.append(list(cmd))
        return _DummyCompletedProcess()

    monkeypatch.setattr(
        impl_quantizer.Quantizer,
        "load_filters",
        staticmethod(_fake_load_filters),
    )
    monkeypatch.setattr(impl_quantizer.subprocess, "run", _fake_run)

    cfg = QuantizerConfig(
        path=str(filters_path),
        filters_path=str(filters_path),
        refresh_on_start=True,
        auto_refresh_days=1,
    )

    QuantizerImpl(cfg)
    QuantizerImpl(cfg)

    assert len(run_calls) == 1


def test_quantizer_accepts_percent_price_without_multipliers():
    filters = {
        "BTCUSDT": {
            "PRICE_FILTER": {"tickSize": "0.01"},
            "LOT_SIZE": {"stepSize": "0.001"},
            "MIN_NOTIONAL": {"minNotional": "10"},
            # Missing multiplierUp/multiplierDown should not raise
            "PERCENT_PRICE_BY_SIDE": {
                "bidMultiplierUp": "1.1",
                "bidMultiplierDown": "0.9",
            },
        }
    }

    quantizer = Quantizer(filters)
    symbol_filters = quantizer._filters["BTCUSDT"]
    assert symbol_filters.multiplier_up is None
    assert symbol_filters.multiplier_down is None
