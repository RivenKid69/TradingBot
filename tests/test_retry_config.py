import json
import types
import sys

import service_signal_runner
from core_config import (
    SimulationConfig,
    SimulationDataConfig,
    ComponentSpec,
    Components,
)


class _Guards:
    def apply(self, ts_ms, symbol, decisions):
        return decisions, None


def _make_stub_module() -> None:
    mod = types.ModuleType("stub_comp")
    class MarketData:  # noqa: D401 - simple stub
        pass
    class FeaturePipe:
        def warmup(self):
            pass
    class Policy:  # noqa: D401 - simple stub
        pass
    class Executor:  # noqa: D401 - simple stub
        pass
    mod.MarketData = MarketData
    mod.FeaturePipe = FeaturePipe
    mod.Policy = Policy
    mod.Executor = Executor
    mod.Guards = _Guards
    sys.modules["stub_comp"] = mod


def test_retry_config_override(tmp_path, monkeypatch):
    _make_stub_module()
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "ops.json").write_text(
        json.dumps({
            "retry": {
                "max_attempts": 7,
                "backoff_base_s": 0.5,
                "max_backoff_s": 5.0,
            }
        })
    )

    comps = Components(
        market_data=ComponentSpec(target="stub_comp:MarketData"),
        executor=ComponentSpec(target="stub_comp:Executor"),
        feature_pipe=ComponentSpec(target="stub_comp:FeaturePipe"),
        policy=ComponentSpec(target="stub_comp:Policy"),
        risk_guards=ComponentSpec(target="stub_comp:Guards"),
    )
    cfg = SimulationConfig(
        components=comps,
        data=SimulationDataConfig(symbols=["BTCUSDT"], timeframe="1m"),
        symbols=["BTCUSDT"],
    )

    class _DummyRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            return iter(())

    monkeypatch.setattr(service_signal_runner, "ServiceSignalRunner", _DummyRunner)

    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        list(service_signal_runner.from_config(cfg))

    assert cfg.retry.max_attempts == 7
    assert cfg.retry.backoff_base_s == 0.5
    assert cfg.retry.max_backoff_s == 5.0
