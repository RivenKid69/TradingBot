import json
from typing import Any, Callable
from types import SimpleNamespace

import pandas as pd

from core_config import MonitoringConfig
from impl_sim_executor import SimExecutor
from impl_slippage import SlippageImpl, load_calibration_artifact
from execution_sim import ExecutionSimulator
from scripts.calibrate_live_slippage import _prepare_dataframe
from service_signal_runner import ServiceSignalRunner, SignalRunnerConfig
from slippage import CalibratedProfilesConfig, SymbolCalibratedProfile


def _write_artifact(tmp_path, payload, name="slippage_calibration.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_load_calibration_artifact_success(tmp_path):
    payload = {
        "generated_at": "2023-01-02T03:04:05Z",
        "total_samples": 42,
        "symbols": {
            "btcusdt": {
                "notional_curve": [
                    {"qty": 1, "impact_bps": 12},
                    {"qty": 5, "impact_bps": 24},
                ],
                "hourly_multipliers": [1.0, 1.1, 1.2],
                "regime_multipliers": {"NORMAL": 1.0},
                "samples": 42,
            }
        },
    }
    artifact_path = _write_artifact(tmp_path, payload)

    config = load_calibration_artifact(
        str(artifact_path),
        default_symbol="btcusdt",
        symbols=["BTCUSDT"],
        enabled=True,
    )

    assert config is not None
    assert config["enabled"] is True
    assert config["default_symbol"] == "BTCUSDT"
    assert config["metadata"]["artifact_path"] == str(artifact_path)
    assert "last_refresh_ts" in config and isinstance(config["last_refresh_ts"], int)
    assert "BTCUSDT" in config["symbols"]


def test_load_calibration_artifact_empty_after_filter(tmp_path):
    payload = {
        "generated_at": "2023-02-03T04:05:06+00:00",
        "symbols": {
            "btcusdt": {
                "notional_curve": [
                    {"qty": 1, "impact_bps": 5},
                ],
            }
        },
    }
    artifact_path = _write_artifact(tmp_path, payload, name="calibration.json")

    config = load_calibration_artifact(
        str(artifact_path),
        default_symbol="ethusdt",
        symbols=["ETHUSDT"],
        enabled=True,
    )

    assert config is not None
    assert config["enabled"] is False
    assert config["symbols"] == {}
    assert config["metadata"]["artifact_path"] == str(artifact_path)


def test_prepare_slippage_payload_merges_artifact(tmp_path):
    payload = {
        "generated_at": "2023-05-06T07:08:09Z",
        "symbols": {
            "foo": {
                "notional_curve": [
                    {"qty": 1, "impact_bps": 10},
                ],
                "hourly_multipliers": [1.0],
            }
        },
    }
    artifact_path = _write_artifact(tmp_path, payload, name="foo.json")
    existing_cfg = {
        "calibrated_profiles": {
            "enabled": False,
            "symbols": {
                "FOO": {"symbol": "FOO", "legacy": True},
            },
            "metadata": {"source": "static"},
        }
    }
    run_config = SimpleNamespace(
        slippage_calibration_enabled=True,
        slippage_calibration_path=str(artifact_path),
        slippage_calibration_default_symbol=None,
        artifacts_dir=str(tmp_path),
    )

    result = SimExecutor._prepare_slippage_payload(
        existing_cfg,
        run_config=run_config,
        symbol="FOO",
    )

    profiles = result["calibrated_profiles"]
    assert profiles["enabled"] is True
    assert "FOO" in profiles["symbols"]
    assert profiles["symbols"]["FOO"]["symbol"] == "FOO"
    metadata = profiles.get("metadata")
    assert metadata is not None
    assert metadata["source"] == "static"
    assert metadata["artifact_path"] == str(artifact_path)
    assert existing_cfg["calibrated_profiles"]["enabled"] is False


def test_prepare_slippage_payload_disabled_keeps_profiles():
    existing_cfg = {
        "calibrated_profiles": {
            "enabled": True,
            "metadata": {"origin": "inline"},
        }
    }
    run_config = SimpleNamespace(slippage_calibration_enabled=False)

    result = SimExecutor._prepare_slippage_payload(
        existing_cfg,
        run_config=run_config,
        symbol="FOO",
    )

    profiles = result["calibrated_profiles"]
    assert profiles["enabled"] is False
    assert existing_cfg["calibrated_profiles"]["enabled"] is True


def test_prepare_dataframe_includes_market_regime_column():
    source = pd.DataFrame(
        {
            "ts_ms": [1, 2],
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "slippage_bps": [1.0, 2.0],
            "spread_bps": [0.1, 0.2],
            "notional": [100.0, 200.0],
            "size": [0.01, 0.02],
            "liquidity": [10.0, 20.0],
            "vol_factor": [0.5, 0.6],
            "execution_profile": ["DEFAULT", "DEFAULT"],
            "market_regime": ["TREND", None],
        }
    )

    prepared = _prepare_dataframe(source)
    assert "market_regime" in prepared.columns
    values = prepared.set_index("symbol")["market_regime"].to_dict()
    assert values["BTCUSDT"] == "TREND"
    assert pd.isna(values["ETHUSDT"])

    without_regime = source.drop(columns=["market_regime"])
    prepared_no_regime = _prepare_dataframe(without_regime)
    assert "market_regime" in prepared_no_regime.columns
    assert prepared_no_regime["market_regime"].isna().all()


def test_execution_simulator_market_regime_listener():
    sim = ExecutionSimulator(symbol="BTCUSDT")
    events: list = []

    sim.register_market_regime_listener(events.append)
    sim.set_market_regime_hint("TREND")
    assert events == ["TREND"]

    sim.set_market_regime_hint("TREND")
    assert events == ["TREND"]

    sim.set_market_regime_hint("FLAT")
    assert events == ["TREND", "FLAT"]

    replay_events: list = []
    sim.register_market_regime_listener(replay_events.append)
    assert replay_events == ["FLAT"]


def test_service_runner_bridges_market_regime_to_slippage(tmp_path):
    logs_dir = tmp_path / "logs"
    artifacts_dir = tmp_path / "artifacts"
    logs_dir.mkdir()
    artifacts_dir.mkdir()

    profile = SymbolCalibratedProfile.from_dict(
        {
            "symbol": "BTCUSDT",
            "impact_curve": [{"qty": 1, "impact_bps": 10}],
        }
    )
    calibrated = CalibratedProfilesConfig.from_dict(
        {
            "enabled": True,
            "symbols": {"BTCUSDT": profile.to_dict()},
        }
    )
    slippage = SlippageImpl.from_dict({"calibrated_profiles": calibrated})
    slippage._calibration_symbols = {"BTCUSDT": profile}
    slippage._calibration_enabled = True

    class _Sim:
        def __init__(self) -> None:
            self._listeners: list[Callable[[Any], None]] = []
            self._last_market_regime: Any = "NORMAL"

        def register_market_regime_listener(
            self, callback: Callable[[Any], None]
        ) -> None:
            self._listeners.append(callback)
            callback(self._last_market_regime)

        def push(self, regime: Any) -> None:
            self._last_market_regime = regime
            for cb in list(self._listeners):
                cb(regime)

    class _Adapter:
        def __init__(self, sim: _Sim, slip: SlippageImpl) -> None:
            self.sim = sim
            self._slippage_impl = slip

        def run_events(self, worker):  # pragma: no cover - not exercised
            return iter(())

    class _Pipe:
        spread_ttl_ms = 0

        def warmup(self) -> None:
            return None

        def reset(self) -> None:
            return None

        def update(self, bar, skip_metrics: bool = False) -> dict:
            return {}

    class _Policy:
        def decide(self, features, ctx):  # pragma: no cover - not exercised
            return []

    sim = _Sim()
    adapter = _Adapter(sim, slippage)
    cfg = SignalRunnerConfig(
        logs_dir=str(logs_dir),
        artifacts_dir=str(artifacts_dir),
        marker_path=str(tmp_path / "marker"),
        snapshot_metrics_json=str(tmp_path / "metrics.json"),
        snapshot_metrics_csv=str(tmp_path / "metrics.csv"),
    )
    run_cfg = SimpleNamespace(
        slippage_regime_updates=True,
        slippage_calibration_enabled=True,
    )

    ServiceSignalRunner(
        adapter,
        _Pipe(),
        _Policy(),
        None,
        cfg,
        monitoring_cfg=MonitoringConfig(enabled=False),
        run_config=run_cfg,
    )

    assert getattr(slippage, "_current_market_regime") == "NORMAL"

    sim.push("TREND")
    assert getattr(slippage, "_current_market_regime") == "TREND"
