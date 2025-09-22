import json
from types import SimpleNamespace

from impl_sim_executor import SimExecutor
from impl_slippage import load_calibration_artifact
from execution_sim import ExecutionSimulator


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
