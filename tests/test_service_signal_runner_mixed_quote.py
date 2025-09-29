from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import service_signal_runner
from service_signal_runner import MixedQuoteError


def test_service_signal_runner_rejects_mixed_quote(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    logs_dir = tmp_path / "logs"
    artifacts_dir = tmp_path / "artifacts"
    logs_dir.mkdir()
    artifacts_dir.mkdir()

    run_config = SimpleNamespace(
        symbols=["BTCUSDT", "ETHBTC"],
        data=SimpleNamespace(symbols=["BTCUSDT", "ETHBTC"]),
        symbol_specs={
            "BTCUSDT": {"quote_asset": "USDT"},
            "ETHBTC": {"quote_asset": "BTC"},
        },
        execution=SimpleNamespace(mode="order"),
        slippage_regime_updates=False,
        slippage_calibration_enabled=False,
        portfolio=None,
        components=SimpleNamespace(),
    )

    cfg = service_signal_runner.SignalRunnerConfig(
        logs_dir=str(logs_dir),
        artifacts_dir=str(artifacts_dir),
    )

    caplog.set_level("INFO")

    with pytest.raises(MixedQuoteError):
        service_signal_runner.ServiceSignalRunner(
            adapter=SimpleNamespace(),
            feature_pipe=SimpleNamespace(),
            policy=SimpleNamespace(),
            risk_guards=None,
            cfg=cfg,
            monitoring_cfg=None,
            monitoring_agg=None,
            run_config=run_config,
        )

    status_path = logs_dir / "runner_status.json"
    assert status_path.exists()
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_payload["init"]["reason"] == "mixed_quote"
    assert status_payload["init"]["quotes"] == {"BTCUSDT": "USDT", "ETHBTC": "BTC"}
    assert "mixed_quote" in caplog.text
