from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from pathlib import Path

import logging

import pytest

import service_signal_runner
from core_config import (
    Components,
    ComponentSpec,
    SimulationConfig,
    SimulationDataConfig,
)
from core_models import Bar
from pipeline import PipelineConfig, PipelineResult, Stage, Reason


class _DummyMetric:
    def labels(self, *args, **kwargs):  # noqa: D401 - simple passthrough
        return self

    def inc(self, *args, **kwargs) -> None:
        return None

    def set(self, *args, **kwargs) -> None:
        return None


def test_worker_forwards_close_lag_to_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, int] = {}

    def _guard(**kwargs):
        recorded["lag_ms"] = kwargs.get("lag_ms")
        return PipelineResult(
            action="drop",
            stage=Stage.CLOSED_BAR,
            reason=Reason.INCOMPLETE_BAR,
        )

    monkeypatch.setattr(service_signal_runner, "closed_bar_guard", _guard)
    dummy_metric = _DummyMetric()
    monkeypatch.setattr(service_signal_runner, "skipped_incomplete_bars", dummy_metric)
    monkeypatch.setattr(service_signal_runner, "pipeline_stage_drop_count", dummy_metric)
    monkeypatch.setattr(service_signal_runner.monitoring, "record_signals", lambda *args, **kwargs: None)
    monkeypatch.setattr(service_signal_runner.monitoring, "signal_error_rate", dummy_metric)
    monkeypatch.setattr(service_signal_runner.monitoring, "inc_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(service_signal_runner.monitoring, "inc_reason", lambda *args, **kwargs: None)

    worker = service_signal_runner._Worker(
        fp=SimpleNamespace(update=lambda bar, skip_metrics=False: {}, signal_quality={}),
        policy=SimpleNamespace(decide=lambda feats, ctx: []),
        logger=logging.getLogger("test"),
        executor=SimpleNamespace(submit=lambda order: None),
        guards=None,
        enforce_closed_bars=True,
        close_lag_ms=321,
        pipeline_cfg=PipelineConfig(),
    )

    bar = Bar(
        ts=1,
        symbol="BTCUSDT",
        open=Decimal("1"),
        high=Decimal("1"),
        low=Decimal("1"),
        close=Decimal("1"),
        volume_quote=Decimal("0"),
        is_final=True,
    )

    worker.process(bar)

    assert recorded["lag_ms"] == 321


def test_from_config_propagates_close_lag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, int | None] = {}

    class _DummyRunner:
        def __init__(self, *args, close_lag_ms: int | None = None, **kwargs):
            captured["close_lag_ms"] = close_lag_ms

        def run(self):
            return iter(())

    class _Adapter:
        pass

    container = {
        "executor": _Adapter(),
        "feature_pipe": SimpleNamespace(),
        "policy": SimpleNamespace(),
        "risk_guards": None,
    }

    def _build_graph(*args, **kwargs):
        return container

    monkeypatch.setattr(service_signal_runner, "ServiceSignalRunner", _DummyRunner)
    monkeypatch.setattr(service_signal_runner.di_registry, "build_graph", _build_graph)

    comps = Components(
        market_data=ComponentSpec(target="stub.module:MarketData"),
        executor=ComponentSpec(target="stub.module:Executor"),
        feature_pipe=ComponentSpec(target="stub.module:FeaturePipe"),
        policy=ComponentSpec(target="stub.module:Policy"),
        risk_guards=ComponentSpec(target="stub.module:Guards"),
    )
    cfg = SimulationConfig(
        components=comps,
        data=SimulationDataConfig(symbols=["BTCUSDT"], timeframe="1m"),
        symbols=["BTCUSDT"],
    )

    cfg.timing.close_lag_ms = 4321
    cfg.logs_dir = str(tmp_path / "logs")
    cfg.artifacts_dir = str(tmp_path / "artifacts")

    (tmp_path / "logs").mkdir()
    (tmp_path / "artifacts").mkdir()

    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        list(service_signal_runner.from_config(cfg))

    assert captured["close_lag_ms"] == 4321
