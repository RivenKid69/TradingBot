import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.run_seasonality_pipeline import (
    PipelineOptions,
    build_pipeline_steps,
    run_pipeline,
)


@pytest.fixture
def base_options(tmp_path: Path) -> PipelineOptions:
    data = tmp_path / "dataset.parquet"
    data.write_text("dummy")
    seasonality = tmp_path / "out" / "multipliers.json"
    return PipelineOptions(
        data=data,
        seasonality_out=seasonality,
        run_plot=False,
        plots_dir=tmp_path / "plots",
        run_validate=False,
        validation_threshold=0.1,
        run_train=True,
        train_config="configs/config_train.yaml",
        regime_config="configs/market_regimes.json",
        dataset_split="none",
        symbol=None,
        build_args=["--smooth", "3"],
        validate_args=[],
        train_args=["--epochs", "2"],
    )


def test_build_pipeline_steps_default(base_options: PipelineOptions) -> None:
    steps = build_pipeline_steps(base_options)
    assert steps[0][0] == "build seasonality"
    build_cmd = steps[0][1]
    assert "scripts/build_hourly_seasonality.py" in build_cmd
    assert "--smooth" in build_cmd

    assert steps[-1][0] == "train model"
    train_cmd = steps[-1][1]
    assert "train_model_multi_patch.py" in train_cmd
    assert "--dataset-split" in train_cmd
    assert "none" in train_cmd
    assert "--epochs" in train_cmd


def test_build_pipeline_steps_skip_optional(base_options: PipelineOptions) -> None:
    opts = replace(
        base_options,
        run_plot=True,
        run_validate=True,
        validate_args=["--symbol", "BTCUSDT"],
        symbol="BTCUSDT",
    )
    steps = build_pipeline_steps(opts)
    titles = [title for title, _ in steps]
    assert titles == ["build seasonality", "plot seasonality", "validate seasonality", "train model"]
    validate_cmd = dict(steps)["validate seasonality"]
    assert "BTCUSDT" in validate_cmd


def test_run_pipeline_requires_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing_data = tmp_path / "missing.parquet"
    options = PipelineOptions(
        data=missing_data,
        seasonality_out=tmp_path / "out.json",
        run_plot=False,
        plots_dir=tmp_path / "plots",
        run_validate=False,
        validation_threshold=0.1,
        run_train=False,
        train_config="configs/config_train.yaml",
        regime_config="configs/market_regimes.json",
        dataset_split=None,
        symbol=None,
        build_args=[],
        validate_args=[],
        train_args=[],
    )

    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return subprocess.CompletedProcess(_args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(FileNotFoundError):
        run_pipeline(options)

    assert not called
