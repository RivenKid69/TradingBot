import numpy as np
import pandas as pd
import pytest
import pathlib
import os
import sys

sys.path.append(os.getcwd())

from service_train import ServiceTrain, TrainConfig


class DummyFeaturePipe:
    def warmup(self):
        pass

    def fit(self, df: pd.DataFrame):
        pass

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[["feat"]]

    def make_targets(self, df: pd.DataFrame) -> pd.Series:
        return df["target"]


class CaptureTrainer:
    def __init__(self):
        self.X = None
        self.y = None
        self.sample_weight = None

    def fit(self, X, y=None, sample_weight=None):
        self.X = X
        self.y = y
        self.sample_weight = sample_weight

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write("dummy")
        return path


def _make_dataset(tmp_path: pathlib.Path) -> pathlib.Path:
    ts = np.arange(0, 24 * 60, dtype=np.int64) * 60_000
    df = pd.DataFrame({"ts_ms": ts, "feat": ts.astype(float), "target": np.ones_like(ts)})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path, df


def test_drop_and_weight_modes_same_effective_samples(tmp_path):
    data_path, df = _make_dataset(tmp_path)
    fp = DummyFeaturePipe()

    trainer_drop = CaptureTrainer()
    cfg_drop = TrainConfig(
        input_path=str(data_path),
        input_format="csv",
        artifacts_dir=str(tmp_path / "art_drop"),
        no_trade_mode="drop",
        no_trade_config_path="configs/legacy_sandbox.yaml",
    )
    service_drop = ServiceTrain(fp, trainer_drop, cfg_drop)
    res_drop = service_drop.run()

    trainer_weight = CaptureTrainer()
    cfg_weight = TrainConfig(
        input_path=str(data_path),
        input_format="csv",
        artifacts_dir=str(tmp_path / "art_weight"),
        no_trade_mode="weight",
        no_trade_config_path="configs/legacy_sandbox.yaml",
    )
    service_weight = ServiceTrain(fp, trainer_weight, cfg_weight)
    res_weight = service_weight.run()

    assert res_drop["effective_samples"] == pytest.approx(res_weight["effective_samples"])
    assert trainer_weight.sample_weight is not None
    assert int(trainer_weight.sample_weight.sum()) == res_weight["effective_samples"]
    assert len(trainer_weight.X) == len(df)
