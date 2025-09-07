# -*- coding: utf-8 -*-
"""
services/service_train.py
Сервис подготовки данных (офлайн) и запуска обучения модели.
Оркестрация: OfflineData -> FeaturePipe(offl) -> Dataset -> Trainer.fit -> сохранение артефактов.

Пример использования
--------------------
```python
from transformers import FeatureSpec
from offline_feature_pipe import OfflineFeaturePipe
from service_train import ServiceTrain, TrainConfig

spec = FeatureSpec(lookbacks_prices=[5, 15, 60], rsi_period=14)
fp = OfflineFeaturePipe(spec, price_col="ref_price")
trainer = ...  # реализация Trainer
cfg = TrainConfig(input_path="data/train.parquet")
ServiceTrain(fp, trainer, cfg).run()
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence, Iterable, Tuple
import os
import time
import pandas as pd

from services.utils_config import snapshot_config  # снапшот конфигурации


class FeaturePipe(Protocol):
    def warmup(self) -> None: ...
    def fit(self, df: pd.DataFrame) -> None: ...
    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame: ...
    # опционально: целевая переменная
    def make_targets(self, df: pd.DataFrame) -> Optional[pd.Series]: ...


class Trainer(Protocol):
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Any: ...
    def save(self, path: str) -> str: ...


@dataclass
class TrainConfig:
    input_path: str                       # путь к исходным данным (csv/parquet)
    input_format: str = "parquet"         # "parquet" | "csv"
    artifacts_dir: str = "artifacts"      # куда складывать датасеты и модель
    dataset_name: str = "train_dataset"   # базовое имя файлов датасета
    model_name: str = "model"             # базовое имя сохранённой модели
    columns_keep: Optional[Sequence[str]] = None  # если нужно отфильтровать
    snapshot_config_path: Optional[str] = None    # путь к YAML конфигу запуска


class ServiceTrain:
    """
    Подготавливает датасет и обучает переданный Trainer.
    Никакой бизнес-логики обучения внутри; только пайплайн.
    """
    def __init__(self, feature_pipe: FeaturePipe, trainer: Trainer, cfg: TrainConfig):
        self.fp = feature_pipe
        self.trainer = trainer
        self.cfg = cfg

    def _load_input(self) -> pd.DataFrame:
        fmt = str(self.cfg.input_format).lower()
        if fmt == "parquet":
            df = pd.read_parquet(self.cfg.input_path)
        elif fmt == "csv":
            df = pd.read_csv(self.cfg.input_path)
        else:
            raise ValueError(f"Unsupported input_format: {self.cfg.input_format}")
        return df

    def run(self) -> Dict[str, Any]:
        os.makedirs(self.cfg.artifacts_dir, exist_ok=True)
        if self.cfg.snapshot_config_path:
            snapshot_config(self.cfg.snapshot_config_path, self.cfg.artifacts_dir)

        # загрузка
        df_raw = self._load_input()

        # прогрев и обучение преобразований
        self.fp.warmup()
        self.fp.fit(df_raw)

        # построение фичей и таргета
        X = self.fp.transform_df(df_raw)
        y = None
        if hasattr(self.fp, "make_targets"):
            try:
                y = self.fp.make_targets(df_raw)  # type: ignore[attr-defined]
            except Exception:
                y = None

        # опциональная фильтрация колонок
        if self.cfg.columns_keep:
            cols = [c for c in self.cfg.columns_keep if c in X.columns]
            X = X[cols]

        # сохранение датасета
        ts = int(time.time())
        ds_base = os.path.join(self.cfg.artifacts_dir, f"{self.cfg.dataset_name}_{ts}")
        X_path = ds_base + "_X.parquet"
        y_path = ds_base + "_y.parquet"
        X.to_parquet(X_path, index=False)
        if y is not None:
            pd.DataFrame({"y": y}).to_parquet(y_path, index=False)

        # обучение модели
        self.trainer.fit(X, y)
        model_path = os.path.join(self.cfg.artifacts_dir, f"{self.cfg.model_name}_{ts}.bin")
        saved_path = self.trainer.save(model_path)

        return {
            "dataset_X": X_path,
            "dataset_y": (y_path if y is not None else None),
            "model_path": saved_path,
            "n_samples": int(len(X)),
            "n_features": int(len(X.columns)),
        }
