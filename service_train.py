# -*- coding: utf-8 -*-
"""
services/service_train.py
Сервис подготовки данных (офлайн) и запуска обучения модели.
Оркестрация: OfflineData -> FeaturePipe(offl) -> Dataset -> Trainer.fit -> сохранение артефактов.

Пример использования через конфиг
---------------------------------
```python
from core_config import CommonRunConfig
from service_train import from_config, TrainConfig

cfg_run = CommonRunConfig(...)
trainer = ...  # реализация Trainer
train_cfg = TrainConfig(input_path="data/train.parquet")
from_config(cfg_run, trainer=trainer, train_cfg=train_cfg)
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Protocol
import os
import time
import pandas as pd

from no_trade import compute_no_trade_mask

from services.utils_config import snapshot_config  # снапшот конфигурации
from core_contracts import FeaturePipe
from core_config import CommonRunConfig
import di_registry


class Trainer(Protocol):
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None,
    ) -> Any:
        ...

    def save(self, path: str) -> str:
        ...


@dataclass
class TrainConfig:
    input_path: str                       # путь к исходным данным (csv/parquet)
    input_format: str = "parquet"         # "parquet" | "csv"
    artifacts_dir: str = "artifacts"      # куда складывать датасеты и модель
    dataset_name: str = "train_dataset"   # базовое имя файлов датасета
    model_name: str = "model"             # базовое имя сохранённой модели
    columns_keep: Optional[Sequence[str]] = None  # если нужно отфильтровать
    snapshot_config_path: Optional[str] = None    # путь к YAML конфигу запуска
    no_trade_mode: Optional[str] = None           # "drop" | "weight" | None
    no_trade_config_path: Optional[str] = None    # путь к sandbox.yaml
    no_trade_ts_col: str = "ts_ms"               # колонка временной метки


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

        weights: Optional[pd.Series] = None
        if self.cfg.no_trade_mode in {"drop", "weight"}:
            mask_block = compute_no_trade_mask(
                df_raw,
                sandbox_yaml_path=self.cfg.no_trade_config_path
                or self.cfg.snapshot_config_path
                or "configs/legacy_sandbox.yaml",
                ts_col=self.cfg.no_trade_ts_col,
            )
            if self.cfg.no_trade_mode == "drop":
                df_raw = df_raw.loc[~mask_block].reset_index(drop=True)
            else:
                df_raw = df_raw.copy()
                df_raw["train_weight"] = 1.0
                df_raw.loc[mask_block, "train_weight"] = 0.0
                weights = df_raw["train_weight"]

        # прогрев и обучение преобразований
        self.fp.warmup()
        self.fp.fit(df_raw)

        # построение фичей и таргета
        X = self.fp.transform_df(df_raw)
        y = None
        try:
            y = self.fp.make_targets(df_raw)
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
        self.trainer.fit(X, y, sample_weight=weights)
        model_path = os.path.join(self.cfg.artifacts_dir, f"{self.cfg.model_name}_{ts}.bin")
        saved_path = self.trainer.save(model_path)

        effective = int(weights.sum()) if weights is not None else int(len(X))

        return {
            "dataset_X": X_path,
            "dataset_y": (y_path if y is not None else None),
            "model_path": saved_path,
            "n_samples": int(len(X)),
            "n_features": int(len(X.columns)),
            "effective_samples": effective,
        }


def from_config(cfg: CommonRunConfig, *, trainer: Trainer, train_cfg: TrainConfig) -> Dict[str, Any]:
    """Build dependencies from ``cfg`` and run :class:`ServiceTrain`.

    Parameters
    ----------
    cfg: CommonRunConfig
        Runtime configuration with component declarations.
    trainer: Trainer
        Instance implementing :class:`Trainer` protocol.
    train_cfg: TrainConfig
        Configuration specific to training process.
    """
    container = di_registry.build_graph(cfg.components, cfg)
    fp: FeaturePipe = container["feature_pipe"]  # type: ignore[assignment]
    service = ServiceTrain(fp, trainer, train_cfg)
    return service.run()


__all__ = ["TrainConfig", "ServiceTrain", "from_config"]
