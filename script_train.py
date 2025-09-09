"""Train model via :mod:`service_train`."""

from __future__ import annotations

import argparse
import importlib

from core_config import load_config
from service_train import from_config, TrainConfig, Trainer


def _load_trainer(module_name: str) -> Trainer:
    """Import trainer instance from module."""
    mod = importlib.import_module(module_name)
    if hasattr(mod, "build_trainer"):
        return mod.build_trainer()  # type: ignore[call-arg]
    if hasattr(mod, "create_trainer"):
        return mod.create_trainer()  # type: ignore[call-arg]
    if hasattr(mod, "Trainer"):
        obj = getattr(mod, "Trainer")
        return obj() if callable(obj) else obj
    if hasattr(mod, "trainer"):
        obj = getattr(mod, "trainer")
        return obj() if callable(obj) else obj
    raise AttributeError(
        "Trainer module must define 'build_trainer', 'create_trainer', 'Trainer' or 'trainer'"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train model via ServiceTrain")
    p.add_argument(
        "--config",
        default="configs/config_train.yaml",
        help="Путь к YAML-конфигу запуска",
    )
    p.add_argument(
        "--trainer-module",
        default=None,
        help="Dotted path to module providing trainer",
    )
    p.add_argument(
        "--no-trade-mode",
        choices=["drop", "weight"],
        default="drop",
        help="How to handle no-trade windows: drop rows or assign train_weight=0",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.trainer_module:
        trainer = _load_trainer(args.trainer_module)
    else:
        class DummyTrainer:
            def fit(self, X, y=None, sample_weight=None):
                return None

            def save(self, path: str) -> str:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("dummy")
                return path

        trainer = DummyTrainer()

    data_path = getattr(getattr(cfg, "data", None), "prices_path", None)
    if not data_path:
        raise ValueError("Config must provide data.prices_path for training input")
    fmt = "parquet" if str(data_path).lower().endswith(".parquet") else "csv"

    train_cfg = TrainConfig(
        input_path=data_path,
        input_format=fmt,
        artifacts_dir=cfg.artifacts_dir,
        snapshot_config_path=args.config,
        no_trade_mode=args.no_trade_mode,
        no_trade_config_path=args.config,
    )

    res = from_config(cfg, trainer=trainer, train_cfg=train_cfg)
    print("dataset_X:", res["dataset_X"])
    if res.get("dataset_y"):
        print("dataset_y:", res["dataset_y"])
    print("model_path:", res["model_path"])


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
