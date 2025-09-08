"""Evaluate strategy performance via :mod:`service_eval`."""

from __future__ import annotations

import argparse

from core_config import load_config
from service_eval import from_config


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate strategy performance via ServiceEval",
    )
    p.add_argument(
        "--config",
        default="configs/config_eval.yaml",
        help="Путь к YAML-конфигу запуска",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    metrics = from_config(cfg, snapshot_config_path=args.config)
    print(metrics)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

