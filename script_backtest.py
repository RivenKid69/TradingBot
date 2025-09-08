"""Run backtest via :mod:`service_backtest`."""

from __future__ import annotations

import argparse

from core_config import load_config
from service_backtest import from_config


def main() -> None:
    p = argparse.ArgumentParser(description="Strategy backtest runner")
    p.add_argument(
        "--config",
        default="configs/config_sim.yaml",
        help="Путь к YAML-конфигу запуска",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    reports = from_config(cfg, snapshot_config_path=args.config)
    print(f"Produced {len(reports)} reports")


if __name__ == "__main__":
    main()
