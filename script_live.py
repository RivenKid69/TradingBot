"""Run realtime signaler using :mod:`service_signal_runner`."""

from __future__ import annotations

import argparse

from core_config import load_config
from service_signal_runner import from_config


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run realtime signaler (public Binance WS, no keys).",
    )
    p.add_argument(
        "--config",
        default="configs/config_live.yaml",
        help="Путь к YAML-конфигу запуска",
    )
    args = p.parse_args()

    cfg = load_config(args.config)

    for report in from_config(cfg, snapshot_config_path=args.config):
        print(report)


if __name__ == "__main__":
    main()
