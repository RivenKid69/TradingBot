"""Run realtime signaler using :mod:`service_signal_runner`.

The script now relies on ``load_config``/``from_config`` pair and does not
manually construct any of the runtime components.
"""

from __future__ import annotations

import argparse

from core_config import load_config
from service_signal_runner import RunnerConfig, from_config


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

    svc_cfg = RunnerConfig(snapshot_config_path=args.config, artifacts_dir=cfg.artifacts_dir)

    for report in from_config(cfg, svc_cfg=svc_cfg):
        print(report)


if __name__ == "__main__":
    main()
