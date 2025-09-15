"""Run realtime signaler using :mod:`service_signal_runner`."""

from __future__ import annotations

import argparse

from pathlib import Path

import yaml

from services.universe import get_symbols
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
    p.add_argument(
        "--state-config",
        default="configs/state.yaml",
        help="Путь к YAML-конфигу состояния",
    )
    p.add_argument(
        "--reset-state",
        action="store_true",
        help="Удалить файлы состояния перед запуском",
    )
    p.add_argument(
        "--symbols",
        default="",
        help="Список символов через запятую; пусто = загрузить из universe",
    )
    args = p.parse_args()
    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else get_symbols()
    )

    try:
        with open(args.state_config, "r", encoding="utf-8") as f:
            state_data = yaml.safe_load(f) or {}
    except Exception:
        state_data = {}

    if args.reset_state:
        for key in ("path", "lock_path"):
            pth = state_data.get(key)
            if pth:
                try:
                    Path(pth).unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass

    cfg = load_config(args.config)
    cfg.data.symbols = symbols
    try:
        cfg.components.executor.params["symbol"] = symbols[0]
    except Exception:
        pass
    if state_data:
        cfg.state = cfg.state.copy(update=state_data)

    for report in from_config(cfg, snapshot_config_path=args.config):
        print(report)


if __name__ == "__main__":
    main()
