"""Run realtime signaler using :mod:`service_signal_runner`."""

from __future__ import annotations

import argparse
from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel

from services.universe import get_symbols
from core_config import StateConfig, load_config
from service_signal_runner import from_config

try:
    from box import Box  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Box = None  # type: ignore


def _merge_state_config(state_obj: Any, payload: Mapping[str, Any]) -> Any:
    if not payload:
        return state_obj
    if isinstance(state_obj, BaseModel):
        return state_obj.copy(update=payload)
    if state_obj is None:
        try:
            return StateConfig.parse_obj(payload)
        except Exception:
            return payload
    if Box is not None and isinstance(state_obj, Box):
        state_obj.update(payload)
        return state_obj
    if isinstance(state_obj, dict):
        state_obj.update(payload)
        return state_obj
    for key, value in payload.items():
        try:
            setattr(state_obj, key, value)
        except Exception:
            continue
    return state_obj


def _reset_state_files(state_obj: Any) -> None:
    path_value = getattr(state_obj, "path", None)
    if path_value:
        p = Path(path_value)
        with suppress(Exception):
            p.unlink()
        for backup in p.parent.glob(f"{p.name}.bak*"):
            with suppress(Exception):
                backup.unlink()
        plain_backup = p.with_name(p.name + ".bak")
        with suppress(Exception):
            if plain_backup.exists():
                plain_backup.unlink()
        derived_lock = p.with_suffix(p.suffix + ".lock")
        with suppress(Exception):
            if derived_lock.exists():
                derived_lock.unlink()
    lock_value = getattr(state_obj, "lock_path", None)
    if lock_value:
        lock_path = Path(lock_value)
        with suppress(Exception):
            if lock_path.exists():
                lock_path.unlink()


def _ensure_state_dir(state_obj: Any) -> None:
    target_dir = getattr(state_obj, "dir", None)
    if not target_dir:
        path_value = getattr(state_obj, "path", None)
        if path_value:
            target_dir = Path(path_value).parent
    if not target_dir:
        return
    Path(target_dir).mkdir(parents=True, exist_ok=True)


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
            state_data_raw = yaml.safe_load(f) or {}
    except Exception:
        state_data_raw = {}
    state_data = state_data_raw if isinstance(state_data_raw, Mapping) else {}

    cfg = load_config(args.config)
    cfg.data.symbols = symbols
    try:
        cfg.components.executor.params["symbol"] = symbols[0]
    except Exception:
        pass
    if state_data:
        merged_state = _merge_state_config(cfg.state, state_data)
        if merged_state is not cfg.state:
            cfg.state = merged_state
    state_cfg = cfg.state

    if args.reset_state:
        _reset_state_files(state_cfg)

    if getattr(state_cfg, "enabled", False):
        _ensure_state_dir(state_cfg)

    for report in from_config(cfg, snapshot_config_path=args.config):
        print(report)


if __name__ == "__main__":
    main()
