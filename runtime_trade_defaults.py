"""Helpers for loading runtime trade defaults shared across tools."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Mapping, MutableMapping

import yaml

DEFAULT_RUNTIME_TRADE_PATH = "configs/runtime_trade.yaml"

RuntimeTradeLoader = Callable[[str], Mapping[str, Any]]


def _default_loader(path: str) -> Mapping[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return data if isinstance(data, Mapping) else {}


def load_runtime_trade_defaults(
    path: str | None = None,
    *,
    loader: RuntimeTradeLoader | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Load runtime trade defaults from YAML and return sanitized sections."""

    target = str(path or DEFAULT_RUNTIME_TRADE_PATH)
    active_loader = loader or _default_loader
    try:
        raw = active_loader(target) or {}
    except Exception:
        raw = {}
    if isinstance(raw, tuple):  # support loaders returning (data, text)
        raw = raw[0] if raw else {}
    if not isinstance(raw, Mapping):
        raw = {}

    defaults: Dict[str, Dict[str, Any]] = {}
    for section in ("portfolio", "execution", "costs"):
        payload = raw.get(section) if isinstance(raw, Mapping) else {}
        if isinstance(payload, Mapping):
            defaults[section] = deepcopy(dict(payload))
        else:
            defaults[section] = {}
    return defaults


def merge_runtime_trade_defaults(
    cfg_dict: MutableMapping[str, Any],
    defaults: Mapping[str, Mapping[str, Any]],
) -> MutableMapping[str, Any]:
    """Apply runtime trade defaults onto a config mapping in-place."""

    if not defaults:
        return cfg_dict

    execution_block = dict(cfg_dict.get("execution") or {})

    for section in ("portfolio", "costs"):
        default_block = defaults.get(section)
        if isinstance(default_block, Mapping) and default_block:
            current_top = dict(cfg_dict.get(section) or {})
            current_top.update(default_block)
            cfg_dict[section] = current_top

            exec_section = dict(execution_block.get(section) or {})
            exec_section.update(default_block)
            execution_block[section] = exec_section

    execution_defaults = defaults.get("execution")
    if isinstance(execution_defaults, Mapping) and execution_defaults:
        execution_block.update(execution_defaults)

    if execution_block:
        cfg_dict["execution"] = execution_block

    return cfg_dict
