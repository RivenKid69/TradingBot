# -*- coding: utf-8 -*-
"""Утилиты для запуска песочницы/бэктеста."""
from __future__ import annotations

import importlib
from typing import Any, Dict, Mapping

import pandas as pd
from core_contracts import SignalPolicy


def read_df(path: str) -> pd.DataFrame:
    """Читает DataFrame из CSV или Parquet."""
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_policy(mod: str, cls: str, params: Dict[str, Any]) -> SignalPolicy:
    """Создаёт политику и при необходимости вызывает ``setup``."""
    m = importlib.import_module(mod)
    Cls = getattr(m, cls)
    p: SignalPolicy = Cls()
    setup = getattr(p, "setup", None)
    if callable(setup):
        setup(params or {})
    return p


def policy_from_config(spec: Mapping[str, Any]) -> SignalPolicy:
    """Построить политику из словаря конфига ``{"module", "class", "params"}``.

    Поддерживает также форму ``{"target": "module:Class", "params": {}}``.
    """
    if "target" in spec:
        mod, cls = str(spec["target"]).split(":", 1)
    else:
        mod = str(spec["module"])
        cls = str(spec["class"])
    params: Dict[str, Any] = dict(spec.get("params") or {})
    return build_policy(mod, cls, params)
