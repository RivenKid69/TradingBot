# -*- coding: utf-8 -*-
"""Утилиты для запуска песочницы/бэктеста."""
from __future__ import annotations

import importlib
from typing import Any, Dict

import pandas as pd

from core_strategy import Strategy


def read_df(path: str) -> pd.DataFrame:
    """Читает DataFrame из CSV или Parquet."""
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_strategy(mod: str, cls: str, params: Dict[str, Any]) -> Strategy:
    """Создаёт стратегию и вызывает setup."""
    m = importlib.import_module(mod)
    Cls = getattr(m, cls)
    s: Strategy = Cls()
    s.setup(params or {})
    return s
