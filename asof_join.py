# data/asof_join.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd


@dataclass
class AsofSpec:
    """
    Описание одной таблицы для asof-джойна.
    """
    name: str
    df: pd.DataFrame
    time_col: str = "ts_ms"
    keys: Sequence[str] = ()
    prefix: Optional[str] = None
    # поведение джойна
    direction: str = "backward"        # "backward" | "forward" | "nearest"
    tolerance_ms: Optional[int] = None # макс. разрыв во времени; если None — без ограничения
    allow_exact_matches: bool = True   # разрешать совпадение по времени

    def normalized(self) -> "AsofSpec":
        d = self.df.copy()
        if self.time_col not in d.columns:
            raise ValueError(f"[{self.name}] нет колонки времени '{self.time_col}'")
        # сортировка по ключам и времени — требование merge_asof
        sort_by = list(self.keys) + [self.time_col]
        d = d.sort_values(sort_by).reset_index(drop=True)
        # приведение времени к int64 (миллисекунды)
        d[self.time_col] = d[self.time_col].astype("int64")
        pfx = self.prefix if (self.prefix not in (None, "")) else f"{self.name}_"
        # запретим коллизии имён
        forbid = set(sort_by)
        rename = {c: (pfx + c) for c in d.columns if c not in forbid}
        d = d.rename(columns=rename)
        # пересчитать time_col/key после переименования
        time_col = self.time_col
        keys = list(self.keys)
        if time_col in rename:
            # time_col не должен попадать в rename, но на всякий случай
            rev = {v: k for k, v in rename.items()}
            time_col = rev.get(rename[time_col], time_col)
        return AsofSpec(
            name=self.name,
            df=d,
            time_col=time_col,
            keys=tuple(keys),
            prefix=pfx,
            direction=self.direction,
            tolerance_ms=self.tolerance_ms,
            allow_exact_matches=self.allow_exact_matches,
        )


class AsofMerger:
    """
    Удобная обёртка над pandas.merge_asof для мульти-табличного asof-джойна.

    Принцип:
      - base_df — «осевая» таблица (обычно — фичи/сигналы), одна строка = один момент принятия решения.
      - Остальные таблицы присоединяются asof по времени и ключам: для каждой строки base_df берётся
        последняя (direction="backward") запись из таблицы-источника не позже текущего времени строки
        base_df (либо согласно выбранному direction).
      - tolerance_ms ограничивает максимальный «разрыв» по времени, иначе значения будут NaN.

    Требования к входам:
      - Колонка времени — int64 (миллисекунды unix).
      - Ключи и время должны быть отсортированы.
    """
    def __init__(self, *, base_df: pd.DataFrame, time_col: str = "ts_ms", keys: Sequence[str] = ()):
        if time_col not in base_df.columns:
            raise ValueError(f"base_df не содержит колонку времени '{time_col}'")
        self.base_time_col = str(time_col)
        self.base_keys = list(keys)
        self.base = base_df.copy()
        sort_by = self.base_keys + [self.base_time_col]
        self.base = self.base.sort_values(sort_by).reset_index(drop=True)
        self.base[self.base_time_col] = self.base[self.base_time_col].astype("int64")

    def merge(self, specs: Sequence[AsofSpec]) -> pd.DataFrame:
        out = self.base
        for spec in specs:
            s = spec.normalized()
            left_by = self.base_keys if self.base_keys else None
            right_by = s.keys if s.keys else None
            # pandas требует одинаковые множества ключей для merge_asof(by=...)
            if (left_by or []) != (right_by or []):
                raise ValueError(f"[{s.name}] несовпадающий набор ключей: base={left_by}, right={right_by}")

            # allow_exact_matches=False имитирует «решение позже источника» (исключить совпадание времени)
            allow = bool(s.allow_exact_matches)
            tol = None if s.tolerance_ms is None else pd.Timedelta(milliseconds=int(s.tolerance_ms))

            out = pd.merge_asof(
                out,
                s.df,
                left_on=self.base_time_col,
                right_on=s.time_col,
                by=self.base_keys if self.base_keys else None,
                direction=s.direction,
                tolerance=tol,
                allow_exact_matches=allow,
            )
        return out
