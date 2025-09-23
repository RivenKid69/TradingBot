# data/leakguard.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api import types as pdt


@dataclass
class LeakConfig:
    """
    Настройки защиты от утечки:
      - decision_delay_ms: задержка между ts фичей и фактическим моментом принятия решения
                           (включает расчёт индикаторов, агрегации, сериализацию и пр.)
      - min_lookback_ms: требуемая минимальная глубина истории источников для ffill (если gap больше — NaN)
    """
    decision_delay_ms: int = 0
    min_lookback_ms: int = 0


class LeakGuard:
    """
    Правила:
      1) Все фичи/источники присоединяются asof с direction='backward' и ограничением tolerance.
      2) Вводится колонка decision_ts = ts_ms + decision_delay_ms — **только** начиная с этого времени
         можно смотреть на рынок для постановки ордеров и построения меток (labels).
      3) Метки рассчитываются только на отрезке [decision_ts, decision_ts + horizon].
    """
    def __init__(self, cfg: Optional[LeakConfig] = None):
        self.cfg = cfg or LeakConfig()

    def attach_decision_time(self, df: pd.DataFrame, *, ts_col: str = "ts_ms") -> pd.DataFrame:
        if ts_col not in df.columns:
            raise ValueError(f"'{ts_col}' не найден в датафрейме")
        d = df.copy()
        d["decision_ts"] = d[ts_col].astype("int64") + int(self.cfg.decision_delay_ms)
        return d

    def validate_ffill_gaps(
        self,
        df: pd.DataFrame,
        *,
        ts_col: str,
        group_keys: list[str],
        value_cols: list[str],
        max_gap_ms: int,
    ) -> pd.DataFrame:
        """
        Выставляет NaN, если «держание» значения длится дольше max_gap_ms (защита от чрезмерного ffill).
        """
        if ts_col not in df.columns:
            raise ValueError(f"'{ts_col}' не найден в датафрейме")
        missing_group = [col for col in group_keys if col not in df.columns]
        if missing_group:
            raise ValueError(f"Отсутствуют ключи группировки: {missing_group}")
        missing_value_cols = [col for col in value_cols if col not in df.columns]
        if missing_value_cols:
            raise ValueError(f"Отсутствуют колонки для проверки: {missing_value_cols}")

        if not value_cols:
            return df.copy()

        try:
            max_gap = int(max_gap_ms)
        except (TypeError, ValueError) as err:
            raise ValueError("max_gap_ms должен быть целым числом") from err
        if max_gap < 0:
            raise ValueError("max_gap_ms должен быть неотрицательным")

        d = df.copy()
        d["__orig_order__"] = np.arange(len(d), dtype="int64")

        ts_numeric = pd.to_numeric(d[ts_col], errors="raise")
        if ts_numeric.isna().any():
            raise ValueError("Обнаружены пропущенные значения в столбце с таймштампами")
        d[ts_col] = ts_numeric.astype("int64")

        sort_columns = list(group_keys) + [ts_col, "__orig_order__"]
        d = d.sort_values(sort_columns, kind="mergesort").copy()

        if group_keys:
            grouped = d.groupby(group_keys, sort=False, dropna=False)
        else:
            grouped = d.groupby(lambda _: 0, sort=False)
        masks: dict[str, pd.Series] = {
            col: pd.Series(False, index=d.index) for col in value_cols
        }

        for _, idx in grouped.groups.items():
            group_df = d.loc[idx]
            ts_values = group_df[ts_col].to_numpy(dtype="int64")

            for col in value_cols:
                values = group_df[col]
                invalid_positions: list[int] = []

                last_value = None
                has_last_value = False
                run_start_ts: Optional[int] = None
                invalid_run = False

                for row_idx, ts_val, cell in zip(values.index, ts_values, values.to_numpy()):
                    if pd.isna(cell):
                        last_value = None
                        has_last_value = False
                        run_start_ts = None
                        invalid_run = False
                        continue

                    if (not has_last_value) or (cell != last_value):
                        last_value = cell
                        has_last_value = True
                        run_start_ts = int(ts_val)
                        invalid_run = False

                    if invalid_run:
                        invalid_positions.append(row_idx)
                        continue

                    if run_start_ts is None:
                        continue

                    if int(ts_val) - run_start_ts > max_gap:
                        invalid_positions.append(row_idx)
                        invalid_run = True

                if invalid_positions:
                    masks[col].loc[invalid_positions] = True

        for col, mask in masks.items():
            if mask.any():
                if pdt.is_numeric_dtype(d[col].dtype):
                    d[col] = d[col].where(~mask, other=np.nan)
                else:
                    d[col] = d[col].where(~mask, other=pd.NA)

        min_lookback = int(getattr(self.cfg, "min_lookback_ms", 0) or 0)
        if min_lookback > 0:
            if group_keys:
                grouped_after = d.groupby(group_keys, sort=False, dropna=False)
            else:
                grouped_after = d.groupby(lambda _: 0, sort=False)
            for _, idx in grouped_after.groups.items():
                group_df = d.loc[idx]
                if group_df.empty:
                    continue
                first_ts = int(group_df[ts_col].iloc[0])
                for col in value_cols:
                    valid_idx = group_df.index[~group_df[col].isna()]
                    if valid_idx.empty:
                        raise ValueError(
                            f"Недостаточно данных для '{col}' внутри группы {group_keys}: всё NaN"
                        )
                    first_valid_ts = int(group_df.loc[valid_idx[0], ts_col])
                    if first_valid_ts - first_ts > min_lookback:
                        if group_keys:
                            group_info = group_df[group_keys].iloc[0].to_dict()
                        else:
                            group_info = {"group": "__all__"}
                        raise ValueError(
                            "Нарушено требование min_lookback_ms: колонка "
                            f"'{col}' для группы {group_info} имеет первую доступную точку "
                            f"только через {first_valid_ts - first_ts} мс"
                        )

        d = d.sort_values("__orig_order__", kind="mergesort")
        d = d.drop(columns="__orig_order__")
        return d
