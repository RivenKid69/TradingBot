# training/splits.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class WFSplit:
    fold_id: int
    train_start_ts: int
    train_end_ts_raw: int
    train_end_ts_effective: int
    val_start_ts: int
    val_end_ts: int
    horizon_bars: int
    embargo_bars: int
    interval_ms: int
    train_count: int = 0
    val_count: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


def _estimate_bar_interval_ms(df: pd.DataFrame, ts_col: str, symbol_col: Optional[str]) -> int:
    """
    Оцениваем интервал бара по медиане разностей. Если есть несколько символов — берём медиану медиан.
    """
    ts = pd.to_numeric(df[ts_col], errors="coerce")
    if symbol_col and (symbol_col in df.columns):
        medians: List[float] = []
        for _, g in df[[symbol_col, ts_col]].dropna().sort_values([symbol_col, ts_col]).groupby(symbol_col):
            d = g[ts_col].diff().dropna()
            if not d.empty:
                medians.append(float(d.median()))
        if medians:
            return int(np.median(medians))
    d = ts.sort_values().diff().dropna()
    if d.empty:
        raise ValueError("Не удалось оценить интервал бара: недостаточно точек времени")
    return int(d.median())


def _time_windows_by_span(
    t0: int,
    t1: int,
    *,
    train_span_ms: int,
    val_span_ms: int,
    step_ms: int,
    horizon_ms: int,
    embargo_ms: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Генерирует последовательность временных окон (train_start, train_end_effective, val_start, val_end],
    где train_end_effective уже урезан на purge (horizon_ms).
    """
    out: List[Tuple[int, int, int, int]] = []
    cur_train_start = int(t0)
    while True:
        train_end_raw = cur_train_start + int(train_span_ms)
        train_end_eff = train_end_raw - int(horizon_ms)
        val_start = train_end_raw + int(embargo_ms)
        val_end = val_start + int(val_span_ms)

        if train_end_eff <= cur_train_start:
            break
        if val_start >= t1 or cur_train_start >= t1:
            break

        out.append((cur_train_start, train_end_eff, val_start, val_end))
        # шаг вперёд
        cur_train_start = cur_train_start + int(step_ms)
        if cur_train_start >= t1:
            break
    return out


def make_walkforward_splits(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts_ms",
    symbol_col: Optional[str] = "symbol",
    interval_ms: Optional[int] = None,
    train_span_bars: int = 7 * 24 * 60,     # 7 дней @ 1m
    val_span_bars: int = 1 * 24 * 60,       # 1 день @ 1m
    step_bars: int = 1 * 24 * 60,           # шаг окна 1 день
    horizon_bars: int = 60,                 # горизонт таргета (purge = h)
    embargo_bars: int = 5,                  # буфер между train и val
) -> Tuple[pd.DataFrame, List[WFSplit]]:
    """
    Возвращает (df_out, manifest), где:
      - df_out = df с новыми колонками:
            wf_fold: int  (номер фолда, для строк train/val; -1 — вне всех окон)
            wf_role: str  ("train"/"val"/"none")
      - manifest: список описаний фолдов (WFSplit)

    Алгоритм:
      - окна задаём по времени (ms), используя оценку интервала бара (или заданный interval_ms);
      - train_end_effective = train_end_raw - horizon_ms  (PURGE);
      - val_start = train_end_raw + embargo_ms           (EMBARGO);
      - роли назначаем по временным границам для каждого фолда по всем символам сразу.
    """
    if ts_col not in df.columns:
        raise ValueError(f"Отсутствует колонка времени '{ts_col}'")
    if symbol_col and symbol_col not in df.columns:
        symbol_col = None

    # оценка интервала
    if interval_ms is None:
        interval_ms = _estimate_bar_interval_ms(df, ts_col=ts_col, symbol_col=symbol_col)

    train_span_ms = int(train_span_bars) * int(interval_ms)
    val_span_ms = int(val_span_bars) * int(interval_ms)
    step_ms = int(step_bars) * int(interval_ms)
    horizon_ms = int(horizon_bars) * int(interval_ms)
    embargo_ms = int(embargo_bars) * int(interval_ms)

    # упорядочим и найдём временной диапазон
    d = df.sort_values([ts_col]).reset_index(drop=True)
    t0 = int(d[ts_col].min())
    t1 = int(d[ts_col].max())

    windows = _time_windows_by_span(
        t0,
        t1,
        train_span_ms=train_span_ms,
        val_span_ms=val_span_ms,
        step_ms=step_ms,
        horizon_ms=horizon_ms,
        embargo_ms=embargo_ms,
    )
    if not windows:
        raise ValueError("Не удалось построить окна walk-forward (возможно, слишком большие train/val/step).")

    # подготовим колонки результата
    out = df.copy()
    out["wf_fold"] = -1
    out["wf_role"] = "none"

    manifest: List[WFSplit] = []
    ts_series = pd.to_numeric(out[ts_col], errors="coerce").astype("int64")

    for fid, (tr_s, tr_e_eff, va_s, va_e) in enumerate(windows):
        # маски по времени
        m_train = (ts_series >= tr_s) & (ts_series < tr_e_eff)
        m_val = (ts_series >= va_s) & (ts_series < va_e)

        # назначим роли (train имеет приоритет над предыдущими фолдами только если строка ещё "none")
        # здесь мы позволяем перекрывать валид. окна разных фолдов по времени, но строка попадает в первый подходящий фолд
        sel_none = out["wf_role"] == "none"
        assign_train = sel_none & m_train
        assign_val = sel_none & m_val

        out.loc[assign_train, "wf_fold"] = fid
        out.loc[assign_train, "wf_role"] = "train"
        out.loc[assign_val, "wf_fold"] = fid
        out.loc[assign_val, "wf_role"] = "val"

        # посчитаем фактический размер выборок
        tr_cnt = int(assign_train.sum())
        va_cnt = int(assign_val.sum())

        manifest.append(WFSplit(
            fold_id=fid,
            train_start_ts=int(tr_s),
            train_end_ts_raw=int(tr_s + train_span_ms),
            train_end_ts_effective=int(tr_e_eff),
            val_start_ts=int(va_s),
            val_end_ts=int(va_e),
            horizon_bars=int(horizon_bars),
            embargo_bars=int(embargo_bars),
            interval_ms=int(interval_ms),
            train_count=tr_cnt,
            val_count=va_cnt,
        ))

    return out, manifest
