# training/threshold_tuner.py
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class TuneConfig:
    score_col: str = "score"
    # для классификации:
    y_col: Optional[str] = None            # бинарная метка (0/1). Если задана — считаем precision/recall/F1.
    # для регрессии:
    ret_col: Optional[str] = None          # эффективный ретёрн (например, eff_ret_60). Если задан — считаем Sharpe/mean.
    ts_col: str = "ts_ms"
    symbol_col: str = "symbol"
    direction: str = "greater"             # "greater" → сигнал если score >= thr; "less" → если score <= thr
    # частотные ограничения:
    target_signals_per_day: float = 1.5    # желаемые сигналы в день (суммарно по всем символам)
    tolerance: float = 0.5                 # допустимое отклонение от целевого значения
    min_signal_gap_s: int = 0              # кулдаун между сигналами в секундах (как в проде)
    # перебор порогов:
    min_thr: float = 0.50
    max_thr: float = 0.99
    steps: int = 50
    # учёт no-trade (опционально):
    sandbox_yaml_for_no_trade: Optional[str] = None
    drop_no_trade: bool = True
    # выбор целевой метрики:
    optimize_for: str = "sharpe"           # "sharpe" для регрессии; "precision" или "f1" для классификации


# -------------------- helpers: YAML --------------------

def load_min_signal_gap_s_from_yaml(yaml_path: str) -> int:
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        v = y.get("min_signal_gap_s", None)
        if v is None:
            # возможно, param находится в другом блоке — не найдено
            return 0
        return int(v)
    except Exception:
        return 0


def _compute_no_trade_mask_from_yaml(df: pd.DataFrame, ts_col: str, sandbox_yaml_path: str) -> pd.Series:
    """
    Читает блок no_trade из sandbox.yaml и строит булеву маску запрещённых строк.
    True = строка в запрещённом окне; False = разрешена.
    """
    try:
        with open(sandbox_yaml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        no_trade = dict(y.get("no_trade", {}) or {})
        daily = list(no_trade.get("daily_utc", []) or [])
        fund_buf = int(no_trade.get("funding_buffer_min", 0))
        custom = list(no_trade.get("custom_ms", []) or [])
    except Exception:
        daily = []
        fund_buf = 0
        custom = []

    # daily windows
    def _parse_daily_windows_min(windows: List[str]) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for w in windows:
            try:
                a, b = str(w).strip().split("-")
                sh, sm = a.split(":")
                eh, em = b.split(":")
                smin = int(sh) * 60 + int(sm)
                emin = int(eh) * 60 + int(em)
                if 0 <= smin <= 1440 and 0 <= emin <= 1440 and smin <= emin:
                    out.append((smin, emin))
            except Exception:
                continue
        return out

    def _in_daily_window(ts_ms: np.ndarray, daily_min: List[Tuple[int, int]]) -> np.ndarray:
        if not daily_min:
            return np.zeros_like(ts_ms, dtype=bool)
        mins = ((ts_ms // 60000) % 1440).astype(np.int64)
        mask = np.zeros_like(mins, dtype=bool)
        for s, e in daily_min:
            mask |= (mins >= s) & (mins < e)
        return mask

    def _in_funding_buffer(ts_ms: np.ndarray, buf_min: int) -> np.ndarray:
        if buf_min <= 0:
            return np.zeros_like(ts_ms, dtype=bool)
        sec_day = ((ts_ms // 1000) % 86400).astype(np.int64)
        marks = np.array([0, 8 * 3600, 16 * 3600], dtype=np.int64)
        mask = np.zeros_like(sec_day, dtype=bool)
        for m in marks:
            mask |= (np.abs(sec_day - m) <= buf_min * 60)
        return mask

    def _in_custom_window(ts_ms: np.ndarray, windows: List[Dict[str, int]]) -> np.ndarray:
        if not windows:
            return np.zeros_like(ts_ms, dtype=bool)
        mask = np.zeros_like(ts_ms, dtype=bool)
        for w in windows:
            try:
                s = int(w.get("start_ts_ms"))
                e = int(w.get("end_ts_ms"))
                mask |= (ts_ms >= s) & (ts_ms <= e)
            except Exception:
                continue
        return mask

    ts = pd.to_numeric(df[ts_col], errors="coerce").astype("Int64").astype("float").astype("int64")
    daily_min = _parse_daily_windows_min(daily)
    m_daily = _in_daily_window(ts, daily_min)
    m_fund = _in_funding_buffer(ts, int(fund_buf or 0))
    m_custom = _in_custom_window(ts, custom)
    blocked = m_daily | m_fund | m_custom
    return pd.Series(blocked, index=df.index, name="no_trade_block")


# -------------------- helpers: cooldown --------------------

def enforce_cooldown(df: pd.DataFrame, *, ts_col: str, symbol_col: str, min_signal_gap_s: int) -> pd.DataFrame:
    """
    Оставляет сигналы с учётом кулдауна между ними по каждому символу.
    Вход df должен быть отсортирован по [symbol, ts_col].
    """
    if min_signal_gap_s <= 0 or df.empty:
        return df

    out_rows = []
    for sym, g in df.groupby(symbol_col, sort=False):
        g = g.sort_values([ts_col])
        last_ts = None
        keep_idx = []
        for i, row in g.iterrows():
            t = int(row[ts_col])
            if last_ts is None or (t - last_ts) >= min_signal_gap_s * 1000:
                keep_idx.append(i)
                last_ts = t
        out_rows.append(g.loc[keep_idx])
    if not out_rows:
        return df.iloc[0:0]
    return pd.concat(out_rows, axis=0).sort_values([symbol_col, ts_col]).reset_index(drop=True)


# -------------------- helpers: metrics --------------------

def _signals_per_day(df: pd.DataFrame, ts_col: str) -> float:
    if df.empty:
        return 0.0
    day = (pd.to_numeric(df[ts_col], errors="coerce") // 86_400_000).astype("Int64")
    grp = df.groupby(day).size()
    return float(grp.mean()) if len(grp) else 0.0


def _classification_metrics(df_sel: pd.DataFrame, y_col: str) -> Tuple[float, float, float]:
    # precision, recall, f1
    if df_sel.empty:
        return 0.0, 0.0, 0.0
    tp = float((df_sel[y_col] == 1).sum())
    fp = float((df_sel[y_col] == 0).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # для recall нам нужно знать общее число позитивов за период
    # предположим, что df_sel — это принятые сигналы; для recall нужно сравнить с общим числом "1" в исходном df.
    # Пусть recall оценивается приближённо как отношение tp к числу позитивов в окрестности (здесь используем tp/(tp+fn) с fn≈0 без информации).
    # Если нужен строгий recall, его стоит считать на полном df с флагом "selected".
    recall = 0.0
    try:
        # хак: если в df_sel есть колонка total_positives, используем её
        tot_pos = float(df_sel.get("_total_positives", pd.Series(dtype=float)).iloc[0])
        recall = tp / tot_pos if tot_pos > 0 else 0.0
    except Exception:
        recall = 0.0

    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _returns_metrics(df_sel: pd.DataFrame, ret_col: str) -> Dict[str, float]:
    if df_sel.empty:
        return {
            "mean_ret": 0.0,
            "median_ret": 0.0,
            "hit_rate": 0.0,
            "trade_sharpe": 0.0,
            "sum_ret": 0.0,
        }
    r = pd.to_numeric(df_sel[ret_col], errors="coerce").astype(float)
    r = r[np.isfinite(r)]
    if r.empty:
        return {
            "mean_ret": 0.0,
            "median_ret": 0.0,
            "hit_rate": 0.0,
            "trade_sharpe": 0.0,
            "sum_ret": 0.0,
        }
    mean_ret = float(r.mean())
    median_ret = float(r.median())
    hit_rate = float((r > 0).mean())
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    trade_sharpe = float(mean_ret / (std + 1e-12))
    sum_ret = float(r.sum())
    return {
        "mean_ret": mean_ret,
        "median_ret": median_ret,
        "hit_rate": hit_rate,
        "trade_sharpe": trade_sharpe,
        "sum_ret": sum_ret,
    }


# -------------------- core: tuning --------------------

def tune_threshold(
    df: pd.DataFrame,
    cfg: TuneConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Возвращает (таблица-результатов-по-порогам, лучшая_строка_dict).
    Требования к df:
      - столбцы: ts_col, symbol_col, score_col;
      - либо y_col (классификация), либо ret_col (регрессия, eff_ret);
      - строки должны покрывать период валидации; желательно отсортирован df.
    """
    # подготовка
    need = [cfg.ts_col, cfg.symbol_col, cfg.score_col]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Отсутствует колонка '{c}'")
    if cfg.y_col is None and cfg.ret_col is None:
        raise ValueError("Нужно указать либо y_col (классификация), либо ret_col (регрессия).")

    d = df.copy()
    d = d.sort_values([cfg.symbol_col, cfg.ts_col]).reset_index(drop=True)

    # учёт no-trade
    if cfg.sandbox_yaml_for_no_trade and cfg.drop_no_trade:
        mask_block = _compute_no_trade_mask_from_yaml(d, ts_col=cfg.ts_col, sandbox_yaml_path=cfg.sandbox_yaml_for_no_trade)
        d = d.loc[~mask_block].reset_index(drop=True)

    # сетка порогов
    thr_list = np.linspace(float(cfg.min_thr), float(cfg.max_thr), int(cfg.steps))
    rows = []

    # общее число позитивов для приблизительного recall, если есть y_col
    total_positives = None
    if cfg.y_col is not None and cfg.y_col in d.columns:
        try:
            total_positives = int((d[cfg.y_col] == 1).sum())
        except Exception:
            total_positives = None

    for thr in thr_list:
        if cfg.direction == "greater":
            sel = d[d[cfg.score_col] >= thr]
        else:
            sel = d[d[cfg.score_col] <= thr]

        # кулдаун
        sel_cd = enforce_cooldown(sel, ts_col=cfg.ts_col, symbol_col=cfg.symbol_col, min_signal_gap_s=int(cfg.min_signal_gap_s))

        spd = _signals_per_day(sel_cd, ts_col=cfg.ts_col)

        row: Dict[str, float] = {
            "threshold": float(thr),
            "signals": float(len(sel_cd)),
            "signals_per_day": float(spd),
        }

        if cfg.y_col is not None and cfg.y_col in d.columns:
            if total_positives is not None and len(sel_cd) > 0:
                sel_cd = sel_cd.copy()
                sel_cd["_total_positives"] = total_positives
            precision, recall, f1 = _classification_metrics(sel_cd, cfg.y_col)
            row.update({
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            })

        if cfg.ret_col is not None and cfg.ret_col in d.columns:
            rmetrics = _returns_metrics(sel_cd, cfg.ret_col)
            row.update(rmetrics)

        rows.append(row)

    res = pd.DataFrame(rows)

    # фильтр по частоте
    lo = float(cfg.target_signals_per_day - cfg.tolerance)
    hi = float(cfg.target_signals_per_day + cfg.tolerance)
    mask_freq = (res["signals_per_day"] >= lo) & (res["signals_per_day"] <= hi)
    candidate = res.loc[mask_freq].copy()

    key_metric = "trade_sharpe" if cfg.optimize_for.lower() == "sharpe" else ("precision" if cfg.optimize_for.lower() == "precision" else "f1")
    if candidate.empty:
        # нет порога, удовлетворяющего частоте — возьмём близкий по частоте, а затем лучший по метрике
        res = res.copy()
        res["freq_dist"] = np.abs(res["signals_per_day"] - float(cfg.target_signals_per_day))
        res = res.sort_values(["freq_dist", key_metric], ascending=[True, False])
        best = dict(res.iloc[0])
    else:
        candidate = candidate.sort_values([key_metric], ascending=[False])
        best = dict(candidate.iloc[0])

    return res, best
