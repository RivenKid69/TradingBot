# training/no_trade.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class NoTradeConfig:
    funding_buffer_min: int = 0
    daily_utc: List[str] = None
    custom_ms: List[Dict[str, int]] = None

    @classmethod
    def from_yaml(cls, sandbox_yaml_path: str) -> "NoTradeConfig":
        with open(sandbox_yaml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        d = dict(y.get("no_trade", {}) or {})
        return cls(
            funding_buffer_min=int(d.get("funding_buffer_min", 0)),
            daily_utc=list(d.get("daily_utc", []) or []),
            custom_ms=list(d.get("custom_ms", []) or []),
        )


def _parse_daily_windows_min(windows: List[str]) -> List[Tuple[int, int]]:
    """
    Преобразует строки "HH:MM-HH:MM" в список (start_minute, end_minute), UTC.
    Не поддерживает окна через полночь (используй два окна).
    """
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
    # для каждого ts ищем близость к любой из меток
    # |sec_day - mark| <= buf*60
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


def compute_no_trade_mask(
    df: pd.DataFrame,
    *,
    sandbox_yaml_path: str = "configs/sandbox.yaml",
    ts_col: str = "ts_ms",
) -> pd.Series:
    """
    Возвращает pd.Series[bool] длины df:
      True  — строка попадает в «запрещённое» окно (no_trade), её надо исключить из обучения;
      False — строку можно использовать в train/val.
    """
    cfg = NoTradeConfig.from_yaml(sandbox_yaml_path)
    ts = pd.to_numeric(df[ts_col], errors="coerce").astype("Int64").astype("float").astype("int64")

    daily_min = _parse_daily_windows_min(cfg.daily_utc or [])
    m_daily = _in_daily_window(ts, daily_min)
    m_funding = _in_funding_buffer(ts, int(cfg.funding_buffer_min or 0))
    m_custom = _in_custom_window(ts, cfg.custom_ms or [])
    blocked = m_daily | m_funding | m_custom
    return pd.Series(blocked, index=df.index, name="no_trade_block")
