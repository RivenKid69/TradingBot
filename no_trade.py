# training/no_trade.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from no_trade_config import NoTradeConfig, get_no_trade_config


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
            s = int(w["start_ts_ms"])
            e = int(w["end_ts_ms"])
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Invalid custom window {w}: expected integer 'start_ts_ms' and 'end_ts_ms'"
            ) from exc

        if s >= e:
            raise ValueError(
                f"Invalid custom window {w}: start_ts_ms ({s}) must be < end_ts_ms ({e})"
            )

        mask |= (ts_ms >= s) & (ts_ms <= e)

    return mask


def estimate_block_ratio(
    df: pd.DataFrame,
    cfg: NoTradeConfig,
    ts_col: str = "ts_ms",
) -> float:
    """Оценивает ожидаемую долю блокированных меток времени.

    Окна из ``daily_utc`` и ``funding_buffer_min`` трактуются как
    периодически повторяющиеся по дням. ``custom_ms`` рассматриваются как
    разовые интервалы поверх них. Возвращает значение в диапазоне [0, 1].
    """

    ts = (
        pd.to_numeric(df[ts_col], errors="coerce")
        .astype("Int64")
        .dropna()
        .astype("float")
        .astype("int64")
        .to_numpy()
    )
    if ts.size == 0:
        return 0.0

    # Строим дискретную маску минут суток
    minutes = np.zeros(1440, dtype=bool)
    for s, e in _parse_daily_windows_min(cfg.daily_utc or []):
        minutes[s:e] = True

    buf = int(cfg.funding_buffer_min or 0)
    if buf > 0:
        for m in (0, 8 * 60, 16 * 60):
            s = max(0, m - buf)
            e = min(1439, m + buf)
            minutes[s : e + 1] = True  # включаем обе границы

    ratio_daily = minutes.mean()

    if cfg.custom_ms:
        mins_idx = ((ts // 60000) % 1440).astype(int)
        mask_df = minutes[mins_idx]
        m_custom = _in_custom_window(ts, cfg.custom_ms)
        ratio = ratio_daily + np.mean(~mask_df & m_custom)
    else:
        ratio = ratio_daily

    return float(min(max(ratio, 0.0), 1.0))


def compute_no_trade_mask(
    df: pd.DataFrame,
    *,
    sandbox_yaml_path: str = "configs/legacy_sandbox.yaml",
    ts_col: str = "ts_ms",
) -> pd.Series:
    """
    Возвращает pd.Series[bool] длины df:
      True  — строка попадает в «запрещённое» окно (no_trade), её надо исключить из обучения;
      False — строку можно использовать в train/val.
    """
    cfg = get_no_trade_config(sandbox_yaml_path)
    ts = pd.to_numeric(df[ts_col], errors="coerce").astype("Int64").astype("float").astype("int64")

    daily_min = _parse_daily_windows_min(cfg.daily_utc or [])
    m_daily = _in_daily_window(ts, daily_min)
    m_funding = _in_funding_buffer(ts, int(cfg.funding_buffer_min or 0))
    m_custom = _in_custom_window(ts, cfg.custom_ms or [])
    blocked = m_daily | m_funding | m_custom
    return pd.Series(blocked, index=df.index, name="no_trade_block")
