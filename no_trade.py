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


def _prepare_ts(df: pd.DataFrame, ts_col: str) -> Tuple[np.ndarray, np.ndarray]:
    ts_series = pd.to_numeric(df[ts_col], errors="coerce")
    valid = ts_series.notna().to_numpy(dtype=bool)
    ts_int = ts_series.fillna(-1).astype(np.int64).to_numpy()
    return ts_int, valid


def _window_mask(ts_ms: np.ndarray, cfg: NoTradeConfig) -> np.ndarray:
    """Return mask for schedule-based no-trade windows."""

    daily_min = _parse_daily_windows_min(cfg.daily_utc or [])
    m_daily = _in_daily_window(ts_ms, daily_min)
    m_funding = _in_funding_buffer(ts_ms, int(cfg.funding_buffer_min or 0))
    m_custom = _in_custom_window(ts_ms, cfg.custom_ms or [])
    return m_daily | m_funding | m_custom


def _symbol_series(df: pd.DataFrame, column: str = "symbol") -> pd.Series:
    if column in df.columns:
        return df[column].fillna("__nan__")
    return pd.Series("__global__", index=df.index)


def _numeric_series(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _rolling_sigma(
    values: pd.Series,
    symbols: pd.Series,
    window: Optional[int],
    *,
    min_periods: Optional[int] = None,
) -> pd.Series:
    if window is None or window <= 1:
        return pd.Series(np.nan, index=values.index, dtype=float)
    if min_periods is None:
        min_periods = min(window, max(2, window // 2))
    result = (
        values.groupby(symbols)
        .rolling(window=window, min_periods=min_periods)
        .std()
        .reset_index(level=0, drop=True)
    )
    return result.reindex(values.index)


def _rolling_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    symbols: pd.Series,
    window: Optional[int],
    *,
    min_periods: Optional[int] = None,
) -> pd.Series:
    if window is None or window <= 1:
        return pd.Series(np.nan, index=close.index, dtype=float)
    if min_periods is None:
        min_periods = min(window, max(1, window // 2))
    prev_close = close.groupby(symbols).shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1, skipna=True)
    atr = (
        tr.groupby(symbols)
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return atr.reindex(close.index)


def _rolling_percentile(
    values: pd.Series,
    symbols: pd.Series,
    window: Optional[int],
    *,
    min_periods: Optional[int] = None,
) -> pd.Series:
    if window is None or window <= 1:
        return pd.Series(np.nan, index=values.index, dtype=float)
    if min_periods is None:
        min_periods = min(window, max(1, window // 5))

    def _percentile(arr: np.ndarray) -> float:
        if arr.size == 0:
            return np.nan
        val = arr[-1]
        if np.isnan(val):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.nan
        return float((valid <= val).sum()) / float(valid.size)

    result = (
        values.groupby(symbols)
        .rolling(window=window, min_periods=min_periods)
        .apply(_percentile, raw=True)
        .reset_index(level=0, drop=True)
    )
    return result.reindex(values.index)


def _dynamic_guard_mask(
    df: pd.DataFrame, dyn_cfg: Any, symbol_col: str = "symbol"
) -> Tuple[pd.Series, pd.DataFrame]:
    symbols = _symbol_series(df, symbol_col)

    price = _numeric_series(df, ["close", "price", "mid", "mid_price", "last_price"])
    high = _numeric_series(df, ["high", "high_price", "max_price"])
    low = _numeric_series(df, ["low", "low_price", "min_price"])
    spread = _numeric_series(df, ["spread_bps", "half_spread_bps"])
    if "half_spread_bps" in df.columns and "spread_bps" not in df.columns:
        spread = spread * 2.0
    if "bid" in df.columns and "ask" in df.columns:
        bid = pd.to_numeric(df["bid"], errors="coerce").astype(float)
        ask = pd.to_numeric(df["ask"], errors="coerce").astype(float)
        mid = (bid + ask) * 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            derived_spread = (ask - bid) / mid * 10000.0
        spread = spread.fillna(derived_spread)
    elif "bid_price" in df.columns and "ask_price" in df.columns:
        bid = pd.to_numeric(df["bid_price"], errors="coerce").astype(float)
        ask = pd.to_numeric(df["ask_price"], errors="coerce").astype(float)
        mid = (bid + ask) * 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            derived_spread = (ask - bid) / mid * 10000.0
        spread = spread.fillna(derived_spread)

    dyn_mask = pd.Series(False, index=df.index, dtype=bool)
    reasons = pd.DataFrame(
        False,
        index=df.index,
        columns=[
            "dyn_vol_abs",
            "dyn_vol_pctile",
            "dyn_spread_abs",
            "dyn_spread_pctile",
            "dyn_guard_raw",
            "dyn_guard_hold",
        ],
    )
    reasons.attrs["meta"] = {}

    thresholds_defined = any(
        x is not None
        for x in (
            dyn_cfg.vol_abs,
            dyn_cfg.vol_pctile,
            dyn_cfg.spread_abs_bps,
            dyn_cfg.spread_pctile,
        )
    )
    if not thresholds_defined:
        return dyn_mask, reasons

    required_metrics: List[str] = []
    if dyn_cfg.vol_abs is not None or dyn_cfg.vol_pctile is not None:
        required_metrics.append("volatility")
    if dyn_cfg.spread_abs_bps is not None or dyn_cfg.spread_pctile is not None:
        required_metrics.append("spread")

    close = price.replace(0, np.nan)
    returns = price.groupby(symbols).pct_change()

    sigma_window = dyn_cfg.sigma_window or 120
    atr_window = dyn_cfg.atr_window or 14

    sigma = _rolling_sigma(returns, symbols, sigma_window)
    atr = _rolling_atr(high, low, close, symbols, atr_window)
    atr_pct = atr / close.abs()
    spread_proxy = spread
    if spread_proxy.isna().all() and atr_pct.notna().any():
        spread_proxy = atr_pct * 10000.0

    vol_metric = sigma.fillna(atr_pct)
    vol_pctile = _rolling_percentile(vol_metric, symbols, sigma_window)
    spread_pctile = _rolling_percentile(spread_proxy, symbols, atr_window)

    missing_requirements: List[str] = []
    if "volatility" in required_metrics and not (
        vol_metric.notna().any() or vol_pctile.notna().any()
    ):
        missing_requirements.append("volatility")
    if "spread" in required_metrics and not (
        spread_proxy.notna().any() or spread_pctile.notna().any()
    ):
        missing_requirements.append("spread")

    if missing_requirements:
        reasons.attrs["meta"] = {
            "skipped": True,
            "reason": "missing_data",
            "missing": missing_requirements,
        }
        return dyn_mask, reasons

    hysteresis = float(dyn_cfg.hysteresis or 0.0)
    if hysteresis < 0:
        hysteresis = 0.0
    cooldown = max(0, int(dyn_cfg.cooldown_bars or 0))

    for _, group in df.groupby(symbols, sort=False):
        blocked = False
        cooldown_left = 0
        for label in group.index:
            trigger = False

            if dyn_cfg.vol_abs is not None:
                val = vol_metric.loc[label]
                if not np.isnan(val) and val >= float(dyn_cfg.vol_abs):
                    reasons.at[label, "dyn_vol_abs"] = True
                    trigger = True

            if dyn_cfg.vol_pctile is not None:
                val = vol_pctile.loc[label]
                if not np.isnan(val) and val >= float(dyn_cfg.vol_pctile):
                    reasons.at[label, "dyn_vol_pctile"] = True
                    trigger = True

            if dyn_cfg.spread_abs_bps is not None:
                val = spread_proxy.loc[label]
                if not np.isnan(val) and val >= float(dyn_cfg.spread_abs_bps):
                    reasons.at[label, "dyn_spread_abs"] = True
                    trigger = True

            if dyn_cfg.spread_pctile is not None:
                val = spread_pctile.loc[label]
                if not np.isnan(val) and val >= float(dyn_cfg.spread_pctile):
                    reasons.at[label, "dyn_spread_pctile"] = True
                    trigger = True

            reasons.at[label, "dyn_guard_raw"] = trigger

            if trigger:
                dyn_mask.loc[label] = True
                blocked = True
                cooldown_left = max(cooldown_left, cooldown)
                continue

            if not blocked:
                continue

            release_ready = True

            if dyn_cfg.vol_abs is not None:
                val = vol_metric.loc[label]
                release_thr = float(dyn_cfg.vol_abs) * (1.0 - hysteresis)
                if not np.isnan(val) and val > release_thr:
                    release_ready = False

            if dyn_cfg.vol_pctile is not None:
                val = vol_pctile.loc[label]
                release_thr = max(0.0, float(dyn_cfg.vol_pctile) - hysteresis)
                if not np.isnan(val) and val > release_thr:
                    release_ready = False

            if dyn_cfg.spread_abs_bps is not None:
                val = spread_proxy.loc[label]
                release_thr = float(dyn_cfg.spread_abs_bps) * (1.0 - hysteresis)
                if not np.isnan(val) and val > release_thr:
                    release_ready = False

            if dyn_cfg.spread_pctile is not None:
                val = spread_pctile.loc[label]
                release_thr = max(0.0, float(dyn_cfg.spread_pctile) - hysteresis)
                if not np.isnan(val) and val > release_thr:
                    release_ready = False

            if not release_ready:
                dyn_mask.loc[label] = True
                reasons.at[label, "dyn_guard_hold"] = True
                continue

            if cooldown_left > 0:
                dyn_mask.loc[label] = True
                reasons.at[label, "dyn_guard_hold"] = True
                cooldown_left -= 1
                continue

            blocked = False

    return dyn_mask, reasons


def estimate_block_ratio(
    df: pd.DataFrame,
    cfg: NoTradeConfig,
    ts_col: str = "ts_ms",
) -> float:
    """Estimate share of rows blocked by schedule and dynamic guard."""

    ts, ts_valid = _prepare_ts(df, ts_col)
    if ts.size == 0:
        return 0.0

    window_mask = _window_mask(ts, cfg) & ts_valid

    dyn_cfg = cfg.dynamic_guard or None
    if dyn_cfg and dyn_cfg.enable:
        dyn_mask, _ = _dynamic_guard_mask(df, dyn_cfg)
        combined = window_mask | dyn_mask.to_numpy(dtype=bool)
        return float(np.mean(combined))
    return float(np.mean(window_mask))


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
    ts, ts_valid = _prepare_ts(df, ts_col)

    window_mask = _window_mask(ts, cfg) & ts_valid

    reasons = pd.DataFrame(index=df.index, data={"window": window_mask})

    dyn_cfg = cfg.dynamic_guard or None
    dyn_mask = pd.Series(False, index=df.index, dtype=bool)
    dyn_reasons: Optional[pd.DataFrame] = None
    meta: Dict[str, Any] = {}
    if dyn_cfg and dyn_cfg.enable:
        dyn_mask, dyn_reasons = _dynamic_guard_mask(df, dyn_cfg)
        reasons = pd.concat([reasons, dyn_reasons], axis=1).fillna(False)
        window_mask = window_mask | dyn_mask.to_numpy(dtype=bool)
        dyn_meta = getattr(dyn_reasons, "attrs", {}).get("meta") if dyn_reasons is not None else None
        if dyn_meta:
            meta["dynamic_guard"] = dyn_meta

    result = pd.Series(window_mask, index=df.index, name="no_trade_block")
    if dyn_cfg and dyn_cfg.enable:
        reasons["dynamic_guard"] = dyn_mask.astype(bool)
    else:
        reasons["dynamic_guard"] = False
    reasons = reasons.reindex(df.index).fillna(False).astype(bool)
    result.attrs["reasons"] = reasons
    result.attrs["reason_labels"] = {col: col for col in reasons.columns}
    if meta:
        result.attrs["meta"] = meta
    return result
