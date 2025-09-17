# training/no_trade.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from no_trade_config import (
    NoTradeConfig,
    NoTradeState,
    get_no_trade_config,
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


def _window_reasons(ts_ms: np.ndarray, cfg: NoTradeConfig) -> pd.DataFrame:
    """Return per-reason mask for schedule-based no-trade windows."""

    if ts_ms.size == 0:
        return pd.DataFrame(
            {
                "maintenance_daily": [],
                "maintenance_funding": [],
                "maintenance_custom": [],
                "window": [],
            }
        )

    daily_min = _parse_daily_windows_min(cfg.daily_utc or [])
    m_daily = _in_daily_window(ts_ms, daily_min)
    m_funding = _in_funding_buffer(ts_ms, int(cfg.funding_buffer_min or 0))
    m_custom = _in_custom_window(ts_ms, cfg.custom_ms or [])

    data = {
        "maintenance_daily": m_daily,
        "maintenance_funding": m_funding,
        "maintenance_custom": m_custom,
    }
    df = pd.DataFrame(data)
    df["window"] = df.any(axis=1)
    return df.astype(bool)


def _symbol_series(df: pd.DataFrame, column: str = "symbol") -> pd.Series:
    if column in df.columns:
        return df[column].fillna("__nan__")
    return pd.Series("__global__", index=df.index)


def _numeric_series(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _coerce_positive_int(value: Any) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return 0
    return ivalue if ivalue > 0 else 0


def _coerce_int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _reason_categories(reason: str) -> List[str]:
    """Return generic categories for a concrete trigger column."""

    base = reason.lower()
    categories: List[str] = []
    if "vol" in base or "ret" in base:
        categories.extend(["volatility", "vol", "return", "ret"])
    if "spread" in base:
        categories.extend(["spread", "spr"])
    if "anomaly" in base:
        categories.append("anomaly")
    return categories


def _resolve_next_block(reasons: Iterable[str], mapping: Mapping[str, Any]) -> int:
    """Return hold duration for triggered *reasons* using *mapping*."""

    keys: List[str] = []
    for reason in reasons:
        if not reason:
            continue
        keys.append(reason)
        keys.extend(_reason_categories(reason))
    keys.extend(["anomaly", "any", "*"])

    result = 0
    for key in keys:
        if key in mapping:
            result = max(result, _coerce_positive_int(mapping[key]))
    return result


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
    df: pd.DataFrame,
    dyn_cfg: Any,
    *,
    ts_int: np.ndarray,
    ts_valid: np.ndarray,
    symbol_col: str = "symbol",
    state_map: Optional[Mapping[str, int]] = None,
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
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
            "dyn_ret_anomaly",
            "dyn_spread_anomaly",
            "dyn_guard_raw",
            "dyn_guard_hold",
            "dyn_guard_next_block",
            "dyn_guard_state",
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

    ts_series = pd.Series(ts_int, index=df.index, dtype=np.int64)
    ts_valid_series = pd.Series(ts_valid, index=df.index, dtype=bool)
    state_map = {str(k): int(v) for k, v in (state_map or {}).items()}

    if not thresholds_defined and not state_map:
        return dyn_mask, reasons, {"anomaly_block_until_ts": {}}

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
    if thresholds_defined:
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
        state_payload = {"anomaly_block_until_ts": dict(state_map)}
        return dyn_mask, reasons, state_payload

    hysteresis = float(dyn_cfg.hysteresis or 0.0)
    if hysteresis < 0:
        hysteresis = 0.0
    cooldown = max(0, int(dyn_cfg.cooldown_bars or 0))
    next_block_cfg: Mapping[str, Any] = getattr(dyn_cfg, "next_bars_block", {}) or {}

    anomaly_state: Dict[str, int] = dict(state_map)
    symbol_states: Dict[str, Dict[str, Any]] = {}

    for symbol, group in df.groupby(symbols, sort=False):
        idx = group.index
        symbol_ts = ts_series.loc[idx]
        symbol_valid = ts_valid_series.loc[idx]
        ts_values = symbol_ts.to_numpy(dtype=np.int64)
        valid_ts = ts_values[symbol_valid.to_numpy(dtype=bool)]
        diffs = np.diff(valid_ts)
        diffs = diffs[diffs > 0]
        median_delta = float(np.median(diffs)) if diffs.size > 0 else 0.0

        blocked = False
        cooldown_left = 0
        next_block_left = 0
        last_trigger: Tuple[str, ...] = ()
        last_snapshot: Dict[str, Any] = {}
        block_deadline = anomaly_state.get(symbol, -1)
        last_valid_ts = block_deadline if block_deadline is not None else -1

        for label in idx:
            ts_val = int(symbol_ts.loc[label])
            ts_ok = bool(symbol_valid.loc[label] and ts_val >= 0)

            blocked_by_state = ts_ok and block_deadline >= 0 and ts_val <= block_deadline
            blocked_by_next = next_block_left > 0

            triggered_reasons: List[str] = []
            if thresholds_defined:
                if dyn_cfg.vol_abs is not None:
                    val = vol_metric.loc[label]
                    if not np.isnan(val) and val >= float(dyn_cfg.vol_abs):
                        reasons.at[label, "dyn_vol_abs"] = True
                        triggered_reasons.append("dyn_vol_abs")

                if dyn_cfg.vol_pctile is not None:
                    val = vol_pctile.loc[label]
                    if not np.isnan(val) and val >= float(dyn_cfg.vol_pctile):
                        reasons.at[label, "dyn_vol_pctile"] = True
                        triggered_reasons.append("dyn_vol_pctile")

                if dyn_cfg.spread_abs_bps is not None:
                    val = spread_proxy.loc[label]
                    if not np.isnan(val) and val >= float(dyn_cfg.spread_abs_bps):
                        reasons.at[label, "dyn_spread_abs"] = True
                        triggered_reasons.append("dyn_spread_abs")

                if dyn_cfg.spread_pctile is not None:
                    val = spread_pctile.loc[label]
                    if not np.isnan(val) and val >= float(dyn_cfg.spread_pctile):
                        reasons.at[label, "dyn_spread_pctile"] = True
                        triggered_reasons.append("dyn_spread_pctile")

            if triggered_reasons:
                reasons.at[label, "dyn_guard_raw"] = True
                if any(r.startswith("dyn_vol") for r in triggered_reasons):
                    reasons.at[label, "dyn_ret_anomaly"] = True
                if any(r.startswith("dyn_spread") for r in triggered_reasons):
                    reasons.at[label, "dyn_spread_anomaly"] = True

            guard_block = False
            hold_reason = False

            if triggered_reasons:
                blocked = True
                cooldown_left = max(cooldown_left, cooldown)
                guard_block = True
                last_trigger = tuple(triggered_reasons)
            elif blocked:
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
                    guard_block = True
                    hold_reason = True
                elif cooldown_left > 0:
                    guard_block = True
                    hold_reason = True
                    cooldown_left -= 1
                else:
                    blocked = False
                    cooldown_left = 0

            if hold_reason:
                reasons.at[label, "dyn_guard_hold"] = True

            if blocked_by_next:
                reasons.at[label, "dyn_guard_next_block"] = True

            if blocked_by_state:
                reasons.at[label, "dyn_guard_state"] = True

            final_block = (
                bool(triggered_reasons)
                or guard_block
                or blocked_by_next
                or blocked_by_state
            )

            if final_block:
                dyn_mask.loc[label] = True
                if ts_ok and (guard_block or triggered_reasons or blocked_by_next):
                    block_deadline = max(block_deadline, ts_val)
                    last_valid_ts = max(last_valid_ts, ts_val)
            else:
                dyn_mask.loc[label] = False

            last_snapshot = {
                "vol_metric": float(vol_metric.loc[label])
                if not np.isnan(vol_metric.loc[label])
                else None,
                "vol_pctile": float(vol_pctile.loc[label])
                if not np.isnan(vol_pctile.loc[label])
                else None,
                "spread": float(spread_proxy.loc[label])
                if not np.isnan(spread_proxy.loc[label])
                else None,
                "spread_pctile": float(spread_pctile.loc[label])
                if not np.isnan(spread_pctile.loc[label])
                else None,
                "ts": int(ts_val) if ts_ok else None,
            }

            if blocked_by_next:
                next_block_left = max(0, next_block_left - 1)

            if triggered_reasons:
                extra = _resolve_next_block(triggered_reasons, next_block_cfg)
                if extra > 0:
                    next_block_left = max(next_block_left, extra)

        if next_block_left > 0 and last_valid_ts >= 0:
            if median_delta > 0:
                future_ts = last_valid_ts + int(median_delta * next_block_left)
                block_deadline = max(block_deadline, future_ts)
            else:
                block_deadline = max(block_deadline, last_valid_ts)

        if block_deadline >= 0:
            anomaly_state[symbol] = int(block_deadline)
        elif symbol in anomaly_state:
            anomaly_state.pop(symbol, None)

        symbol_states[symbol] = {
            "blocked": bool(blocked),
            "cooldown_left": int(max(0, cooldown_left)),
            "next_block_left": int(max(0, next_block_left)),
            "block_until_ts": int(block_deadline) if block_deadline >= 0 else None,
            "last_trigger": list(last_trigger),
            "last_snapshot": last_snapshot,
            "median_bar_ms": int(median_delta) if median_delta > 0 else None,
        }

    state_payload = {
        "anomaly_block_until_ts": anomaly_state,
        "dynamic_guard": symbol_states,
    }
    return dyn_mask, reasons, state_payload


def _extract_anomaly_state(state: Optional[Any]) -> Dict[str, int]:
    """Normalise anomaly state input into ``symbol -> timestamp`` map."""

    if state is None:
        return {}
    if isinstance(state, NoTradeState):
        source = state.anomaly_block_until_ts or {}
    elif isinstance(state, Mapping):
        raw = state.get("anomaly_block_until_ts") if isinstance(state, Mapping) else None
        if isinstance(raw, Mapping):
            source = raw
        else:
            source = state
    else:
        return {}

    result: Dict[str, int] = {}
    if isinstance(source, Mapping):
        for symbol, value in source.items():
            ts = _coerce_int_or_none(value)
            if ts is not None:
                result[str(symbol)] = ts
    return result


def _compute_no_trade_components(
    df: pd.DataFrame,
    cfg: NoTradeConfig,
    *,
    ts_col: str = "ts_ms",
    state: Optional[Any] = None,
) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, str]]:
    ts_int, ts_valid = _prepare_ts(df, ts_col)
    state_map = _extract_anomaly_state(state)

    window_reasons = _window_reasons(ts_int, cfg)
    window_reasons.index = df.index
    window_reasons = window_reasons.astype(bool)
    valid_series = pd.Series(ts_valid, index=df.index, dtype=bool)
    window_reasons.loc[~valid_series, :] = False
    window_mask = window_reasons["window"].to_numpy(dtype=bool)

    dyn_cfg = cfg.dynamic_guard if hasattr(cfg, "dynamic_guard") else None
    dyn_mask = pd.Series(False, index=df.index, dtype=bool)
    dyn_reasons = pd.DataFrame(index=df.index)
    dyn_state: Dict[str, Any] = {
        "anomaly_block_until_ts": dict(state_map),
        "dynamic_guard": {},
    }
    meta: Dict[str, Any] = {}

    expected_dyn_cols = [
        "dyn_vol_abs",
        "dyn_vol_pctile",
        "dyn_spread_abs",
        "dyn_spread_pctile",
        "dyn_ret_anomaly",
        "dyn_spread_anomaly",
        "dyn_guard_raw",
        "dyn_guard_hold",
        "dyn_guard_next_block",
        "dyn_guard_state",
    ]

    if dyn_cfg and (getattr(dyn_cfg, "enable", False) or state_map):
        dyn_mask, dyn_reasons, dyn_state = _dynamic_guard_mask(
            df,
            dyn_cfg,
            ts_int=ts_int,
            ts_valid=ts_valid,
            state_map=state_map,
        )
        dyn_meta = getattr(dyn_reasons, "attrs", {}).get("meta") if isinstance(dyn_reasons, pd.DataFrame) else None
        if dyn_meta:
            meta["dynamic_guard"] = dyn_meta
    elif state_map:
        dyn_reasons = pd.DataFrame(False, index=df.index, columns=expected_dyn_cols)
        symbols = _symbol_series(df)
        ts_series = pd.Series(ts_int, index=df.index, dtype=np.int64)
        for label in df.index:
            symbol = symbols.loc[label]
            ts_val = ts_series.loc[label]
            if symbol in state_map and ts_val >= 0 and ts_val <= state_map[symbol]:
                dyn_mask.loc[label] = True
                dyn_reasons.at[label, "dyn_guard_state"] = True
    else:
        dyn_reasons = pd.DataFrame(False, index=df.index, columns=expected_dyn_cols)

    if not dyn_reasons.empty:
        for col in expected_dyn_cols:
            if col not in dyn_reasons.columns:
                dyn_reasons[col] = False
        dyn_reasons = dyn_reasons[expected_dyn_cols]
        dyn_reasons = dyn_reasons.reindex(df.index).fillna(False).astype(bool)
    else:
        dyn_reasons = pd.DataFrame(False, index=df.index, columns=expected_dyn_cols)

    reasons = pd.concat([window_reasons, dyn_reasons], axis=1)
    if not dyn_reasons.empty:
        reasons["dynamic_guard"] = dyn_mask.astype(bool)
    else:
        reasons["dynamic_guard"] = False
    reasons = reasons.reindex(df.index).fillna(False).astype(bool)

    combined = window_mask | dyn_mask.to_numpy(dtype=bool)
    mask = pd.Series(combined, index=df.index, name="no_trade_block")

    reason_labels: Dict[str, str] = {
        "window": "Maintenance windows",
        "maintenance_daily": "Maintenance: daily schedule",
        "maintenance_funding": "Maintenance: funding buffer",
        "maintenance_custom": "Maintenance: custom window",
        "dynamic_guard": "Dynamic guard",  # aggregated column
        "dyn_vol_abs": "Dynamic guard: volatility >= abs",
        "dyn_vol_pctile": "Dynamic guard: volatility percentile",
        "dyn_spread_abs": "Dynamic guard: spread >= abs",
        "dyn_spread_pctile": "Dynamic guard: spread percentile",
        "dyn_ret_anomaly": "Dynamic guard: return anomaly",
        "dyn_spread_anomaly": "Dynamic guard: spread anomaly",
        "dyn_guard_raw": "Dynamic guard triggered",
        "dyn_guard_hold": "Dynamic guard hold",
        "dyn_guard_next_block": "Dynamic guard next-bar block",
        "dyn_guard_state": "Dynamic guard state carry",
    }

    return mask, reasons, meta, dyn_state, reason_labels


def estimate_block_ratio(
    df: pd.DataFrame,
    cfg: NoTradeConfig,
    ts_col: str = "ts_ms",
    state: Optional[Any] = None,
) -> float:
    """Estimate share of rows blocked by schedule and dynamic guard."""

    if df.empty:
        return 0.0

    mask, _, _, _, _ = _compute_no_trade_components(
        df,
        cfg,
        ts_col=ts_col,
        state=state,
    )
    return float(mask.mean())


def compute_no_trade_mask(
    df: pd.DataFrame,
    *,
    sandbox_yaml_path: str = "configs/legacy_sandbox.yaml",
    ts_col: str = "ts_ms",
    config: Optional[NoTradeConfig] = None,
    state: Optional[Any] = None,
) -> pd.Series:
    """
    Возвращает pd.Series[bool] длины df:
      True  — строка попадает в «запрещённое» окно (no_trade), её надо исключить из обучения;
      False — строку можно использовать в train/val.
    """
    cfg = config or get_no_trade_config(sandbox_yaml_path)

    mask, reasons, meta, state_payload, reason_labels = _compute_no_trade_components(
        df,
        cfg,
        ts_col=ts_col,
        state=state,
    )

    mask.attrs["reasons"] = reasons
    mask.attrs["reason_labels"] = reason_labels
    if meta:
        mask.attrs["meta"] = meta
    if state_payload:
        mask.attrs["state"] = state_payload
    return mask
