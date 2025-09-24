# training/no_trade.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from no_trade_config import NoTradeConfig, NoTradeState, get_no_trade_config


LOGGER = logging.getLogger(__name__)
DEFAULT_MAINTENANCE_MAX_AGE_SEC = 24 * 3600


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
        diff = np.abs(sec_day - m)
        wrapped = 86400 - diff
        mask |= np.minimum(diff, wrapped) <= buf_min * 60
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


def _window_reasons(
    ts_ms: np.ndarray,
    cfg: NoTradeConfig,
    *,
    symbols: Optional[pd.Series] = None,
    calendar: Optional[Dict[str, Any]] = None,
    calendar_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Return per-reason mask for schedule-based no-trade windows."""

    maintenance_state: Dict[str, Any] = {
        "funding_buffer_min": int(getattr(cfg, "funding_buffer_min", 0) or 0),
        "daily_utc": list(getattr(cfg, "daily_utc", []) or []),
        "custom_ms": [],
    }
    for item in getattr(cfg, "custom_ms", []) or []:
        if isinstance(item, Mapping):
            start = _coerce_timestamp_ms(item.get("start_ts_ms"))
            end = _coerce_timestamp_ms(item.get("end_ts_ms"))
            if start is not None and end is not None and start < end:
                payload = {"start_ts_ms": int(start), "end_ts_ms": int(end)}
                if "symbol" in item:
                    symbol = _coerce_str_or_none(item.get("symbol"))
                    if symbol:
                        payload["symbol"] = symbol
                maintenance_state["custom_ms"].append(payload)

    calendar_state: Dict[str, Any] = {}
    if calendar_meta:
        for key in (
            "path",
            "source",
            "format",
            "exists",
            "mtime",
            "age_sec",
            "stale",
            "error",
            "total",
            "discarded",
        ):
            if key in calendar_meta and calendar_meta[key] is not None:
                calendar_state[key] = calendar_meta[key]
    if calendar:
        if calendar.get("windows"):
            calendar_state["windows"] = calendar.get("windows")
        if calendar.get("global"):
            calendar_state["global"] = calendar.get("global")
        if calendar.get("per_symbol"):
            calendar_state["per_symbol"] = calendar.get("per_symbol")
    if calendar_state:
        maintenance_state["calendar"] = calendar_state

    if ts_ms.size == 0:
        df = pd.DataFrame(
            {
                "maintenance_daily": [],
                "maintenance_funding": [],
                "maintenance_custom": [],
                "maintenance_calendar": [],
                "window": [],
            }
        )
        return df.astype(bool), maintenance_state, calendar_meta or {}

    daily_min = _parse_daily_windows_min(cfg.daily_utc or [])
    m_daily = _in_daily_window(ts_ms, daily_min)
    m_funding = _in_funding_buffer(ts_ms, int(cfg.funding_buffer_min or 0))
    m_custom = _in_custom_window(ts_ms, cfg.custom_ms or [])
    m_calendar = _apply_calendar_windows(ts_ms, calendar, symbols)

    data = {
        "maintenance_daily": m_daily,
        "maintenance_funding": m_funding,
        "maintenance_custom": m_custom,
        "maintenance_calendar": m_calendar,
    }
    df = pd.DataFrame(data)
    df["window"] = df.any(axis=1)
    return df.astype(bool), maintenance_state, calendar_meta or {}


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


def _coerce_timestamp_ms(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            ts = _coerce_timestamp_ms(item)
            if ts is not None:
                return ts
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text, 10)
        except ValueError:
            try:
                parsed = float(text)
            except ValueError:
                return None
            if np.isnan(parsed):
                return None
            return int(parsed)
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(parsed):
            return None
        return int(parsed)


def _coerce_str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
    try:
        if isinstance(value, pd.Series):  # defensive guard
            value = value.iloc[0]
    except Exception:  # pragma: no cover - defensive
        pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"nan", "none", "null"}:
            return None
        return text
    try:
        text = str(value)
    except Exception:  # pragma: no cover - defensive
        return None
    text = text.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return None
    return text


def _parse_symbol_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: List[str] = []
        for item in value:
            result.extend(_parse_symbol_list(item))
        return result
    text = _coerce_str_or_none(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            loaded = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, list):
            return [sym for sym in (_coerce_str_or_none(item) for item in loaded) if sym]
    parts = [part.strip() for part in re.split(r"[;,\\s/]+", text) if part.strip()]
    if parts:
        return parts
    return [text]


def _flatten_calendar_payload(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        if isinstance(payload.get("windows"), list):
            return [item for item in payload.get("windows", []) if isinstance(item, Mapping)]
        if isinstance(payload.get("data"), list):
            return [item for item in payload.get("data", []) if isinstance(item, Mapping)]

        flattened: List[Mapping[str, Any]] = []
        for key, value in payload.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, Mapping):
                        entry = dict(item)
                        entry.setdefault("symbol", key)
                        flattened.append(entry)
            elif isinstance(value, Mapping):
                entry = dict(value)
                entry.setdefault("symbol", key)
                flattened.append(entry)

        if flattened:
            return flattened
        return [payload]
    return []


def _normalise_calendar_records(records: Iterable[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    normalised: List[Dict[str, Any]] = []
    total = 0
    discarded = 0

    start_keys = (
        "start_ts_ms",
        "start_ts",
        "start_ms",
        "start",
        "from_ts_ms",
        "from_ts",
    )
    end_keys = (
        "end_ts_ms",
        "end_ts",
        "end_ms",
        "end",
        "to_ts_ms",
        "to_ts",
        "finish_ts_ms",
    )

    for item in records:
        if not isinstance(item, Mapping):
            continue

        total += 1

        start_ts: Optional[int] = None
        end_ts: Optional[int] = None

        for key in start_keys:
            if key in item:
                start_ts = _coerce_timestamp_ms(item.get(key))
                if start_ts is not None:
                    break

        for key in end_keys:
            if key in item:
                end_ts = _coerce_timestamp_ms(item.get(key))
                if end_ts is not None:
                    break

        if start_ts is None or end_ts is None or start_ts >= end_ts:
            discarded += 1
            continue

        reason: Optional[str] = None
        for key in ("reason", "label", "tag", "description", "title"):
            reason = _coerce_str_or_none(item.get(key))
            if reason:
                break

        symbols: List[str] = []
        for key in ("symbol", "symbols", "pair", "pairs", "instrument"):
            if key in item:
                symbols = _parse_symbol_list(item.get(key))
                if symbols:
                    break

        base_payload: Dict[str, Any] = {
            "start_ts_ms": int(start_ts),
            "end_ts_ms": int(end_ts),
        }
        if reason:
            base_payload["reason"] = reason

        if not symbols:
            normalised.append(dict(base_payload))
        else:
            for symbol in symbols:
                entry = dict(base_payload)
                entry["symbol"] = symbol
                normalised.append(entry)

    meta = {"total": total, "discarded": discarded}
    return normalised, meta


def _build_calendar_structure(windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_symbol: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    global_windows: List[Tuple[int, int]] = []

    for entry in windows:
        start = int(entry["start_ts_ms"])
        end = int(entry["end_ts_ms"])
        symbol = _coerce_str_or_none(entry.get("symbol"))
        if symbol:
            per_symbol[symbol].append((start, end))
        else:
            global_windows.append((start, end))

    def _merge(items: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not items:
            return []
        items.sort()
        merged: List[Tuple[int, int]] = [items[0]]
        for start, end in items[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    for key, items in list(per_symbol.items()):
        per_symbol[key] = _merge(items)
    global_windows = _merge(global_windows)

    return {
        "global": global_windows,
        "per_symbol": dict(per_symbol),
        "windows": windows,
    }


def _apply_calendar_windows(
    ts_ms: np.ndarray,
    calendar: Optional[Dict[str, Any]],
    symbols: Optional[pd.Series] = None,
) -> np.ndarray:
    if ts_ms.size == 0:
        return np.zeros_like(ts_ms, dtype=bool)
    if not calendar:
        return np.zeros_like(ts_ms, dtype=bool)

    mask = np.zeros_like(ts_ms, dtype=bool)

    for start, end in calendar.get("global", []) or []:
        mask |= (ts_ms >= int(start)) & (ts_ms <= int(end))

    if symbols is not None and calendar.get("per_symbol"):
        symbol_values = symbols.to_numpy(dtype=object)
        for symbol, windows in calendar.get("per_symbol", {}).items():
            if not windows:
                continue
            symbol_mask = symbol_values == symbol
            if not symbol_mask.any():
                continue
            symbol_mask = symbol_mask.astype(bool)
            for start, end in windows:
                mask |= symbol_mask & (ts_ms >= int(start)) & (ts_ms <= int(end))

    return mask


def _load_maintenance_calendar(cfg: NoTradeConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    maintenance_cfg = getattr(cfg, "maintenance", None)
    if maintenance_cfg is None:
        return {"global": [], "per_symbol": {}, "windows": []}, {}

    path_value = getattr(maintenance_cfg, "path", None)
    if not path_value:
        return {"global": [], "per_symbol": {}, "windows": []}, {}

    path_hint = _coerce_str_or_none(path_value)
    if not path_hint:
        return {"global": [], "per_symbol": {}, "windows": []}, {}

    base_dir = _coerce_str_or_none(getattr(maintenance_cfg, "_config_base_dir", None))
    if not base_dir:
        base_dir = _coerce_str_or_none(getattr(cfg, "_config_base_dir", None))
    config_path_hint = _coerce_str_or_none(getattr(cfg, "_config_path", None))
    if base_dir is None and config_path_hint:
        try:
            base_dir = str(Path(config_path_hint).resolve().parent)
        except Exception:  # pragma: no cover - defensive
            base_dir = None

    raw_source = _coerce_str_or_none(getattr(maintenance_cfg, "_path_source", None))
    if raw_source is None:
        raw_source = path_hint

    path = Path(path_hint).expanduser()
    if not path.is_absolute() and base_dir:
        try:
            path = Path(base_dir) / path
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        resolved_path = path.resolve(strict=False)
    except Exception:  # pragma: no cover - defensive
        resolved_path = path

    meta: Dict[str, Any] = {"path": str(resolved_path)}
    if raw_source and str(resolved_path) != str(raw_source):
        meta["source"] = raw_source
    if base_dir:
        meta["base_dir"] = base_dir

    format_hint = _coerce_str_or_none(getattr(maintenance_cfg, "format", None))
    format_mode = "auto"
    if format_hint:
        lowered = format_hint.strip().lower()
        if lowered in {"json", "csv"}:
            format_mode = lowered
        elif lowered in {"auto", ""} or "hh:mm" in lowered:
            format_mode = "auto"
        else:
            LOGGER.warning(
                "Unknown maintenance.format hint '%s', falling back to auto", format_hint
            )
    meta["format"] = format_mode

    if not path.exists():
        LOGGER.warning("Maintenance calendar file not found: path=%s", resolved_path)
        meta["exists"] = False
        return {"global": [], "per_symbol": {}, "windows": []}, meta

    meta["exists"] = True

    try:
        stat = resolved_path.stat()
    except OSError as exc:
        LOGGER.warning("Failed to stat maintenance calendar file %s: %s", resolved_path, exc)
        stat = None
    if stat is not None:
        mtime = stat.st_mtime
        meta["mtime"] = mtime
        age_sec = max(0.0, time.time() - mtime)
        meta["age_sec"] = age_sec

        max_age_sec = getattr(maintenance_cfg, "max_age_sec", None)
        if max_age_sec is None:
            max_age_hours = getattr(maintenance_cfg, "max_age_hours", None)
            if max_age_hours is not None:
                try:
                    max_age_sec = float(max_age_hours) * 3600.0
                except (TypeError, ValueError):
                    max_age_sec = None
        if max_age_sec is None:
            max_age_sec = DEFAULT_MAINTENANCE_MAX_AGE_SEC

        try:
            threshold = float(max_age_sec) if max_age_sec is not None else None
        except (TypeError, ValueError):  # pragma: no cover - defensive
            threshold = None

        if threshold is not None and age_sec > threshold:
            meta["stale"] = True
            LOGGER.warning(
                "Maintenance calendar file looks stale: path=%s age_sec=%.0f max_age_sec=%s",
                resolved_path,
                age_sec,
                threshold,
            )
        else:
            meta["stale"] = False

    suffix = resolved_path.suffix.lower()
    load_format = format_mode
    if format_mode == "auto":
        if suffix in {".json", ""}:
            load_format = "json"
        elif suffix == ".csv":
            load_format = "csv"
        else:
            LOGGER.warning(
                "Unsupported maintenance calendar format: path=%s suffix=%s", 
                resolved_path,
                suffix or "<none>",
            )
            meta["error"] = "unsupported_format"
            return {"global": [], "per_symbol": {}, "windows": []}, meta
    meta["format"] = load_format

    try:
        if load_format == "json":
            text = resolved_path.read_text(encoding="utf-8")
            if text.strip():
                payload = json.loads(text)
                records = _flatten_calendar_payload(payload)
            else:
                records = []
        elif load_format == "csv":
            df = pd.read_csv(resolved_path)
            records = df.to_dict(orient="records")
        else:
            LOGGER.warning(
                "Unsupported maintenance calendar format hint: path=%s format=%s",
                resolved_path,
                load_format or "<none>",
            )
            meta["error"] = "unsupported_format"
            return {"global": [], "per_symbol": {}, "windows": []}, meta
    except Exception as exc:  # pragma: no cover - parsing errors
        LOGGER.warning("Failed to read maintenance calendar file %s: %s", resolved_path, exc)
        meta["error"] = str(exc)
        return {"global": [], "per_symbol": {}, "windows": []}, meta

    windows, stats = _normalise_calendar_records(records)
    meta.update(stats)
    calendar = _build_calendar_structure(windows)
    return calendar, meta


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
    prev_symbol_states: Optional[Mapping[str, Mapping[str, Any]]] = None,
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
            "dyn_vol_extreme",
            "dyn_spread_wide",
            "dyn_guard_warmup",
            "dyn_cooldown",
            "dyn_guard_raw",
            "dyn_guard_hold",
            "dyn_guard_next_block",
            "dyn_guard_state",
        ],
    )
    reasons.attrs["meta"] = {}

    vol_cfg = getattr(dyn_cfg, "volatility", None)
    spread_cfg = getattr(dyn_cfg, "spread", None)

    thresholds_defined = any(
        x is not None
        for x in (
            dyn_cfg.vol_abs,
            dyn_cfg.vol_pctile,
            dyn_cfg.spread_abs_bps,
            dyn_cfg.spread_pctile,
            getattr(vol_cfg, "upper_multiplier", None) if vol_cfg is not None else None,
            getattr(spread_cfg, "upper_pctile", None) if spread_cfg is not None else None,
        )
    )

    ts_series = pd.Series(ts_int, index=df.index, dtype=np.int64)
    ts_valid_series = pd.Series(ts_valid, index=df.index, dtype=bool)
    state_map = {str(k): int(v) for k, v in (state_map or {}).items()}

    if not thresholds_defined and not state_map:
        return dyn_mask, reasons, {"anomaly_block_until_ts": {}}

    required_metrics: List[str] = []
    if (
        dyn_cfg.vol_abs is not None
        or dyn_cfg.vol_pctile is not None
        or (vol_cfg is not None and getattr(vol_cfg, "upper_multiplier", None) is not None)
    ):
        required_metrics.append("volatility")
    if (
        dyn_cfg.spread_abs_bps is not None
        or dyn_cfg.spread_pctile is not None
        or (spread_cfg is not None and getattr(spread_cfg, "upper_pctile", None) is not None)
    ):
        required_metrics.append("spread")

    close = price.replace(0, np.nan)
    returns = price.groupby(symbols).pct_change()

    sigma_window = _coerce_positive_int(
        getattr(vol_cfg, "window", None) or getattr(dyn_cfg, "sigma_window", None)
    )
    if sigma_window <= 1:
        sigma_window = 120
    sigma_min_periods = _coerce_positive_int(
        getattr(vol_cfg, "min_periods", None) or getattr(dyn_cfg, "sigma_min_periods", None)
    )
    if sigma_min_periods <= 0:
        sigma_min_periods = min(sigma_window, max(2, sigma_window // 2))

    sigma = _rolling_sigma(returns, symbols, sigma_window, min_periods=sigma_min_periods)
    if sigma_window > 1:
        sigma_ready = (
            returns.notna()
            .astype(float)
            .groupby(symbols)
            .rolling(window=sigma_window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        sigma_ready = sigma_ready.reindex(returns.index).fillna(0.0) >= float(sigma_min_periods)
    else:
        sigma_ready = pd.Series(False, index=returns.index, dtype=bool)

    atr_window = _coerce_positive_int(
        getattr(dyn_cfg, "atr_window", None) or getattr(spread_cfg, "window", None)
    )
    if atr_window <= 1:
        atr_window = 14
    atr_min_periods = _coerce_positive_int(
        getattr(dyn_cfg, "atr_min_periods", None) or getattr(spread_cfg, "min_periods", None)
    )
    if atr_min_periods <= 0:
        atr_min_periods = min(atr_window, max(1, atr_window // 2))

    atr = _rolling_atr(high, low, close, symbols, atr_window, min_periods=atr_min_periods)
    atr_pct = atr / close.abs()
    spread_proxy = spread
    if spread_proxy.isna().all() and atr_pct.notna().any():
        spread_proxy = atr_pct * 10000.0

    vol_metric = sigma.fillna(atr_pct)

    vol_pctile_window = _coerce_positive_int(
        getattr(vol_cfg, "pctile_window", None) or getattr(dyn_cfg, "vol_pctile_window", None)
    )
    if vol_pctile_window <= 1:
        vol_pctile_window = sigma_window
    vol_pctile_min_periods = _coerce_positive_int(
        getattr(vol_cfg, "pctile_min_periods", None)
        or getattr(dyn_cfg, "vol_pctile_min_periods", None)
    )
    if vol_pctile_min_periods <= 0:
        vol_pctile_min_periods = min(vol_pctile_window, sigma_min_periods)

    vol_pctile = _rolling_percentile(
        vol_metric,
        symbols,
        vol_pctile_window,
        min_periods=vol_pctile_min_periods,
    )

    spread_pctile_window = _coerce_positive_int(
        getattr(spread_cfg, "pctile_window", None) or getattr(dyn_cfg, "spread_pctile_window", None)
    )
    if spread_pctile_window <= 1:
        spread_pctile_window = atr_window
    spread_pctile_min_periods = _coerce_positive_int(
        getattr(spread_cfg, "pctile_min_periods", None)
        or getattr(dyn_cfg, "spread_pctile_min_periods", None)
    )
    if spread_pctile_min_periods <= 0:
        spread_pctile_min_periods = min(spread_pctile_window, atr_min_periods)

    spread_pctile = _rolling_percentile(
        spread_proxy,
        symbols,
        spread_pctile_window,
        min_periods=spread_pctile_min_periods,
    )

    if spread_pctile_window > 1:
        spread_ready = (
            spread_proxy.notna()
            .astype(float)
            .groupby(symbols)
            .rolling(window=spread_pctile_window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        spread_ready = spread_ready.reindex(spread_proxy.index).fillna(0.0) >= float(
            spread_pctile_min_periods
        )
    else:
        spread_ready = pd.Series(False, index=spread_proxy.index, dtype=bool)

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
    cooldown_candidates = [
        int(dyn_cfg.cooldown_bars or 0),
    ]
    if vol_cfg is not None:
        cooldown_candidates.append(int(getattr(vol_cfg, "cooldown_bars", 0) or 0))
    if spread_cfg is not None:
        cooldown_candidates.append(int(getattr(spread_cfg, "cooldown_bars", 0) or 0))
    cooldown = max(0, max(cooldown_candidates))
    next_block_cfg: Mapping[str, Any] = getattr(dyn_cfg, "next_bars_block", {}) or {}

    prev_symbol_states = {str(k): dict(v) for k, v in (prev_symbol_states or {}).items()}
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

        previous_state = prev_symbol_states.get(str(symbol), {})

        blocked = bool(previous_state.get("blocked", False))
        cooldown_left = max(0, int(previous_state.get("cooldown_left", 0)))
        next_block_left = max(0, int(previous_state.get("next_block_left", 0)))
        last_trigger: Tuple[str, ...] = tuple(previous_state.get("last_trigger", []) or [])
        last_snapshot: Dict[str, Any] = dict(previous_state.get("last_snapshot", {}) or {})
        block_deadline = max(
            anomaly_state.get(symbol, -1),
            _coerce_positive_int(previous_state.get("block_until_ts")) if previous_state else -1,
        )
        last_valid_ts = block_deadline if block_deadline is not None else -1

        for label in idx:
            ts_val = int(symbol_ts.loc[label])
            ts_ok = bool(symbol_valid.loc[label] and ts_val >= 0)

            blocked_by_state = ts_ok and block_deadline >= 0 and ts_val <= block_deadline
            blocked_by_next = next_block_left > 0

            triggered_reasons: List[str] = []
            guard_ready = True
            vol_trigger = False
            spread_trigger = False

            vol_upper_mult = getattr(vol_cfg, "upper_multiplier", None) if vol_cfg is not None else None
            vol_lower_mult = getattr(vol_cfg, "lower_multiplier", None) if vol_cfg is not None else None

            spread_upper_pct = getattr(spread_cfg, "upper_pctile", None) if spread_cfg is not None else None
            spread_lower_pct = getattr(spread_cfg, "lower_pctile", None) if spread_cfg is not None else None
            if spread_lower_pct is None and spread_upper_pct is not None and hysteresis > 0:
                spread_lower_pct = max(0.0, float(spread_upper_pct) - hysteresis)

            if thresholds_defined:

                vol_checks = (
                    dyn_cfg.vol_abs is not None
                    or dyn_cfg.vol_pctile is not None
                    or vol_upper_mult is not None
                )
                spread_checks = (
                    dyn_cfg.spread_abs_bps is not None
                    or dyn_cfg.spread_pctile is not None
                    or spread_upper_pct is not None
                )

                vol_ready_row = True
                spread_ready_row = True
                if vol_checks:
                    vol_ready_row = bool(sigma_ready.loc[label])
                if spread_checks:
                    spread_ready_row = bool(spread_ready.loc[label])
                guard_ready = bool(vol_ready_row and spread_ready_row)

                if not guard_ready:
                    reasons.at[label, "dyn_guard_warmup"] = True
                else:
                    ret_val = returns.loc[label]
                    sigma_val = sigma.loc[label]
                    abs_ret = abs(ret_val) if not np.isnan(ret_val) else np.nan

                    if dyn_cfg.vol_abs is not None:
                        val = vol_metric.loc[label]
                        if not np.isnan(val) and val >= float(dyn_cfg.vol_abs):
                            reasons.at[label, "dyn_vol_abs"] = True
                            triggered_reasons.append("dyn_vol_abs")
                            vol_trigger = True

                    if dyn_cfg.vol_pctile is not None:
                        val = vol_pctile.loc[label]
                        if not np.isnan(val) and val >= float(dyn_cfg.vol_pctile):
                            reasons.at[label, "dyn_vol_pctile"] = True
                            triggered_reasons.append("dyn_vol_pctile")
                            vol_trigger = True

                    if (
                        vol_upper_mult is not None
                        and not np.isnan(abs_ret)
                        and not np.isnan(sigma_val)
                        and float(sigma_val) > 0
                    ):
                        if abs_ret >= float(vol_upper_mult) * float(sigma_val):
                            reasons.at[label, "dyn_vol_extreme"] = True
                            triggered_reasons.append("dyn_vol_extreme")
                            vol_trigger = True

                    if dyn_cfg.spread_abs_bps is not None:
                        val = spread_proxy.loc[label]
                        if not np.isnan(val) and val >= float(dyn_cfg.spread_abs_bps):
                            reasons.at[label, "dyn_spread_abs"] = True
                            triggered_reasons.append("dyn_spread_abs")
                            spread_trigger = True

                    if dyn_cfg.spread_pctile is not None:
                        val = spread_pctile.loc[label]
                        if not np.isnan(val) and val >= float(dyn_cfg.spread_pctile):
                            reasons.at[label, "dyn_spread_pctile"] = True
                            triggered_reasons.append("dyn_spread_pctile")
                            spread_trigger = True

                    if spread_upper_pct is None and dyn_cfg.spread_pctile is not None:
                        spread_upper_pct = float(dyn_cfg.spread_pctile)

                    if spread_upper_pct is not None:
                        val = spread_pctile.loc[label]
                        if not np.isnan(val) and val >= float(spread_upper_pct):
                            reasons.at[label, "dyn_spread_wide"] = True
                            triggered_reasons.append("dyn_spread_wide")
                            spread_trigger = True

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
                if not guard_ready:
                    blocked = False
                    cooldown_left = 0
                else:
                    release_ready = True
                    sigma_val = sigma.loc[label]
                    ret_val = returns.loc[label]
                    abs_ret = abs(ret_val) if not np.isnan(ret_val) else np.nan

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

                    if (
                        vol_lower_mult is not None
                        and not np.isnan(abs_ret)
                        and not np.isnan(sigma_val)
                        and float(sigma_val) > 0
                    ):
                        if abs_ret > float(vol_lower_mult) * float(sigma_val):
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

                    if spread_lower_pct is not None:
                        val = spread_pctile.loc[label]
                        if not np.isnan(val) and val > float(spread_lower_pct):
                            release_ready = False

                    if not release_ready:
                        guard_block = True
                        hold_reason = True
                    elif cooldown_left > 0:
                        guard_block = True
                        hold_reason = True
                        cooldown_left -= 1
                        reasons.at[label, "dyn_cooldown"] = True
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
                "sigma": float(sigma.loc[label]) if not np.isnan(sigma.loc[label]) else None,
                "ret_last": float(returns.loc[label]) if not np.isnan(returns.loc[label]) else None,
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
                "ready": bool(guard_ready),
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


def _extract_dynamic_guard_state(state: Optional[Any]) -> Dict[str, Dict[str, Any]]:
    if state is None:
        return {}
    if isinstance(state, NoTradeState):
        raw = state.dynamic_guard or {}
    elif isinstance(state, Mapping):
        raw = state.get("dynamic_guard") if isinstance(state, Mapping) else {}
    else:
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        for symbol, payload in raw.items():
            if isinstance(payload, Mapping):
                result[str(symbol)] = dict(payload)
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
    prev_dynamic_state = _extract_dynamic_guard_state(state)

    symbols = _symbol_series(df)
    calendar, calendar_meta = _load_maintenance_calendar(cfg)
    window_reasons, maintenance_state, maintenance_meta = _window_reasons(
        ts_int,
        cfg,
        symbols=symbols,
        calendar=calendar,
        calendar_meta=calendar_meta,
    )
    window_reasons.index = df.index
    window_reasons = window_reasons.astype(bool)
    valid_series = pd.Series(ts_valid, index=df.index, dtype=bool)
    window_reasons.loc[~valid_series, :] = False
    window_mask = window_reasons["window"].to_numpy(dtype=bool)

    dyn_cfg = cfg.dynamic_guard if hasattr(cfg, "dynamic_guard") else None
    dyn_mask = pd.Series(False, index=df.index, dtype=bool)
    dyn_reasons = pd.DataFrame(index=df.index)
    meta: Dict[str, Any] = {}

    expected_dyn_cols = [
        "dyn_vol_abs",
        "dyn_vol_pctile",
        "dyn_spread_abs",
        "dyn_spread_pctile",
        "dyn_ret_anomaly",
        "dyn_spread_anomaly",
        "dyn_vol_extreme",
        "dyn_spread_wide",
        "dyn_guard_warmup",
        "dyn_cooldown",
        "dyn_guard_raw",
        "dyn_guard_hold",
        "dyn_guard_next_block",
        "dyn_guard_state",
    ]

    if dyn_cfg and (getattr(dyn_cfg, "enable", False) or state_map or prev_dynamic_state):
        dyn_mask, dyn_reasons, dyn_state = _dynamic_guard_mask(
            df,
            dyn_cfg,
            ts_int=ts_int,
            ts_valid=ts_valid,
            state_map=state_map,
            prev_symbol_states=prev_dynamic_state,
        )
        dyn_state = dict(dyn_state or {})
        dyn_state.setdefault("anomaly_block_until_ts", dict(state_map))
        dyn_state.setdefault("dynamic_guard", {})
        dyn_meta = getattr(dyn_reasons, "attrs", {}).get("meta") if isinstance(dyn_reasons, pd.DataFrame) else None
        if dyn_meta:
            meta["dynamic_guard"] = dyn_meta
    elif state_map or prev_dynamic_state:
        dyn_reasons = pd.DataFrame(False, index=df.index, columns=expected_dyn_cols)
        ts_series = pd.Series(ts_int, index=df.index, dtype=np.int64)
        for label in df.index:
            symbol = symbols.loc[label]
            ts_val = ts_series.loc[label]
            if symbol in state_map and ts_val >= 0 and ts_val <= state_map[symbol]:
                dyn_mask.loc[label] = True
                dyn_reasons.at[label, "dyn_guard_state"] = True
            prev_info = (
                prev_dynamic_state.get(str(symbol))
                if isinstance(prev_dynamic_state, Mapping)
                else None
            )
            if prev_info and bool(prev_info.get("blocked")):
                dyn_reasons.at[label, "dyn_guard_hold"] = True
        dyn_state = {
            "anomaly_block_until_ts": dict(state_map),
            "dynamic_guard": dict(prev_dynamic_state or {}),
        }
    else:
        dyn_reasons = pd.DataFrame(False, index=df.index, columns=expected_dyn_cols)
        dyn_state = {
            "anomaly_block_until_ts": dict(state_map),
            "dynamic_guard": dict(prev_dynamic_state or {}),
        }

    if not dyn_reasons.empty:
        for col in expected_dyn_cols:
            if col not in dyn_reasons.columns:
                dyn_reasons[col] = False
        dyn_reasons = dyn_reasons[expected_dyn_cols]
        dyn_reasons = dyn_reasons.reindex(df.index).fillna(False).astype(bool)
    else:
        dyn_reasons = pd.DataFrame(False, index=df.index, columns=expected_dyn_cols)

    dyn_state["maintenance"] = maintenance_state

    if maintenance_meta:
        filtered_meta = {
            key: value
            for key, value in maintenance_meta.items()
            if key not in {"windows"} and value is not None
        }
        if filtered_meta:
            meta["maintenance_calendar"] = filtered_meta

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
        "maintenance_calendar": "Maintenance: calendar schedule",
        "dynamic_guard": "Dynamic guard",  # aggregated column
        "dyn_vol_abs": "Dynamic guard: volatility >= abs",
        "dyn_vol_pctile": "Dynamic guard: volatility percentile",
        "dyn_spread_abs": "Dynamic guard: spread >= abs",
        "dyn_spread_pctile": "Dynamic guard: spread percentile",
        "dyn_ret_anomaly": "Dynamic guard: return anomaly",
        "dyn_spread_anomaly": "Dynamic guard: spread anomaly",
        "dyn_vol_extreme": "Dynamic guard: return vs sigma",
        "dyn_spread_wide": "Dynamic guard: spread percentile high",
        "dyn_guard_warmup": "Dynamic guard warm-up",
        "dyn_cooldown": "Dynamic guard cooldown",
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
