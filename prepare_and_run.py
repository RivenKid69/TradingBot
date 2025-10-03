# -*- coding: utf-8 -*-
"""
prepare_and_run.py
---------------------------------------------------------------
Merge raw 1h candles from data/candles/ with Fear & Greed (data/fear_greed.csv)
and write per-symbol Feather files to data/processed/ expected by training.
Also enforces column schema and avoids renaming 'volume'.
"""
import glob
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from impl_offline_data import timeframe_to_ms

RAW_DIR = os.path.join("data","candles")
KLINES_DIR = os.path.join("data", "klines")
FNG = os.path.join("data","fear_greed.csv")
EVENTS = os.path.join("data","economic_events.csv")
EVENT_HORIZON_HOURS = 96
OUT_DIR = os.path.join("data","processed")
os.makedirs(OUT_DIR, exist_ok=True)


RAW_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "symbol",
]


def _normalize_raw_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    df["symbol"] = df.get("symbol", symbol).fillna(symbol).astype(str)
    for col in ("open_time", "close_time"):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in raw candles for {symbol}")
        series = pd.to_numeric(df[col], errors="coerce")
        if series.max() > 10_000_000_000:
            series = (series // 1000).astype("int64")
        else:
            series = series.astype("int64")
        df[col] = series
    float_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for col in float_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["quote_asset_volume"] = df["quote_asset_volume"].fillna(
        df["close"].astype(float) * df["volume"].astype(float)
    )
    df["number_of_trades"] = pd.to_numeric(
        df.get("number_of_trades", 0), errors="coerce"
    ).fillna(0).astype("int64")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    df["taker_buy_base_asset_volume"] = df["taker_buy_base_asset_volume"].fillna(0.0)
    df["taker_buy_quote_asset_volume"] = df["taker_buy_quote_asset_volume"].fillna(0.0)
    df["timestamp"] = (df["close_time"] // 3600) * 3600
    df = df[
        [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]
    ].drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df


def _read_raw(path: str) -> Tuple[str, pd.DataFrame]:
    df = pd.read_csv(path)
    # Convert open/close time to seconds
    sym = os.path.splitext(os.path.basename(path))[0]
    return sym, _normalize_raw_df(df, sym)


def _parse_interval_ms(stem: str) -> Tuple[str, int]:
    if "_" in stem:
        sym, interval = stem.split("_", 1)
    else:
        sym, interval = stem, "1h"
    try:
        interval_ms = timeframe_to_ms(interval)
    except Exception:
        interval_ms = 3_600_000
    return sym.upper(), int(interval_ms)


def _infer_interval_ms(open_time_ms: pd.Series, fallback: int) -> int:
    values = np.sort(open_time_ms.dropna().unique())
    if len(values) <= 1:
        return fallback
    diffs = np.diff(values)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return fallback
    return int(diffs.min())


def _read_klines_parquet(path: str, *, write_csv: bool = True) -> Tuple[str, pd.DataFrame]:
    df = pd.read_parquet(path)
    stem = Path(path).stem
    symbol, interval_hint_ms = _parse_interval_ms(stem)
    if df.empty:
        return symbol, df
    df = df.copy()
    if "ts_ms" not in df.columns:
        raise ValueError(f"Missing ts_ms column in {path}")
    df["ts_ms"] = pd.to_numeric(df["ts_ms"], errors="coerce")
    df = df.dropna(subset=["ts_ms"])
    df["ts_ms"] = df["ts_ms"].astype("int64")
    interval_ms = _infer_interval_ms(df["ts_ms"], interval_hint_ms)
    raw = pd.DataFrame()
    raw["open_time"] = df["ts_ms"]
    raw["close_time"] = raw["open_time"] + int(interval_ms)
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
        raw[col] = pd.to_numeric(df[col], errors="coerce")
    if "quote_asset_volume" in df.columns:
        raw["quote_asset_volume"] = pd.to_numeric(df["quote_asset_volume"], errors="coerce")
    else:
        raw["quote_asset_volume"] = np.nan
    raw["number_of_trades"] = pd.to_numeric(
        df.get("number_of_trades", 0), errors="coerce"
    )
    raw["taker_buy_base_asset_volume"] = pd.to_numeric(
        df.get("taker_buy_base_asset_volume", df.get("taker_buy_base", 0)),
        errors="coerce",
    )
    raw["taker_buy_quote_asset_volume"] = pd.to_numeric(
        df.get("taker_buy_quote_asset_volume", df.get("taker_buy_quote", 0)),
        errors="coerce",
    )
    if "symbol" in df.columns:
        raw["symbol"] = df["symbol"].astype(str).fillna(symbol)
    else:
        raw["symbol"] = symbol
    raw = raw[RAW_COLUMNS]
    if raw["quote_asset_volume"].isna().all():
        raw["quote_asset_volume"] = raw["close"].astype(float) * raw["volume"].astype(float)
    raw["number_of_trades"] = raw["number_of_trades"].fillna(0)
    raw["taker_buy_base_asset_volume"] = raw["taker_buy_base_asset_volume"].fillna(0.0)
    raw["taker_buy_quote_asset_volume"] = raw["taker_buy_quote_asset_volume"].fillna(0.0)
    if write_csv:
        os.makedirs(RAW_DIR, exist_ok=True)
        csv_path = os.path.join(RAW_DIR, f"{symbol}.csv")
        raw.to_csv(csv_path, index=False)
    return symbol, _normalize_raw_df(raw, symbol)


def _gather_raw_frames() -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}

    def _register(symbol: str, frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            return
        prev = frames.get(symbol)
        if prev is None or len(frame) > len(prev):
            frames[symbol] = frame

    for path in sorted(glob.glob(os.path.join(KLINES_DIR, "*.parquet"))):
        try:
            symbol, frame = _read_klines_parquet(path)
        except Exception as exc:
            print(f"! Failed to convert {path}: {exc}")
            continue
        _register(symbol, frame)

    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
        try:
            symbol, frame = _read_raw(path)
        except Exception as exc:
            print(f"! Failed to read {path}: {exc}")
            continue
        _register(symbol, frame)

    return frames


def _read_fng() -> pd.DataFrame:
    if not os.path.exists(FNG):
        return pd.DataFrame(columns=["timestamp","fear_greed_value","fear_greed_value_norm"])
    f = pd.read_csv(FNG)
    if f["timestamp"].max() > 10_000_000_000:
        f["timestamp"] = (f["timestamp"] // 1000).astype("int64")
    else:
        f["timestamp"] = f["timestamp"].astype("int64")
    f["timestamp"] = (f["timestamp"] // 3600) * 3600
    if "fear_greed_value" not in f.columns and "value" in f.columns:
        f = f.rename(columns={"value":"fear_greed_value"})
    f["fear_greed_value_norm"] = f["fear_greed_value"].astype(float) / 100.0
    f = f.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")[["timestamp","fear_greed_value","fear_greed_value_norm"]]
    return f


def _read_events() -> pd.DataFrame:
    if not os.path.exists(EVENTS):
        return pd.DataFrame(columns=["timestamp","importance_level"])
    e = pd.read_csv(EVENTS)
    if e["timestamp"].max() > 10_000_000_000:
        e["timestamp"] = (e["timestamp"] // 1000).astype("int64")
    else:
        e["timestamp"] = e["timestamp"].astype("int64")
    e["timestamp"] = (e["timestamp"] // 3600) * 3600
    e = e.sort_values("timestamp")[["timestamp","importance_level"]]
    return e


def prepare() -> list[str]:
    """Process raw candles and return list of written paths."""
    fng = _read_fng()
    events = _read_events()
    written: list[str] = []

    frames = _gather_raw_frames()
    for sym in sorted(frames.keys()):
        df = frames[sym].copy()
        if not fng.empty:
            fng_sorted = fng.sort_values("timestamp")[["timestamp","fear_greed_value"]].copy()
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                fng_sorted,
                on="timestamp",
                direction="backward"
            )
            df["fear_greed_value"] = df["fear_greed_value"].ffill()

        if not events.empty:
            df_sorted = df.sort_values("timestamp").copy()
            df_sorted["timestamp_dt"] = pd.to_datetime(df_sorted["timestamp"], unit="s").astype("datetime64[ns]")
            df_sorted = df_sorted.sort_values("timestamp_dt")

            ev = events.rename(columns={"timestamp": "event_ts"}).sort_values("event_ts").copy()
            ev["event_ts_dt"] = pd.to_datetime(ev["event_ts"], unit="s").astype("datetime64[ns]")
            ev = ev.sort_values("event_ts_dt")

            df = pd.merge_asof(
                df_sorted,
                ev,
                left_on="timestamp_dt",
                right_on="event_ts_dt",
                direction="backward",
                tolerance=pd.Timedelta(hours=EVENT_HORIZON_HOURS),
            )
            df["time_since_last_event_hours"] = (
                (df["timestamp_dt"] - df["event_ts_dt"]).dt.total_seconds() / 3600.0
            )
            df["is_high_importance"] = ((df["importance_level"] == 2) & df["event_ts"].notna()).astype(int)
            df = df.drop(columns=["timestamp_dt", "event_ts_dt", "event_ts", "importance_level"])

        out = os.path.join(OUT_DIR, f"{sym}.feather")
        prefix = ["timestamp","symbol","open","high","low","close","volume","quote_asset_volume",
                  "number_of_trades","taker_buy_base_asset_volume","taker_buy_quote_asset_volume"]
        other = [c for c in df.columns if c not in prefix]
        df = df[prefix + other]
        tmp = out + ".tmp"
        df.reset_index(drop=True).to_feather(tmp)
        os.replace(tmp, out)
        written.append(out)
        print(f"✓ Wrote {out} ({len(df)} rows)")

    if len(written) != len(set(written)):
        raise ValueError("Duplicate output paths detected")
    return sorted(written)


def main():
    paths = prepare()
    print(f"Prepared {len(paths)} files in {OUT_DIR}")


if __name__ == "__main__":
    main()
