# -*- coding: utf-8 -*-
"""
fetch_all_data_patch.py
------------------------------------------------------------------
Loader for preprocessed OHLCV datasets and optional Fear & Greed
index. Produces two mappings expected by train_model_multi_patch.py:
  - all_dfs_dict: {key -> pd.DataFrame}
  - all_obs_dict: {key -> np.ndarray or dict} (optional, may be {})

Conventions
===========
• Preprocessed files live in data/processed/*.feather
• Fear & Greed lives at data/fear_greed.csv
• Required OHLCV columns (Binance 12 without loss):
    timestamp, open, high, low, close, volume,
    quote_asset_volume, number_of_trades,
    taker_buy_base_asset_volume, taker_buy_quote_asset_volume
  (close_time and ignore can be kept as metadata if present)
• Symbol column must be present as 'symbol'.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List

FNG_PATH = os.path.join("data", "fear_greed.csv")

def _read_fng() -> pd.DataFrame:
    if not os.path.exists(FNG_PATH):
        # No F&G available — return empty frame
        return pd.DataFrame(columns=["timestamp", "fear_greed_value"])
    fng = pd.read_csv(FNG_PATH)
    # expected columns: timestamp (unix seconds or ms), value (0..100) or 'fear_greed_value'
    cols = {c.lower(): c for c in fng.columns}
    # normalize timestamp to seconds (int) UTC
    ts_col = "timestamp" if "timestamp" in cols else next((c for c in fng.columns if "time" in c.lower()), "timestamp")
    val_col = "fear_greed_value" if "fear_greed_value" in fng.columns else ("value" if "value" in fng.columns else None)
    fng = fng.rename(columns={ts_col: "timestamp"})
    if val_col and val_col != "fear_greed_value":
        fng = fng.rename(columns={val_col: "fear_greed_value"})
    # convert timestamp to seconds integer
    if fng["timestamp"].max() > 10_000_000_000:  # ms heuristic
        fng["timestamp"] = (fng["timestamp"] // 1000).astype("int64")
    else:
        fng["timestamp"] = fng["timestamp"].astype("int64")
    # keep only two columns
    fng = fng[["timestamp", "fear_greed_value"]].copy()
    # add normalized column for convenience
    fng["fear_greed_value_norm"] = fng["fear_greed_value"].astype(float) / 100.0
    # round to hour boundary (floor) to align with 1h candles
    fng["timestamp"] = (fng["timestamp"] // 3600) * 3600
    fng = fng.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return fng

def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Minimal set; tolerate extra columns
    required = [
        "timestamp", "open", "high", "low", "close", "volume",
        "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "symbol"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in preprocessed file: {missing}")
    # enforce types
    df = df.copy()
    float_cols = ["open","high","low","close","volume","quote_asset_volume",
                  "taker_buy_base_asset_volume","taker_buy_quote_asset_volume"]
    int_cols = ["number_of_trades"]
    for c in float_cols:
        df[c] = df[c].astype(float)
    # Normalize timestamp column (detect milliseconds and convert to seconds)
    ts = pd.to_numeric(df["timestamp"], errors="raise")
    if ts.max() > 10_000_000_000:
        ts = ts // 1000
    df["timestamp"] = ts.astype("int64")
    for c in int_cols:
        df[c] = df[c].astype("int64")
    df["symbol"] = df["symbol"].astype(str)
    # hour alignment & dedup
    df["timestamp"] = (df["timestamp"] // 3600) * 3600
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df

def load_all_data(feather_paths: List[str], synthetic_fraction: float = 0.0, seed: int = 42) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
    """Load all .feather files and perform safe merge with Fear & Greed if available.
    Keys are derived from file stem (e.g., BTCUSDT).
    synthetic_fraction/seed are accepted for compatibility and ignored here.
    """
    all_dfs: Dict[str, pd.DataFrame] = {}
    all_obs: Dict[str, np.ndarray] = {}
    # read F&G once
    fng = _read_fng()
    for p in feather_paths:
        sym = os.path.splitext(os.path.basename(p))[0]
        df = pd.read_feather(p)
        # Standardize/ensure columns
        if "symbol" not in df.columns:
            df["symbol"] = sym
        # Timestamp column: support both 'timestamp' or 'close_time'
        if "timestamp" not in df.columns:
            if "close_time" in df.columns:
                ts = df["close_time"].astype("int64")
                if ts.max() > 10_000_000_000:
                    ts = ts // 1000
                df["timestamp"] = ts
            elif "open_time" in df.columns:
                ts = df["open_time"].astype("int64")
                if ts.max() > 10_000_000_000:
                    ts = ts // 1000
                df["timestamp"] = ts + 3600  # shift to close
            else:
                raise ValueError(f"{p}: neither 'timestamp' nor close/open_time present")
        # Ensure required columns exist; if quote_asset_volume missing, try derive
        if "quote_asset_volume" not in df.columns and {"close","volume"}.issubset(df.columns):
            df["quote_asset_volume"] = df["close"].astype(float) * df["volume"].astype(float)
        if "number_of_trades" not in df.columns:
            df["number_of_trades"] = 0
        if "taker_buy_base_asset_volume" not in df.columns:
            df["taker_buy_base_asset_volume"] = 0.0
        if "taker_buy_quote_asset_volume" not in df.columns:
            df["taker_buy_quote_asset_volume"] = 0.0
        df = _ensure_required_columns(df)
        # Не ломаем OHLC: оставляем close как есть, а «прошлый close» кладём отдельно
        if "close" in df.columns:
            df["close_orig"] = df["close"].astype(float)
            df["close_prev"] = df["close_orig"].shift(1)  # используйте это в фичах вместо сдвига самого close
        # Merge Fear & Greed on the same hour (left join to preserve OHLCV)
        orig_fear_col = None
        if "fear_greed_value" in df.columns:
            orig_fear_col = "fear_greed_value_orig"
            df = df.rename(columns={"fear_greed_value": orig_fear_col})
        if not fng.empty:
            fng_sorted = fng.sort_values("timestamp")[["timestamp","fear_greed_value"]].copy()
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                fng_sorted,
                on="timestamp",
                direction="backward"
            )
            if "fear_greed_value" in df.columns:
                df["fear_greed_value"] = df["fear_greed_value"].ffill()
        if "fear_greed_value" not in df.columns and orig_fear_col:
            df["fear_greed_value"] = df[orig_fear_col]
        elif "fear_greed_value" in df.columns and orig_fear_col:
            df["fear_greed_value"] = df["fear_greed_value"].fillna(df[orig_fear_col])
        if orig_fear_col and orig_fear_col in df.columns:
            df = df.drop(columns=[orig_fear_col])
        # Maintain stable column order
        base_cols = [
            "timestamp","symbol","open","high","low","close","volume","quote_asset_volume",
            "number_of_trades","taker_buy_base_asset_volume","taker_buy_quote_asset_volume"
        ]
        other_cols = [c for c in df.columns if c not in base_cols]
        # enforce stable order for trainer
        df = df[base_cols + other_cols]
        all_dfs[sym] = df
        # No additional obs tensors provided here
    return all_dfs, all_obs
