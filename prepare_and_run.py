# -*- coding: utf-8 -*-
"""
prepare_and_run.py
---------------------------------------------------------------
Merge raw 1h candles from data/candles/ with Fear & Greed (data/fear_greed.csv)
and write per-symbol Feather files to data/processed/ expected by training.
Also enforces column schema and avoids renaming 'volume'.
"""
import os
import glob
import re
import argparse

import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data","candles")  # оставим как дефолт для обратной совместимости
FNG = os.path.join("data","fear_greed.csv")
EVENTS = os.path.join("data","economic_events.csv")
EVENT_HORIZON_HOURS = 96
OUT_DIR = os.path.join("data","processed")
os.makedirs(OUT_DIR, exist_ok=True)


def _read_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Convert open/close time to seconds
    for c in ["open_time","close_time"]:
        if df[c].max() > 10_000_000_000:
            df[c] = (df[c] // 1000).astype("int64")
        else:
            df[c] = df[c].astype("int64")
    # Canonical timestamp = close_time floored to hour
    df["timestamp"] = (df["close_time"] // 3600) * 3600
    # Ensure symbol
    if "symbol" not in df.columns:
        sym = os.path.splitext(os.path.basename(path))[0]
        df["symbol"] = sym
    # Ensure quote_asset_volume
    if "quote_asset_volume" not in df.columns:
        df["quote_asset_volume"] = df["close"].astype(float) * df["volume"].astype(float)
    # Minimal schema
    keep = ["timestamp","symbol","open","high","low","close","volume","quote_asset_volume",
            "number_of_trades","taker_buy_base_asset_volume","taker_buy_quote_asset_volume"]
    for c in keep:
        if c not in df.columns:
            df[c] = 0 if c in ["number_of_trades"] else 0.0
    df = df[keep].drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df


def _to_seconds(ts: pd.Series) -> pd.Series:
    ts = pd.to_numeric(ts, errors="coerce")
    ts_max = ts.max(skipna=True)
    # поддержка ms → сек
    if pd.notna(ts_max) and ts_max > 10_000_000_000:
        ts = ts // 1000
    return ts


def _infer_symbol(path: str, df: pd.DataFrame) -> str:
    if "symbol" in df.columns and df["symbol"].notna().any():
        try:
            v = str(df["symbol"].dropna().iloc[0])
            if v:
                return v
        except Exception:
            pass
    base = os.path.basename(path)
    return re.split(r"[_.]", base)[0]  # BTCUSDT_1h.parquet → BTCUSDT


def _normalize_ohlcv(df: pd.DataFrame, path: str) -> pd.DataFrame:
    # timestamp
    if "timestamp" in df.columns:
        ts = _to_seconds(df["timestamp"])
    elif "close_time" in df.columns:
        ts = _to_seconds(df["close_time"])
    elif "open_time" in df.columns:
        ts = _to_seconds(df["open_time"]) + 3600  # сместим к закрытию часа
    else:
        raise ValueError(f"{path}: no 'timestamp'/'close_time'/'open_time'")

    ts = ((ts // 3600) * 3600).astype("Int64")

    # базовые поля
    def num(col, fallback=None, dtype=float):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        if fallback is not None:
            return pd.Series(fallback, index=df.index, dtype=dtype)
        return pd.Series(np.nan, index=df.index, dtype="float64")

    open_ = num("open")
    high_ = num("high")
    low_ = num("low")
    close_ = num("close")
    vol = num("volume")
    qvol = num("quote_asset_volume")
    if qvol.isna().all():
        qvol = close_.astype(float) * vol.astype(float)
    ntr = num("number_of_trades", 0, dtype=int).fillna(0).astype(int)
    tb_base = num("taker_buy_base_asset_volume", 0.0).fillna(0.0)
    tb_quote = num("taker_buy_quote_asset_volume")
    if tb_quote.isna().all():
        tb_quote = (tb_base.astype(float) * close_.astype(float)).fillna(0.0)

    sym = _infer_symbol(path, df)
    out = pd.DataFrame({
        "timestamp": ts,
        "symbol": sym,
        "open": open_.astype(float),
        "high": high_.astype(float),
        "low": low_.astype(float),
        "close": close_.astype(float),
        "volume": vol.astype(float),
        "quote_asset_volume": qvol.astype(float),
        "number_of_trades": ntr.astype(int),
        "taker_buy_base_asset_volume": tb_base.astype(float),
        "taker_buy_quote_asset_volume": tb_quote.astype(float),
    })
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    out["timestamp"] = out["timestamp"].astype("int64")
    return out


def _read_any_raw(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _read_raw(path)  # существующая функция
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported raw extension: {path}")


def _discover_raw_paths(raw_dirs: list[str]) -> list[str]:
    """Собираем все CSV/Parquet из указанных директорий."""
    patterns = ("*.csv", "*_1h.parquet", "*.parquet")
    paths = set()
    for d in raw_dirs:
        if not d:
            continue
        for pat in patterns:
            paths.update(glob.glob(os.path.join(d, pat)))
    return sorted(paths)


def _parse_args():
    ap = argparse.ArgumentParser(description="Prepare processed feathers from raw candles")
    ap.add_argument(
        "--raw-dir",
        help="Comma-separated list of directories with raw candles (csv/parquet). "
             "If omitted, uses ENV RAW_DIR or defaults to 'data/candles,data/klines'.",
        default=os.environ.get("RAW_DIR", "")
    )
    ap.add_argument(
        "--out-dir",
        help="Output directory for processed feather files (default: data/processed or ENV OUT_DIR).",
        default=os.environ.get("OUT_DIR", OUT_DIR),
    )
    return ap.parse_args()


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

    # 1) выбираем директории для поиска raw
    raw_dirs_env = os.environ.get("RAW_DIR", "")
    # резервные директории по умолчанию: и candles, и klines
    default_dirs = [RAW_DIR, os.path.join("data","klines")]
    raw_dirs = [p for p in raw_dirs_env.split(",") if p] or default_dirs

    # 2) собираем пути raw
    raw_paths = _discover_raw_paths(raw_dirs)
    if not raw_paths:
        raise FileNotFoundError(
            f"No raw files found. Checked: {', '.join(raw_dirs)}. "
            f"Provide --raw-dir or set RAW_DIR, or place files into one of defaults."
        )
    for path in raw_paths:
        df_raw = _read_any_raw(path)
        df = _normalize_ohlcv(df_raw, path)
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

        sym = _infer_symbol(path, df)
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
    args = _parse_args()
    # если пользователь указал иной out-dir — применим
    global OUT_DIR
    if args.out_dir and args.out_dir != OUT_DIR:
        OUT_DIR = args.out_dir
        os.makedirs(OUT_DIR, exist_ok=True)
    # прокинем RAW_DIR через окружение для совместимости с prepare()
    if args.raw_dir:
        os.environ["RAW_DIR"] = args.raw_dir
    paths = prepare()
    print(f"Prepared {len(paths)} files in {OUT_DIR}")


if __name__ == "__main__":
    main()
