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
import pandas as pd

RAW_DIR = os.path.join("data","candles")
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
            # default fallbacks
            df[c] = 0 if c in ["number_of_trades"] else 0.0
    df = df[keep].drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df

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
    # normalize ts to seconds and floor to hour
    if e["timestamp"].max() > 10_000_000_000:
        e["timestamp"] = (e["timestamp"] // 1000).astype("int64")
    else:
        e["timestamp"] = e["timestamp"].astype("int64")
    e["timestamp"] = (e["timestamp"] // 3600) * 3600
    e = e.sort_values("timestamp")[["timestamp","importance_level"]]
    return e

def main():
    fng = _read_fng()
    events = _read_events()

    for path in glob.glob(os.path.join(RAW_DIR, "*.csv")):
        df = _read_raw(path)
        # merge F&G
        if not fng.empty:
            # merge F&G as-of (backward) with daily values; then carry forward within the day
            fng_sorted = fng.sort_values("timestamp")[["timestamp","fear_greed_value"]].copy()
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                fng_sorted,
                on="timestamp",
                direction="backward"
            )
            # ensure daily F&G present on every hourly bar by forward-fill
            df["fear_greed_value"] = df["fear_greed_value"].ffill()

        # merge recent events (asof backward) -> flags within 96h
        if not events.empty:
            ev = events.rename(columns={"timestamp": "event_ts"})
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                ev.sort_values("event_ts"),
                left_on="timestamp",
                right_on="event_ts",
                direction="backward",
                tolerance=pd.Timedelta(hours=EVENT_HORIZON_HOURS),
            )
            # time since last event in hours (NaN if none within horizon)
            df["time_since_last_event_hours"] = (df["timestamp"] - df["event_ts"]) / 3600.0

            # high-importance flag within 96h window (0 if no matched event within tolerance)
            df["is_high_importance"] = ((df["importance_level"] == 2) & df["event_ts"].notna()).astype(int)

            # drop helper columns from merge
            df = df.drop(columns=["event_ts", "importance_level"])

        sym = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(OUT_DIR, f"{sym}.feather")
        # enforce stable column order: canonical prefix first, then the rest (as-is)
        prefix = ["timestamp","symbol","open","high","low","close","volume","quote_asset_volume",
          "number_of_trades","taker_buy_base_asset_volume","taker_buy_quote_asset_volume"]
        other = [c for c in df.columns if c not in prefix]
        df = df[prefix + other]
        tmp = out + ".tmp"
        df.reset_index(drop=True).to_feather(tmp)
        os.replace(tmp, out)
        print(f"✓ Wrote {out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
