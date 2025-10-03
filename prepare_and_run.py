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

    for path in glob.glob(os.path.join(RAW_DIR, "*.csv")):
        df = _read_raw(path)
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

        sym = os.path.splitext(os.path.basename(path))[0]
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
