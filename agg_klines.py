# scripts/agg_klines.py
from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd


def _read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _floor_ts(ts_ms: int, step_ms: int) -> int:
    return (int(ts_ms) // step_ms) * step_ms


def _agg(df: pd.DataFrame, interval: str, drop_partial: bool = False) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    step_map = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    if interval not in step_map:
        raise ValueError(f"unsupported interval: {interval}")
    step = step_map[interval]
    d = df.copy()

    # проверка минимального/максимального ts_ms для каждого символа
    cleaned: List[pd.DataFrame] = []
    problems: List[str] = []
    for sym, g in d.groupby("symbol", group_keys=False):
        ts = g["ts_ms"].astype("int64").sort_values()
        min_ts = int(ts.iloc[0])
        max_ts = int(ts.iloc[-1])
        diffs = ts.diff().dropna()
        diffs = diffs[diffs > 0]
        base_step = int(diffs.min()) if not diffs.empty else step
        start_ok = (min_ts % step) == 0
        end_ok = ((max_ts + base_step) % step) == 0
        if start_ok and end_ok:
            cleaned.append(g)
            continue
        if not drop_partial:
            parts = []
            if not start_ok:
                parts.append(f"min_ts={min_ts}")
            if not end_ok:
                parts.append(f"max_ts={max_ts}")
            problems.append(f"{sym}: {'; '.join(parts)}")
            cleaned.append(g)
            continue
        # drop_partial=True: отбрасываем неполные окна
        start_bound = _floor_ts(min_ts, step) + (0 if start_ok else step)
        end_bound = _floor_ts(max_ts, step) + (step if end_ok else 0)
        g2 = g[(g["ts_ms"] >= start_bound) & (g["ts_ms"] < end_bound)]
        cleaned.append(g2)
    if problems:
        raise ValueError("partial windows detected: " + "; ".join(problems))
    d = pd.concat(cleaned, ignore_index=True)

    d["bucket"] = d["ts_ms"].astype("int64").map(lambda x: _floor_ts(int(x), step))
    g = d.groupby(["symbol", "bucket"])
    out = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        number_of_trades=("number_of_trades", "sum"),
        taker_buy_base=("taker_buy_base", "sum"),
        taker_buy_quote=("taker_buy_quote", "sum"),
    ).reset_index().rename(columns={"bucket": "ts_ms"})
    out = out.sort_values(["symbol", "ts_ms"]).reset_index(drop=True)
    # приведение типов
    for c in ["open", "high", "low", "close", "volume", "taker_buy_base", "taker_buy_quote"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["number_of_trades"] = pd.to_numeric(out["number_of_trades"], errors="coerce").astype("Int64")

    # валидации: ts_ms кратно step и нет пропусков/дубликатов
    if ((out["ts_ms"] % step) != 0).any():
        bad = out.loc[(out["ts_ms"] % step) != 0, ["symbol", "ts_ms"]]
        raise ValueError(f"ts_ms not aligned to {step}: {bad.to_dict(orient='records')[:5]}")

    problems: List[str] = []
    for sym, g in out.groupby("symbol"):
        ts = g["ts_ms"].tolist()
        for prev, curr in zip(ts, ts[1:]):
            delta = int(curr) - int(prev)
            if delta != step:
                kind = "dup" if delta == 0 else "gap"
                problems.append(f"{sym}: {prev}->{curr} (Δ={delta}, {kind})")
    if problems:
        raise ValueError("ts_ms gaps/duplicates detected: " + "; ".join(problems))
    return out


def main():
    p = argparse.ArgumentParser(description="Aggregate klines to a higher timeframe.")
    p.add_argument("--in-path", required=True, help="Входной parquet 1m (или другой низкой частоты)")
    p.add_argument("--interval", required=True, help="Целевой интервал: 5m/15m/1h/4h/1d ...")
    p.add_argument("--out-path", required=True, help="Куда сохранить parquet агрегации")
    p.add_argument(
        "--drop-partial",
        action="store_true",
        help="Отбрасывать неполные начальные/конечные окна вместо ошибки",
    )
    args = p.parse_args()

    df = _read_parquet(args.in_path)
    out = _agg(df, args.interval, drop_partial=args.drop_partial)
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    out.to_parquet(args.out_path, index=False)
    print(f"Wrote {len(out)} rows to {args.out_path}")


if __name__ == "__main__":
    main()
