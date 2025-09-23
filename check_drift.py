# scripts/check_drift.py
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from drift import compute_psi, load_baseline_json, default_feature_list


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Неизвестный формат: {ext}")


def _write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Проверить дрифт (PSI) относительно baseline JSON.")
    ap.add_argument("--data", required=True, help="Текущий датасет (CSV/Parquet), например, последние дни онлайна.")
    ap.add_argument("--baseline", required=True, help="Baseline JSON, созданный make_drift_baseline.py.")
    ap.add_argument("--features", default="", help="Явный список фичей через запятую. Если пусто — возьмём из baseline.")
    ap.add_argument("--ts_col", default="ts_ms", help="Колонка времени (UTC мс).")
    ap.add_argument("--last_days", type=int, default=14, help="Сколько последних дней взять из data (если 0 — берём всё).")
    ap.add_argument("--out_csv", default="", help="Куда сохранить таблицу PSI (CSV). Если пусто — рядом с суффиксом _psi.csv.")
    args = ap.parse_args()

    df = _read_table(args.data)

    if int(args.last_days) > 0 and args.ts_col in df.columns:
        max_ts = int(pd.to_numeric(df[args.ts_col], errors="coerce").max())
        cutoff = max_ts - int(args.last_days) * 86400000
        df = df.loc[pd.to_numeric(df[args.ts_col], errors="coerce") >= cutoff].reset_index(drop=True)

    baseline = load_baseline_json(args.baseline)

    if args.features.strip():
        feats = [s.strip() for s in args.features.split(",") if s.strip()]
    else:
        feats = list(baseline.keys())

    res = compute_psi(df, baseline, features=feats)

    if not args.out_csv.strip():
        base, ext = os.path.splitext(args.data)
        args.out_csv = f"{base}_psi.csv"
    _write_csv(res, args.out_csv)

    print(f"Готово. PSI записан: {args.out_csv}")
    if not res.empty:
        avg_psi = float(res["psi"].replace([np.inf, -np.inf], np.nan).dropna().mean())
        worst = res.iloc[0]
        print(f"Средний PSI: {avg_psi:.4f}. Худшая фича: {worst['feature']} = {worst['psi']:.4f}.")
        print("Границы интерпретации: <0.1 — ок; 0.1–0.25 — умеренный; >0.25 — сильный дрифт.")
