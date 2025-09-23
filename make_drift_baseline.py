# scripts/make_drift_baseline.py
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from drift import make_baseline, save_baseline_json, default_feature_list


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Неизвестный формат: {ext}")


def main():
    ap = argparse.ArgumentParser(description="Сформировать baseline для дрифт-контроля (PSI) по выбранным фичам.")
    ap.add_argument("--data", required=True, help="Файл CSV/Parquet (обычно валид. срез последнего фолда).")
    ap.add_argument("--features", default="", help="Через запятую список колонок. Если пусто — автодетект (f_* и score).")
    ap.add_argument("--bins", type=int, default=10, help="Число бинов для числовых фичей (квантили).")
    ap.add_argument("--top_k_cats", type=int, default=20, help="Top-K категорий (остальные → OTHER).")
    ap.add_argument("--out", default="models/drift_baseline.json", help="Куда сохранить baseline JSON.")
    args = ap.parse_args()

    df = _read_table(args.data)

    if args.features.strip():
        feats = [s.strip() for s in args.features.split(",") if s.strip()]
    else:
        feats = default_feature_list(df)
        if not feats:
            raise ValueError("Не удалось автодетектить фичи. Укажи их через --features.")

    spec = make_baseline(df, feats, bins=int(args.bins), categorical=None, top_k_cats=int(args.top_k_cats))
    save_baseline_json(spec, args.out)

    print(f"Готово. Baseline сохранён: {args.out}")
    print(f"Фичи: {', '.join(feats)}")


if __name__ == "__main__":
    main()
