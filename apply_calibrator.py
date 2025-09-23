# scripts/apply_calibrator.py
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from calibration import BaseCalibrator


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Неизвестный формат файла данных: {ext}")


def _write_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
        return
    if ext in (".csv", ".txt"):
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Неизвестный формат файла вывода: {ext}")


def main():
    ap = argparse.ArgumentParser(description="Применить калибратор (Platt/Isotonic) к колонке score в датасете.")
    ap.add_argument("--data", required=True, help="Файл с предсказаниями (CSV/Parquet). Должна быть колонка score.")
    ap.add_argument("--model", required=True, help="JSON калибратора (models/calibrator.json).")
    ap.add_argument("--score_col", default="score", help="Имя колонки со скором.")
    ap.add_argument("--out_col", default="score_calibrated", help="Имя новой колонки для калиброванной вероятности.")
    ap.add_argument("--out", default="", help="Куда сохранить (по умолчанию рядом с суффиксом _calibrated).")
    args = ap.parse_args()

    df = _read_table(args.data)
    if args.score_col not in df.columns:
        raise ValueError(f"Нет колонки: {args.score_col}")

    cal = BaseCalibrator.load_json(args.model)
    s = pd.to_numeric(df[args.score_col], errors="coerce").astype(float).to_numpy()
    p = cal.predict_proba(s)
    df[args.out_col] = p

    out_path = args.out.strip()
    if not out_path:
        base, ext = os.path.splitext(args.data)
        out_path = f"{base}_calibrated{ext if ext.lower() in ('.csv', '.parquet', '.pq', '.txt') else '.parquet'}"

    _write_table(df, out_path)
    print("Готово. Записано:", out_path)


if __name__ == "__main__":
    main()
