# scripts/train_calibrator.py
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from calibration import (
    BaseCalibrator,
    calibration_table,
    evaluate_before_after,
    fit_calibrator,
)


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Неизвестный формат файла данных: {ext}")


def _write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Обучить калибратор вероятностей (Platt/Isotonic) на валидационных предсказаниях.")
    ap.add_argument("--data", required=True, help="Файл с предсказаниями и метками (CSV/Parquet). Нужны колонки: score, y (0/1).")
    ap.add_argument("--score_col", default="score", help="Колонка со скором/вероятностью модели.")
    ap.add_argument("--y_col", default="y", help="Колонка бинарной метки (0/1).")
    ap.add_argument("--filter_val", action="store_true", help="Оставить только строки wf_role=='val'.")
    ap.add_argument("--wf_role_col", default="wf_role", help="Имя колонки роли (если фильтруем).")
    ap.add_argument("--method", choices=["platt", "isotonic"], default="platt", help="Метод калибровки.")
    ap.add_argument("--out_model", default="models/calibrator.json", help="Куда сохранить JSON калибратора.")
    ap.add_argument("--report_csv", default="", help="Куда сохранить calibration-table (.csv). Опционально.")
    args = ap.parse_args()

    df = _read_table(args.data)
    if args.filter_val and args.wf_role_col in df.columns:
        df = df.loc[df[args.wf_role_col].astype(str) == "val"].reset_index(drop=True)

    if args.score_col not in df.columns or args.y_col not in df.columns:
        raise ValueError(f"Нет нужных колонок: {args.score_col}, {args.y_col}")

    s = pd.to_numeric(df[args.score_col], errors="coerce").astype(float).to_numpy()
    y = pd.to_numeric(df[args.y_col], errors="coerce").astype(float).to_numpy()

    cal = fit_calibrator(s, y, method=args.method)

    # метрики до/после
    metrics = evaluate_before_after(s, y, cal, bins=10)

    # таблица калибровки (после)
    p_after = np.clip(cal.predict_proba(s), 0.0, 1.0)
    calib_tbl = calibration_table(p_after, y, bins=10)
    if args.report_csv.strip():
        _write_csv(calib_tbl, args.report_csv.strip())

    # сохранить модель
    cal.save_json(args.out_model)

    print("Готово. Калибратор сохранён:", args.out_model)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
