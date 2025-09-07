# scripts/apply_no_trade_mask.py
from __future__ import annotations

import argparse
import os

import pandas as pd

from training.no_trade import compute_no_trade_mask


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
    ap = argparse.ArgumentParser(description="Применить no_trade-маску к датасету: удалить запрещённые строки или пометить weight=0.")
    ap.add_argument("--data", required=True, help="Входной датасет (CSV/Parquet) с колонкой ts_ms (UTC, миллисекунды).")
    ap.add_argument("--out", default="", help="Выходной файл. По умолчанию рядом, с суффиксом _masked.")
    ap.add_argument("--sandbox_config", default="configs/legacy_sandbox.yaml", help="Путь к legacy_sandbox.yaml (раздел no_trade).")
    ap.add_argument("--ts_col", default="ts_ms", help="Колонка метки времени в мс UTC.")
    ap.add_argument("--mode", choices=["drop", "weight"], default="drop", help="drop — удалить строки; weight — оставить и добавить train_weight=0.")
    args = ap.parse_args()

    df = _read_table(args.data)

    mask_block = compute_no_trade_mask(df, sandbox_yaml_path=args.sandbox_config, ts_col=args.ts_col)
    if args.mode == "drop":
        out_df = df.loc[~mask_block].reset_index(drop=True)
    else:
        out_df = df.copy()
        out_df["train_weight"] = 1.0
        out_df.loc[mask_block, "train_weight"] = 0.0

    base, ext = os.path.splitext(args.data)
    out_path = args.out.strip() or f"{base}_masked{ext if ext.lower() in ('.csv', '.parquet', '.pq', '.txt') else '.parquet'}"
    _write_table(out_df, out_path)

    total = int(len(df))
    blocked = int(mask_block.sum())
    kept = int(len(out_df))
    print(f"Готово. Всего строк: {total}. Запрещённых (no_trade): {blocked}. Вышло: {kept}.")
    if args.mode == "weight":
        z = int((out_df.get('train_weight', pd.Series(dtype=float)) == 0.0).sum())
        print(f"Режим weight: назначено train_weight=0 для {z} строк.")


if __name__ == "__main__":
    main()
