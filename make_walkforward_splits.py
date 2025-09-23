# scripts/make_walkforward_splits.py
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import yaml

from splits import make_walkforward_splits


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


def _write_manifest(manifest, json_path: str, yaml_path: str) -> None:
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    data = [m.to_dict() for m in manifest]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _write_phase_tables(df_out: pd.DataFrame, base: str, ext: str) -> tuple[list[str], list[str]]:
    train_df = df_out[df_out["wf_role"] == "train"].copy()
    val_df = df_out[df_out["wf_role"] == "val"].copy()
    if set(train_df.index) & set(val_df.index):
        raise ValueError("Train/validation rows overlap")
    train_path = f"{base}_train{ext}"
    val_path = f"{base}_val{ext}"
    _write_table(train_df, train_path)
    _write_table(val_df, val_path)
    return [train_path], [val_path]


def main():
    ap = argparse.ArgumentParser(description="Сгенерировать walk-forward сплиты с PURGE (горизонт h) и EMBARGO (буфер).")
    ap.add_argument("--data", required=True, help="Входной датасет (CSV/Parquet) с колонкой ts_ms (UTC миллисекунды).")
    ap.add_argument("--out", default="", help="Путь к выходному датасету с колонками wf_fold,wf_role. По умолчанию рядом с суффиксом _wf.")
    ap.add_argument("--ts_col", default="ts_ms", help="Имя колонки времени.")
    ap.add_argument("--symbol_col", default="symbol", help="Имя колонки символа (может отсутствовать).")
    ap.add_argument("--interval_ms", type=int, default=None, help="Интервал бара в мс (если не задан — оценим автоматически).")
    ap.add_argument("--train_span_bars", type=int, default=7*24*60, help="Длина train-окна в барах.")
    ap.add_argument("--val_span_bars", type=int, default=24*60, help="Длина val-окна в барах.")
    ap.add_argument("--step_bars", type=int, default=24*60, help="Шаг окна в барах.")
    ap.add_argument("--horizon_bars", type=int, default=60, help="Горизонт таргета в барах (PURGE).")
    ap.add_argument("--embargo_bars", type=int, default=5, help="Буфер EMBARGO в барах.")
    ap.add_argument("--manifest_dir", default="logs/walkforward", help="Куда записать манифесты (JSON/YAML).")
    args = ap.parse_args()

    df = _read_table(args.data)

    df_out, manifest = make_walkforward_splits(
        df,
        ts_col=args.ts_col,
        symbol_col=(args.symbol_col if args.symbol_col in df.columns else None),
        interval_ms=args.interval_ms,
        train_span_bars=int(args.train_span_bars),
        val_span_bars=int(args.val_span_bars),
        step_bars=int(args.step_bars),
        horizon_bars=int(args.horizon_bars),
        embargo_bars=int(args.embargo_bars),
    )

    base, ext = os.path.splitext(args.data)
    out_path = args.out.strip() or f"{base}_wf{ext if ext.lower() in ('.csv', '.parquet', '.pq', '.txt') else '.parquet'}"
    _write_table(df_out, out_path)
    train_paths, val_paths = _write_phase_tables(df_out, base, ext)

    json_path = os.path.join(args.manifest_dir, "walkforward_manifest.json")
    yaml_path = os.path.join(args.manifest_dir, "walkforward_manifest.yaml")
    _write_manifest(manifest, json_path=json_path, yaml_path=yaml_path)

    total = int(len(df_out))
    used = int((df_out["wf_role"] != "none").sum())
    n_train = int((df_out["wf_role"] == "train").sum())
    n_val = int((df_out["wf_role"] == "val").sum())
    print(f"Готово. Записан датасет со сплитами: {out_path}")
    print(f"Всего строк: {total}. В сплитах train: {n_train}, val: {n_val}, вне окон: {total - used}.")
    print(f"Train path: {train_paths[0]}, Val path: {val_paths[0]}")
    print(f"Манифесты: {json_path} и {yaml_path}")
    return train_paths, val_paths


if __name__ == "__main__":
    main()
