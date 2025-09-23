# scripts/apply_no_trade_mask.py
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

import numpy as np
import pandas as pd

import clock
from utils_time import is_bar_closed
from impl_offline_data import timeframe_to_ms

from no_trade import compute_no_trade_mask, estimate_block_ratio
from no_trade_config import get_no_trade_config


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Неизвестный формат файла данных: {ext}")


def _write_table(df: pd.DataFrame | Sequence[bool], path: str) -> None:
    """Save dataframe or boolean mask to a file."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame({"no_trade_block": np.asarray(df, dtype=bool)})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
        return
    if ext in (".csv", ".txt"):
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Неизвестный формат файла вывода: {ext}")


def _blocked_durations(
    ts_ms: Sequence[int], mask: Sequence[bool], tf_ms: int | None = None
) -> np.ndarray:
    """Return durations of consecutive blocked intervals in minutes."""
    ts = np.asarray(pd.to_numeric(ts_ms, errors="coerce"), dtype=np.int64)
    m = np.asarray(mask, dtype=bool)
    if ts.size == 0 or not m.any():
        return np.empty(0, dtype=float)

    diff = np.diff(m.astype(int))
    start_idx = np.where(diff == 1)[0] + 1
    end_idx = np.where(diff == -1)[0] + 1
    if m[0]:
        start_idx = np.r_[0, start_idx]
    if m[-1]:
        end_idx = np.r_[end_idx, len(m)]

    inferred_tf_ms: int | None = None
    if tf_ms is None and ts.size > 1:
        deltas = np.diff(ts)
        positive_deltas = deltas[deltas > 0]
        if positive_deltas.size:
            inferred_tf_ms = int(np.median(positive_deltas))

    durations_ms = np.empty_like(start_idx, dtype=np.int64)
    last_index = len(ts)
    for i, (start, end) in enumerate(zip(start_idx, end_idx)):
        start_ts = ts[start]
        if end < last_index:
            end_ts = ts[end]
        else:
            tf = tf_ms if tf_ms is not None else inferred_tf_ms
            if tf is None:
                end_ts = ts[end - 1]
            else:
                end_ts = ts[end - 1] + int(tf)
        durations_ms[i] = max(0, end_ts - start_ts)

    return durations_ms.astype(float) / 60_000.0


def main():
    ap = argparse.ArgumentParser(description="Применить no_trade-маску к датасету: удалить запрещённые строки или пометить weight=0.")
    ap.add_argument("--data", required=True, help="Входной датасет (CSV/Parquet) с колонкой ts_ms (UTC, миллисекунды).")
    ap.add_argument("--out", default="", help="Выходной файл. По умолчанию рядом, с суффиксом _masked.")
    ap.add_argument("--sandbox_config", default="configs/legacy_sandbox.yaml", help="Путь к legacy_sandbox.yaml (раздел no_trade).")
    ap.add_argument(
        "--no-trade-config",
        default="",
        help="Путь к YAML с секцией no_trade (по умолчанию используется --sandbox_config).",
    )
    ap.add_argument(
        "--with-reasons",
        "--with-reason",
        dest="with_reasons",
        action="store_true",
        help="Добавить колонки причин блокировки в вывод.",
    )
    ap.add_argument(
        "--reason-labels",
        action="store_true",
        help="Добавить колонку no_trade_reason с перечислением причин (подразумевает --with-reasons).",
    )
    ap.add_argument("--ts_col", default="ts_ms", help="Колонка метки времени в мс UTC.")
    ap.add_argument("--mode", choices=["drop", "weight"], default="drop", help="drop — удалить строки; weight — оставить и добавить train_weight=0.")
    ap.add_argument("--mask-only", action="store_true", help="Сохранить только колонку no_trade_block для всех строк.")
    ap.add_argument("--timeframe", required=True, help="Баровый таймфрейм, например 1m или 1h.")
    ap.add_argument(
        "--close-lag-ms",
        type=int,
        default=0,
        help="Допустимое запаздывание закрытия бара в миллисекундах.",
    )
    ap.add_argument(
        "--histogram",
        nargs="?",
        const="",
        metavar="PATH",
        help="Вывести гистограмму длительностей блоков (в минутах) в stdout или сохранить в файл PATH.",
    )
    args = ap.parse_args()

    df = _read_table(args.data)

    config_path = (args.no_trade_config or "").strip() or args.sandbox_config
    cfg = get_no_trade_config(config_path)
    mask_nt = compute_no_trade_mask(
        df,
        sandbox_yaml_path=config_path,
        ts_col=args.ts_col,
        config=cfg,
    )
    est_ratio = estimate_block_ratio(df, cfg, ts_col=args.ts_col)
    actual_nt_ratio = float(mask_nt.mean())
    if abs(actual_nt_ratio - est_ratio) > 0.01:
        print(
            f"Blocked ratio {actual_nt_ratio:.4f} differs from expected {est_ratio:.4f}",
            file=sys.stderr,
        )

    tf_ms = timeframe_to_ms(args.timeframe)
    ts = pd.to_numeric(df[args.ts_col], errors="coerce").to_numpy(dtype=np.int64)
    close_ts = (ts // tf_ms) * tf_ms + tf_ms
    now_ms = clock.now_ms()
    closed = np.array(
        [is_bar_closed(int(ct), now_ms, args.close_lag_ms) for ct in close_ts],
        dtype=bool,
    )
    closed_series = pd.Series(closed, index=df.index)
    mask_block = mask_nt | ~closed_series
    actual_ratio = float(mask_block.mean())

    include_reasons = bool(args.with_reasons or args.reason_labels)
    reasons_attr = mask_nt.attrs.get("reasons")
    if isinstance(reasons_attr, pd.DataFrame):
        reasons_df = reasons_attr.reindex(df.index).fillna(False).astype(bool)
    else:
        reasons_df = pd.DataFrame(index=df.index)
    reason_columns = reasons_df.reindex(df.index).fillna(False).astype(bool)
    reason_columns["bar_not_closed"] = ~closed_series
    reason_export = pd.concat(
        [
            pd.DataFrame(
                {"no_trade_block": mask_block.astype(bool)}, index=df.index
            ),
            reason_columns,
        ],
        axis=1,
    )

    raw_labels = mask_nt.attrs.get("reason_labels")
    if isinstance(raw_labels, dict):
        label_map = {str(k): str(v) for k, v in raw_labels.items()}
    else:
        label_map = {}
    for col in reason_columns.columns:
        label_map.setdefault(col, col)

    if args.reason_labels:
        include_reasons = True

        if reason_columns.empty:
            reason_export["no_trade_reason"] = pd.Series("", index=df.index)
        else:

            def _join_labels(row: pd.Series) -> str:
                labels = [label_map.get(col, col) for col, val in row.items() if bool(val)]
                return ";".join(labels)

            reason_export["no_trade_reason"] = reason_columns.apply(
                _join_labels, axis=1
            )

    total = int(len(df))
    blocked = int(mask_block.sum())
    reason_summary = []
    if total > 0 and not reason_columns.empty:
        for col in reason_columns.columns:
            count = int(reason_columns[col].sum())
            if count:
                reason_summary.append((label_map.get(col, col), count, count / total))

    summary_lines = []
    if not reason_columns.empty:
        if reason_summary:
            summary_lines.append("Причины блокировки:")
            for label, count, ratio in reason_summary:
                summary_lines.append(f"  {label}: {count} ({ratio:.2%})")
        else:
            summary_lines.append("Причины блокировки: нет срабатываний.")

    meta = mask_nt.attrs.get("meta") or {}
    dyn_meta = meta.get("dynamic_guard") if isinstance(meta, dict) else None
    dyn_message = ""
    if isinstance(dyn_meta, dict) and dyn_meta.get("skipped"):
        missing = dyn_meta.get("missing") or []
        if missing:
            missing_text = ", ".join(str(m) for m in missing)
            dyn_message = (
                f"Динамический guard пропущен: нет данных ({missing_text})."
            )
        else:
            dyn_message = "Динамический guard пропущен: нет входных данных."

    def _emit_summary_lines() -> None:
        if dyn_message:
            print(dyn_message)
        for line in summary_lines:
            print(line)

    def _maybe_emit_histogram(mask: Sequence[bool]) -> None:
        if args.histogram is None:
            return
        durations = _blocked_durations(df[args.ts_col], mask, tf_ms=tf_ms)
        if durations.size:
            hist, bin_edges = np.histogram(durations, bins="auto")
            lines = ["Гистограмма длительностей блоков (минуты):"]
            for count, start, end in zip(hist, bin_edges[:-1], bin_edges[1:]):
                lines.append(f"{start:.1f}-{end:.1f}: {int(count)}")
        else:
            lines = ["Гистограмма длительностей блоков (минуты):", "(пусто)"]
        out = "\n".join(lines)
        if args.histogram:
            with open(args.histogram, "w", encoding="utf-8") as f:
                f.write(out + "\n")
        else:
            print(out)

    if args.mask_only:
        base, ext = os.path.splitext(args.data)
        out_path = args.out.strip() or f"{base}_mask{ext if ext.lower() in ('.csv', '.parquet', '.pq', '.txt') else '.parquet'}"
        if include_reasons:
            mask_to_write = reason_export
        else:
            mask_to_write = pd.DataFrame(
                {"no_trade_block": mask_block.astype(bool)}, index=df.index
            )
        _write_table(mask_to_write, out_path)
        print(
            f"Готово. Всего строк: {total}. Запрещённых (no_trade): {blocked} ({actual_ratio:.2%}).",
        )
        print(f"Маска сохранена в {out_path}.")
        print(f"NoTradeConfig: {cfg.dict()}")
        _emit_summary_lines()
        _maybe_emit_histogram(mask_block)
        return

    df_out = df.copy()
    if include_reasons:
        for col in reason_export.columns:
            df_out[col] = reason_export[col]

    if args.mode == "drop":
        out_df = df_out.loc[~mask_block].reset_index(drop=True)
    else:
        out_df = df_out.copy()
        out_df["train_weight"] = 1.0
        out_df.loc[mask_block, "train_weight"] = 0.0

    base, ext = os.path.splitext(args.data)
    out_path = args.out.strip() or f"{base}_masked{ext if ext.lower() in ('.csv', '.parquet', '.pq', '.txt') else '.parquet'}"
    _write_table(out_df, out_path)

    kept = int(len(out_df))
    print(
        f"Готово. Всего строк: {total}. Запрещённых (no_trade): {blocked} ({actual_ratio:.2%}). Вышло: {kept}.",
    )
    print(f"NoTradeConfig: {cfg.dict()}")
    if args.mode == "weight":
        z = int((out_df.get('train_weight', pd.Series(dtype=float)) == 0.0).sum())
        print(f"Режим weight: назначено train_weight=0 для {z} строк.")
    _emit_summary_lines()
    _maybe_emit_histogram(mask_block)


if __name__ == "__main__":
    main()
