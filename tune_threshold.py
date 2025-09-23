import argparse
import os

import pandas as pd

from threshold_tuner import (
    TuneConfig,
    tune_threshold,
    load_min_signal_gap_s_from_yaml,
)


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
    ap = argparse.ArgumentParser(
        description=(
            "Подбор порога под целевую частоту сигналов "
            "с учётом кулдауна и no-trade."
        )
    )
    ap.add_argument(
        "--data",
        required=True,
        help=(
            "Файл с предсказаниями и таргетом (CSV/Parquet). "
            "Должны быть колонки ts_ms,symbol,score и y или eff_ret."
        ),
    )
    ap.add_argument(
        "--score_col", default="score", help="Имя колонки со скором/вероятностью."
    )
    ap.add_argument(
        "--y_col", default="", help="Колонка бинарной метки 0/1 (для классификации)."
    )
    ap.add_argument(
        "--ret_col",
        default="",
        help="Колонка эффективного ретёрна (для регрессии), например eff_ret_60.",
    )
    ap.add_argument("--ts_col", default="ts_ms", help="Колонка времени")
    ap.add_argument("--symbol_col", default="symbol", help="Колонка символа")
    ap.add_argument(
        "--direction",
        choices=["greater", "less"],
        default="greater",
        help="Правило: сигнал если score >= thr (greater) или <= thr (less)",
    )
    ap.add_argument(
        "--target_signals_per_day",
        type=float,
        default=1.5,
        help="Желаемая частота сигналов в день.",
    )
    ap.add_argument(
        "--tolerance", type=float, default=0.5, help="Допустимое отклонение частоты."
    )
    ap.add_argument(
        "--min_signal_gap_s",
        type=int,
        default=None,
        help=(
            "Кулдаун между сигналами (сек). Если не задан — "
            "пробуем прочитать из realtime.yaml."
        ),
    )
    ap.add_argument(
        "--realtime_config",
        default="configs/config_live.yaml",
        help="Файл для чтения min_signal_gap_s (если не задан явно).",
    )
    ap.add_argument(
        "--sandbox_config",
        default="configs/legacy_sandbox.yaml",
        help="Файл с no_trade (если нужно фильтровать).",
    )
    ap.add_argument(
        "--drop_no_trade",
        action="store_true",
        help="Фильтровать no-trade окна (по legacy_sandbox.yaml).",
    )
    ap.add_argument(
        "--min_thr", type=float, default=0.50, help="Минимальный порог сетки."
    )
    ap.add_argument(
        "--max_thr", type=float, default=0.99, help="Максимальный порог сетки."
    )
    ap.add_argument("--steps", type=int, default=50, help="Число порогов в сетке.")
    ap.add_argument(
        "--optimize_for",
        choices=["sharpe", "precision", "f1"],
        default="sharpe",
        help="Целевая метрика для выбора порога.",
    )
    ap.add_argument(
        "--out_csv",
        default="",
        help=(
            "Куда сохранить таблицу результатов (если пусто — "
            "рядом, суффикс _thrscan.csv)."
        ),
    )
    args = ap.parse_args()

    df = _read_table(args.data)

    min_gap = args.min_signal_gap_s
    if min_gap is None:
        min_gap = load_min_signal_gap_s_from_yaml(args.realtime_config)

    cfg = TuneConfig(
        score_col=args.score_col,
        y_col=(args.y_col if args.y_col.strip() else None),
        ret_col=(args.ret_col if args.ret_col.strip() else None),
        ts_col=args.ts_col,
        symbol_col=args.symbol_col,
        direction=args.direction,
        target_signals_per_day=float(args.target_signals_per_day),
        tolerance=float(args.tolerance),
        min_signal_gap_s=int(min_gap or 0),
        min_thr=float(args.min_thr),
        max_thr=float(args.max_thr),
        steps=int(args.steps),
        sandbox_yaml_for_no_trade=(args.sandbox_config if args.drop_no_trade else None),
        drop_no_trade=bool(args.drop_no_trade),
        optimize_for=args.optimize_for,
    )

    res, best = tune_threshold(df, cfg)

    base, ext = os.path.splitext(args.data)
    out_csv = args.out_csv.strip() or f"{base}_thrscan.csv"
    _write_table(res, out_csv)

    print("Готово. Таблица порогов записана:", out_csv)
    print("Рекомендуемый порог:")
    for k, v in best.items():
        try:
            print(f"  {k}: {float(v):.6f}")
        except Exception:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
