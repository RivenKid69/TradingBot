# validate_processed.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Строгий префикс колонок Binance + symbol
REQUIRED_PREFIX: List[str] = [
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
]

# Какие колонки считаем ключевыми числовыми для NaN/Inf/диапазонов
NUMERIC_KEY_COLS: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
]


def _fail(msg: str) -> None:
    raise ValueError(msg)


def _check_schema_and_order(df: pd.DataFrame) -> None:
    cols = list(df.columns)
    need = REQUIRED_PREFIX
    if len(cols) < len(need):
        _fail(
            f"Columns too few: have={len(cols)}, need at least {len(need)}; cols={cols}"
        )
    # порядок первых N должен совпадать точь-в-точь
    if cols[: len(need)] != need:
        _fail(f"Prefix/order mismatch.\nGot:   {cols[:len(need)]}\nWant:  {need}")


def _check_types_and_ranges(df: pd.DataFrame) -> None:
    # timestamp должен быть int (секунды UTC)
    if not np.issubdtype(df["timestamp"].dtype, np.integer):
        _fail(f"'timestamp' must be integer seconds, got dtype={df['timestamp'].dtype}")
    # symbol — строковый
    if not pd.api.types.is_object_dtype(
        df["symbol"].dtype
    ) and not pd.api.types.is_string_dtype(df["symbol"].dtype):
        _fail(f"'symbol' must be string dtype, got dtype={df['symbol'].dtype}")
    # Числовые колонки и диапазоны
    for c in NUMERIC_KEY_COLS:
        if c not in df.columns:
            _fail(f"Missing numeric column: {c}")
        if not pd.api.types.is_numeric_dtype(df[c].dtype):
            _fail(f"Column '{c}' must be numeric dtype, got {df[c].dtype}")
    # Положительность ключевых метрик
    if (
        (df["open"] <= 0).any()
        or (df["high"] <= 0).any()
        or (df["low"] <= 0).any()
        or (df["close"] <= 0).any()
    ):
        _fail("OHLC must be > 0")
    # Объёмы/счётчики не отрицательны
    for c in (
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ):
        if (df[c] < 0).any():
            _fail(f"'{c}' must be >= 0")


def _check_for_nulls(df: pd.DataFrame) -> None:
    # NaN/Inf в ключевых числовых колонках
    for c in NUMERIC_KEY_COLS + ["timestamp"]:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s.dtype):
            continue
        if np.isinf(s.to_numpy(dtype=np.float64, copy=False)).any():
            _fail(f"Column '{c}' contains +/-Inf")
        if s.isna().any():
            _fail(f"Column '{c}' contains NaN")


def _check_ohlc(df: pd.DataFrame) -> None:
    bad_high = df["high"] < df[["open", "close", "low"]].max(
        axis=1
    )  # high ≥ {open,close,low}
    bad_low = df["low"] > df[["open", "close"]].min(axis=1)  # low ≤ {open,close}
    if bad_high.any():
        idx = int(bad_high.idxmax())
        row = df.loc[idx, ["timestamp", "open", "high", "low", "close"]]
        _fail(f"OHLC invariant (high) broken at index={idx}, row={row.to_dict()}")
    if bad_low.any():
        idx = int(bad_low.idxmax())
        row = df.loc[idx, ["timestamp", "open", "high", "low", "close"]]
        _fail(f"OHLC invariant (low) broken at index={idx}, row={row.to_dict()}")


def _check_sorted_unique_ts(df: pd.DataFrame) -> None:
    ts = df["timestamp"].to_numpy()
    if not (np.all(ts[1:] > ts[:-1])):
        # найдём первое нарушение
        bad_idx = int(np.where(~(ts[1:] > ts[:-1]))[0][0] + 1)
        _fail(
            f"'timestamp' must be strictly increasing; first violation at position "
            f"{bad_idx} (ts[{bad_idx-1}]={ts[bad_idx-1]} -> ts[{bad_idx}]={ts[bad_idx]})"
        )


def _check_ts_continuity(df: pd.DataFrame, step_sec: int = 3600) -> None:
    ts = df["timestamp"].to_numpy()
    diffs = ts[1:] - ts[:-1]
    if not (np.all(diffs == step_sec)):
        bad_idx = int(np.where(diffs != step_sec)[0][0] + 1)
        _fail(
            f"Timestamp continuity broken at position {bad_idx}: "
            f"delta={int(diffs[bad_idx-1])}, expected {step_sec}"
        )


def _check_utc_alignment(df: pd.DataFrame) -> None:
    ts = df["timestamp"].to_numpy()
    mis = ts % 3600
    if (mis != 0).any():
        bad_idx = int(np.where(mis != 0)[0][0])
        _fail(f"UTC alignment failed at index {bad_idx}: ts%3600={int(mis[bad_idx])}")


def _check_freshness(df: pd.DataFrame, max_age_sec: int = 3600) -> None:
    if len(df) == 0:
        _fail("Empty dataframe")
    last_ts = int(df["timestamp"].iloc[-1])
    now = int(datetime.now(timezone.utc).timestamp())
    age = now - last_ts
    if age > max_age_sec:
        _fail(f"Stale data: last_ts={last_ts} ({age}s old) > max_age_sec={max_age_sec}")


def _infer_symbol_from_filename(path: Path) -> str | None:
    name = path.name
    if name.endswith(".feather"):
        return name[:-8].upper()  # strip .feather
    return None


def _check_symbol_consistency(df: pd.DataFrame, path: Path) -> None:
    # один и тот же symbol в файле
    syms = df["symbol"].unique().tolist()
    if len(syms) != 1:
        _fail(f"Multiple symbols in file: {syms}")
    sym_df = str(syms[0]).upper()
    # если можно — сверим с именем файла
    sym_file = _infer_symbol_from_filename(path)
    if sym_file and sym_file != sym_df:
        _fail(f"Symbol mismatch: filename expects {sym_file}, but data has {sym_df}")


def _check_no_duplicates(df: pd.DataFrame) -> None:
    # дублей по (timestamp,symbol) быть не должно
    dup = df.duplicated(subset=["timestamp", "symbol"])
    if dup.any():
        idx = int(np.where(dup.to_numpy())[0][0])
        row = df.loc[idx, ["timestamp", "symbol"]].to_dict()
        _fail(f"Duplicate row by (timestamp,symbol) at index {idx}: {row}")


def _check_same_columns(cols: Sequence[str], ref_cols: Sequence[str]) -> None:
    if list(cols) != list(ref_cols):
        _fail(
            f"Column set/order must be the same across files.\n"
            f"Got:  {list(cols)}\nRef:  {list(ref_cols)}"
        )


def _validate_file(
    path: Path,
    ref_cols: List[str] | None,
    *,
    max_age_sec: Optional[int],
    skip_freshness: bool,
) -> Tuple[bool, List[str], List[str]]:
    """Возвращает (ok, cols, errors)."""
    errs: List[str] = []
    try:
        df = pd.read_feather(path)
        # Базовые проверки
        _check_schema_and_order(df)
        _check_types_and_ranges(df)
        _check_for_nulls(df)
        _check_ohlc(df)
        _check_sorted_unique_ts(df)
        _check_ts_continuity(df, step_sec=3600)
        _check_utc_alignment(df)
        _check_symbol_consistency(df, path)
        _check_no_duplicates(df)
        if not skip_freshness:
            if max_age_sec is None:
                eff_age = 3600
            else:
                eff_age = max_age_sec
            if eff_age > 0:
                _check_freshness(df, max_age_sec=eff_age)
        cols = list(df.columns)
        # Единообразие схемы (если есть ref)
        if ref_cols is not None:
            _check_same_columns(cols, ref_cols)
        return True, cols, []
    except Exception as e:
        errs.append(str(e))
        try:
            cols = list(pd.read_feather(path, columns=None).columns)
        except Exception:
            cols = []
        return False, cols, errs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate processed feature tables produced by prepare_and_run.py",
    )
    parser.add_argument(
        "--base-dir",
        default="data/processed",
        help="Каталог с .feather файлами (по умолчанию data/processed)",
    )
    parser.add_argument(
        "--max-age-sec",
        type=int,
        default=3600,
        help=(
            "Максимальный допустимый возраст последнего бара в секундах. "
            "Значение <=0 отключает проверку."
        ),
    )
    parser.add_argument(
        "--skip-freshness",
        action="store_true",
        help="Полностью пропустить проверку свежести данных",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base = Path(args.base_dir)
    if not base.exists():
        print(f"[FAIL] {base} does not exist", file=sys.stderr)
        sys.exit(1)

    files = sorted(base.glob("*.feather"))
    if not files:
        print(f"[FAIL] No .feather files in {base}", file=sys.stderr)
        sys.exit(1)

    ref_cols: List[str] | None = None
    ok_total = 0
    fail_total = 0

    for i, path in enumerate(files, 1):
        ok, cols, errs = _validate_file(
            path,
            ref_cols,
            max_age_sec=args.max_age_sec,
            skip_freshness=args.skip_freshness or (args.max_age_sec <= 0),
        )
        if ok:
            print(f"[OK]   {path.name}")
            ok_total += 1
            if ref_cols is None:
                ref_cols = cols
        else:
            print(f"[FAIL] {path.name}")
            for msg in errs:
                print(f"   - {msg}")
            fail_total += 1

    print(f"\nSummary: OK={ok_total}, FAIL={fail_total}, TOTAL={len(files)}")
    if fail_total > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
