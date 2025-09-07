# prepare_events.py
"""
Сбор и обновление макро-событий (economic calendar) в UTC.

Источник: investpy.economic_calendar (если модуль недоступен или API упадёт —
скрипт завершится "мягко": вернёт 0-й код и оставит существующий CSV без изменений).

Выход: data/economic_events.csv  со столбцами:
  - timestamp (UTC, seconds)
  - date (YYYY-MM-DD)
  - time (HH:MM or 'All Day'/'Tentative')
  - country (str)
  - name (event, str)
  - importance (original: 'low'|'medium'|'high'|other)
  - importance_level (int: low=0, medium=1, high=2, unknown=-1)
  - actual (str)
  - forecast (str)
  - previous (str)

CLI:
  python prepare_events.py --from 2024-01-01 --to 2024-12-31 \
      --countries "united states,euro zone,china" --min-importance medium
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


# ------------------------- утилиты логирования -------------------------

def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"[{ts}] {msg}", flush=True)


# ------------------------- investpy с фолбэками -------------------------

def _try_import_investpy():
    try:
        import investpy  # type: ignore
        return investpy
    except Exception as e:
        _log(f"~ investpy не найден или не импортируется: {e!r}. Пропущу обновление событий.")
        return None


def _to_ddmmyyyy(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _parse_hhmm_or_default(s: str) -> Tuple[int, int, str]:
    """
    Парсит 24ч время 'HH:MM'. Возвращает (hour, minute, tag) где tag — 'ok'|'allday'|'tentative'|'empty'.
    """
    if not s:
        return 0, 0, "empty"
    ss = s.strip().lower()
    if ss in ("all day", "allday"):
        return 0, 0, "allday"
    if ss in ("tentative",):
        return 0, 0, "tentative"
    m = re.match(r"^(\d{1,2}):(\d{2})$", ss)
    if m:
        hh = max(0, min(23, int(m.group(1))))
        mm = max(0, min(59, int(m.group(2))))
        return hh, mm, "ok"
    return 0, 0, "empty"


_TZ_RE = re.compile(r"GMT\s*([+-])\s*(\d{1,2})(?::(\d{2}))?$", re.IGNORECASE)


def _parse_gmt_offset(s: Optional[str]) -> int:
    """
    Возвращает смещение в секундах для строки вида 'GMT+3' / 'GMT-04:30'.
    Если строка пустая или не распознана — 0 (UTC).
    """
    if not s:
        return 0
    m = _TZ_RE.search(s.strip())
    if not m:
        return 0
    sign = -1 if m.group(1) == "-" else 1
    hh = int(m.group(2))
    mm = int(m.group(3) or "0")
    return sign * (hh * 3600 + mm * 60)


def _norm_importance(x: str) -> Tuple[str, int]:
    """
    Нормализует важность в текст и уровень: ('low'|'medium'|'high'|raw, level)
    """
    if not isinstance(x, str):
        return str(x), -1
    s = x.strip().lower()
    if s == "low":
        return "low", 0
    if s == "medium":
        return "medium", 1
    if s == "high":
        return "high", 2
    return s, -1


def _fetch_calendar(inv, d_from: date, d_to: date, countries: List[str], importances: List[str],
                    retries: int = 3, backoff: float = 0.8) -> Optional[pd.DataFrame]:
    """
    Надёжная загрузка календаря: несколько попыток, разумные паузы.
    Возвращает DataFrame или None при окончательной неудаче.
    """
    from_s = _to_ddmmyyyy(d_from)
    to_s = _to_ddmmyyyy(d_to)

    for attempt in range(1, retries + 1):
        try:
            # API investpy ожидает даты в формате DD/MM/YYYY
            df = inv.economic_calendar(
                from_date=from_s,
                to_date=to_s,
                countries=countries or None,
                importances=importances or None,
            )
            if not isinstance(df, pd.DataFrame):
                raise RuntimeError("investpy.economic_calendar returned non-DataFrame")
            # Иногда investpy возвращает пустой DF без колонок — нормализуем
            if df.empty:
                cols = ["date", "time", "country", "event", "importance", "zone", "actual", "forecast", "previous"]
                for c in cols:
                    if c not in df.columns:
                        df[c] = pd.Series(dtype="object")
            return df
        except Exception as e:
            if attempt == retries:
                _log(f"! investpy.economic_calendar: попытки исчерпаны: {e!r}")
                return None
            sleep_s = backoff * attempt
            _log(f"~ retry {attempt}/{retries-1} after error: {e!r}; sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)
    return None


def _normalize_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит сырые данные investpy к унифицированной схеме и считает timestamp (UTC, сек).
    """
    # Базовые имена столбцов (с защитой от вариаций)
    rename_map = {}
    for cand in ("event", "name"):
        if cand in df.columns:
            rename_map[cand] = "name"
            break
    if "country" in df.columns:
        rename_map["country"] = "country"
    if "importance" in df.columns:
        rename_map["importance"] = "importance"
    elif "impact" in df.columns:
        rename_map["impact"] = "importance"
    if "zone" in df.columns:
        rename_map["zone"] = "timezone"
    elif "timezone" in df.columns:
        rename_map["timezone"] = "timezone"
    if "actual" in df.columns:
        rename_map["actual"] = "actual"
    if "forecast" in df.columns:
        rename_map["forecast"] = "forecast"
    if "previous" in df.columns:
        rename_map["previous"] = "previous"

    # Убедимся, что есть date/time
    if "date" not in df.columns:
        raise ValueError("investpy payload missing 'date' column")
    if "time" not in df.columns:
        df["time"] = ""

    df = df.rename(columns=rename_map).copy()

    # Обязательные по итогам
    for col in ("country", "name", "importance", "actual", "forecast", "previous", "timezone"):
        if col not in df.columns:
            df[col] = ""

    # Нормализуем важность
    imp_norm = df["importance"].map(lambda x: _norm_importance(str(x)))
    df["importance"] = imp_norm.map(lambda t: t[0])
    df["importance_level"] = imp_norm.map(lambda t: t[1])

    # Разбираем время и зону
    # date в investpy обычно 'DD/MM/YYYY' (строка). Преобразуем в date.
    def _safe_parse_date(s: str) -> date:
        s = str(s).strip()
        # пробуем DD/MM/YYYY
        try:
            return datetime.strptime(s, "%d/%m/%Y").date()
        except Exception:
            pass
        # пробуем уже ISO
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            pass
        # fallback — сегодняшняя дата (не должна срабатывать в норме)
        return datetime.now(timezone.utc).date()

    dd = df["date"].map(_safe_parse_date)

    hhmm = df["time"].astype(str).map(_parse_hhmm_or_default)
    hours = hhmm.map(lambda t: t[0])
    minutes = hhmm.map(lambda t: t[1])

    tz_off = df["timezone"].astype(str).map(_parse_gmt_offset)

    # Считаем UTC timestamp (секунды)
    ts_utc = []
    for d, h, m, off in zip(dd, hours, minutes, tz_off):
        # локальное время -> utc
        local_dt = datetime(d.year, d.month, d.day, h, m, tzinfo=timezone.utc)  # временно как UTC
        # теперь учтём смещение: если было GMT+3, реальное UTC = local_dt - 3ч
        ts = int(local_dt.timestamp()) - int(off)
        ts_utc.append(ts)
    df["timestamp"] = pd.Series(ts_utc, dtype="int64")

    # Дополняем ISO дату: пригодится для удобства чтения
    df["date"] = dd.map(lambda d: d.strftime("%Y-%m-%d"))

    # Финальная выборка столбцов
    cols = [
        "timestamp",
        "date",
        "time",
        "country",
        "name",
        "importance",
        "importance_level",
        "actual",
        "forecast",
        "previous",
    ]
    df = df.loc[:, cols].sort_values(["timestamp", "importance_level", "country", "name"], ascending=[True, False, True, True])
    df = df.drop_duplicates(subset=["timestamp", "name", "country", "importance_level"], keep="last").reset_index(drop=True)
    return df


def _filter_by_min_importance(df: pd.DataFrame, min_imp: str) -> pd.DataFrame:
    lvl_map = {"low": 0, "medium": 1, "high": 2}
    thr = lvl_map.get(min_imp.strip().lower(), 0)
    return df[df["importance_level"] >= thr].copy()


def _merge_with_existing(new_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    if out_path.exists():
        try:
            old = pd.read_csv(out_path)
            need_cols = list(new_df.columns)
            for c in need_cols:
                if c not in old.columns:
                    old[c] = pd.Series(dtype=new_df[c].dtype)
            old = old[need_cols]
            merged = pd.concat([old, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["timestamp", "name", "country", "importance_level"], keep="last")
            merged = merged.sort_values(["timestamp", "importance_level"], ascending=[True, False]).reset_index(drop=True)
            return merged
        except Exception as e:
            _log(f"~ failed to read/merge existing CSV ({out_path}): {e!r}; will overwrite with new data")
    return new_df


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch & normalize macro events from investpy into data/economic_events.csv")
    p.add_argument("--from", dest="date_from", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--countries", type=str, default="united states,euro zone,china", help="comma-separated investpy country names")
    p.add_argument("--min-importance", type=str, default="medium", choices=["low", "medium", "high"], help="minimum importance to keep")
    p.add_argument("--out", type=str, default="data/economic_events.csv")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Парсинг дат
    try:
        d_from = datetime.strptime(args.date_from, "%Y-%m-%d").date()
        d_to = datetime.strptime(args.date_to, "%Y-%m-%d").date()
    except Exception:
        _log("! bad date format; expected YYYY-MM-DD for --from/--to")
        return 0

    if d_to < d_from:
        _log("! 'to' date earlier than 'from'; swapping")
        d_from, d_to = d_to, d_from

    countries = [c.strip().lower() for c in args.countries.split(",") if c.strip()]
    investpy = _try_import_investpy()
    if investpy is None:
        _log("~ skipping events update (no investpy).")
        return 0

    raw = _fetch_calendar(investpy, d_from, d_to, countries, importances=["medium", "high"])
    if raw is None:
        _log("~ skipping events update (investpy fetch failed).")
        return 0

    try:
        norm = _normalize_calendar(raw)
        norm = _filter_by_min_importance(norm, args.min_importance)
        merged = _merge_with_existing(norm, out_path)
        _atomic_write_csv(merged, out_path)
        _log(f"✓ events updated: {len(norm)} new rows, total {len(merged)} -> {out_path}")
    except Exception as e:
        _log(f"! failed to normalize/write events: {e!r}; skipping write")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
