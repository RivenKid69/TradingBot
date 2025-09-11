# update_and_infer.py
"""
Полный цикл обновления данных и инференса сигналов.

Шаги (single pass):
  1) Догрузить последние закрытые свечи по каждому символу
     (Binance, 1h, limit=3 — берём предпоследнюю)
     -> data/candles/{SYMBOL}.csv    [скрипт: incremental_klines.py]
  2) Обновить экономический календарь за окно дат
     -> data/economic_events.csv     [скрипт: prepare_events.py]
        (мягко: при ошибке — лог и продолжить)
  3) Сборка/обогащение фич -> data/processed/*.feather
     [скрипт: prefer prepare_advanced_data.py, иначе fallback prepare_and_run.py]
  4) Валидация processed-таблиц ->
     validate_processed.py        (жёстко: при ошибке — прервать цикл кодом 1)
  5) Инференс сигналов -> data/signals/{SYMBOL}.csv
     [скрипт: infer_signals.py]
  6) Лог «✓ Cycle completed»

ENV:
  SYMS=BTCUSDT,ETHUSDT    — список символов для шага 1
  LOOP=0|1                — бесконечный цикл
  SLEEP_MIN=15            — пауза между проходами (мин)
  EVENTS_DAYS=90          — окно дней для prepare_events.py (по умолчанию 90)
  SKIP_EVENTS=0|1         — пропустить шаг 2 (экономкалендарь)
  EXTRA_ARGS_PREPARE=...  — дополнительные аргументы к
      prepare_advanced_data.py / prepare_and_run.py
  EXTRA_ARGS_INFER=...    — дополнительные аргументы к infer_signals.py
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import List


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"[{ts}] {msg}", flush=True)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except Exception:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    val = os.getenv(name, "")
    if not val:
        return default
    return [x.strip().upper() for x in val.split(",") if x.strip()]


def _exists_script(fname: str) -> bool:
    return os.path.exists(fname) and fname.endswith(".py")


def _run(cmd: str, *, check: bool = True) -> int:
    _log(f"$ {cmd}")
    try:
        res = subprocess.run(shlex.split(cmd), check=check)
        return int(res.returncode or 0)
    except subprocess.CalledProcessError as e:
        _log(f"! command failed (code={e.returncode}): {cmd}")
        if check:
            raise
        return int(e.returncode or 1)
    except FileNotFoundError:
        _log(f"! command not found: {cmd}")
        if check:
            raise
        return 127


def _date_str(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _step1_incremental_klines(symbols: List[str]) -> None:
    if not _exists_script("incremental_klines.py"):
        _log("! skip step1: incremental_klines.py not found")
        return
    syms_arg = ",".join(symbols)
    _run(f"{sys.executable} incremental_klines.py --symbols {syms_arg}", check=False)


def _step2_prepare_events(days: int) -> None:
    if _env_bool("SKIP_EVENTS", False):
        _log("~ step2: SKIP_EVENTS=1 — пропускаем обновление экономкалендаря")
        return
    if not _exists_script("prepare_events.py"):
        _log("! skip step2: prepare_events.py not found")
        return
    to = datetime.now(timezone.utc).date()
    frm = to - timedelta(days=max(1, days))
    cmd = (
        f"{sys.executable} prepare_events.py "
        f"--from {_date_str(frm)} --to {_date_str(to)}"
    )
    _run(cmd, check=False)


def _step3_build_features() -> None:
    extra = os.getenv("EXTRA_ARGS_PREPARE", "")
    if _exists_script("prepare_advanced_data.py"):
        _run(f"{sys.executable} prepare_advanced_data.py {extra}", check=True)
        return
    if _exists_script("prepare_and_run.py"):
        _run(f"{sys.executable} prepare_and_run.py {extra}", check=True)
        return
    _log(
        "! step3 skipped: neither prepare_advanced_data.py nor prepare_and_run.py found"
    )


def _step4_validate_processed() -> None:
    if not _exists_script("validate_processed.py"):
        _log("! step4 skipped: validate_processed.py not found")
        return
    rc = _run(f"{sys.executable} validate_processed.py", check=False)
    if rc != 0:
        raise RuntimeError("validate_processed.py reported failures")


def _step5_infer_signals() -> None:
    extra = os.getenv("EXTRA_ARGS_INFER", "")
    if not _exists_script("infer_signals.py"):
        _log("! step5 skipped: infer_signals.py not found")
        return
    _run(f"{sys.executable} infer_signals.py {extra}", check=True)


def once() -> None:
    symbols = _env_list("SYMS", ["BTCUSDT", "ETHUSDT"])
    events_days = _env_int("EVENTS_DAYS", 90)

    _log("=== CYCLE START ===")
    _log(f"symbols={symbols}")
    try:
        _step1_incremental_klines(symbols)
        _step2_prepare_events(events_days)
        _step3_build_features()
        _step4_validate_processed()
        _step5_infer_signals()
    except Exception as e:
        _log(f"! cycle failed: {e}")
        raise
    finally:
        _log("=== CYCLE END ===")
    _log("✓ Cycle completed")
