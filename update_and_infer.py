# update_and_infer.py
"""
Полный цикл обновления данных и инференса сигналов.

Функции ``run_single_cycle`` и ``run_continuous`` позволяют переиспользовать
логику без настройки переменных окружения.

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

ENV (для CLI по умолчанию):
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
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Sequence


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


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _run(cmd: Iterable[str] | str, *, check: bool = True) -> int:
    if isinstance(cmd, str):
        args = shlex.split(cmd)
    else:
        args = list(cmd)
    _log(f"$ {_format_cmd(args)}")
    try:
        res = subprocess.run(args, check=check)
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


def _normalize_symbols(symbols: Sequence[str]) -> List[str]:
    return [s.strip().upper() for s in symbols if s and s.strip()]


def _step1_incremental_klines(symbols: Sequence[str]) -> None:
    if not _exists_script("incremental_klines.py"):
        _log("! skip step1: incremental_klines.py not found")
        return
    syms_arg = ",".join(_normalize_symbols(symbols))
    _run(
        [
            sys.executable,
            "incremental_klines.py",
            "--symbols",
            syms_arg,
        ],
        check=False,
    )


def _step2_prepare_events(days: int, *, skip_events: bool) -> None:
    if skip_events:
        _log("~ step2: SKIP_EVENTS=1 — пропускаем обновление экономкалендаря")
        return
    if not _exists_script("prepare_events.py"):
        _log("! skip step2: prepare_events.py not found")
        return
    to = datetime.now(timezone.utc).date()
    frm = to - timedelta(days=max(1, days))
    _run(
        [
            sys.executable,
            "prepare_events.py",
            "--from",
            _date_str(frm),
            "--to",
            _date_str(to),
        ],
        check=False,
    )


def _step3_build_features(extra_args: Sequence[str]) -> None:
    extra = list(extra_args)
    ran_any = False
    if _exists_script("prepare_advanced_data.py"):
        ran_any = True
        try:
            _run([sys.executable, "prepare_advanced_data.py", *extra], check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
            msg = (
                "prepare_advanced_data.py failed during step3 "
                f"(exit code {exc.returncode})"
            )
            _log(f"! {msg}")
            raise RuntimeError(msg) from exc
    if _exists_script("prepare_and_run.py"):
        ran_any = True
        try:
            _run([sys.executable, "prepare_and_run.py", *extra], check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
            msg = (
                "prepare_and_run.py failed during step3 "
                f"(exit code {exc.returncode})"
            )
            _log(f"! {msg}")
            raise RuntimeError(msg) from exc
    if not ran_any:
        _log(
            "! step3 skipped: neither prepare_advanced_data.py nor prepare_and_run.py found"
        )


def _step4_validate_processed(
    *, max_age_sec: int | None = 3600, skip_freshness: bool = False
) -> None:
    if not _exists_script("validate_processed.py"):
        _log("! step4 skipped: validate_processed.py not found")
        return
    cmd = [sys.executable, "validate_processed.py"]
    if skip_freshness or (max_age_sec is not None and max_age_sec <= 0):
        cmd.append("--skip-freshness")
    elif max_age_sec is not None:
        cmd.extend(["--max-age-sec", str(max_age_sec)])
    rc = _run(cmd, check=False)
    if rc != 0:
        raise RuntimeError("validate_processed.py reported failures")


def _step5_infer_signals(extra_args: Sequence[str]) -> None:
    extra = list(extra_args)
    if not _exists_script("infer_signals.py"):
        _log("! step5 skipped: infer_signals.py not found")
        return
    _run([sys.executable, "infer_signals.py", *extra], check=True)


def run_single_cycle(
    symbols: Sequence[str],
    *,
    events_days: int = 90,
    skip_events: bool = False,
    extra_prepare_args: Sequence[str] | None = None,
    extra_infer_args: Sequence[str] | None = None,
    validate_max_age_sec: int | None = 3600,
    skip_validate_freshness: bool = False,
) -> None:
    symbols_list = _normalize_symbols(symbols) or ["BTCUSDT", "ETHUSDT"]
    prepare_args = list(extra_prepare_args or [])
    infer_args = list(extra_infer_args or [])
    _log("=== CYCLE START ===")
    _log(f"symbols={symbols_list}")
    try:
        _step1_incremental_klines(symbols_list)
        _step2_prepare_events(events_days, skip_events=skip_events)
        _step3_build_features(prepare_args)
        _step4_validate_processed(
            max_age_sec=validate_max_age_sec,
            skip_freshness=skip_validate_freshness,
        )
        _step5_infer_signals(infer_args)
    except Exception as e:
        _log(f"! cycle failed: {e}")
        raise
    finally:
        _log("=== CYCLE END ===")
    _log("✓ Cycle completed")


def run_continuous(
    symbols: Sequence[str],
    *,
    events_days: int = 90,
    sleep_minutes: float = 15.0,
    skip_events: bool = False,
    extra_prepare_args: Sequence[str] | None = None,
    extra_infer_args: Sequence[str] | None = None,
    validate_max_age_sec: int | None = 3600,
    skip_validate_freshness: bool = False,
) -> None:
    while True:
        run_single_cycle(
            symbols,
            events_days=events_days,
            skip_events=skip_events,
            extra_prepare_args=extra_prepare_args,
            extra_infer_args=extra_infer_args,
            validate_max_age_sec=validate_max_age_sec,
            skip_validate_freshness=skip_validate_freshness,
        )
        pause = max(0.0, float(sleep_minutes))
        if pause <= 0:
            continue
        _log(f"sleeping for {pause:.2f} minutes ...")
        time.sleep(pause * 60)


def _extra_from_env(name: str) -> List[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return shlex.split(raw)


def once() -> None:
    symbols = _env_list("SYMS", ["BTCUSDT", "ETHUSDT"])
    events_days = _env_int("EVENTS_DAYS", 90)
    skip_events = _env_bool("SKIP_EVENTS", False)
    validate_max_age_sec = _env_int("VALIDATE_MAX_AGE_SEC", 3600)
    skip_validate_freshness = _env_bool("SKIP_VALIDATE_FRESHNESS", False)
    run_single_cycle(
        symbols,
        events_days=events_days,
        skip_events=skip_events,
        extra_prepare_args=_extra_from_env("EXTRA_ARGS_PREPARE"),
        extra_infer_args=_extra_from_env("EXTRA_ARGS_INFER"),
        validate_max_age_sec=validate_max_age_sec,
        skip_validate_freshness=skip_validate_freshness,
    )


def main() -> None:
    loop = _env_bool("LOOP", False)
    symbols = _env_list("SYMS", ["BTCUSDT", "ETHUSDT"])
    events_days = _env_int("EVENTS_DAYS", 90)
    sleep_min = _env_int("SLEEP_MIN", 15)
    skip_events = _env_bool("SKIP_EVENTS", False)
    extra_prepare = _extra_from_env("EXTRA_ARGS_PREPARE")
    extra_infer = _extra_from_env("EXTRA_ARGS_INFER")
    validate_max_age_sec = _env_int("VALIDATE_MAX_AGE_SEC", 3600)
    skip_validate_freshness = _env_bool("SKIP_VALIDATE_FRESHNESS", False)

    if loop:
        run_continuous(
            symbols,
            events_days=events_days,
            sleep_minutes=sleep_min,
            skip_events=skip_events,
            extra_prepare_args=extra_prepare,
            extra_infer_args=extra_infer,
            validate_max_age_sec=validate_max_age_sec,
            skip_validate_freshness=skip_validate_freshness,
        )
    else:
        run_single_cycle(
            symbols,
            events_days=events_days,
            skip_events=skip_events,
            extra_prepare_args=extra_prepare,
            extra_infer_args=extra_infer,
            validate_max_age_sec=validate_max_age_sec,
            skip_validate_freshness=skip_validate_freshness,
        )


if __name__ == "__main__":
    main()
