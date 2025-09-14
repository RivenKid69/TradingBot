# -*- coding: utf-8 -*-
"""Мини-шина для публикации сигналов с защитой от повторов."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict
from dataclasses import dataclass
from collections import defaultdict
import os

import utils_time
from .utils_app import append_row_csv
from . import ops_kill_switch

# Путь к файлу состояния
_STATE_PATH = Path("state/seen_signals.json")

# Глобальное состояние: id -> expires_at_ms
_SEEN: Dict[str, int] = {}
# Drop counters by reason
dropped_by_reason: Dict[str, int] = defaultdict(int)
_lock = threading.Lock()
_loaded = False

# Optional CSV output paths
OUT_CSV: str | None = None
DROPS_CSV: str | None = None


@dataclass
class _Config:
    enabled: bool = True


config = _Config()


def signal_id(symbol: str, bar_close_ms: int) -> str:
    """Построить уникальный идентификатор сигнала."""
    return f"{symbol}:{int(bar_close_ms)}"


def _atomic_write(path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    # ensure data is flushed to disk before replace
    with tmp.open("w") as f:
        f.write(json.dumps(_SEEN, separators=(",", ":")))
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    tmp.replace(path)
    # attempt to fsync directory for durability
    try:
        dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        os.fsync(dir_fd)
        os.close(dir_fd)
    except Exception:
        pass


def _purge(now_ms: int | None = None) -> None:
    now = now_ms or int(time.time() * 1000)
    expired = [sid for sid, exp in _SEEN.items() if exp < now]
    for sid in expired:
        _SEEN.pop(sid, None)


def _flush() -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(_STATE_PATH)


def load_state(path: str | Path | None = None) -> None:
    """Загрузить состояние из JSON-файла, очищая устаревшие записи."""
    global _STATE_PATH, _loaded
    if path is not None:
        _STATE_PATH = Path(path)
    p = Path(_STATE_PATH)

    try:
        data = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        # В случае ошибки стартуем с пустым состоянием и перезаписываем файл
        _SEEN.clear()
        _loaded = True
        try:
            flush_state(p)
        except Exception:
            pass
        return

    now = int(time.time() * 1000)
    with _lock:
        _SEEN.clear()
        for sid, exp in data.items():
            try:
                exp_int = int(exp)
            except Exception:
                continue
            if exp_int >= now:
                _SEEN[str(sid)] = exp_int
    _loaded = True
    if len(_SEEN) != len(data):
        flush_state()


def flush_state(path: str | Path | None = None) -> None:
    """Сохранить текущее состояние на диск."""
    global _STATE_PATH
    if path is not None:
        _STATE_PATH = Path(path)
    with _lock:
        _flush()


def _ensure_loaded() -> None:
    if not _loaded:
        load_state()


def already_emitted(sid: str, *, now_ms: int | None = None) -> bool:
    """Проверить, публиковался ли сигнал ``sid`` ранее и не истёк ли его срок."""
    _ensure_loaded()
    now = now_ms or int(time.time() * 1000)
    with _lock:
        exp = _SEEN.get(sid)
        if exp is None:
            return False
        if exp < now:
            _SEEN.pop(sid, None)
            _flush()
            return False
        return True


def mark_emitted(
    sid: str,
    expires_at_ms: int,
    *,
    now_ms: int | None = None,
) -> None:
    """Отметить сигнал как опубликованный до ``expires_at_ms`` (ms since epoch)."""
    _ensure_loaded()
    now = now_ms or int(time.time() * 1000)
    with _lock:
        _purge(now)
        _SEEN[sid] = int(expires_at_ms)
        _flush()


def log_drop(symbol: str, bar_close_ms: int, payload: Any, reason: str) -> None:
    """Log a dropped signal to ``DROPS_CSV`` with a reason.

    The CSV mirrors the structure of ``publish_signal``: ``symbol``,
    ``bar_close_ms``, ``payload`` and ``reason``.  Any exceptions during
    logging are silenced just like in other helpers.
    """
    if reason == "duplicate":
        try:
            ops_kill_switch.record_duplicate()
        except Exception:
            pass
    if not DROPS_CSV:
        dropped_by_reason[str(reason)] += 1
        return
    try:
        header = ["symbol", "bar_close_ms", "payload", "reason"]
        row = [symbol, int(bar_close_ms), json.dumps(payload), str(reason)]
        append_row_csv(DROPS_CSV, header, row)
        dropped_by_reason[str(reason)] += 1
    except Exception:
        # Silently ignore logging errors
        pass


def publish_signal(
    symbol: str,
    bar_close_ms: int,
    payload: Any,
    send_fn: Callable[[Any], None],
    *,
    expires_at_ms: int,
    now_ms: int | None = None,
    dedup_key: str | None = None,
) -> bool:
    """Опубликовать сигнал, если он ещё не публиковался и не истёк TTL.

    Parameters
    ----------
    dedup_key:
        Optional explicit deduplication key.  If provided, it overrides
        :func:`signal_id`.

    Возвращает ``True``, если сигнал был отправлен, иначе ``False``.
    """
    if not config.enabled:
        log_drop(symbol, bar_close_ms, payload, "disabled")
        return False

    _ensure_loaded()
    sid = dedup_key or signal_id(symbol, bar_close_ms)
    now = now_ms if now_ms is not None else utils_time.now_ms()
    if now >= int(expires_at_ms):
        log_drop(symbol, bar_close_ms, payload, "expired")
        return False

    with _lock:
        _purge(now)
        if sid in _SEEN:
            log_drop(symbol, bar_close_ms, payload, "duplicate")
            return False

    send_fn(payload)

    with _lock:
        _SEEN[sid] = int(expires_at_ms)
        _flush()

    try:
        ops_kill_switch.reset_duplicates()
    except Exception:
        pass

    if OUT_CSV:
        try:
            header = ["symbol", "bar_close_ms", "payload", "expires_at_ms"]
            row = [symbol, int(bar_close_ms), json.dumps(payload), int(expires_at_ms)]
            append_row_csv(OUT_CSV, header, row)
        except Exception:
            pass
    return True


# Загрузить состояние при импорте
try:
    load_state()
except Exception:
    _SEEN.clear()
    _loaded = True
