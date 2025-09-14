# -*- coding: utf-8 -*-
"""Мини-шина для публикации сигналов с защитой от повторов."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict

# Путь к файлу состояния
_STATE_PATH = Path("state/seen_signals.json")

# Глобальное состояние: id -> expires_at_ms
_SEEN: Dict[str, int] = {}
_lock = threading.Lock()


def signal_id(symbol: str, bar_close_ms: int) -> str:
    """Построить уникальный идентификатор сигнала."""
    return f"{symbol}:{int(bar_close_ms)}"


def _atomic_write(path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_SEEN, separators=(",", ":")))
    tmp.replace(path)


def _purge(now_ms: int | None = None) -> None:
    now = now_ms or int(time.time() * 1000)
    expired = [sid for sid, exp in _SEEN.items() if exp < now]
    for sid in expired:
        _SEEN.pop(sid, None)


def _flush() -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(_STATE_PATH)


def already_emitted(symbol: str, bar_close_ms: int, *, now_ms: int | None = None) -> bool:
    """Проверить, публиковался ли сигнал ранее и не истёк ли его срок."""
    sid = signal_id(symbol, bar_close_ms)
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
    symbol: str,
    bar_close_ms: int,
    *,
    ttl_ms: int,
    now_ms: int | None = None,
) -> None:
    """Отметить сигнал как опубликованный на ``ttl_ms`` миллисекунд."""
    sid = signal_id(symbol, bar_close_ms)
    now = now_ms or int(time.time() * 1000)
    exp = now + int(ttl_ms)
    with _lock:
        _purge(now)
        _SEEN[sid] = exp
        _flush()


def publish_signal(
    symbol: str,
    bar_close_ms: int,
    payload: Any,
    send_fn: Callable[[Any], None],
    *,
    ttl_ms: int,
    now_ms: int | None = None,
) -> bool:
    """Опубликовать сигнал, если он ещё не публиковался.

    Возвращает True, если сигнал был отправлен, иначе False.
    """
    if already_emitted(symbol, bar_close_ms, now_ms=now_ms):
        return False
    send_fn(payload)
    mark_emitted(symbol, bar_close_ms, ttl_ms=ttl_ms, now_ms=now_ms)
    return True


# Загрузить состояние при импорте
try:
    if _STATE_PATH.exists():
        data = json.loads(_STATE_PATH.read_text())
        now = int(time.time() * 1000)
        for sid, exp in data.items():
            exp_int = int(exp)
            if exp_int >= now:
                _SEEN[str(sid)] = exp_int
        if len(_SEEN) != len(data):
            _flush()
    else:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _flush()
except Exception:
    # В случае ошибки стартуем с пустым состоянием
    _SEEN.clear()
