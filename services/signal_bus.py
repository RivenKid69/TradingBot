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
from collections.abc import Mapping
from datetime import datetime, timezone
import math
import os

from api.spot_signals import SpotSignalEnvelope, build_envelope, sign_envelope
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
_SIGNING_SECRET: bytes | None = None

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


def configure_signing(secret: str | bytes | None) -> None:
    """Configure HMAC signing for published envelopes."""

    global _SIGNING_SECRET
    if not secret:
        _SIGNING_SECRET = None
        return
    _SIGNING_SECRET = secret if isinstance(secret, bytes) else secret.encode("utf-8")


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


def log_drop(envelope: SpotSignalEnvelope, reason: str) -> None:
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
        header = ["symbol", "bar_close_ms", "envelope", "reason"]
        row = [
            envelope.symbol,
            int(envelope.bar_close_ms),
            json.dumps(envelope.to_wire(), separators=(",", ":")),
            str(reason),
        ]
        append_row_csv(DROPS_CSV, header, row)
        dropped_by_reason[str(reason)] += 1
    except Exception:
        # Silently ignore logging errors
        pass


def _coerce_timestamp_ms(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        dt = value
    elif hasattr(value, "to_pydatetime"):
        try:
            dt = value.to_pydatetime()
        except Exception:
            dt = None
        if not isinstance(dt, datetime):
            dt = None
    else:
        dt = None
    if dt is not None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return int(round(dt.timestamp() * 1000.0))
    if isinstance(value, (int, float)):
        if isinstance(value, bool):
            return None
        if not math.isfinite(float(value)):
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                normalized = text
                if text.endswith("Z") or text.endswith("z"):
                    normalized = text[:-1] + "+00:00"
                try:
                    dt = datetime.fromisoformat(normalized)
                except ValueError:
                    return None
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return int(round(dt.timestamp() * 1000.0))
    timestamp = getattr(value, "timestamp", None)
    if callable(timestamp):
        try:
            ts = float(timestamp())
        except Exception:
            return None
        if not math.isfinite(ts):
            return None
        return int(round(ts * 1000.0))
    return None


def _extract_valid_until_ms(payload: Any, provided: Any = None) -> int | None:
    ms = _coerce_timestamp_ms(provided)
    if ms is not None:
        return ms

    def _from_mapping(mapping: Mapping[str, Any]) -> int | None:
        for key in ("valid_until_ms", "valid_until"):
            if key not in mapping:
                continue
            candidate = _coerce_timestamp_ms(mapping.get(key))
            if candidate is not None:
                return candidate
        return None

    if isinstance(payload, Mapping):
        mapped = _from_mapping(payload)
        if mapped is not None:
            return mapped

    for accessor in ("model_dump", "dict"):
        getter = getattr(payload, accessor, None)
        if not callable(getter):
            continue
        try:
            data = getter()
        except Exception:
            continue
        if isinstance(data, Mapping):
            mapped = _from_mapping(data)
            if mapped is not None:
                return mapped

    for attr in ("valid_until_ms", "valid_until"):
        if not hasattr(payload, attr):
            continue
        try:
            value = getattr(payload, attr)
        except Exception:
            continue
        candidate = _coerce_timestamp_ms(value)
        if candidate is not None:
            return candidate
    return None


def publish_signal(
    symbol: str,
    bar_close_ms: int,
    payload: Any,
    send_fn: Callable[[Any], None],
    *,
    expires_at_ms: int,
    now_ms: int | None = None,
    dedup_key: str | None = None,
    valid_until_ms: int | None = None,
) -> bool:
    """Опубликовать сигнал, если он ещё не публиковался и не истёк TTL.

    Parameters
    ----------
    dedup_key:
        Optional explicit deduplication key.  If provided, it overrides
        :func:`signal_id`.
    valid_until_ms:
        Optional signal validity horizon. When provided (or embedded in the
        payload) signals past this timestamp are rejected with a dedicated
        metric even if ``expires_at_ms`` is still in the future.

    Возвращает ``True``, если сигнал был отправлен, иначе ``False``.
    """

    bar_close_ms_int = int(bar_close_ms)
    expires_at_ms_int = int(expires_at_ms)

    envelope_cache: SpotSignalEnvelope | None = None
    valid_until_ms_int = _extract_valid_until_ms(payload, valid_until_ms)

    def _envelope() -> SpotSignalEnvelope:
        nonlocal envelope_cache
        if envelope_cache is None:
            envelope_cache = build_envelope(
                symbol=symbol,
                bar_close_ms=bar_close_ms_int,
                expires_at_ms=expires_at_ms_int,
                payload=payload,
            )
        return envelope_cache

    if not config.enabled:
        log_drop(_envelope(), "disabled")
        return False

    _ensure_loaded()
    sid = dedup_key or signal_id(symbol, bar_close_ms_int)
    now = now_ms if now_ms is not None else utils_time.now_ms()
    if valid_until_ms_int is not None and now >= valid_until_ms_int:
        log_drop(_envelope(), "valid_until_expired")
        return False
    if valid_until_ms_int is not None:
        expires_at_ms_int = min(expires_at_ms_int, valid_until_ms_int)
    if now >= expires_at_ms_int:
        log_drop(_envelope(), "expired")
        return False

    with _lock:
        _purge(now)
        if sid in _SEEN:
            log_drop(_envelope(), "duplicate")
            return False

    envelope = _envelope()
    if _SIGNING_SECRET is not None:
        envelope = sign_envelope(envelope, _SIGNING_SECRET)

    wire_payload = envelope.to_wire()
    send_fn(wire_payload)

    with _lock:
        _SEEN[sid] = expires_at_ms_int
        _flush()

    try:
        ops_kill_switch.reset_duplicates()
    except Exception:
        pass

    if OUT_CSV:
        try:
            header = ["symbol", "bar_close_ms", "envelope", "expires_at_ms"]
            row = [
                symbol,
                bar_close_ms_int,
                json.dumps(wire_payload, separators=(",", ":")),
                expires_at_ms_int,
            ]
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
