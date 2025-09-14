from __future__ import annotations

"""Operational kill switch with persistent counters and flag file support.

The switch can be manually reset by deleting the flag file or by calling
``ops_kill_switch.manual_reset()``.
"""

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .utils_app import atomic_write_with_retry

# ---------------------------------------------------------------------------
# Persistent state and configuration
_state_path = Path("state/ops_state.json")
_flag_path = Path("state/ops_kill_switch.flag")
_alert_command: Optional[Sequence[str]] = None
_reset_cooldown_sec = 60.0
_limits: Dict[str, int] = {"rest": 0, "ws": 0, "duplicates": 0, "stale": 0}

_counters: Dict[str, int] = {"rest": 0, "ws": 0, "duplicates": 0, "stale": 0}
_last_ts: Dict[str, float] = {"rest": 0.0, "ws": 0.0, "duplicates": 0.0, "stale": 0.0}
_tripped = False
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# State persistence helpers

def _load_state() -> None:
    """Load counters and timestamps from :data:`_state_path`."""
    global _tripped
    data: Dict[str, Any] = {}
    p = _state_path
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = {}
    with _lock:
        cnt = data.get("counters", {}) or {}
        ts = data.get("last_ts", {}) or {}
        for k in _counters:
            _counters[k] = int(cnt.get(k, 0))
            _last_ts[k] = float(ts.get(k, 0.0))
        _tripped = bool(data.get("tripped", False))
        try:
            if _flag_path.exists():
                _tripped = True
        except Exception:
            pass


def _save_state() -> None:
    data = {"counters": _counters, "last_ts": _last_ts, "tripped": _tripped}
    atomic_write_with_retry(_state_path, json.dumps(data, separators=(",", ":")), retries=3, backoff=0.1)


# ---------------------------------------------------------------------------
# Configuration

def init(cfg: Dict[str, Any]) -> None:
    """Initialise kill switch settings from *cfg*.

    The configuration may contain the following keys::

        rest_limit / rest_error_limit
        ws_limit / ws_error_limit
        duplicate_limit
        stale_limit
        reset_cooldown_sec
        flag_path
        state_path
        alert_command
    """
    global _reset_cooldown_sec, _flag_path, _alert_command, _limits, _state_path

    _limits = {
        "rest": int(
            cfg.get("rest_limit")
            or cfg.get("rest_error_limit")
            or cfg.get("rest_errors")
            or 0
        ),
        "ws": int(
            cfg.get("ws_limit")
            or cfg.get("ws_error_limit")
            or cfg.get("ws_errors")
            or 0
        ),
        "duplicates": int(cfg.get("duplicate_limit") or cfg.get("duplicates") or 0),
        "stale": int(cfg.get("stale_limit") or cfg.get("stale") or 0),
    }
    _reset_cooldown_sec = float(cfg.get("reset_cooldown_sec", _reset_cooldown_sec))
    _flag_path = Path(cfg.get("flag_path", _flag_path))
    _state_path = Path(cfg.get("state_path", _state_path))
    _alert_command = cfg.get("alert_command")
    _load_state()


# ---------------------------------------------------------------------------
# Internal helpers

def _maybe_reset_all(now: float) -> None:
    for k in list(_counters.keys()):
        if now - _last_ts[k] > _reset_cooldown_sec:
            _counters[k] = 0
            _last_ts[k] = now


def _trip_if_needed(kind: str) -> None:
    limit = _limits.get(kind) or 0
    if limit and _counters[kind] >= limit:
        _trip()


def _trip() -> None:
    global _tripped
    if _tripped:
        return
    _tripped = True
    try:
        _flag_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_with_retry(_flag_path, "1", retries=3, backoff=0.1)
    except Exception:
        pass
    _save_state()


# ---------------------------------------------------------------------------
# Public API

def record_error(kind: str) -> None:
    """Record a REST or websocket error."""
    if kind not in ("rest", "ws"):
        raise ValueError("kind must be 'rest' or 'ws'")
    now = time.time()
    tripped_now = False
    with _lock:
        _maybe_reset_all(now)
        _counters[kind] += 1
        _last_ts[kind] = now
        _save_state()
        was_tripped = _tripped
        _trip_if_needed(kind)
        tripped_now = _tripped and not was_tripped
    if tripped_now and _alert_command:
        try:
            proc = subprocess.run(list(_alert_command))
            if proc.returncode != 0:
                logging.getLogger(__name__).warning(
                    "alert_command exited with code %s", proc.returncode
                )
        except Exception:
            logging.getLogger(__name__).exception("alert_command execution failed")


def _record_generic(kind: str) -> None:
    now = time.time()
    tripped_now = False
    with _lock:
        _maybe_reset_all(now)
        _counters[kind] += 1
        _last_ts[kind] = now
        _save_state()
        was_tripped = _tripped
        _trip_if_needed(kind)
        tripped_now = _tripped and not was_tripped
    if tripped_now and _alert_command:
        try:
            proc = subprocess.run(list(_alert_command))
            if proc.returncode != 0:
                logging.getLogger(__name__).warning(
                    "alert_command exited with code %s", proc.returncode
                )
        except Exception:
            logging.getLogger(__name__).exception("alert_command execution failed")


def record_duplicate() -> None:
    """Record a duplicate message."""
    _record_generic("duplicates")


def reset_duplicates() -> None:
    """Reset duplicate message counter."""
    now = time.time()
    with _lock:
        _maybe_reset_all(now)
        _counters["duplicates"] = 0
        _last_ts["duplicates"] = now
        _save_state()


def record_stale() -> None:
    """Record a stale interval event."""
    _record_generic("stale")


def tripped() -> bool:
    with _lock:
        return _tripped


def manual_reset() -> None:
    """Clear counters and reset the kill switch."""
    global _tripped
    with _lock:
        for k in _counters:
            _counters[k] = 0
            _last_ts[k] = 0.0
        _tripped = False
        try:
            _flag_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        _save_state()


def tick() -> None:
    """Persist counters periodically and apply cooldown resets."""
    now = time.time()
    with _lock:
        _maybe_reset_all(now)
        _save_state()
