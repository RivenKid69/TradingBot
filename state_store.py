"""Simple JSON-based persistence for runner state.

This module exposes a couple of mutable global containers that other parts of
application may update.  The :func:`load` and :func:`save` helpers restore and
persist these containers to disk using an atomic file replace to avoid partial
writes.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

from services.utils_app import atomic_write_with_retry

# Path used when no explicit destination is provided
DEFAULT_PATH = Path("state/state_store.json")

# Exposed mutable state containers
last_seen_close_ms: Dict[str, int] = {}
no_trade_state: Dict[str, Any] = {}
rolling_caches: Dict[str, Any] = {}
kill_switch_counters: Dict[str, Any] = {}
throttle_last_refill: float | int | None = None

_lock = threading.Lock()


def load(path: str | Path | None = None) -> None:
    """Load state from *path* if it exists.

    Missing files are ignored.  Any malformed content results in the state
    being reset to empty defaults.
    """
    p = Path(path or DEFAULT_PATH)
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text())
    except Exception:
        # Corrupted file -> start with empty state
        return
    with _lock:
        last_seen_close_ms.clear()
        last_seen_close_ms.update(data.get("last_seen_close_ms", {}) or {})
        no_trade_state.clear()
        no_trade_state.update(data.get("no_trade_state", {}) or {})
        rolling_caches.clear()
        rolling_caches.update(data.get("rolling_caches", {}) or {})
        kill_switch_counters.clear()
        kill_switch_counters.update(data.get("kill_switch_counters", {}) or {})
        global throttle_last_refill
        throttle_last_refill = data.get("throttle_last_refill")


def save(path: str | Path | None = None) -> None:
    """Persist current state to *path* using an atomic replace."""
    p = Path(path or DEFAULT_PATH)
    with _lock:
        data = {
            "last_seen_close_ms": last_seen_close_ms,
            "no_trade_state": no_trade_state,
            "rolling_caches": rolling_caches,
            "kill_switch_counters": kill_switch_counters,
            "throttle_last_refill": throttle_last_refill,
        }
        data_str = json.dumps(data, separators=(",", ":"))
    atomic_write_with_retry(p, data_str, retries=3, backoff=0.1)
