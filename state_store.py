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

# Paths used when no explicit destination is provided
DEFAULT_PATH = Path("state/state_store.json")
OPS_STATE_PATH = Path("state/ops_state.json")

# Exposed mutable state containers
last_seen_close_ms: Dict[str, int] = {}
no_trade_state: Dict[str, Any] = {}
rolling_caches: Dict[str, Any] = {}
kill_switch_counters: Dict[str, Any] = {}
throttle_last_refill: float | int | None = None

_lock = threading.Lock()


def load(
    path: str | Path | None = None, ops_path: str | Path | None = None
) -> None:
    """Load state from *path* if it exists.

    Missing files are ignored.  Any malformed content results in the state
    being reset to empty defaults.  ``kill_switch_counters`` are loaded from
    ``ops_path``.
    """
    p = Path(path or DEFAULT_PATH)
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = {}
    else:
        data = {}

    ops_p = Path(ops_path or OPS_STATE_PATH)
    if ops_p.exists():
        try:
            ops_data = json.loads(ops_p.read_text())
        except Exception:
            ops_data = {}
    else:
        ops_data = {}

    with _lock:
        last_seen_close_ms.clear()
        last_seen_close_ms.update(data.get("last_seen_close_ms", {}) or {})
        no_trade_state.clear()
        no_trade_state.update(data.get("no_trade_state", {}) or {})
        rolling_caches.clear()
        rolling_caches.update(data.get("rolling_caches", {}) or {})
        kill_switch_counters.clear()
        kill_switch_counters.update(ops_data.get("counters", {}) or {})
        global throttle_last_refill
        throttle_last_refill = data.get("throttle_last_refill")


def save(
    path: str | Path | None = None, ops_path: str | Path | None = None
) -> None:
    """Persist current state to *path* using an atomic replace."""
    p = Path(path or DEFAULT_PATH)
    ops_p = Path(ops_path or OPS_STATE_PATH)
    with _lock:
        data = {
            "last_seen_close_ms": last_seen_close_ms,
            "no_trade_state": no_trade_state,
            "rolling_caches": rolling_caches,
            "throttle_last_refill": throttle_last_refill,
        }
        data_str = json.dumps(data, separators=(",", ":"))

        ops_existing: Dict[str, Any] = {}
        if ops_p.exists():
            try:
                ops_existing = json.loads(ops_p.read_text())
            except Exception:
                ops_existing = {}
        ops_existing["counters"] = kill_switch_counters
        ops_str = json.dumps(ops_existing, separators=(",", ":"))

    atomic_write_with_retry(p, data_str, retries=3, backoff=0.1)
    atomic_write_with_retry(ops_p, ops_str, retries=3, backoff=0.1)
