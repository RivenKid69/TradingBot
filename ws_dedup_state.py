from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict
import sys
import sysconfig

# Ensure we use stdlib logging despite local logging module
_stdlib_path = sysconfig.get_path("stdlib")
if _stdlib_path:
    sys.path.insert(0, _stdlib_path)
import logging as _std_logging
if _stdlib_path:
    sys.path.pop(0)

logger = _std_logging.getLogger(__name__)

# Global state mapping symbol -> last close timestamp in ms
STATE: Dict[str, int] = {}
_lock = threading.Lock()

# Background flusher
_flush_thread: threading.Thread | None = None
_flush_stop = threading.Event()
FLUSH_INTERVAL_SECONDS: float = 60.0

# Runtime configuration
ENABLED: bool = True
TTL_SECONDS: int = 0
OUT_CSV: str | None = None

# Default persistence location
PERSIST_PATH = Path("state/close_state.json")


def _flush_worker() -> None:
    while not _flush_stop.wait(FLUSH_INTERVAL_SECONDS):
        try:
            flush()
        except Exception:
            logger.exception("Periodic flush failed")

def init(
    *,
    enabled: bool = False,
    ttl_seconds: int = 0,
    persist_path: str | Path | None = None,
    out_csv: str | None = None,
    flush_interval_s: float = 60.0,
) -> None:
    """Configure signal bus runtime parameters."""
    global ENABLED, TTL_SECONDS, PERSIST_PATH, OUT_CSV, FLUSH_INTERVAL_SECONDS, _flush_thread

    shutdown()
    STATE.clear()
    ENABLED = bool(enabled)
    TTL_SECONDS = int(ttl_seconds)
    OUT_CSV = out_csv
    FLUSH_INTERVAL_SECONDS = float(flush_interval_s)
    if persist_path is not None:
        PERSIST_PATH = Path(persist_path)
    if ENABLED:
        try:
            load_state(PERSIST_PATH)
        except Exception:
            logger.exception("Failed loading state file %s", PERSIST_PATH)
        _flush_stop.clear()
        _flush_thread = threading.Thread(target=_flush_worker, daemon=True)
        _flush_thread.start()

def load_state(path: str | Path | None = None) -> None:
    """Load state dictionary from JSON file if it exists.

    Parameters
    ----------
    path: str | Path
        Path to JSON file storing state.
    """
    p = Path(path or PERSIST_PATH)
    if not p.exists():
        logger.info("State file %s does not exist; starting empty", p)
        with _lock:
            STATE.clear()
        return

    try:
        raw = p.read_text()
        data = json.loads(raw)
    except Exception:
        logger.exception("Failed reading state file %s", p)
        with _lock:
            STATE.clear()
        try:
            _atomic_write(p)
        except Exception:
            pass
        return

    with _lock:
        STATE.clear()
        for k, v in data.items():
            try:
                STATE[str(k)] = int(v)
            except Exception:
                continue
    if len(STATE) != len(data):
        try:
            _atomic_write(p)
        except Exception:
            pass
    logger.info("Loaded %d symbols from %s", len(STATE), p)

def should_skip(symbol: str, close_ms: int) -> bool:
    """Return True if ``close_ms`` is not newer than stored value for ``symbol``."""
    if not ENABLED:
        return False
    with _lock:
        prev = STATE.get(symbol)
    return prev is not None and close_ms <= prev

def _atomic_write(path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(STATE, separators=(",", ":")))
    tmp.replace(path)

def flush(path: str | Path | None = None) -> None:
    """Persist current state to disk using atomic replace."""
    p = Path(path or PERSIST_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        _atomic_write(p)


def update(
    symbol: str,
    close_ms: int,
    *,
    path: str | Path | None = None,
    auto_flush: bool = True,
) -> None:
    """Update state for symbol and optionally flush to disk."""
    if not ENABLED:
        return
    with _lock:
        STATE[symbol] = close_ms
        if auto_flush:
            p = Path(path or PERSIST_PATH)
            p.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(p)


def shutdown() -> None:
    """Stop background flusher and persist state."""
    global _flush_thread
    if not ENABLED:
        return
    _flush_stop.set()
    if _flush_thread is not None:
        try:
            _flush_thread.join(timeout=FLUSH_INTERVAL_SECONDS)
        except Exception:
            pass
        _flush_thread = None
    try:
        flush()
    except Exception:
        logger.exception("Failed to flush state on shutdown")

