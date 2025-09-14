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

# Default persistence location
PERSIST_PATH = Path("state/close_state.json")

def load_state(path: str | Path = PERSIST_PATH) -> None:
    """Load state dictionary from JSON file if it exists.

    Parameters
    ----------
    path: str | Path
        Path to JSON file storing state.
    """
    p = Path(path)
    if not p.exists():
        logger.info("State file %s does not exist; starting empty", p)
        return

    try:
        data = json.loads(p.read_text())
    except Exception:
        logger.exception("Failed reading state file %s", p)
        return

    with _lock:
        STATE.clear()
        STATE.update({str(k): int(v) for k, v in data.items()})
    logger.info("Loaded %d symbols from %s", len(STATE), p)

def should_skip(symbol: str, close_ms: int) -> bool:
    """Return True if ``close_ms`` is not newer than stored value for ``symbol``."""
    with _lock:
        prev = STATE.get(symbol)
    return prev is not None and close_ms <= prev

def _atomic_write(path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(STATE, separators=(",", ":")))
    tmp.replace(path)

def flush(path: str | Path = PERSIST_PATH) -> None:
    """Persist current state to disk using atomic replace."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        _atomic_write(p)


def update(
    symbol: str,
    close_ms: int,
    *,
    path: str | Path = PERSIST_PATH,
    auto_flush: bool = True,
) -> None:
    """Update state for symbol and optionally flush to disk."""
    with _lock:
        STATE[symbol] = close_ms
        if auto_flush:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(p)
