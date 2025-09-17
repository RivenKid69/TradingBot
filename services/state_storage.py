from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema


@dataclass
class TradingState:
    """In-memory representation of runner state."""

    positions: Dict[str, Any] = field(default_factory=dict)
    open_orders: Dict[str, Any] = field(default_factory=dict)
    cash: float = 0.0
    last_processed_bar_ms: int | None = None
    seen_signals: Iterable[Any] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    signal_states: Dict[str, Any] = field(default_factory=dict)
    entry_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    git_hash: str | None = None
    version: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingState":
        try:
            return cls(
                positions=data.get("positions", {}) or {},
                open_orders=data.get("open_orders", {}) or {},
                cash=data.get("cash", 0.0) or 0.0,
                last_processed_bar_ms=data.get("last_processed_bar_ms"),
                seen_signals=data.get("seen_signals", []) or [],
                config_snapshot=data.get("config_snapshot", {}) or {},
                signal_states=data.get("signal_states", {}) or {},
                entry_limits=data.get("entry_limits", {}) or {},
                git_hash=data.get("git_hash"),
                version=data.get("version", 1) or 1,
            )
        except Exception:
            logger.warning("Corrupt state data, using defaults")
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Backend abstraction


class StateBackend(Protocol):
    """Backend API for persisting :class:`TradingState`."""

    def load(self, path: Path) -> TradingState: ...

    def save(self, path: Path, state: TradingState) -> None: ...


# ---------------------------------------------------------------------------
# JSON backend


class JsonBackend:
    def load(self, path: Path) -> TradingState:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return TradingState.from_dict(data)

    def save(self, path: Path, state: TradingState) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        data = json.dumps(state.to_dict(), separators=(",", ":"))
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        bak = path.with_suffix(path.suffix + ".bak")
        if path.exists():
            try:
                os.replace(path, bak)
            except Exception:
                logger.warning("Unable to rotate backup for %s", path, exc_info=True)
        os.replace(tmp, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
            os.fsync(dir_fd)
            os.close(dir_fd)
        except Exception:
            pass
        try:
            os.remove(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# SQLite backend


class SQLiteBackend:
    TABLE_SQL = (
        "CREATE TABLE IF NOT EXISTS state ("
        "id INTEGER PRIMARY KEY CHECK (id = 1),"
        "positions TEXT,"
        "open_orders TEXT,"
        "cash REAL,"
        "last_processed_bar_ms INTEGER,"
        "seen_signals TEXT,"
        "config_snapshot TEXT,"
        "signal_states TEXT,"
        "entry_limits TEXT,"
        "git_hash TEXT,"
        "version INTEGER"
        ")"
    )

    def load(self, path: Path) -> TradingState:
        if not path.exists():
            raise FileNotFoundError(path)
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        try:
            with con:
                con.execute("PRAGMA journal_mode=WAL;")
                con.execute(self.TABLE_SQL)
                cur = con.execute(
                    "SELECT * FROM state WHERE id = 1"
                )
                row = cur.fetchone()
        finally:
            con.close()
        if not row:
            return TradingState()
        data = {
            "positions": json.loads(row["positions"] or "{}"),
            "open_orders": json.loads(row["open_orders"] or "{}"),
            "cash": row["cash"] or 0.0,
            "last_processed_bar_ms": row["last_processed_bar_ms"],
            "seen_signals": json.loads(row["seen_signals"] or "[]"),
            "config_snapshot": json.loads(row["config_snapshot"] or "{}"),
            "signal_states": json.loads(
                (row["signal_states"] if "signal_states" in row.keys() else "{}")
                or "{}"
            ),
            "entry_limits": json.loads(
                (row["entry_limits"] if "entry_limits" in row.keys() else "{}")
                or "{}"
            ),
            "git_hash": row["git_hash"],
            "version": row["version"] or 1,
        }
        return TradingState.from_dict(data)

    def save(self, path: Path, state: TradingState) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        bak = path.with_suffix(path.suffix + ".bak")
        if path.exists():
            try:
                shutil.copy2(path, bak)
            except Exception:
                logger.warning("Unable to rotate backup for %s", path, exc_info=True)
        con = sqlite3.connect(path)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            cur = con.cursor()
            cur.execute("BEGIN IMMEDIATE;")
            cur.execute(self.TABLE_SQL)
            try:
                columns = {
                    row[1] for row in cur.execute("PRAGMA table_info(state)")
                }
            except Exception:
                columns = set()
            if "signal_states" not in columns:
                cur.execute("ALTER TABLE state ADD COLUMN signal_states TEXT")
            if "entry_limits" not in columns:
                cur.execute("ALTER TABLE state ADD COLUMN entry_limits TEXT")
            cur.execute(
                "REPLACE INTO state (id, positions, open_orders, cash, last_processed_bar_ms,"
                " seen_signals, config_snapshot, signal_states, entry_limits, git_hash, version)"
                " VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    json.dumps(state.positions, separators=(",", ":")),
                    json.dumps(state.open_orders, separators=(",", ":")),
                    state.cash,
                    state.last_processed_bar_ms,
                    json.dumps(list(state.seen_signals), separators=(",", ":")),
                    json.dumps(state.config_snapshot, separators=(",", ":")),
                    json.dumps(state.signal_states, separators=(",", ":")),
                    json.dumps(state.entry_limits, separators=(",", ":")),
                    state.git_hash,
                    state.version,
                ),
            )
            con.commit()
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Thread-safe helpers


_state = TradingState()
_state_lock = threading.RLock()


def get_state() -> TradingState:
    with _state_lock:
        return TradingState.from_dict(_state.to_dict())


def update_state(**kwargs: Any) -> None:
    with _state_lock:
        for key, value in kwargs.items():
            if hasattr(_state, key):
                setattr(_state, key, value)
            else:
                raise AttributeError(key)


# ---------------------------------------------------------------------------
# File locking


@contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _rotate_backups(p: Path, keep: int) -> None:
    bak = p.with_suffix(p.suffix + ".bak")
    pattern = re.compile(re.escape(p.name) + r"\.bak(\d+)$")
    if keep <= 0:
        for old in p.parent.glob(p.name + ".bak*"):
            try:
                old.unlink()
            except Exception:
                pass
        if bak.exists():
            try:
                bak.unlink()
            except Exception:
                pass
        return
    for old in p.parent.glob(p.name + ".bak*"):
        m = pattern.match(old.name)
        if m and int(m.group(1)) > keep:
            try:
                old.unlink()
            except Exception:
                pass
    for i in range(keep, 0, -1):
        src = p.with_suffix(p.suffix + f".bak{i}")
        if src.exists():
            if i == keep:
                try:
                    src.unlink()
                except Exception:
                    pass
            else:
                dst = p.with_suffix(p.suffix + f".bak{i+1}")
                try:
                    src.rename(dst)
                except Exception:
                    pass
    if bak.exists():
        try:
            bak.rename(p.with_suffix(p.suffix + ".bak1"))
        except Exception:
            pass


def _get_backend(backend: str | StateBackend) -> StateBackend:
    if isinstance(backend, str):
        if backend == "json":
            return JsonBackend()
        if backend == "sqlite":
            return SQLiteBackend()
        raise ValueError(f"Unknown backend: {backend}")
    return backend


def load_state(
    path: str | Path,
    backend: str | StateBackend = "json",
    lock_path: str | Path | None = None,
    backup_keep: int = 0,
) -> TradingState:
    p = Path(path)
    backend_obj = _get_backend(backend)
    lock_p = Path(lock_path) if lock_path else p.with_suffix(p.suffix + ".lock")
    with _file_lock(lock_p):
        try:
            state = backend_obj.load(p)
        except Exception:
            logger.warning("Failed to load state from %s", p, exc_info=True)
            state = None
            for i in range(1, backup_keep + 1):
                bak = p.with_suffix(p.suffix + f".bak{i}")
                try:
                    state = backend_obj.load(bak)
                    logger.warning("Recovered state from backup %s", bak)
                    break
                except Exception:
                    continue
            if state is None:
                state = TradingState()
    with _state_lock:
        for k, v in state.to_dict().items():
            setattr(_state, k, v)
    return state


def save_state(
    path: str | Path,
    backend: str | StateBackend = "json",
    lock_path: str | Path | None = None,
    backup_keep: int = 0,
    state: TradingState | None = None,
) -> None:
    p = Path(path)
    backend_obj = _get_backend(backend)
    lock_p = Path(lock_path) if lock_path else p.with_suffix(p.suffix + ".lock")
    with _state_lock:
        state_obj = state or TradingState.from_dict(_state.to_dict())
    with _file_lock(lock_p):
        backend_obj.save(p, state_obj)
        _rotate_backups(p, backup_keep)
