from __future__ import annotations

import json
import logging
import os
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

    def load(self, path: Path) -> TradingState:
        ...

    def save(self, path: Path, state: TradingState) -> None:
        ...


# ---------------------------------------------------------------------------
# JSON backend


class JsonBackend:
    def load(self, path: Path) -> TradingState:
        if not path.exists():
            return TradingState()
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return TradingState.from_dict(data)
        except Exception:
            logger.warning("Failed to load JSON state from %s", path, exc_info=True)
            bak = path.with_suffix(path.suffix + ".bak")
            if bak.exists():
                try:
                    with bak.open("r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    logger.warning("Recovered state from backup %s", bak)
                    return TradingState.from_dict(data)
                except Exception:
                    logger.warning("Backup state %s corrupted", bak, exc_info=True)
            return TradingState()

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
        "git_hash TEXT,"
        "version INTEGER"
        ")"
    )

    def load(self, path: Path) -> TradingState:
        if not path.exists():
            return TradingState()
        try:
            con = sqlite3.connect(path)
            with con:
                con.execute("PRAGMA journal_mode=WAL;")
                con.execute(self.TABLE_SQL)
                cur = con.execute(
                    "SELECT positions, open_orders, cash, last_processed_bar_ms,"
                    " seen_signals, config_snapshot, git_hash, version FROM state"
                    " WHERE id = 1"
                )
                row = cur.fetchone()
            con.close()
            if not row:
                return TradingState()
            data = {
                "positions": json.loads(row[0] or "{}"),
                "open_orders": json.loads(row[1] or "{}"),
                "cash": row[2] or 0.0,
                "last_processed_bar_ms": row[3],
                "seen_signals": json.loads(row[4] or "[]"),
                "config_snapshot": json.loads(row[5] or "{}"),
                "git_hash": row[6],
                "version": row[7] or 1,
            }
            return TradingState.from_dict(data)
        except Exception:
            logger.warning("Failed to load SQLite state from %s", path, exc_info=True)
            bak = path.with_suffix(path.suffix + ".bak")
            if bak.exists():
                try:
                    con = sqlite3.connect(bak)
                    with con:
                        cur = con.execute(
                            "SELECT positions, open_orders, cash, last_processed_bar_ms,"
                            " seen_signals, config_snapshot, git_hash, version FROM state"
                            " WHERE id = 1"
                        )
                        row = cur.fetchone()
                    con.close()
                    if row:
                        data = {
                            "positions": json.loads(row[0] or "{}"),
                            "open_orders": json.loads(row[1] or "{}"),
                            "cash": row[2] or 0.0,
                            "last_processed_bar_ms": row[3],
                            "seen_signals": json.loads(row[4] or "[]"),
                            "config_snapshot": json.loads(row[5] or "{}"),
                            "git_hash": row[6],
                            "version": row[7] or 1,
                        }
                        logger.warning("Recovered state from backup %s", bak)
                        return TradingState.from_dict(data)
                except Exception:
                    logger.warning("Backup SQLite state %s corrupted", bak, exc_info=True)
            return TradingState()

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
            cur.execute(
                "REPLACE INTO state (id, positions, open_orders, cash, last_processed_bar_ms,"
                " seen_signals, config_snapshot, git_hash, version)"
                " VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    json.dumps(state.positions, separators=(",", ":")),
                    json.dumps(state.open_orders, separators=(",", ":")),
                    state.cash,
                    state.last_processed_bar_ms,
                    json.dumps(list(state.seen_signals), separators=(",", ":")),
                    json.dumps(state.config_snapshot, separators=(",", ":")),
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


def _get_backend(backend: str | StateBackend) -> StateBackend:
    if isinstance(backend, str):
        if backend == "json":
            return JsonBackend()
        if backend == "sqlite":
            return SQLiteBackend()
        raise ValueError(f"Unknown backend: {backend}")
    return backend


def load_state(path: str | Path, backend: str | StateBackend = "json", lock_path: str | Path | None = None) -> TradingState:
    p = Path(path)
    backend_obj = _get_backend(backend)
    lock_p = Path(lock_path) if lock_path else p.with_suffix(p.suffix + ".lock")
    with _file_lock(lock_p):
        state = backend_obj.load(p)
    with _state_lock:
        for k, v in state.to_dict().items():
            setattr(_state, k, v)
    return state


def save_state(path: str | Path, backend: str | StateBackend = "json", lock_path: str | Path | None = None, state: TradingState | None = None) -> None:
    p = Path(path)
    backend_obj = _get_backend(backend)
    lock_p = Path(lock_path) if lock_path else p.with_suffix(p.suffix + ".lock")
    with _state_lock:
        state_obj = state or TradingState.from_dict(_state.to_dict())
    with _file_lock(lock_p):
        backend_obj.save(p, state_obj)
