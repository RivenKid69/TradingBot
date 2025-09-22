from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
import sqlite3
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema


CURRENT_STATE_VERSION = 2
LAST_PROCESSED_GLOBAL_KEY = "__all__"


def _coerce_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


@dataclass(eq=False)
class PositionState:
    qty: float = 0.0
    avg_price: float = 0.0
    last_update_ms: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qty": float(self.qty),
            "avg_price": float(self.avg_price),
            "last_update_ms": self.last_update_ms,
        }

    def copy(self) -> "PositionState":
        return PositionState(self.qty, self.avg_price, self.last_update_ms)

    def update(self, **kwargs: Any) -> None:
        if "qty" in kwargs:
            self.qty = _coerce_float(kwargs["qty"])
        if "avg_price" in kwargs:
            self.avg_price = _coerce_float(kwargs["avg_price"])
        if "last_update_ms" in kwargs:
            self.last_update_ms = _coerce_int(kwargs["last_update_ms"])

    def __eq__(self, other: Any) -> bool:  # pragma: no cover - helper
        if isinstance(other, PositionState):
            return (
                math.isclose(self.qty, other.qty, rel_tol=1e-12, abs_tol=1e-12)
                and math.isclose(
                    self.avg_price, other.avg_price, rel_tol=1e-12, abs_tol=1e-12
                )
                and self.last_update_ms == other.last_update_ms
            )
        if isinstance(other, (int, float)):
            return math.isclose(self.qty, float(other), rel_tol=1e-12, abs_tol=1e-12)
        return False

    def __float__(self) -> float:  # pragma: no cover - helper
        return float(self.qty)

    @classmethod
    def from_any(cls, value: Any) -> "PositionState":
        if isinstance(value, PositionState):
            return value.copy()
        if isinstance(value, Mapping):
            qty = value.get("qty")
            if qty is None:
                qty = value.get("quantity") or value.get("position_qty") or value.get("size")
            avg_price = value.get("avg_price") or value.get("avgPrice") or value.get("price")
            last_update = (
                value.get("last_update_ms")
                or value.get("timestamp")
                or value.get("ts_ms")
                or value.get("time")
            )
            return cls(
                qty=_coerce_float(qty),
                avg_price=_coerce_float(avg_price),
                last_update_ms=_coerce_int(last_update),
            )
        if isinstance(value, (list, tuple)):
            qty = value[0] if len(value) > 0 else 0.0
            avg_price = value[1] if len(value) > 1 else 0.0
            last_update = value[2] if len(value) > 2 else None
            return cls(
                qty=_coerce_float(qty),
                avg_price=_coerce_float(avg_price),
                last_update_ms=_coerce_int(last_update),
            )
        if isinstance(value, (int, float)):
            return cls(qty=float(value))
        return cls()


@dataclass
class OrderState:
    symbol: str = ""
    client_order_id: str | None = None
    order_id: str | None = None
    side: str | None = None
    qty: float = 0.0
    price: float | None = None
    status: str | None = None
    ts_ms: int | None = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "symbol": self.symbol,
            "clientOrderId": self.client_order_id,
            "orderId": self.order_id,
            "side": self.side,
            "qty": self.qty,
            "price": self.price,
            "status": self.status,
            "ts_ms": self.ts_ms,
        }
        data.update(self.extra)
        return {k: v for k, v in data.items() if v is not None}

    def copy(self) -> "OrderState":
        return OrderState(
            symbol=self.symbol,
            client_order_id=self.client_order_id,
            order_id=self.order_id,
            side=self.side,
            qty=self.qty,
            price=self.price,
            status=self.status,
            ts_ms=self.ts_ms,
            extra=copy.deepcopy(self.extra),
        )

    def update(self, data: Mapping[str, Any]) -> None:
        for key, value in data.items():
            if key in {"qty", "quantity", "origQty"}:
                self.qty = _coerce_float(value)
            elif key in {"price", "avg_price"}:
                self.price = _coerce_float(value)
            elif key in {"ts_ms", "timestamp", "time"}:
                self.ts_ms = _coerce_int(value)
            elif key in {"clientOrderId", "client_order_id", "client_id"}:
                self.client_order_id = str(value) if value is not None else None
            elif key in {"orderId", "order_id"}:
                self.order_id = str(value) if value is not None else None
            elif key == "symbol":
                self.symbol = str(value or "")
            elif key == "side":
                self.side = str(value or "") or None
            elif key == "status":
                self.status = str(value or "") or None
            else:
                self.extra[key] = value

    @classmethod
    def from_any(cls, value: Any) -> "OrderState":
        if isinstance(value, OrderState):
            return value.copy()
        if isinstance(value, Mapping):
            order = cls()
            order.update(value)
            return order
        if hasattr(value, "_asdict"):
            return cls.from_any(value._asdict())  # type: ignore[attr-defined]
        return cls()


@dataclass
class TradingState:
    """In-memory representation of runner state."""

    positions: Dict[str, PositionState] = field(default_factory=dict)
    open_orders: list[OrderState] = field(default_factory=list)
    cash: float = 0.0
    equity: float | None = None
    last_processed_bar_ms: Dict[str, int] = field(default_factory=dict)
    seen_signals: Iterable[Any] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    signal_states: Dict[str, Any] = field(default_factory=dict)
    entry_limits: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_prices: Dict[str, float] = field(default_factory=dict)
    exposure_state: Dict[str, Any] = field(default_factory=dict)
    total_notional: float = 0.0
    git_hash: str | None = None
    version: int = CURRENT_STATE_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_update_ms: int | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingState":
        if not isinstance(data, Mapping):
            logger.warning("Corrupt state data, using defaults")
            return cls()

        known_keys = {
            "positions",
            "open_orders",
            "cash",
            "equity",
            "last_processed_bar_ms",
            "seen_signals",
            "config_snapshot",
            "signal_states",
            "entry_limits",
            "last_prices",
            "exposure_state",
            "total_notional",
            "git_hash",
            "version",
            "metadata",
            "last_update_ms",
        }

        positions: Dict[str, PositionState] = {}
        raw_positions = data.get("positions") or {}
        if isinstance(raw_positions, Mapping):
            for key, value in raw_positions.items():
                symbol = str(key)
                positions[symbol] = PositionState.from_any(value)
        elif isinstance(raw_positions, Iterable):
            for item in raw_positions:
                if isinstance(item, Mapping):
                    symbol = str(item.get("symbol") or item.get("asset") or "")
                    if not symbol:
                        continue
                    positions[symbol] = PositionState.from_any(item)

        orders: list[OrderState] = []
        raw_orders = data.get("open_orders") or []
        if isinstance(raw_orders, Mapping):
            values = list(raw_orders.values())
        elif isinstance(raw_orders, Iterable) and not isinstance(raw_orders, (str, bytes)):
            values = list(raw_orders)
        else:
            values = []
        for idx, value in enumerate(values):
            order = OrderState.from_any(value)
            if not order.order_id and not order.client_order_id:
                order.order_id = str(idx)
            orders.append(order)

        def _ensure_dict(value: Any) -> Dict[str, Any]:
            if isinstance(value, Mapping):
                return json.loads(json.dumps(value))
            return {}

        def _ensure_list(value: Any) -> Iterable[Any]:
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
                return list(value)
            if isinstance(value, Mapping):
                return list(value.items())
            return []

        last_prices_raw = data.get("last_prices") or {}
        last_prices: Dict[str, float] = {}
        if isinstance(last_prices_raw, Mapping):
            for key, value in last_prices_raw.items():
                try:
                    last_prices[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        last_processed_raw = data.get("last_processed_bar_ms")
        last_processed: Dict[str, int] = {}
        if isinstance(last_processed_raw, Mapping):
            for key, value in last_processed_raw.items():
                ts = _coerce_int(value)
                if ts is None:
                    continue
                last_processed[str(key)] = ts
        else:
            ts = _coerce_int(last_processed_raw)
            if ts is not None:
                last_processed[LAST_PROCESSED_GLOBAL_KEY] = ts

        state = cls(
            positions=positions,
            open_orders=orders,
            cash=_coerce_float(data.get("cash")),
            equity=(
                float(data.get("equity"))
                if isinstance(data.get("equity"), (int, float))
                else None
            ),
            last_processed_bar_ms=last_processed,
            seen_signals=_ensure_list(data.get("seen_signals")),
            config_snapshot=_ensure_dict(data.get("config_snapshot")),
            signal_states=_ensure_dict(data.get("signal_states")),
            entry_limits=_ensure_dict(data.get("entry_limits")),
            last_prices=last_prices,
            exposure_state=_ensure_dict(data.get("exposure_state")),
            total_notional=_coerce_float(data.get("total_notional")),
            git_hash=(str(data.get("git_hash")) if data.get("git_hash") else None),
            version=CURRENT_STATE_VERSION,
            metadata=_ensure_dict(data.get("metadata")),
            last_update_ms=_coerce_int(data.get("last_update_ms")),
        )

        leftovers = {
            str(key): copy.deepcopy(value)
            for key, value in data.items()
            if key not in known_keys
        }
        if leftovers:
            state.metadata.setdefault("_legacy", {}).update(leftovers)

        return state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "open_orders": [order.to_dict() for order in self.open_orders],
            "cash": float(self.cash),
            "equity": float(self.equity) if self.equity is not None else None,
            "last_processed_bar_ms": {
                str(sym): int(ts)
                for sym, ts in self.last_processed_bar_ms.items()
                if ts is not None
            },
            "seen_signals": list(self.seen_signals),
            "config_snapshot": copy.deepcopy(self.config_snapshot),
            "signal_states": copy.deepcopy(self.signal_states),
            "entry_limits": copy.deepcopy(self.entry_limits),
            "last_prices": {k: float(v) for k, v in self.last_prices.items()},
            "exposure_state": copy.deepcopy(self.exposure_state),
            "total_notional": float(self.total_notional),
            "git_hash": self.git_hash,
            "version": CURRENT_STATE_VERSION,
            "metadata": copy.deepcopy(self.metadata),
            "last_update_ms": self.last_update_ms,
        }

    def copy(self) -> "TradingState":
        return TradingState.from_dict(self.to_dict())

    def apply_updates(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if key == "positions":
                self.positions = TradingState.from_dict({"positions": value}).positions
            elif key == "open_orders":
                self.open_orders = TradingState.from_dict({"open_orders": value}).open_orders
            elif key == "seen_signals":
                if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
                    self.seen_signals = list(value)
                elif isinstance(value, Mapping):
                    self.seen_signals = list(value.items())
                else:
                    self.seen_signals = []
            elif key == "last_prices":
                if isinstance(value, Mapping):
                    self.last_prices = {
                        str(sym): _coerce_float(price) for sym, price in value.items()
                    }
                else:
                    self.last_prices = {}
            elif key in {
                "config_snapshot",
                "signal_states",
                "entry_limits",
                "exposure_state",
                "metadata",
            }:
                setattr(self, key, copy.deepcopy(value) if isinstance(value, Mapping) else {})
            elif key in {"cash", "total_notional", "equity"}:
                setattr(self, key, _coerce_float(value))
            elif key == "last_processed_bar_ms":
                mapping: Dict[str, int] = {}
                if isinstance(value, Mapping):
                    for sym, ts in value.items():
                        ts_val = _coerce_int(ts)
                        if ts_val is None:
                            continue
                        mapping[str(sym)] = ts_val
                else:
                    ts_val = _coerce_int(value)
                    if ts_val is not None:
                        mapping[LAST_PROCESSED_GLOBAL_KEY] = ts_val
                self.last_processed_bar_ms = mapping
            elif key == "last_update_ms":
                self.last_update_ms = _coerce_int(value)
            elif hasattr(self, key):
                setattr(self, key, copy.deepcopy(value))
            else:
                raise AttributeError(key)
        self.version = CURRENT_STATE_VERSION


# ---------------------------------------------------------------------------
# Backend abstraction


class StateBackend(Protocol):
    """Backend API for persisting :class:`TradingState`."""

    def load(self, path: Path) -> TradingState: ...

    def save(self, path: Path, state: TradingState, backup_keep: int = 0) -> None: ...


# ---------------------------------------------------------------------------
# JSON backend


class JsonBackend:
    def load(self, path: Path) -> TradingState:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return TradingState.from_dict(data)

    def save(self, path: Path, state: TradingState, backup_keep: int = 0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        payload = json.dumps(state.to_dict(), separators=(",", ":"), sort_keys=True)
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
            with suppress(OSError):
                dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        except Exception:
            logger.warning("Failed to persist JSON state to %s", path, exc_info=True)
            with suppress(Exception):
                if not path.exists():
                    for i in range(1, backup_keep + 1):
                        bak = path.with_suffix(path.suffix + f".bak{i}")
                        if bak.exists():
                            shutil.copy2(bak, path)
                            break
            raise
        finally:
            with suppress(OSError):
                tmp_path.unlink()


# ---------------------------------------------------------------------------
# SQLite backend


class SQLiteBackend:
    TABLE = "state"
    COLUMNS: Dict[str, str] = {
        "positions": "TEXT",
        "open_orders": "TEXT",
        "cash": "REAL",
        "equity": "REAL",
        "last_processed_bar_ms": "TEXT",
        "seen_signals": "TEXT",
        "config_snapshot": "TEXT",
        "signal_states": "TEXT",
        "entry_limits": "TEXT",
        "last_prices": "TEXT",
        "exposure_state": "TEXT",
        "total_notional": "REAL",
        "git_hash": "TEXT",
        "version": "INTEGER",
        "metadata": "TEXT",
        "last_update_ms": "INTEGER",
    }

    def _ensure_schema(self, con: sqlite3.Connection) -> None:
        columns_sql = ",".join(f"{name} {ctype}" for name, ctype in self.COLUMNS.items())
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {self.TABLE} ("
            "id INTEGER PRIMARY KEY CHECK (id = 1),"
            f"{columns_sql}"
            ")"
        )
        existing = {row[1] for row in con.execute(f"PRAGMA table_info({self.TABLE})")}
        for name, ctype in self.COLUMNS.items():
            if name not in existing:
                con.execute(f"ALTER TABLE {self.TABLE} ADD COLUMN {name} {ctype}")

    def load(self, path: Path) -> TradingState:
        if not path.exists():
            raise FileNotFoundError(path)
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            self._ensure_schema(con)
            cur = con.execute(f"SELECT * FROM {self.TABLE} WHERE id = 1")
            row = cur.fetchone()
        finally:
            con.close()
        if not row:
            return TradingState()
        keys = set(row.keys())
        def _load_json(value: Any, default: Any) -> Any:
            if value is None:
                return default
            if isinstance(value, (bytes, bytearray)):
                try:
                    value = value.decode()
                except Exception:
                    return default
            text = str(value)
            if not text:
                return default
            try:
                return json.loads(text)
            except Exception:
                return default

        raw_last_processed = (
            row["last_processed_bar_ms"]
            if "last_processed_bar_ms" in keys
            else None
        )
        if isinstance(raw_last_processed, (bytes, bytearray)):
            try:
                raw_last_processed = raw_last_processed.decode()
            except Exception:
                raw_last_processed = None
        if isinstance(raw_last_processed, str):
            raw_last_processed = raw_last_processed.strip()
        if isinstance(raw_last_processed, str) and raw_last_processed:
            try:
                last_processed_payload: Any = json.loads(raw_last_processed)
            except Exception:
                try:
                    last_processed_payload = int(raw_last_processed)
                except Exception:
                    last_processed_payload = None
        else:
            last_processed_payload = raw_last_processed

        data: Dict[str, Any] = {
            "positions": _load_json(row["positions"], {}),
            "open_orders": _load_json(row["open_orders"], []),
            "cash": row["cash"] if "cash" in keys else 0.0,
            "equity": row["equity"] if "equity" in keys else None,
            "last_processed_bar_ms": last_processed_payload,
            "seen_signals": json.loads(row["seen_signals"] or "[]"),
            "config_snapshot": json.loads(row["config_snapshot"] or "{}"),
            "signal_states": json.loads(row["signal_states"] or "{}"),
            "entry_limits": json.loads(row["entry_limits"] or "{}"),
            "last_prices": json.loads(row["last_prices"] or "{}"),
            "exposure_state": json.loads(row["exposure_state"] or "{}"),
            "total_notional": row["total_notional"] if "total_notional" in keys and row["total_notional"] is not None else 0.0,
            "git_hash": row["git_hash"] if "git_hash" in keys else None,
            "version": row["version"] if "version" in keys and row["version"] else CURRENT_STATE_VERSION,
            "metadata": json.loads(row["metadata"] or "{}") if "metadata" in keys else {},
            "last_update_ms": row["last_update_ms"] if "last_update_ms" in keys else None,
        }
        return TradingState.from_dict(data)

    def save(self, path: Path, state: TradingState, backup_keep: int = 0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            cur = con.cursor()
            cur.execute("BEGIN IMMEDIATE;")
            self._ensure_schema(con)
            payload = state.to_dict()
            cur.execute(
                f"REPLACE INTO {self.TABLE} ("
                "id, positions, open_orders, cash, equity, last_processed_bar_ms,"
                " seen_signals, config_snapshot, signal_states, entry_limits, last_prices,"
                " exposure_state, total_notional, git_hash, version, metadata, last_update_ms"
                ") VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    json.dumps(payload["positions"], separators=(",", ":")),
                    json.dumps(payload["open_orders"], separators=(",", ":")),
                    payload.get("cash", 0.0),
                    payload.get("equity"),
                    json.dumps(
                        payload.get("last_processed_bar_ms", {}),
                        separators=(",", ":"),
                    ),
                    json.dumps(payload.get("seen_signals", []), separators=(",", ":")),
                    json.dumps(payload.get("config_snapshot", {}), separators=(",", ":")),
                    json.dumps(payload.get("signal_states", {}), separators=(",", ":")),
                    json.dumps(payload.get("entry_limits", {}), separators=(",", ":")),
                    json.dumps(payload.get("last_prices", {}), separators=(",", ":")),
                    json.dumps(payload.get("exposure_state", {}), separators=(",", ":")),
                    payload.get("total_notional", 0.0),
                    payload.get("git_hash"),
                    payload.get("version", CURRENT_STATE_VERSION),
                    json.dumps(payload.get("metadata", {}), separators=(",", ":")),
                    payload.get("last_update_ms"),
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
        return _state.copy()


def update_state(**kwargs: Any) -> None:
    with _state_lock:
        _state.apply_updates(**kwargs)


def update_position(symbol: str, data: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
    sym = str(symbol or "")
    if not sym:
        raise ValueError("symbol must be non-empty")
    payload: Dict[str, Any] = {}
    if isinstance(data, PositionState):
        payload.update(data.to_dict())
    elif data:
        payload.update(dict(data))
    payload.update(kwargs)
    with _state_lock:
        if not payload:
            _state.positions.pop(sym, None)
        else:
            current = _state.positions.get(sym)
            position = current.copy() if current else PositionState()
            position.update(**payload)
            _state.positions[sym] = position
        _state.version = CURRENT_STATE_VERSION


def update_open_order(
    order_key: str,
    data: Mapping[str, Any] | OrderState | None = None,
    **kwargs: Any,
) -> None:
    key = str(order_key or "")
    if not key:
        raise ValueError("order_key must be non-empty")
    with _state_lock:
        def _matches(order: OrderState, needle: str) -> bool:
            if not needle:
                return False
            return needle in {
                str(order.order_id or ""),
                str(order.client_order_id or ""),
            }

        if isinstance(data, OrderState):
            order = data.copy()
            if kwargs:
                order.update(kwargs)
        else:
            payload: Dict[str, Any] = {}
            if data:
                payload.update(dict(data))
            payload.update(kwargs)
            if not payload:
                _state.open_orders = [
                    existing
                    for existing in _state.open_orders
                    if not _matches(existing, key)
                ]
                _state.version = CURRENT_STATE_VERSION
                return
            order = OrderState()
            order.update(payload)
        if not order.order_id and key:
            order.order_id = key
        identifiers = {
            str(order.order_id or ""),
            str(order.client_order_id or ""),
            key,
        }
        _state.open_orders = [
            existing
            for existing in _state.open_orders
            if not any(
                _matches(existing, ident) for ident in identifiers if ident
            )
        ]
        _state.open_orders.append(order)
        _state.version = CURRENT_STATE_VERSION


# ---------------------------------------------------------------------------
# File locking


@contextmanager
def _file_lock(lock_path: Path | str):
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            with suppress(Exception):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _rotate_backups(path: Path, keep: int, *, create_new: bool) -> None:
    path = Path(path)
    bak_plain = path.with_suffix(path.suffix + ".bak")
    with suppress(Exception):
        if bak_plain.exists():
            bak_plain.unlink()
    alt_plain = path.with_name(path.name + ".bak")
    with suppress(Exception):
        if alt_plain.exists():
            alt_plain.unlink()
    if keep <= 0:
        for old in path.parent.glob(f"{path.name}.bak*"):
            with suppress(Exception):
                old.unlink()
        return
    for index in range(keep, 0, -1):
        src = path.with_suffix(path.suffix + f".bak{index}")
        if src.exists():
            if index == keep:
                with suppress(Exception):
                    src.unlink()
            else:
                dst = path.with_suffix(path.suffix + f".bak{index + 1}")
                with suppress(Exception):
                    os.replace(src, dst)
    if create_new and path.exists():
        dst = path.with_suffix(path.suffix + ".bak1")
        with suppress(Exception):
            shutil.copy2(path, dst)


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
    enabled: bool = True,
) -> TradingState:
    global _state
    if not enabled:
        state = TradingState()
        with _state_lock:
            _state = state.copy()
        return state.copy()
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
                    logger.warning("Backup %s is not usable", bak, exc_info=True)
                    continue
            if state is None:
                state = TradingState()
    loaded = state.copy()
    with _state_lock:
        _state = loaded.copy()
    return loaded


def save_state(
    path: str | Path,
    backend: str | StateBackend = "json",
    lock_path: str | Path | None = None,
    backup_keep: int = 0,
    state: TradingState | None = None,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    p = Path(path)
    backend_obj = _get_backend(backend)
    lock_p = Path(lock_path) if lock_path else p.with_suffix(p.suffix + ".lock")
    with _state_lock:
        state_obj = state.copy() if state is not None else _state.copy()
        state_obj.version = CURRENT_STATE_VERSION
    with _file_lock(lock_p):
        try:
            _rotate_backups(p, backup_keep, create_new=backup_keep > 0 and p.exists())
        except Exception:
            logger.warning("Failed to rotate backups for %s", p, exc_info=True)
        try:
            backend_obj.save(p, state_obj, backup_keep=backup_keep)
        except Exception:
            logger.warning("Failed to save state to %s", p, exc_info=True)
            if backup_keep > 0:
                bak = p.with_suffix(p.suffix + ".bak1")
                if bak.exists():
                    with suppress(Exception):
                        shutil.copy2(bak, p)
            raise


def clear_state(
    path: str | Path,
    *,
    backend: str | StateBackend = "json",
    lock_path: str | Path | None = None,
    backup_keep: int = 0,
) -> None:
    """Remove persistent state files and reset in-memory snapshot."""

    p = Path(path)
    backend_obj = _get_backend(backend)
    lock_p = Path(lock_path) if lock_path else p.with_suffix(p.suffix + ".lock")
    with _file_lock(lock_p):
        # Remove primary file and known artefacts
        with suppress(Exception):
            if p.exists():
                p.unlink()
        with suppress(Exception):
            tmp_path = p.with_name(p.name + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
        if backup_keep != 0:
            for candidate in p.parent.glob(f"{p.name}.bak*"):
                with suppress(Exception):
                    candidate.unlink()
            plain_backup = p.with_name(p.name + ".bak")
            with suppress(Exception):
                if plain_backup.exists():
                    plain_backup.unlink()
        # Backend specific clean-up (sqlite writes auxiliary files)
        if isinstance(backend_obj, SQLiteBackend):
            for suffix in ("-wal", "-shm"):
                sidecar = p.with_suffix(p.suffix + suffix)
                with suppress(Exception):
                    if sidecar.exists():
                        sidecar.unlink()
    global _state
    with _state_lock:
        _state = TradingState()
