# domainorder_manager.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Dict, Optional, Iterable, Tuple, Any


class OrderState(IntEnum):
    UNKNOWN = 0
    NEW = 1
    PARTIAL = 2
    FILLED = 3
    CANCELED = 4
    REJECTED = 5


# Допустимые текстовые статусы из разных источников
_STATUS_ALIASES = {
    "new": OrderState.NEW,
    "accepted": OrderState.NEW,
    "open": OrderState.NEW,

    "partial": OrderState.PARTIAL,
    "partially_filled": OrderState.PARTIAL,
    "partially-filled": OrderState.PARTIAL,

    "filled": OrderState.FILLED,
    "done": OrderState.FILLED,
    "closed": OrderState.FILLED,

    "canceled": OrderState.CANCELED,
    "cancelled": OrderState.CANCELED,
    "void": OrderState.CANCELED,

    "rejected": OrderState.REJECTED,
    "reject": OrderState.REJECTED,
    "error": OrderState.REJECTED,
}


def _normalize_status(s: Any) -> OrderState:
    if isinstance(s, OrderState):
        return s
    if isinstance(s, (int,)):
        try:
            return OrderState(int(s))
        except Exception:
            return OrderState.UNKNOWN
    if isinstance(s, str):
        key = s.strip().lower()
        if key in _STATUS_ALIASES:
            return _STATUS_ALIASES[key]
    return OrderState.UNKNOWN


@dataclass
class OrderRecord:
    state: OrderState = OrderState.UNKNOWN
    last_ts: int = 0        # монотонный таймштамп события (если источник его передаёт)
    updates: int = 0        # количество апдейтов


class OrderManager:
    """Лёгкий менеджер состояний ордеров по потокам событий."""
    def __init__(self) -> None:
        self._state: Dict[int, OrderRecord] = {}

    # ---------- базовые операции ----------

    def clear(self) -> None:
        self._state.clear()

    def remove(self, order_id: int) -> bool:
        return self._state.pop(int(order_id), None) is not None

    def contains(self, order_id: int) -> bool:
        return int(order_id) in self._state

    def size(self) -> int:
        return len(self._state)

    # ---------- запись/чтение ----------

    def set(self, order_id: int, state: OrderState, *, ts: int = 0) -> OrderState:
        oid = int(order_id)
        rec = self._state.get(oid)
        if rec is None:
            rec = OrderRecord(state=OrderState.UNKNOWN, last_ts=0, updates=0)
            self._state[oid] = rec
        # Обновление только если ts монотонно не меньше прошлого или ts=0 (без таймштампа)
        if ts == 0 or ts >= rec.last_ts:
            rec.state = _normalize_status(state)
            rec.last_ts = int(ts)
            rec.updates += 1
        return rec.state

    def get(self, order_id: int) -> OrderState:
        rec = self._state.get(int(order_id))
        return rec.state if rec is not None else OrderState.UNKNOWN

    def get_record(self, order_id: int) -> Optional[OrderRecord]:
        return self._state.get(int(order_id))

    def get_many(self, order_ids: Iterable[int]) -> Dict[int, OrderState]:
        return {int(oid): self.get(int(oid)) for oid in order_ids}

    # ---------- обработка событий ----------

    def on_event(self, ev: Dict[str, Any]) -> OrderState:
        """
        Унифицированная точка входа для внешних событий.
        Ожидаемые ключи словаря:
          - 'order_id' (int|str)
          - 'status' (str|int|OrderState) — допускаются алиасы ('cancelled', 'partially_filled', 'done', ...)
          - 'ts' (int, опционально) — монотонный таймштамп
        Возвращает новое состояние.
        """
        if not isinstance(ev, dict):
            raise TypeError("event must be a dict")

        if "order_id" not in ev:
            raise KeyError("event missing 'order_id'")
        oid = int(ev["order_id"])

        raw_status = ev.get("status", OrderState.UNKNOWN)
        state = _normalize_status(raw_status)
        ts = int(ev.get("ts", 0) or 0)

        return self.set(oid, state, ts=ts)

    # ---------- утилиты ----------

    def stats(self) -> Dict[str, int]:
        """Сводка по текущим состояниям."""
        out: Dict[str, int] = {
            "total": len(self._state),
            "unknown": 0,
            "new": 0,
            "partial": 0,
            "filled": 0,
            "canceled": 0,
            "rejected": 0,
        }
        for rec in self._state.values():
            st = rec.state
            if st == OrderState.UNKNOWN:
                out["unknown"] += 1
            elif st == OrderState.NEW:
                out["new"] += 1
            elif st == OrderState.PARTIAL:
                out["partial"] += 1
            elif st == OrderState.FILLED:
                out["filled"] += 1
            elif st == OrderState.CANCELED:
                out["canceled"] += 1
            elif st == OrderState.REJECTED:
                out["rejected"] += 1
        return out
