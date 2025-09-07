# sim/risk.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import deque
from datetime import datetime, timezone


@dataclass
class RiskEvent:
    ts_ms: int
    code: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskConfig:
    """
    Базовые правила риска для среднечастотного бота.
      - max_abs_position_qty: максимальный абсолютный размер позиции (штук). 0 = выключено.
      - max_abs_position_notional: максимальный абсолютный размер позиции в котируемой валюте. 0 = выключено.
      - max_order_notional: максимум на одну заявку по нотионалу (price * qty). 0 = выключено.
      - max_orders_per_min: ограничение интенсивности заявок в скользящем окне.
      - max_orders_window_s: длина окна для лимита интенсивности (сек). Обычно 60.
      - daily_loss_limit: лимит дневного убытка (в котируемой валюте). Если equity - equity_at_day_start <= -limit → пауза.
      - pause_seconds_on_violation: сколько секунд держать торговлю на паузе при нарушении правил/лимитов.
      - daily_reset_utc_hour: час UTC, когда начинается новый «торговый день» (пересчёт equity_at_day_start).
      - enabled: общий флаг.
    """
    enabled: bool = True
    max_abs_position_qty: float = 0.0
    max_abs_position_notional: float = 0.0
    max_order_notional: float = 0.0
    max_orders_per_min: int = 60
    max_orders_window_s: int = 60
    daily_loss_limit: float = 0.0
    pause_seconds_on_violation: int = 300
    daily_reset_utc_hour: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RiskConfig":
        return cls(
            enabled=bool(d.get("enabled", True)),
            max_abs_position_qty=float(d.get("max_abs_position_qty", 0.0)),
            max_abs_position_notional=float(d.get("max_abs_position_notional", 0.0)),
            max_order_notional=float(d.get("max_order_notional", 0.0)),
            max_orders_per_min=int(d.get("max_orders_per_min", 60)),
            max_orders_window_s=int(d.get("max_orders_window_s", 60)),
            daily_loss_limit=float(d.get("daily_loss_limit", 0.0)),
            pause_seconds_on_violation=int(d.get("pause_seconds_on_violation", 300)),
            daily_reset_utc_hour=int(d.get("daily_reset_utc_hour", 0)),
        )


class RiskManager:
    """
    Минимально необходимый «менеджер рисков»:
      - дросселирование заявок (rate limit)
      - ограничение позиции (qty и/или notional)
      - дневной лосс → пауза

    Взаимодействие:
      - pre_trade_adjust() → корректирует целевой размер перед планированием детей
      - can_send_order() / on_new_order() → ограничение частоты
      - on_mark() → обновляет дневной PnL и выставляет паузу
    """
    def __init__(self, cfg: Optional[RiskConfig] = None):
        self.cfg = cfg or RiskConfig()
        self._paused_until_ms: int = 0
        self._orders_ts: deque[int] = deque()
        self._day_key: Optional[str] = None
        self._equity_day_start: Optional[float] = None
        self._events: List[RiskEvent] = []

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RiskManager":
        return cls(RiskConfig.from_dict(d or {}))

    @property
    def paused_until_ms(self) -> int:
        return int(self._paused_until_ms)

    def pop_events(self) -> List[RiskEvent]:
        ev = list(self._events)
        self._events.clear()
        return ev

    def _emit(self, ts_ms: int, code: str, message: str, **data: Any) -> None:
        self._events.append(RiskEvent(ts_ms=int(ts_ms), code=str(code), message=str(message), data=dict(data)))

    def _day_bucket(self, ts_ms: int) -> str:
        # Ключ «дня» по UTC с учётом смещения начала дня (daily_reset_utc_hour)
        h = int(self.cfg.daily_reset_utc_hour)
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        # сместим время так, чтобы "день" начинался в h:00
        adj = dt.replace(hour=h, minute=0, second=0, microsecond=0)
        if dt < adj:
            adj = adj.replace(day=(adj.day - 1))
        return adj.strftime("%Y-%m-%dT%H")

    def is_paused(self, ts_ms: int) -> bool:
        if not self.cfg.enabled:
            return False
        return int(ts_ms) < int(self._paused_until_ms)

    def _ensure_day(self, ts_ms: int, equity: Optional[float]) -> None:
        key = self._day_bucket(ts_ms)
        if key != self._day_key:
            self._day_key = key
            if equity is not None and math.isfinite(float(equity)):
                self._equity_day_start = float(equity)
            else:
                # если equity неизвестен, старт примем за 0 до следующего обновления on_mark
                self._equity_day_start = 0.0

    def on_mark(self, ts_ms: int, equity: Optional[float]) -> None:
        if not self.cfg.enabled:
            return
        self._ensure_day(ts_ms, equity)
        if equity is None or not math.isfinite(float(equity)):
            return
        if self._equity_day_start is None:
            self._equity_day_start = float(equity)
            return
        daily_pnl = float(equity) - float(self._equity_day_start)
        if float(self.cfg.daily_loss_limit) > 0.0 and daily_pnl <= -float(self.cfg.daily_loss_limit):
            pause_s = max(0, int(self.cfg.pause_seconds_on_violation))
            self._paused_until_ms = max(self._paused_until_ms, int(ts_ms + pause_s * 1000))
            self._emit(ts_ms, "DAILY_LOSS_PAUSE", f"Daily loss limit breached: PnL={daily_pnl:.2f} <= -{self.cfg.daily_loss_limit:.2f}",
                       equity=float(equity), equity_day_start=float(self._equity_day_start), paused_until_ms=int(self._paused_until_ms))

    def can_send_order(self, ts_ms: int) -> bool:
        if not self.cfg.enabled:
            return True
        # очистить старые таймстемпы
        window_ms = max(1, int(self.cfg.max_orders_window_s)) * 1000
        limit = max(1, int(self.cfg.max_orders_per_min))
        while self._orders_ts and (ts_ms - self._orders_ts[0]) > window_ms:
            self._orders_ts.popleft()
        return len(self._orders_ts) < limit

    def on_new_order(self, ts_ms: int) -> None:
        if not self.cfg.enabled:
            return
        self._orders_ts.append(int(ts_ms))

    def pre_trade_adjust(self, *, ts_ms: int, side: str, intended_qty: float, price: Optional[float], position_qty: float) -> float:
        """
        Возвращает допустимое количество (qty) с учётом ограничений позиции и заявки.
        Может вернуть 0.0, если торговля на паузе или лимит не позволяет увеличивать позицию.
        """
        if not self.cfg.enabled:
            return float(intended_qty)
        if self.is_paused(ts_ms):
            self._emit(ts_ms, "PAUSED", "Trading paused by risk manager", paused_until_ms=int(self._paused_until_ms))
            return 0.0

        q = max(0.0, float(intended_qty))
        if q == 0.0:
            return 0.0

        # Лимит на заявку по нотионалу
        if float(self.cfg.max_order_notional) > 0.0 and price is not None and price > 0.0:
            max_q_by_order = float(self.cfg.max_order_notional) / float(price)
            if max_q_by_order <= 0.0:
                self._emit(ts_ms, "ORDER_NOTIONAL_BLOCK", "max_order_notional too small", max_order_notional=float(self.cfg.max_order_notional))
                return 0.0
            if q > max_q_by_order:
                self._emit(ts_ms, "ORDER_NOTIONAL_CLAMP", "clamped by max_order_notional", requested_qty=float(q), allowed_qty=float(max_q_by_order))
                q = max_q_by_order

        # Лимит на абсолютную позицию (по qty)
        pos_after = float(position_qty) + (q if str(side).upper() == "BUY" else -q)
        if float(self.cfg.max_abs_position_qty) > 0.0 and abs(pos_after) > float(self.cfg.max_abs_position_qty):
            # допустимый инкремент до границы
            if str(side).upper() == "BUY":
                room = float(self.cfg.max_abs_position_qty) - max(0.0, float(position_qty))
            else:
                room = float(self.cfg.max_abs_position_qty) - max(0.0, -float(position_qty))
            allowed = max(0.0, float(room))
            if allowed <= 0.0:
                self._emit(ts_ms, "POS_QTY_BLOCK", "position qty limit blocks increase",
                           limit=float(self.cfg.max_abs_position_qty), position=float(position_qty), side=str(side))
                return 0.0
            if q > allowed:
                self._emit(ts_ms, "POS_QTY_CLAMP", "clamped by position qty limit",
                           requested_qty=float(q), allowed_qty=float(allowed), limit=float(self.cfg.max_abs_position_qty), position=float(position_qty))
                q = float(allowed)

        # Лимит на абсолютную позицию (по нотионалу)
        if float(self.cfg.max_abs_position_notional) > 0.0 and price is not None and price > 0.0:
            notional_after = abs(pos_after) * float(price)
            if notional_after > float(self.cfg.max_abs_position_notional):
                # сколько ещё можно добавить
                current_notional = abs(float(position_qty)) * float(price)
                room_notional = max(0.0, float(self.cfg.max_abs_position_notional) - current_notional)
                allowed = room_notional / float(price)
                if allowed <= 0.0:
                    self._emit(ts_ms, "POS_NOTIONAL_BLOCK", "position notional limit blocks increase",
                               limit=float(self.cfg.max_abs_position_notional), position=float(position_qty), price=float(price))
                    return 0.0
                if q > allowed:
                    self._emit(ts_ms, "POS_NOTIONAL_CLAMP", "clamped by position notional limit",
                               requested_qty=float(q), allowed_qty=float(allowed), limit=float(self.cfg.max_abs_position_notional), position=float(position_qty), price=float(price))
                    q = float(allowed)

        return float(q)

    def reset(self) -> None:
        self._paused_until_ms = 0
        self._orders_ts.clear()
        self._day_key = None
        self._equity_day_start = None
        self._events.clear()
