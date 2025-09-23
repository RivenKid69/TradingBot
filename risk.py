# sim/risk.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import deque
from utils.time import daily_reset_key


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
      - daily_reset_utc_hour: час UTC, когда начинается новый «торговый день» (пересчёт equity_at_day_start и дневных лимитов).
      - max_entries_per_day: максимум на количество новых входов в позицию за «торговый день». ``None``/``-1`` = без лимита.
      - max_total_notional: лимит на суммарный нотионал портфеля (0 = выключено).
      - max_total_exposure_pct: лимит на суммарный нотионал как долю equity (0 = выключено).
      - exposure_buffer_frac: дополнительный буфер к приросту экспозиции (0 = выключено).
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
    max_entries_per_day: Optional[int] = None
    max_total_notional: float = 0.0
    max_total_exposure_pct: float = 0.0
    exposure_buffer_frac: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RiskConfig":
        raw_entries = d.get("max_entries_per_day")
        entries_per_day: Optional[int]
        if raw_entries is None:
            entries_per_day = None
        else:
            entries_per_day = int(raw_entries)
            if entries_per_day < 0:
                entries_per_day = None
        def _coerce_float(value: Any, default: float = 0.0, *, minimum: float | None = None) -> float:
            if value is None:
                return float(default)
            try:
                coerced = float(value)
            except (TypeError, ValueError):
                return float(default)
            if not math.isfinite(coerced):
                return float(default)
            if minimum is not None and coerced < minimum:
                return float(default)
            return float(coerced)

        max_total_notional = _coerce_float(d.get("max_total_notional", 0.0), 0.0, minimum=0.0)
        max_total_exposure_pct = _coerce_float(
            d.get("max_total_exposure_pct", 0.0), 0.0, minimum=0.0
        )
        exposure_buffer = _coerce_float(
            d.get("exposure_buffer_frac", 0.0), 0.0, minimum=0.0
        )

        return cls(
            enabled=bool(d.get("enabled", True)),
            max_abs_position_qty=_coerce_float(d.get("max_abs_position_qty", 0.0), 0.0, minimum=0.0),
            max_abs_position_notional=_coerce_float(
                d.get("max_abs_position_notional", 0.0), 0.0, minimum=0.0
            ),
            max_order_notional=_coerce_float(d.get("max_order_notional", 0.0), 0.0, minimum=0.0),
            max_orders_per_min=int(d.get("max_orders_per_min", 60)),
            max_orders_window_s=int(d.get("max_orders_window_s", 60)),
            daily_loss_limit=_coerce_float(d.get("daily_loss_limit", 0.0), 0.0, minimum=0.0),
            pause_seconds_on_violation=int(d.get("pause_seconds_on_violation", 300)),
            daily_reset_utc_hour=int(d.get("daily_reset_utc_hour", 0)),
            max_entries_per_day=entries_per_day,
            max_total_notional=max_total_notional,
            max_total_exposure_pct=max_total_exposure_pct,
            exposure_buffer_frac=exposure_buffer,
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
      - Методы принимают сезонные коэффициенты ликвидности/латентности для
        масштабирования лимитов при необходимости.
    """
    def __init__(self, cfg: Optional[RiskConfig] = None):
        self.cfg = cfg or RiskConfig()
        self._max_entries_per_day: Optional[int] = self._normalize_entries_limit(
            getattr(self.cfg, "max_entries_per_day", None)
        )
        # нормализованное значение прокинем обратно в конфиг для консистентности
        self.cfg.max_entries_per_day = self._max_entries_per_day
        self._paused_until_ms: int = 0
        self._orders_ts: deque[int] = deque()
        self._day_key: Optional[str] = None
        self._equity_day_start: Optional[float] = None
        self._events: List[RiskEvent] = []
        self._entries_day_key: Optional[str] = None
        self._entries_day_count: int = 0
        self._last_equity: Optional[float] = None

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

    @staticmethod
    def _normalize_entries_limit(value: Optional[Any]) -> Optional[int]:
        if value is None:
            return None
        try:
            limit = int(value)
        except (TypeError, ValueError):
            return None
        if limit < 0:
            return None
        return limit

    def _day_bucket(self, ts_ms: int) -> str:
        # Ключ «дня» по UTC с учётом смещения начала дня (daily_reset_utc_hour)
        return daily_reset_key(ts_ms, self.cfg.daily_reset_utc_hour)

    def _ensure_entry_day(self, ts_ms: int) -> None:
        key = self._day_bucket(ts_ms)
        if key != self._entries_day_key:
            self._entries_day_key = key
            self._entries_day_count = 0

    def _is_entry(self, side: str, position_qty: float, qty: float) -> bool:
        eps = 1e-9
        q = float(qty)
        if q <= 0.0:
            return False
        pos_before = float(position_qty)
        side_up = str(side).upper()
        if side_up == "BUY":
            pos_after = pos_before + q
            return pos_after > eps and pos_before <= eps
        if side_up == "SELL":
            pos_after = pos_before - q
            return pos_after < -eps and pos_before >= -eps
        return False

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
                # если equity неизвестен, базовое значение станет известным позже
                self._equity_day_start = None
            self._entries_day_key = key
            self._entries_day_count = 0

    def on_mark(self, ts_ms: int, equity: Optional[float]) -> None:
        if not self.cfg.enabled:
            return
        self._ensure_day(ts_ms, equity)
        if equity is None or not math.isfinite(float(equity)):
            self._last_equity = None
            return
        if self._equity_day_start is None:
            self._equity_day_start = float(equity)
            self._last_equity = float(equity)
            return
        daily_pnl = float(equity) - float(self._equity_day_start)
        self._last_equity = float(equity)
        if float(self.cfg.daily_loss_limit) > 0.0 and daily_pnl <= -float(self.cfg.daily_loss_limit):
            pause_s = max(0, int(self.cfg.pause_seconds_on_violation))
            self._paused_until_ms = max(self._paused_until_ms, int(ts_ms + pause_s * 1000))
            self._emit(ts_ms, "DAILY_LOSS_PAUSE", f"Daily loss limit breached: PnL={daily_pnl:.2f} <= -{self.cfg.daily_loss_limit:.2f}",
                       equity=float(equity), equity_day_start=float(self._equity_day_start), paused_until_ms=int(self._paused_until_ms))

    def can_send_order(self, ts_ms: int, *, latency_mult: float = 1.0) -> bool:
        """Check whether a new order can be sent respecting rate limits.

        Parameters
        ----------
        ts_ms : int
            Current timestamp in milliseconds.
        latency_mult : float, optional
            Seasonal multiplier for latency where ``>1`` means slower execution
            (fewer orders allowed per unit time). ``1.0`` keeps the original
            configuration unchanged. Values ``<=0`` are treated as ``1.0``.
        """
        if not self.cfg.enabled:
            return True
        # очистить старые таймстемпы
        base_window_ms = max(1, int(self.cfg.max_orders_window_s)) * 1000
        base_limit = max(1, int(self.cfg.max_orders_per_min))
        lm = float(latency_mult)
        if not math.isfinite(lm) or lm <= 0.0:
            lm = 1.0
        window_ms = int(base_window_ms * max(1.0, lm))
        limit = max(1, int(base_limit / max(lm, 1e-9)))
        while self._orders_ts and (ts_ms - self._orders_ts[0]) > window_ms:
            self._orders_ts.popleft()
        return len(self._orders_ts) < limit

    def on_new_order(self, ts_ms: int) -> None:
        if not self.cfg.enabled:
            return
        self._orders_ts.append(int(ts_ms))

    def pre_trade_adjust(
        self,
        *,
        ts_ms: int,
        side: str,
        intended_qty: float,
        price: Optional[float],
        position_qty: float,
        liquidity_mult: float = 1.0,
        total_notional: Optional[float] = None,
        equity: Optional[float] = None,
    ) -> float:
        """Return allowed quantity given risk limits.

        The optional ``liquidity_mult`` parameter allows scaling position and
        notional limits based on seasonal liquidity assumptions. Values ``>1``
        increase limits while values ``<1`` tighten them. Non-finite or
        non-positive values are treated as ``1.0``.
        """
        if not self.cfg.enabled:
            return float(intended_qty)
        if self.is_paused(ts_ms):
            self._emit(ts_ms, "PAUSED", "Trading paused by risk manager", paused_until_ms=int(self._paused_until_ms))
            return 0.0

        if self._max_entries_per_day is not None:
            self._ensure_entry_day(ts_ms)

        q = max(0.0, float(intended_qty))
        if q == 0.0:
            return 0.0

        lm = float(liquidity_mult)
        if not math.isfinite(lm) or lm <= 0.0:
            lm = 1.0

        max_order_notional = float(self.cfg.max_order_notional) * lm
        max_pos_qty = float(self.cfg.max_abs_position_qty) * lm
        max_pos_notional = float(self.cfg.max_abs_position_notional) * lm
        side_up = str(side).upper()

        # Лимит на заявку по нотионалу
        if max_order_notional > 0.0 and price is not None and price > 0.0:
            max_q_by_order = max_order_notional / float(price)
            if max_q_by_order <= 0.0:
                self._emit(ts_ms, "ORDER_NOTIONAL_BLOCK", "max_order_notional too small", max_order_notional=max_order_notional)
                return 0.0
            if q > max_q_by_order:
                self._emit(ts_ms, "ORDER_NOTIONAL_CLAMP", "clamped by max_order_notional", requested_qty=float(q), allowed_qty=float(max_q_by_order))
                q = max_q_by_order

        # Лимит на абсолютную позицию (по qty)
        pos_after = float(position_qty) + (q if side_up == "BUY" else -q)
        if max_pos_qty > 0.0 and abs(pos_after) > max_pos_qty:
            # допустимый инкремент до границы
            if side_up == "BUY":
                room = max_pos_qty - max(0.0, float(position_qty))
            else:
                room = max_pos_qty - max(0.0, -float(position_qty))
            allowed = max(0.0, float(room))
            if allowed <= 0.0:
                self._emit(ts_ms, "POS_QTY_BLOCK", "position qty limit blocks increase",
                           limit=max_pos_qty, position=float(position_qty), side=str(side))
                return 0.0
            if q > allowed:
                self._emit(ts_ms, "POS_QTY_CLAMP", "clamped by position qty limit",
                           requested_qty=float(q), allowed_qty=float(allowed), limit=max_pos_qty, position=float(position_qty))
                q = float(allowed)

        # Лимит на абсолютную позицию (по нотионалу)
        if max_pos_notional > 0.0 and price is not None and price > 0.0:
            notional_after = abs(pos_after) * float(price)
            if notional_after > max_pos_notional:
                # сколько ещё можно добавить
                current_notional = abs(float(position_qty)) * float(price)
                room_notional = max(0.0, max_pos_notional - current_notional)
                allowed = room_notional / float(price)
                if allowed <= 0.0:
                    self._emit(ts_ms, "POS_NOTIONAL_BLOCK", "position notional limit blocks increase",
                               limit=max_pos_notional, position=float(position_qty), price=float(price))
                    return 0.0
                if q > allowed:
                    self._emit(ts_ms, "POS_NOTIONAL_CLAMP", "clamped by position notional limit",
                               requested_qty=float(q), allowed_qty=float(allowed), limit=max_pos_notional, position=float(position_qty), price=float(price))
                    q = float(allowed)
                    pos_after = float(position_qty) + (q if side_up == "BUY" else -q)

        # Портфельные лимиты по экспозиции
        buffer_raw = float(self.cfg.exposure_buffer_frac or 0.0)
        if not math.isfinite(buffer_raw) or buffer_raw < 0.0:
            buffer_raw = 0.0
        buffer_mult = 1.0 + buffer_raw

        price_val: Optional[float]
        try:
            price_val = float(price) if price is not None else None
        except (TypeError, ValueError):
            price_val = None
        if price_val is not None:
            if not math.isfinite(price_val) or price_val <= 0.0:
                price_val = None

        total_notional_val: Optional[float]
        try:
            total_notional_val = float(total_notional) if total_notional is not None else None
        except (TypeError, ValueError):
            total_notional_val = None
        if total_notional_val is not None:
            if not math.isfinite(total_notional_val) or total_notional_val < 0.0:
                total_notional_val = None

        instrument_notional: Optional[float] = None
        if price_val is not None:
            instrument_notional = abs(float(position_qty)) * price_val

        if instrument_notional is not None:
            if total_notional_val is None:
                total_notional_val = instrument_notional
            else:
                total_notional_val = max(total_notional_val, instrument_notional)

        equity_candidate = equity if equity is not None else self._last_equity
        equity_val: Optional[float]
        try:
            equity_val = float(equity_candidate) if equity_candidate is not None else None
        except (TypeError, ValueError):
            equity_val = None
        if equity_val is not None:
            if not math.isfinite(equity_val) or equity_val <= 0.0:
                equity_val = None

        exposure_limits: List[Dict[str, Any]] = []
        if total_notional_val is not None:
            current_total = float(total_notional_val)
        else:
            current_total = None

        delta_abs_units = abs(pos_after) - abs(float(position_qty))
        if delta_abs_units < 0.0:
            delta_abs_units = 0.0

        if current_total is not None and delta_abs_units > 0.0:
            if float(self.cfg.max_total_notional) > 0.0:
                exposure_limits.append(
                    {
                        "type": "TOTAL_NOTIONAL",
                        "limit": float(self.cfg.max_total_notional),
                    }
                )
            if float(self.cfg.max_total_exposure_pct) > 0.0 and equity_val is not None:
                limit_equity = float(equity_val) * float(self.cfg.max_total_exposure_pct)
                if math.isfinite(limit_equity) and limit_equity > 0.0:
                    exposure_limits.append(
                        {
                            "type": "TOTAL_EXPOSURE",
                            "limit": float(limit_equity),
                        }
                    )

            limit_q_pairs: List[Dict[str, Any]] = []
            buffered_delta_notional: Optional[float]
            if price_val is None:
                buffered_delta_notional = None
            else:
                buffered_delta_notional = delta_abs_units * price_val * buffer_mult

            for info in exposure_limits:
                limit_value = float(info["limit"])
                headroom = limit_value - current_total
                if headroom <= 0.0:
                    code = (
                        "TOTAL_NOTIONAL_BLOCK"
                        if info["type"] == "TOTAL_NOTIONAL"
                        else "TOTAL_EXPOSURE_BLOCK"
                    )
                    message = (
                        "total notional limit blocks increase"
                        if info["type"] == "TOTAL_NOTIONAL"
                        else "total exposure limit blocks increase"
                    )
                    self._emit(
                        ts_ms,
                        code,
                        message,
                        limit=float(limit_value),
                        total=float(current_total),
                        buffer=float(buffer_raw),
                    )
                    return 0.0

                if buffered_delta_notional is None:
                    # Cannot evaluate the effect of the order without price data.
                    continue

                projected_total = current_total + buffered_delta_notional
                if projected_total <= limit_value + 1e-12:
                    continue

                allowed_delta_units = headroom / (price_val * buffer_mult)
                if allowed_delta_units <= 0.0:
                    code = (
                        "TOTAL_NOTIONAL_BLOCK"
                        if info["type"] == "TOTAL_NOTIONAL"
                        else "TOTAL_EXPOSURE_BLOCK"
                    )
                    message = (
                        "total notional limit blocks increase"
                        if info["type"] == "TOTAL_NOTIONAL"
                        else "total exposure limit blocks increase"
                    )
                    self._emit(
                        ts_ms,
                        code,
                        message,
                        limit=float(limit_value),
                        total=float(current_total),
                        buffer=float(buffer_raw),
                    )
                    return 0.0

                max_abs_after = abs(float(position_qty)) + allowed_delta_units

                if side_up == "BUY":
                    qty_limit = max(0.0, max_abs_after - float(position_qty))
                elif side_up == "SELL":
                    if float(position_qty) <= 0.0:
                        qty_limit = max(0.0, max_abs_after - abs(float(position_qty)))
                    else:
                        qty_limit = max(0.0, float(position_qty) + max_abs_after)
                else:
                    qty_limit = 0.0

                limit_q_pairs.append(
                    {
                        "type": str(info["type"]),
                        "limit": float(limit_value),
                        "qty_limit": float(qty_limit),
                    }
                )

            if limit_q_pairs:
                limiting = min(limit_q_pairs, key=lambda item: item["qty_limit"])
                qty_cap = float(limiting["qty_limit"])
                limit_value = float(limiting["limit"])
                limit_type = str(limiting["type"])
                if qty_cap <= 0.0:
                    code = (
                        "TOTAL_NOTIONAL_BLOCK"
                        if limit_type == "TOTAL_NOTIONAL"
                        else "TOTAL_EXPOSURE_BLOCK"
                    )
                    message = (
                        "total notional limit blocks increase"
                        if limit_type == "TOTAL_NOTIONAL"
                        else "total exposure limit blocks increase"
                    )
                    self._emit(
                        ts_ms,
                        code,
                        message,
                        limit=float(limit_value),
                        total=float(current_total),
                        buffer=float(buffer_raw),
                    )
                    return 0.0
                if q > qty_cap + 1e-12:
                    code = (
                        "TOTAL_NOTIONAL_CLAMP"
                        if limit_type == "TOTAL_NOTIONAL"
                        else "TOTAL_EXPOSURE_CLAMP"
                    )
                    message = (
                        "clamped by total notional limit"
                        if limit_type == "TOTAL_NOTIONAL"
                        else "clamped by total exposure limit"
                    )
                    self._emit(
                        ts_ms,
                        code,
                        message,
                        requested_qty=float(q),
                        allowed_qty=float(qty_cap),
                        limit=float(limit_value),
                        total=float(current_total),
                        buffer=float(buffer_raw),
                    )
                    q = float(qty_cap)
                    pos_after = float(position_qty) + (q if side_up == "BUY" else -q)

        if self._max_entries_per_day is not None and q > 0.0:
            if self._is_entry(side, position_qty, q):
                if self._entries_day_count >= int(self._max_entries_per_day):
                    self._emit(
                        ts_ms,
                        "ENTRY_LIMIT_BLOCK",
                        "daily entry limit reached",
                        limit=int(self._max_entries_per_day),
                        entries_today=int(self._entries_day_count),
                        day_key=str(self._entries_day_key),
                    )
                    return 0.0
                self._entries_day_count += 1

        return float(q)

    def reset(self) -> None:
        self._paused_until_ms = 0
        self._orders_ts.clear()
        self._day_key = None
        self._equity_day_start = None
        self._events.clear()
        self._entries_day_key = None
        self._entries_day_count = 0
        self._last_equity = None
