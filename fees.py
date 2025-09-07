# sim/fees.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List


@dataclass
class FeesModel:
    """
    Модель комиссий Binance: maker/taker в б.п. (basis points), опциональная скидка BNB.
    Расчёт идёт от НОМИНАЛА сделки (price * qty).
    """
    maker_bps: float = 1.0    # 0.01%
    taker_bps: float = 5.0    # 0.05%
    use_bnb_discount: bool = False
    maker_discount_mult: float = 1.0  # например 0.75 если включена оплата BNB со скидкой
    taker_discount_mult: float = 1.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeesModel":
        maker_bps = float(d.get("maker_bps", 1.0))
        taker_bps = float(d.get("taker_bps", 5.0))
        use_bnb = bool(d.get("use_bnb_discount", False))
        # По умолчанию скидок не даём; если нужно — параметризуем явно в конфиге
        maker_mult = float(d.get("maker_discount_mult", 0.75 if use_bnb else 1.0))
        taker_mult = float(d.get("taker_discount_mult", 0.75 if use_bnb else 1.0))
        return cls(
            maker_bps=maker_bps,
            taker_bps=taker_bps,
            use_bnb_discount=use_bnb,
            maker_discount_mult=maker_mult,
            taker_discount_mult=taker_mult,
        )

    def compute(self, *, side: str, price: float, qty: float, liquidity: str) -> float:
        """
        :param side: "BUY"|"SELL" (не влияет на комиссию)
        :param price: цена сделки
        :param qty: количество (абсолютное)
        :param liquidity: "maker"|"taker"
        :return: абсолютная комиссия (>0)
        """
        notional = abs(float(price) * float(qty))
        if notional <= 0.0:
            return 0.0
        if str(liquidity).lower() == "maker":
            rate_bps = float(self.maker_bps) * float(self.maker_discount_mult)
        else:
            rate_bps = float(self.taker_bps) * float(self.taker_discount_mult)
        fee = notional * (rate_bps / 1e4)
        # Комиссия всегда положительная, её знак учитывается отдельно в учёте PnL
        return float(fee)


@dataclass
class FundingEvent:
    ts_ms: int
    rate: float
    position_qty: float
    mark_price: float
    cashflow: float  # положительно — получили, отрицательно — заплатили


class FundingCalculator:
    """
    Упрощённый калькулятор funding для перпетуалов.
    Модель: дискретные события каждые interval_seconds (по умолчанию 8 часов).
    Ставка фиксированная (const) на каждое событие. Для гибкости допускаем таблицу ставок.

    Знак cashflow:
      - Для long (qty>0) при rate>0 — платёж (cashflow < 0)
      - Для short (qty<0) при rate>0 — получение (cashflow > 0)
      - При отрицательной ставке — наоборот.
    """
    def __init__(
        self,
        *,
        enabled: bool = False,
        rate_source: str = "const",  # "const" | "curve"
        const_rate_per_interval: float = 0.0,  # например 0.0001 = 1 б.п. за интервал
        interval_seconds: int = 8 * 60 * 60,
        curve: Optional[Dict[int, float]] = None,  # {timestamp_ms->rate}, если rate_source="curve"
        align_to_epoch: bool = True,  # привязка к кратным интервала Epoch (даёт 00:00/08:00/16:00 UTC для 8h)
    ):
        self.enabled = bool(enabled)
        self.rate_source = str(rate_source)
        self.const_rate_per_interval = float(const_rate_per_interval)
        self.interval_seconds = int(interval_seconds)
        self.curve = dict(curve or {})
        self.align_to_epoch = bool(align_to_epoch)
        self._next_ts_ms: Optional[int] = None

    def _next_boundary(self, ts_ms: int) -> int:
        if not self.align_to_epoch:
            return int(ts_ms + self.interval_seconds * 1000)
        sec = int(ts_ms // 1000)
        next_sec = ((sec // self.interval_seconds) + 1) * self.interval_seconds
        return int(next_sec * 1000)

    def _rate_for_ts(self, ts_ms: int) -> float:
        if self.rate_source == "curve":
            # Берём точную ставку на этот момент; если нет — 0
            return float(self.curve.get(int(ts_ms), 0.0))
        # const
        return float(self.const_rate_per_interval)

    def reset(self) -> None:
        self._next_ts_ms = None

    def accrue(self, *, position_qty: float, mark_price: Optional[float], now_ts_ms: int) -> Tuple[float, List[FundingEvent]]:
        """
        Начисляет funding за все прошедшие дискретные моменты с предыдущего вызова.
        :param position_qty: текущая чистая позиция (штук)
        :param mark_price: текущая справедливая цена (для оценки notional)
        :param now_ts_ms: текущее время (мс)
        :return: (total_cashflow, [events...])
        """
        if not self.enabled:
            return 0.0, []
        if mark_price is None or not math.isfinite(float(mark_price)) or abs(position_qty) <= 0.0:
            # Нет цены или позиции — funding не начисляем
            self._next_ts_ms = None if self._next_ts_ms is None else self._next_ts_ms
            return 0.0, []

        total = 0.0
        events: List[FundingEvent] = []

        now_ts_ms = int(now_ts_ms)
        if self._next_ts_ms is None:
            self._next_ts_ms = self._next_boundary(now_ts_ms)

        # Если успели пройти сразу несколько интервалов — начислим несколько событий
        while now_ts_ms >= int(self._next_ts_ms):
            rate = self._rate_for_ts(int(self._next_ts_ms))
            notional = abs(float(position_qty)) * float(mark_price)
            # cashflow = - sign(position) * rate * notional
            sign = 1.0 if position_qty > 0 else -1.0
            cf = float(-sign * rate * notional)
            total += cf
            events.append(FundingEvent(
                ts_ms=int(self._next_ts_ms),
                rate=float(rate),
                position_qty=float(position_qty),
                mark_price=float(mark_price),
                cashflow=float(cf),
            ))
            # следующий интервал
            self._next_ts_ms = int(self._next_ts_ms + self.interval_seconds * 1000)

        return float(total), events
