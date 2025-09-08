# sandbox/sim_adapter.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Iterator, Protocol

from execution_sim import ExecutionSimulator  # type: ignore
from action_proto import ActionProto, ActionType
from core_models import Bar, Order, Side, as_dict
from compat_shims import sim_report_dict_to_core_exec_reports
from event_bus import log_trade_exec as _bus_log_trade_exec
from core_contracts import MarketDataSource

_TF_MS = {
    "1s": 1_000,
    "5s": 5_000,
    "10s": 10_000,
    "15s": 15_000,
    "30s": 30_000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def _ensure_timeframe(tf: str) -> str:
    tf = str(tf).lower()
    if tf not in _TF_MS:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return tf


def _timeframe_to_ms(tf: str) -> int:
    tf = _ensure_timeframe(tf)
    return _TF_MS[tf]

class OrdersProvider(Protocol):
    def on_bar(self, bar: Bar) -> Sequence[Order]: ...

class SimAdapter:
    """
    Тонкий мост: превращает решения стратегии в список экшенов симулятора.
    Требуется ExecutionSimulator с публичным методом run_step(...) (ниже добавим в execution_sim.py).
    """
    def __init__(self, sim: ExecutionSimulator, *, symbol: str, timeframe: str, source: MarketDataSource):
        self.sim = sim
        self.symbol = str(symbol).upper()
        self.timeframe = _ensure_timeframe(timeframe)
        self.interval_ms = _timeframe_to_ms(self.timeframe)
        self.source = source


    def _to_actions(self, orders: Sequence[Order]) -> List[Tuple[ActionType, ActionProto]]:
        actions: List[Tuple[ActionType, ActionProto]] = []
        for o in orders:
            vol = float(o.quantity)
            if o.side == Side.SELL:
                vol = -abs(vol)
            proto = ActionProto(action_type=ActionType.MARKET, volume_frac=vol)
            actions.append((ActionType.MARKET, proto))
        return actions

    def step(self,
             *,
             ts_ms: int,
             ref_price: Optional[float],
             bid: Optional[float],
             ask: Optional[float],
             vol_factor: Optional[float],
             liquidity: Optional[float],
             orders: Sequence[Order]) -> Dict[str, Any]:
        actions = self._to_actions(orders)
        report = self.sim.run_step(
            ts=ts_ms,
            ref_price=ref_price,
            bid=bid,
            ask=ask,
            vol_factor=vol_factor,
            liquidity=liquidity,
            actions=actions,
        )
        d = report.to_dict()

        # Пишем унифицированный лог построчно (без изменения возврата)
        try:
            exec_reports = sim_report_dict_to_core_exec_reports(
                d, symbol=self.symbol, client_order_id=None
            )
        except Exception:
            exec_reports = []

        for _er in exec_reports:
            try:
                _bus_log_trade_exec(_er)
            except Exception:
                pass

        # формируем core_exec_reports (унифицированные отчёты исполнения) без изменения существующего интерфейса
        d["core_exec_reports"] = [as_dict(er) for er in exec_reports]
        return d

    def run_events(self, provider: "OrdersProvider") -> Iterator[Dict[str, Any]]:
        """
        Итерация по источнику баров.
        Для каждого BAR:
          - получаем решения из provider.on_bar(bar)
          - выполняем шаг симуляции через self.step(...)
          - возвращаем отчёт симулятора, расширенный служебными полями
        """
        prev_close: Optional[float] = None
        for bar in self.source.stream_bars([self.symbol], self.interval_ms):
            if bar.symbol != self.symbol:
                continue

            orders: Sequence[Order] = list(provider.on_bar(bar) or [])

            high = float(bar.high)
            low = float(bar.low)
            close = float(bar.close)

            vol_factor: Optional[float] = None
            if prev_close is not None and prev_close > 0.0:
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                atr_pct = tr / prev_close if prev_close != 0.0 else None
                log_ret = abs(math.log(close / prev_close)) if close > 0.0 else None
                if atr_pct is not None and log_ret is not None:
                    vol_factor = max(atr_pct, log_ret)
                else:
                    vol_factor = atr_pct if atr_pct is not None else log_ret

            liquidity: Optional[float] = None
            if bar.volume_base is not None:
                liquidity = float(bar.volume_base)
            elif bar.trades is not None:
                liquidity = float(bar.trades)

            rep = self.step(
                ts_ms=int(bar.ts),
                ref_price=close,
                bid=None,
                ask=None,
                vol_factor=vol_factor,
                liquidity=liquidity,
                orders=orders,
            )

            rep["symbol"] = bar.symbol
            rep["ts_ms"] = int(bar.ts)
            rep["core_orders"] = ([as_dict(o) for o in orders] or [])

            prev_close = close
            yield rep
