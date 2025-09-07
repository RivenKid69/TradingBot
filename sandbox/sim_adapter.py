# sandbox/sim_adapter.py
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterator, Protocol, Callable

from execution_sim import ExecutionSimulator, SimStepReport as ExecReport  # type: ignore
from action_proto import ActionType
from core_models import as_dict
import event_bus
from compat_shims import sim_report_dict_to_core_exec_reports
from decimal import Decimal
from event_bus import log_trade_exec as _bus_log_trade_exec
from market_data_port import MarketDataSource, MarketEvent, EventKind, Bar, ensure_timeframe
from order_shims import OrderContext, decisions_to_orders

class DecisionsProvider(Protocol):
    def on_bar(self, bar: Bar) -> Sequence[Any]: ...

class SimAdapter:
    """
    Тонкий мост: превращает решения стратегии в список экшенов симулятора.
    Требуется ExecutionSimulator с публичным методом run_step(...) (ниже добавим в execution_sim.py).
    """
    def __init__(self, sim: ExecutionSimulator, *, symbol: str, timeframe: str, source: MarketDataSource):
        self.sim = sim
        self.symbol = str(symbol).upper()
        self.timeframe = ensure_timeframe(timeframe)
        self.source = source


    def _to_actions(self, decisions: List[Dict[str, Any]]) -> List[Tuple[Any, Any]]:
        actions: List[Tuple[Any, Any]] = []
        for d in decisions:
            kind = str(d.get("kind", "MARKET")).upper()
            if kind == "MARKET":
                side = str(d.get("side", "BUY")).upper()
                vol = float(d.get("volume_frac", 0.0))
                proto = SimpleNamespace(volume_frac=(vol if side == "BUY" else -abs(vol)))
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
             decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        actions = self._to_actions(decisions)
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

    def run_events(self, provider: "DecisionsProvider") -> Iterator[Dict[str, Any]]:
        """
        Итерация по источнику событий.
        Для каждого BAR:
          - получаем решения из provider.on_bar(bar)
          - выполняем шаг симуляции через self.step(...)
          - возвращаем отчёт симулятора, расширенный служебными полями
        """
        self.source.open()
        self.source.subscribe([self.symbol], timeframe=self.timeframe)
        try:
            for ev in self.source.iter_events():
                if ev.kind != EventKind.BAR or ev.bar is None:
                    continue
                bar = ev.bar
                if bar.symbol != self.symbol or bar.timeframe != self.timeframe:
                    continue

                decisions = list(provider.on_bar(bar) or [])

                # эвристики для vol_factor и liquidity
                vol_factor = (float(bar.volume) if bar.volume is not None else None)
                liquidity = "NORMAL"

                rep = self.step(
                    ts_ms=int(bar.ts),
                    ref_price=float(bar.close),
                    bid=(None if bar.bid is None else float(bar.bid)),
                    ask=(None if bar.ask is None else float(bar.ask)),
                    vol_factor=vol_factor,
                    liquidity=liquidity,
                    decisions=decisions,
                )

                # дополнительно прикладываем стандартизованные заказы
                ctx = OrderContext(
                    ts_ms=int(bar.ts),
                    symbol=bar.symbol,
                    ref_price=float(bar.close),
                    max_position_abs_base=1.0,
                    tick_size=None,
                    price_offset_ticks=0,
                    tif="GTC",
                    client_tag=None,
                    round_qty_fn=None,
                )
                orders = decisions_to_orders(decisions, ctx)
                rep["symbol"] = bar.symbol
                rep["ts_ms"] = int(bar.ts)
                rep["core_orders"] = ([as_dict(o) for o in orders] if orders else [])
                yield rep
        finally:
            self.source.close()
