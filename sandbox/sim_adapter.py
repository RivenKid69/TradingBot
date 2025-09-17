# sandbox/sim_adapter.py
from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, Iterator, List, Optional, Sequence, Tuple, Protocol

from execution_sim import ExecutionSimulator  # type: ignore
from action_proto import ActionProto, ActionType
from core_models import Bar, Order, Side, as_dict
from compat_shims import sim_report_dict_to_core_exec_reports
from event_bus import log_trade_exec as _bus_log_trade_exec
from core_contracts import MarketDataSource
from core_config import CommonRunConfig
from services.monitoring import skipped_incomplete_bars


class _VolEstimator:
    """Rolling volatility estimator used by simulation and backtests.

    Tracks both logarithmic returns and true range (normalised by price)
    per symbol.  The configured ``vol_metric`` controls which series is
    exposed via :meth:`observe`, while the other series is still
    maintained for potential fallbacks.  The window is a simple rolling
    window with equal weights.
    """

    def __init__(self, *, vol_metric: str = "sigma", vol_window: int = 120) -> None:
        self._metric = str(vol_metric or "sigma").lower()
        if self._metric not in {"sigma", "atr", "atr_pct", "atr/price"}:
            self._metric = "sigma"
        self._window = max(1, int(vol_window or 1))
        self._returns: Dict[str, Deque[float]] = {}
        self._ret_sumsq: Dict[str, float] = {}
        self._tranges: Dict[str, Deque[float]] = {}
        self._tr_sum: Dict[str, float] = {}
        self._last_close: Dict[str, float] = {}
        self._last_value: Dict[str, Optional[float]] = {}

    @staticmethod
    def _to_float(val: Any) -> Optional[float]:
        try:
            out = float(val)
        except (TypeError, ValueError):
            return None
        if math.isfinite(out):
            return out
        return None

    def _append_return(self, symbol: str, value: float) -> None:
        dq = self._returns.get(symbol)
        if dq is None:
            dq = deque(maxlen=self._window)
            self._returns[symbol] = dq
        total = self._ret_sumsq.get(symbol, 0.0)
        if dq.maxlen is not None and len(dq) == dq.maxlen:
            old = dq.popleft()
            total -= old * old
        dq.append(value)
        total += value * value
        self._ret_sumsq[symbol] = total

    def _append_trange(self, symbol: str, value: float) -> None:
        dq = self._tranges.get(symbol)
        if dq is None:
            dq = deque(maxlen=self._window)
            self._tranges[symbol] = dq
        total = self._tr_sum.get(symbol, 0.0)
        if dq.maxlen is not None and len(dq) == dq.maxlen:
            old = dq.popleft()
            total -= old
        dq.append(value)
        total += value
        self._tr_sum[symbol] = total

    def _compute(self, symbol: str, metric: Optional[str] = None) -> Optional[float]:
        metric_key = (metric or self._metric or "").lower()
        metric = metric_key
        if metric == "sigma":
            dq = self._returns.get(symbol)
            if dq:
                total = max(0.0, self._ret_sumsq.get(symbol, 0.0))
                return math.sqrt(total / len(dq))
        elif metric in {"atr", "atr_pct", "atr/price"}:
            dq = self._tranges.get(symbol)
            if dq:
                total = self._tr_sum.get(symbol, 0.0)
                return max(0.0, total / len(dq))

        # Fallbacks: try sigma first, then ATR, whichever has data.
        dq_ret = self._returns.get(symbol)
        if dq_ret:
            total = max(0.0, self._ret_sumsq.get(symbol, 0.0))
            return math.sqrt(total / len(dq_ret))
        dq_tr = self._tranges.get(symbol)
        if dq_tr:
            total = self._tr_sum.get(symbol, 0.0)
            return max(0.0, total / len(dq_tr))
        return None

    def observe(
        self,
        *,
        symbol: str,
        high: Any,
        low: Any,
        close: Any,
    ) -> Optional[float]:
        sym = str(symbol).upper()
        prev_close = self._last_close.get(sym)
        hi = self._to_float(high)
        lo = self._to_float(low)
        cl = self._to_float(close)

        if prev_close is not None and cl is not None and prev_close > 0.0 and cl > 0.0:
            try:
                log_ret = math.log(cl / prev_close)
            except ValueError:
                log_ret = None
            if log_ret is not None and math.isfinite(log_ret):
                self._append_return(sym, log_ret)

        if (
            prev_close is not None
            and hi is not None
            and lo is not None
            and prev_close > 0.0
        ):
            tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
            if prev_close != 0.0:
                tr_pct = tr / prev_close
                if math.isfinite(tr_pct):
                    self._append_trange(sym, max(0.0, tr_pct))

        if cl is not None:
            self._last_close[sym] = cl

        value = self._compute(sym)
        self._last_value[sym] = value
        return value

    def value(self, symbol: str, *, metric: Optional[str] = None) -> Optional[float]:
        return self._compute(str(symbol).upper(), metric)

    def last(self, symbol: str, metric: Optional[str] = None) -> Optional[float]:
        sym = str(symbol).upper()
        if metric is None or metric.lower() == self._metric:
            return self._last_value.get(sym)
        return self._compute(sym, metric)

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
    def __init__(
        self,
        sim: ExecutionSimulator,
        *,
        symbol: str,
        timeframe: str,
        source: MarketDataSource,
        run_config: CommonRunConfig | None = None,
    ):
        self.sim = sim
        self.symbol = str(symbol).upper()
        self.timeframe = _ensure_timeframe(timeframe)
        self.interval_ms = _timeframe_to_ms(self.timeframe)
        self.source = source
        self.enforce_closed_bars = (
            run_config.timing.enforce_closed_bars if run_config is not None else True
        )

        latency_cfg = getattr(run_config, "latency", None) if run_config is not None else None
        metric = "sigma"
        window = 120
        if latency_cfg is not None:
            if isinstance(latency_cfg, dict):
                metric = latency_cfg.get("vol_metric", metric) or metric
                window = latency_cfg.get("vol_window", window) or window
            else:
                metric = getattr(latency_cfg, "vol_metric", metric) or metric
                window = getattr(latency_cfg, "vol_window", window) or window
        try:
            window_int = int(window)
        except (TypeError, ValueError):
            window_int = 120
        self._vol_estimator = _VolEstimator(vol_metric=metric, vol_window=window_int)


    @property
    def vol_estimator(self) -> _VolEstimator:
        return self._vol_estimator


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
        try:
            for bar in self.source.stream_bars([self.symbol], self.interval_ms):
                if bar.symbol != self.symbol:
                    continue
                if self.enforce_closed_bars and not getattr(bar, "is_final", True):
                    try:
                        skipped_incomplete_bars.labels(bar.symbol).inc()
                    except Exception:
                        pass
                    continue

                orders: Sequence[Order] = list(provider.on_bar(bar) or [])

                high = float(bar.high)
                low = float(bar.low)
                close = float(bar.close)

                vol_factor = self._vol_estimator.observe(
                    symbol=bar.symbol,
                    high=high,
                    low=low,
                    close=close,
                )

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

                yield rep
        except ValueError as e:
            raise ValueError(f"Market data error: {e}") from e
