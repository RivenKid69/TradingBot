# sandbox/sim_adapter.py
from __future__ import annotations

import logging
import math
from collections import deque
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Protocol,
    Mapping,
)

from execution_sim import ExecutionSimulator  # type: ignore
from adv_store import ADVStore
from action_proto import ActionProto, ActionType
from core_models import Bar, Order, Side, as_dict
from compat_shims import sim_report_dict_to_core_exec_reports
from event_bus import log_trade_exec as _bus_log_trade_exec
from core_contracts import MarketDataSource
from core_config import CommonRunConfig
from services.monitoring import skipped_incomplete_bars


logger = logging.getLogger(__name__)


def _log_adv_runtime_warnings(
    store: ADVStore,
    symbol: Any,
    adv_cfg: Any,
    *,
    context: str,
) -> None:
    try:
        sym = str(symbol).strip().upper() if symbol is not None else ""
    except Exception:
        sym = ""
    path = store.path
    if not path:
        logger.warning(
            "%s: ADV runtime enabled but dataset path is not configured; using default quote=%s",
            context,
            store.default_quote,
        )
    elif store.is_dataset_stale:
        refresh_days = getattr(adv_cfg, "refresh_days", None)
        logger.warning(
            "%s: ADV dataset %s appears stale; refresh recommended (refresh_days=%s)",
            context,
            path,
            refresh_days,
        )
    base_quote = store.get_adv_quote(sym) if sym else None
    if base_quote is None:
        default_q = store.default_quote
        if default_q is not None:
            logger.warning(
                "%s: ADV quote missing for %s; falling back to default %.3f",
                context,
                sym or "<unknown>",
                default_q,
            )
        else:
            logger.warning(
                "%s: ADV quote missing for %s and no default configured",
                context,
                sym or "<unknown>",
            )


def _attach_adv_runtime(
    sim: ExecutionSimulator,
    run_cfg: CommonRunConfig | None,
    *,
    context: str,
) -> Optional[ADVStore]:
    if run_cfg is None:
        return None
    adv_cfg = getattr(run_cfg, "adv", None)
    if adv_cfg is None or not getattr(adv_cfg, "enabled", False):
        return None
    set_store = getattr(sim, "set_adv_store", None)
    if not callable(set_store):
        logger.warning(
            "%s: ExecutionSimulator lacks set_adv_store(); ADV runtime disabled",
            context,
        )
        return None
    capacity_fraction = getattr(adv_cfg, "capacity_fraction", None)
    bars_override = getattr(adv_cfg, "bars_per_day_override", None)
    extra_block = getattr(adv_cfg, "extra", None)
    if capacity_fraction is None and isinstance(extra_block, Mapping):
        capacity_fraction = extra_block.get("capacity_fraction")
    if bars_override is None and isinstance(extra_block, Mapping):
        bars_override = extra_block.get("bars_per_day_override")
        if bars_override is None:
            bars_override = extra_block.get("bars_per_day")
    existing_store: Optional[ADVStore] = None
    has_store_fn = getattr(sim, "has_adv_store", None)
    if callable(has_store_fn):
        try:
            if bool(has_store_fn()):
                existing_store = getattr(sim, "_adv_store", None)
        except Exception:
            existing_store = None
    if isinstance(existing_store, ADVStore):
        try:
            set_store(
                existing_store,
                enabled=True,
                capacity_fraction=capacity_fraction,
                bars_per_day_override=bars_override,
            )
        except Exception:
            logger.exception(
                "%s: failed to refresh ADV runtime settings on existing store",
                context,
            )
        else:
            _log_adv_runtime_warnings(
                existing_store, getattr(sim, "symbol", None), adv_cfg, context
            )
        return existing_store
    try:
        store = ADVStore(adv_cfg)
    except Exception:
        logger.exception("%s: failed to initialise ADV store from config", context)
        return None
    try:
        set_store(
            store,
            enabled=True,
            capacity_fraction=capacity_fraction,
            bars_per_day_override=bars_override,
        )
    except Exception:
        logger.exception("%s: failed to attach ADV store to simulator", context)
        return None
    _log_adv_runtime_warnings(store, getattr(sim, "symbol", None), adv_cfg, context)
    return store


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
        allowed_metrics = {
            "sigma",
            "atr",
            "atr_pct",
            "atr/price",
            "range",
            "range_ratio",
            "range_ratio_bps",
        }
        if self._metric not in allowed_metrics:
            self._metric = "sigma"
        self._window = max(1, int(vol_window or 1))
        self._returns: Dict[str, Deque[float]] = {}
        self._ret_sumsq: Dict[str, float] = {}
        self._tranges: Dict[str, Deque[float]] = {}
        self._tr_sum: Dict[str, float] = {}
        self._last_close: Dict[str, float] = {}
        self._last_value: Dict[str, Optional[float]] = {}
        self._last_sigma: Dict[str, Optional[float]] = {}
        self._last_atr_pct: Dict[str, Optional[float]] = {}
        self._last_range_ratio: Dict[str, Optional[float]] = {}

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

    def _compute_sigma_raw(self, symbol: str) -> Optional[float]:
        dq = self._returns.get(symbol)
        if not dq:
            return None
        total = max(0.0, self._ret_sumsq.get(symbol, 0.0))
        try:
            return math.sqrt(total / len(dq))
        except ZeroDivisionError:
            return None

    def _compute_atr_raw(self, symbol: str) -> Optional[float]:
        dq = self._tranges.get(symbol)
        if not dq:
            return None
        total = self._tr_sum.get(symbol, 0.0)
        if len(dq) == 0:
            return None
        return max(0.0, total / len(dq))

    def _compute(self, symbol: str, metric: Optional[str] = None) -> Optional[float]:
        metric_key = (metric or self._metric or "").lower()
        if metric_key == "sigma":
            sigma_val = self._compute_sigma_raw(symbol)
            if sigma_val is not None:
                return sigma_val
        elif metric_key in {"atr", "atr_pct", "atr/price"}:
            atr_val = self._compute_atr_raw(symbol)
            if atr_val is not None:
                return atr_val
        elif metric_key in {"range", "range_ratio"}:
            range_val = self._last_range_ratio.get(symbol)
            if range_val is not None:
                return range_val
        elif metric_key == "range_ratio_bps":
            range_val = self._last_range_ratio.get(symbol)
            if range_val is not None:
                return max(0.0, range_val) * 1e4

        # Fallbacks: try sigma first, then ATR, whichever has data.
        sigma_val = self._compute_sigma_raw(symbol)
        if sigma_val is not None:
            return sigma_val
        atr_val = self._compute_atr_raw(symbol)
        if atr_val is not None:
            return atr_val
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

        range_ratio: Optional[float] = None
        if hi is not None and lo is not None:
            span = hi - lo
            if span < 0.0:
                span = abs(span)
            mid_value = (hi + lo) * 0.5
            if mid_value is not None and mid_value != 0.0:
                try:
                    ratio_candidate = span / abs(mid_value)
                except ZeroDivisionError:
                    ratio_candidate = None
                if (
                    ratio_candidate is not None
                    and math.isfinite(ratio_candidate)
                    and ratio_candidate >= 0.0
                ):
                    range_ratio = max(0.0, ratio_candidate)

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

        self._last_range_ratio[sym] = range_ratio
        sigma_val = self._compute_sigma_raw(sym)
        atr_val = self._compute_atr_raw(sym)
        range_val = range_ratio if range_ratio is not None else self._last_range_ratio.get(sym)
        if self._metric == "sigma":
            value = sigma_val if sigma_val is not None else atr_val
        elif self._metric in {"atr", "atr_pct", "atr/price"}:
            value = atr_val if atr_val is not None else sigma_val
        elif self._metric in {"range", "range_ratio"}:
            value = range_val if range_val is not None else (
                sigma_val if sigma_val is not None else atr_val
            )
        elif self._metric == "range_ratio_bps":
            if range_val is not None:
                value = max(0.0, range_val) * 1e4
            else:
                fallback = sigma_val if sigma_val is not None else atr_val
                value = fallback
        else:
            value = atr_val if atr_val is not None else sigma_val
        if value is None:
            value = self._compute(sym)
        self._last_sigma[sym] = sigma_val
        self._last_atr_pct[sym] = atr_val
        self._last_value[sym] = value
        return value

    def value(self, symbol: str, *, metric: Optional[str] = None) -> Optional[float]:
        return self._compute(str(symbol).upper(), metric)

    def last(self, symbol: str, metric: Optional[str] = None) -> Optional[float]:
        sym = str(symbol).upper()
        metric_key = (metric or self._metric or "").lower()
        if metric_key == "sigma":
            val = self._last_sigma.get(sym)
            if val is not None:
                return val
            return self._compute_sigma_raw(sym)
        if metric_key in {"atr", "atr_pct", "atr/price"}:
            val = self._last_atr_pct.get(sym)
            if val is not None:
                return val
            return self._compute_atr_raw(sym)
        if metric_key in {"range", "range_ratio"}:
            val = self._last_range_ratio.get(sym)
            if val is not None:
                return val
            return None
        if metric_key == "range_ratio_bps":
            val = self._last_range_ratio.get(sym)
            if val is not None:
                return max(0.0, val) * 1e4
            return None
        return self._last_value.get(sym)

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
        if run_config is None:
            run_config = getattr(sim, "run_config", None)
        if run_config is None:
            run_config = getattr(sim, "_run_config", None)
        self.run_config = run_config
        self._adv_store = _attach_adv_runtime(
            self.sim, self.run_config, context="sandbox.sim_adapter"
        )
        self.symbol = str(symbol).upper()
        self.timeframe = _ensure_timeframe(timeframe)
        self.interval_ms = _timeframe_to_ms(self.timeframe)
        try:
            if hasattr(sim, "set_intrabar_timeframe_ms"):
                sim.set_intrabar_timeframe_ms(self.interval_ms)
            else:
                setattr(sim, "_intrabar_timeframe_ms", self.interval_ms)
        except Exception:
            setattr(sim, "_intrabar_timeframe_ms", self.interval_ms)
        self.source = source
        self.enforce_closed_bars = (
            run_config.timing.enforce_closed_bars if run_config is not None else True
        )

        close_lag_value: Optional[int] = None
        if run_config is not None:
            timing_cfg = getattr(run_config, "timing", None)
            if timing_cfg is not None:
                candidate = getattr(timing_cfg, "close_lag_ms", None)
                if candidate is None and isinstance(timing_cfg, Mapping):
                    candidate = timing_cfg.get("close_lag_ms")
                try:
                    if candidate is not None:
                        close_lag_value = int(candidate)
                except (TypeError, ValueError):
                    close_lag_value = None
        if close_lag_value is None:
            existing_lag = getattr(self.sim, "close_lag_ms", None)
            try:
                if existing_lag is not None:
                    close_lag_value = int(existing_lag)
            except (TypeError, ValueError):
                close_lag_value = None
        if close_lag_value is not None and close_lag_value < 0:
            close_lag_value = 0
        self.close_lag_ms = close_lag_value if close_lag_value is not None else 0
        if close_lag_value is not None:
            for attr in ("close_lag_ms", "_timing_close_lag_ms"):
                try:
                    setattr(self.sim, attr, int(close_lag_value))
                except Exception:
                    continue

        latency_cfg = (
            getattr(run_config, "latency", None) if run_config is not None else None
        )
        metric = "sigma"
        window = 120
        dyn_metric: Optional[Any] = None
        dyn_window: Optional[Any] = None
        if run_config is not None:
            slippage_cfg = getattr(run_config, "slippage", None)
            dyn_block: Optional[Any] = None
            if isinstance(slippage_cfg, dict):
                dyn_block = slippage_cfg.get("dynamic") or slippage_cfg.get("dynamic_spread")
            else:
                dyn_block = getattr(slippage_cfg, "dynamic", None) or getattr(
                    slippage_cfg, "dynamic_spread", None
                )
            if dyn_block is not None:
                if isinstance(dyn_block, dict):
                    dyn_metric = dyn_block.get("vol_metric", dyn_metric)
                    dyn_window = dyn_block.get("vol_window", dyn_window)
                else:
                    dyn_metric = getattr(dyn_block, "vol_metric", dyn_metric)
                    dyn_window = getattr(dyn_block, "vol_window", dyn_window)
        if dyn_metric:
            metric = dyn_metric
        if dyn_window:
            window = dyn_window
        if (dyn_metric is None or dyn_window is None) and latency_cfg is not None:
            if isinstance(latency_cfg, dict):
                if dyn_metric is None:
                    metric = latency_cfg.get("vol_metric", metric) or metric
                if dyn_window is None:
                    window = latency_cfg.get("vol_window", window) or window
            else:
                if dyn_metric is None:
                    metric = getattr(latency_cfg, "vol_metric", metric) or metric
                if dyn_window is None:
                    window = getattr(latency_cfg, "vol_window", window) or window
        metric_norm = str(metric or "sigma").lower()
        try:
            window_int = int(window)
        except (TypeError, ValueError):
            window_int = 120
        if window_int < 1:
            window_int = 1
        self._vol_metric = metric_norm if metric_norm else "sigma"
        self._vol_window = window_int
        self._vol_estimator = _VolEstimator(
            vol_metric=self._vol_metric, vol_window=self._vol_window
        )


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
             orders: Sequence[Order],
             bar_open: Optional[float] = None,
             bar_high: Optional[float] = None,
             bar_low: Optional[float] = None,
             bar_close: Optional[float] = None,
             bar_timeframe_ms: Optional[int] = None) -> Dict[str, Any]:
        actions = self._to_actions(orders)
        sigma_last = self._vol_estimator.last(self.symbol, metric="sigma")
        atr_last = self._vol_estimator.last(self.symbol, metric="atr_pct")
        range_last = self._vol_estimator.last(self.symbol, metric="range")
        vol_raw_payload: Dict[str, float] = {}
        if sigma_last is not None:
            try:
                vol_raw_payload["sigma"] = float(sigma_last)
            except (TypeError, ValueError):
                pass
        if atr_last is not None:
            try:
                atr_value = float(atr_last)
            except (TypeError, ValueError):
                atr_value = None
            if atr_value is not None:
                vol_raw_payload["atr_pct"] = atr_value
                vol_raw_payload["atr"] = atr_value
                vol_raw_payload["atr/price"] = atr_value
        if range_last is not None:
            try:
                range_value = float(range_last)
            except (TypeError, ValueError):
                range_value = None
            else:
                if math.isfinite(range_value) and range_value >= 0.0:
                    vol_raw_payload["range"] = range_value
                    vol_raw_payload.setdefault("range_ratio", range_value)
                    vol_raw_payload["range_ratio_bps"] = range_value * 1e4
        vol_raw_arg = vol_raw_payload or None
        close_arg = bar_close if bar_close is not None else ref_price
        report = self.sim.run_step(
            ts=ts_ms,
            ref_price=ref_price,
            bid=bid,
            ask=ask,
            vol_factor=vol_factor,
            vol_raw=vol_raw_arg,
            liquidity=liquidity,
            bar_open=bar_open,
            bar_high=bar_high,
            bar_low=bar_low,
            bar_close=close_arg,
            bar_timeframe_ms=bar_timeframe_ms if bar_timeframe_ms is not None else self.interval_ms,
            actions=actions,
        )
        d = report.to_dict()
        if "market_regime" not in d:
            d["market_regime"] = getattr(report, "market_regime", None)

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

                open_price = float(bar.open)
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
                    bar_open=open_price,
                    bar_high=high,
                    bar_low=low,
                    bar_close=close,
                    bar_timeframe_ms=self.interval_ms,
                )

                rep["symbol"] = bar.symbol
                rep["ts_ms"] = int(bar.ts)
                rep["core_orders"] = ([as_dict(o) for o in orders] or [])

                yield rep
        except ValueError as e:
            raise ValueError(f"Market data error: {e}") from e

    # ------------------------------------------------------------------
    # Market regime helpers
    # ------------------------------------------------------------------
    def _market_regime_owner(self) -> Any:
        """Return the underlying object managing market regime state."""

        if hasattr(self.sim, "register_market_regime_listener"):
            return self.sim
        candidate = getattr(self, "_sim", None)
        if candidate is not None and hasattr(
            candidate, "register_market_regime_listener"
        ):
            return candidate
        return None

    def register_market_regime_listener(
        self, callback: Callable[[Any], None]
    ) -> None:
        """Proxy registration to the simulator if it supports regime hooks."""

        owner = self._market_regime_owner()
        if owner is None:
            return
        register = getattr(owner, "register_market_regime_listener", None)
        if callable(register):
            register(callback)

    def current_market_regime(self) -> Any:
        """Return the last known market regime if available."""

        owner = self._market_regime_owner()
        if owner is None:
            return None
        if hasattr(owner, "current_market_regime"):
            try:
                return owner.current_market_regime  # type: ignore[attr-defined]
            except Exception:
                return None
        getter = getattr(owner, "get_market_regime", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                return None
        return getattr(owner, "_last_market_regime", None)
