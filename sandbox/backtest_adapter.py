# sandbox/backtest_adapter.py
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace, field
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import pandas as pd
import clock
from utils_time import bar_close_ms, is_bar_closed

from core_contracts import SignalPolicy, PolicyCtx
from core_models import Order, Side
if TYPE_CHECKING:  # pragma: no cover - typing helper
    from sandbox.sim_adapter import SimAdapter  # type: ignore
else:  # pragma: no cover - runtime placeholder for annotations
    SimAdapter = Any  # type: ignore
from exchange.specs import load_specs, round_price_to_tick
from services.monitoring import skipped_incomplete_bars


@dataclass
class DynSpreadConfig:
    enabled: bool = True
    alpha_bps: Optional[float] = None
    beta_coef: Optional[float] = None
    min_spread_bps: Optional[float] = None
    max_spread_bps: Optional[float] = None
    smoothing_alpha: Optional[float] = None
    fallback_spread_bps: Optional[float] = None
    vol_metric: Optional[str] = None
    vol_window: Optional[int] = None
    vol_mode: Optional[str] = None
    liq_col: Optional[str] = None
    liq_ref: Optional[float] = None
    # legacy aliases kept for compatibility with older configs
    base_bps: Optional[float] = None
    alpha_vol: Optional[float] = None
    beta_illiquidity: Optional[float] = None
    min_bps: Optional[float] = None
    max_bps: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    volatility_metric: Optional[str] = None
    volatility_window: Optional[int] = None
    use_volatility: bool = False
    ema_state: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DynSpreadConfig":
        if not isinstance(d, dict):
            d = {}

        def _first_non_null(*keys: str) -> Any:
            for key in keys:
                if key in d and d[key] is not None:
                    return d[key]
            return None

        def _to_float(value: Any) -> Optional[float]:
            try:
                out = float(value)
            except (TypeError, ValueError):
                return None
            if math.isfinite(out):
                return out
            return None

        def _to_int(value: Any) -> Optional[int]:
            try:
                out = int(value)
            except (TypeError, ValueError):
                return None
            return out

        alpha_val = _to_float(_first_non_null("alpha_bps", "alpha", "base_bps"))
        beta_val = _to_float(_first_non_null("beta_coef", "beta", "alpha_vol"))
        min_val = _to_float(_first_non_null("min_spread_bps", "min_bps"))
        max_val = _to_float(_first_non_null("max_spread_bps", "max_bps"))
        smoothing_val = _to_float(_first_non_null("smoothing_alpha", "smoothing"))
        fallback_val = _to_float(d.get("fallback_spread_bps"))
        vol_metric_val = _first_non_null("vol_metric", "volatility_metric")
        vol_window_val = _to_int(_first_non_null("vol_window", "volatility_window"))
        vol_mode_val = _first_non_null("vol_mode", None)
        liq_col_val = _first_non_null("liq_col", None)
        liq_ref_val = _to_float(d.get("liq_ref"))

        if vol_mode_val is None and vol_metric_val is None:
            vol_mode_val = "hl"
        if liq_col_val is None:
            liq_col_val = "number_of_trades"
        if liq_ref_val is None:
            liq_ref_val = 1000.0

        cfg = cls(
            enabled=bool(d.get("enabled", True)),
            alpha_bps=alpha_val,
            beta_coef=beta_val,
            min_spread_bps=min_val,
            max_spread_bps=max_val,
            smoothing_alpha=smoothing_val,
            fallback_spread_bps=fallback_val,
            vol_metric=str(vol_metric_val) if vol_metric_val is not None else None,
            vol_window=vol_window_val,
            vol_mode=str(vol_mode_val) if vol_mode_val is not None else None,
            liq_col=str(liq_col_val) if liq_col_val is not None else None,
            liq_ref=liq_ref_val,
            base_bps=_to_float(_first_non_null("base_bps", "alpha_bps", "alpha")),
            alpha_vol=_to_float(d.get("alpha_vol")),
            beta_illiquidity=_to_float(d.get("beta_illiquidity")),
            min_bps=_to_float(d.get("min_bps")),
            max_bps=_to_float(d.get("max_bps")),
            alpha=_to_float(_first_non_null("alpha", "alpha_bps", "base_bps")),
            beta=_to_float(_first_non_null("beta", "beta_coef", "alpha_vol")),
            volatility_metric=str(vol_metric_val) if vol_metric_val is not None else None,
            volatility_window=vol_window_val,
            use_volatility=bool(d.get("use_volatility", False)),
        )

        if cfg.base_bps is None:
            cfg.base_bps = cfg.alpha_bps
        if cfg.alpha is None:
            cfg.alpha = cfg.alpha_bps
        if cfg.beta is None:
            cfg.beta = cfg.beta_coef
        if cfg.min_bps is None:
            cfg.min_bps = cfg.min_spread_bps
        if cfg.max_bps is None:
            cfg.max_bps = cfg.max_spread_bps
        if cfg.volatility_metric is None:
            cfg.volatility_metric = cfg.vol_metric
        if cfg.volatility_window is None:
            cfg.volatility_window = cfg.vol_window

        return cfg


@dataclass
class GuardsConfig:
    min_history_bars: int = 0
    gap_cooldown_bars: int = 0
    gap_threshold_ms: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GuardsConfig":
        return cls(
            min_history_bars=int(d.get("min_history_bars", 0)),
            gap_cooldown_bars=int(d.get("gap_cooldown_bars", 0)),
            gap_threshold_ms=int(d["gap_threshold_ms"]) if d.get("gap_threshold_ms") is not None else None,
        )


@dataclass
class NoTradeConfig:
    funding_buffer_min: int = 0
    daily_utc: List[str] = None
    custom_ms: List[Dict[str, int]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NoTradeConfig":
        return cls(
            funding_buffer_min=int(d.get("funding_buffer_min", 0)),
            daily_utc=list(d.get("daily_utc", []) or []),
            custom_ms=list(d.get("custom_ms", []) or []),
        )


@dataclass
class TimingConfig:
    enforce_closed_bars: bool = True
    timeframe_ms: int = 60_000
    close_lag_ms: int = 2000

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TimingConfig":
        return cls(
            enforce_closed_bars=bool(d.get("enforce_closed_bars", True)),
            timeframe_ms=int(d.get("timeframe_ms", 60_000)),
            close_lag_ms=int(d.get("close_lag_ms", 2000)),
        )


class BacktestAdapter:
    """
    Простой бэктестер по уже собранной таблице (например, data/train.parquet):
      Требуемые колонки:
        - ts_ms: int
        - symbol: str (можно один символ)
        - ref_price: float (или mid/close — главное, что передаём это же в сим)
      Остальные колонки считаются фичами и прокидываются в стратегию.

    Доработки:
      - Динамический спред/слиппедж без стакана.
      - Биржевые ограничения: bid/ask к tickSize; order_qty к stepSize; notional < minNotional — отбрасываем.
      - Гварды: холодный старт и пауза после гэпа.
      - Частотный кулдаун: блок новых сигналов чаще, чем раз в X секунд.
      - Чёрные окна (no_trade): ежедневные окна UTC, буфер вокруг funding (00:00/08:00/16:00 UTC), кастомные окна по ts_ms.
    """
    def __init__(
        self,
        policy: SignalPolicy,
        sim_bridge: SimAdapter,
        dynamic_spread_config: Optional[Dict[str, Any]] = None,
        exchange_specs_path: Optional[str] = None,
        guards_config: Optional[Dict[str, Any]] = None,
        signal_cooldown_s: int = 0,
        no_trade_config: Optional[Dict[str, Any]] = None,
        timing_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.policy = policy
        self.sim = sim_bridge
        self._keep_zero_quantity_orders = self._detect_keep_zero_quantity_orders(sim_bridge)
        sim_obj = getattr(self.sim, "sim", None)
        getter = getattr(sim_obj, "get_spread_bps", None)
        fallback_getter = getattr(sim_obj, "_slippage_get_spread", None)
        self._sim_has_spread_getter = callable(getter) or callable(fallback_getter)
        self._dyn = DynSpreadConfig.from_dict(dynamic_spread_config or {})
        self._guards = GuardsConfig.from_dict(guards_config or {})
        self._no_trade = NoTradeConfig.from_dict(no_trade_config or {})
        self._timing = TimingConfig.from_dict(timing_config or {})

        # спецификации биржи
        self._specs, self._specs_meta = load_specs(exchange_specs_path or "")
        if self._specs_meta:
            logging.getLogger(__name__).info("Loaded exchange specs metadata: %s", self._specs_meta)

        # состояние гвардов
        self._hist_bars: Dict[str, int] = {}
        self._cooldown_left: Dict[str, int] = {}
        self._last_ts: Dict[str, int] = {}

        # частотный кулдаун (сек → мс) и время последнего разрешённого сигнала
        self._signal_cooldown_ms: int = max(0, int(signal_cooldown_s)) * 1000
        self._last_signal_ts: Dict[str, int] = {}

        # распарсенные ежедневные окна (минуты от начала суток, UTC)
        self._daily_windows_min: List[tuple] = self._parse_daily_windows(self._no_trade.daily_utc or [])

    # --------------------- helpers: spread, liquidity ---------------------

    @staticmethod
    def _detect_keep_zero_quantity_orders(sim_bridge: SimAdapter) -> bool:
        """Detect whether zero-quantity orders should be preserved."""

        def _type_name(obj: Any) -> str:
            try:
                return type(obj).__name__
            except Exception:
                return ""

        def _module_name(obj: Any) -> str:
            try:
                module = type(obj).__module__
            except Exception:
                module = ""
            return str(module or "")

        explicit_flag = getattr(sim_bridge, "keep_zero_quantity_orders", None)
        if explicit_flag is None:
            explicit_flag = getattr(sim_bridge, "allow_zero_quantity_orders", None)
        if explicit_flag is not None:
            return bool(explicit_flag)

        bridge_name = _type_name(sim_bridge).lower()
        bridge_module = _module_name(sim_bridge).lower()
        if bridge_name == "barbacktestsimbridge":
            return True
        if "barbacktest" in bridge_name or "bar_backtest" in bridge_module:
            return True

        executor = getattr(sim_bridge, "executor", None)
        executor_name = _type_name(executor).lower()
        executor_module = _module_name(executor).lower()
        if executor_name == "barexecutor":
            return True
        if executor_name.endswith("barexecutor"):
            return True
        if "impl_bar_executor" in executor_module or "bar_executor" in executor_module:
            return True

        return False


    def _compute_vol_factor(self, row: pd.Series, *, ref: float, has_hl: bool) -> float:
        estimator = getattr(self.sim, "vol_estimator", None)
        if estimator is None:
            return 0.0

        sym = str(row.get("symbol", self.sim.symbol)).upper()
        hi = None
        lo = None
        if has_hl:
            try:
                if "high" in row.index:
                    hi = float(row["high"])
                if "low" in row.index:
                    lo = float(row["low"])
            except Exception:
                hi = None
                lo = None

        close_val = ref
        if "close" in row.index:
            try:
                close_val = float(row["close"])
            except Exception:
                close_val = ref

        estimator.observe(symbol=sym, high=hi, low=lo, close=close_val)

        metric_key = getattr(self._dyn, "vol_metric", None) or getattr(
            self._dyn, "volatility_metric", None
        )
        if not metric_key:
            vol_mode = str(getattr(self._dyn, "vol_mode", "")).lower()
            if vol_mode == "hl":
                metric_key = "range_ratio_bps"
            elif vol_mode == "ret":
                metric_key = "sigma"
        if metric_key is not None:
            metric_key = str(metric_key).strip().lower()
            if not metric_key:
                metric_key = None
        value = estimator.value(sym, metric=metric_key)
        if value is None:
            last = estimator.last(sym, metric=metric_key)
            if last is not None:
                return float(last)
            return 0.0
        return float(value)

    def _compute_liquidity(self, row: pd.Series) -> float:
        try:
            key = str(self._dyn.liq_col)
            if key in row.index:
                return float(row[key])
            if "volume" in row.index:
                return float(row["volume"])
        except Exception:
            pass
        return 1.0

    def _synth_quotes(
        self, *, symbol: str, ref: float, vol_factor: float, liquidity: float
    ) -> (float, float, float):
        # ``vol_factor`` already corresponds to the configured volatility metric
        # (for example ``range_ratio_bps``) in ``SimAdapter``.  Legacy configs may
        # still provide ``base_bps``/``alpha_vol`` which we treat as aliases for
        # the new ``alpha_bps``/``beta_coef`` knobs.

        def _finite_float(value: Any) -> Optional[float]:
            try:
                out = float(value)
            except (TypeError, ValueError):
                return None
            if math.isfinite(out):
                return out
            return None

        alpha = _finite_float(self._dyn.alpha_bps)
        if alpha is None:
            alpha = _finite_float(self._dyn.base_bps)

        fallback = _finite_float(self._dyn.fallback_spread_bps)
        if fallback is None:
            fallback = alpha

        beta = _finite_float(self._dyn.beta_coef)
        if beta is None:
            beta = _finite_float(self._dyn.alpha_vol)
        if beta is None:
            beta = 0.0

        metric_value = _finite_float(vol_factor)
        if metric_value is not None and metric_value < 0.0:
            metric_value = 0.0

        if metric_value is None:
            spread_candidate = fallback if fallback is not None else alpha
        else:
            base_component = alpha if alpha is not None else fallback
            if base_component is None:
                base_component = 0.0
            spread_candidate = base_component + beta * metric_value

        if spread_candidate is None:
            spread_candidate = fallback if fallback is not None else 0.0

        min_bps = _finite_float(self._dyn.min_spread_bps)
        if min_bps is None:
            min_bps = _finite_float(self._dyn.min_bps)
        if min_bps is not None:
            spread_candidate = max(min_bps, spread_candidate)

        max_bps = _finite_float(self._dyn.max_spread_bps)
        if max_bps is None:
            max_bps = _finite_float(self._dyn.max_bps)
        if max_bps is not None:
            spread_candidate = min(max_bps, spread_candidate)

        smoothing = _finite_float(self._dyn.smoothing_alpha)
        if smoothing is not None:
            smoothing = min(max(smoothing, 0.0), 1.0)
            prev = self._dyn.ema_state.get(symbol)
            if prev is None or not math.isfinite(prev):
                ema_val = spread_candidate
            else:
                ema_val = smoothing * spread_candidate + (1.0 - smoothing) * prev
            self._dyn.ema_state[symbol] = float(ema_val)
            spread_candidate = ema_val
        else:
            self._dyn.ema_state.pop(symbol, None)

        spread_bps = _finite_float(spread_candidate)
        if spread_bps is None:
            spread_bps = fallback if fallback is not None else 0.0
        if spread_bps < 0.0:
            spread_bps = 0.0

        ref_price = _finite_float(ref)
        if ref_price is None or ref_price <= 0.0:
            ref_price = max(ref_price or 0.0, 0.0)

        half = spread_bps / 20000.0
        raw_bid = float(ref_price) * (1.0 - half)
        raw_ask = float(ref_price) * (1.0 + half)
        bid = round_price_to_tick(symbol, raw_bid, self._specs, side="BID")
        ask = round_price_to_tick(symbol, raw_ask, self._specs, side="ASK")
        if ask <= bid:
            rb = round_price_to_tick(symbol, bid, self._specs, side="BID")
            ra = round_price_to_tick(symbol, bid, self._specs, side="ASK")
            ask = max(ra, bid * 1.000001)
        return bid, ask, spread_bps

    # --------------------- helpers: exchange constraints ---------------------

    def _apply_exchange_rules_to_orders(
        self,
        symbol: str,
        ref_price: float,
        orders: Sequence[Order],
    ) -> List[Order]:
        """Применяет биржевые ограничения к ордерам.

        Фильтрует ордера с нулевым количеством и нормализует ``side``, ``quantity``
        и ``price_offset_ticks`` (в ``meta``).
        """

        out: List[Order] = []
        logger = logging.getLogger(__name__)
        for o in orders:
            try:
                qty = abs(o.quantity)
            except Exception:
                logger.warning("Failed to normalise order quantity: %s", o, exc_info=True)
                continue

            if qty == 0 and not self._keep_zero_quantity_orders:
                continue

            side_raw = getattr(o.side, "value", o.side)
            try:
                side = Side(str(side_raw).upper())
            except Exception:
                logger.warning("Unable to determine order side for %s", o, exc_info=True)
                continue

            po = 0
            if isinstance(o.meta, dict) and "price_offset_ticks" in o.meta:
                try:
                    po = int(o.meta.get("price_offset_ticks", 0))
                except Exception:
                    logger.debug("Invalid price_offset_ticks for %s", o, exc_info=True)
                    po = 0

            try:
                meta = dict(o.meta)
            except Exception:
                logger.debug("Failed to copy order meta for %s", o, exc_info=True)
                meta = {}

            meta["price_offset_ticks"] = po

            try:
                normalised = replace(o, side=side, quantity=qty, meta=meta)
            except Exception:
                logger.warning("Failed to normalise order %s", o, exc_info=True)
                continue

            out.append(normalised)
        return out

    # --------------------- helpers: guards & cooldown ---------------------

    def _apply_guards(self, sym: str, ts: int) -> bool:
        h = self._hist_bars.get(sym, 0)
        cd = self._cooldown_left.get(sym, 0)
        last = self._last_ts.get(sym)

        if last is not None:
            dt = int(ts) - int(last)
            thr = self._guards.gap_threshold_ms if self._guards.gap_threshold_ms is not None else 90000
            if dt > max(0, int(thr)):
                self._cooldown_left[sym] = int(self._guards.gap_cooldown_bars or 0)

        self._hist_bars[sym] = h + 1
        self._last_ts[sym] = int(ts)

        if self._hist_bars[sym] < int(self._guards.min_history_bars or 0):
            return False
        if self._cooldown_left.get(sym, 0) > 0:
            self._cooldown_left[sym] = max(0, self._cooldown_left[sym] - 1)
            return False
        return True

    def _apply_signal_cooldown(
        self, sym: str, ts: int, orders: Sequence[Order]
    ) -> List[Order]:
        if self._signal_cooldown_ms <= 0 or not orders:
            return list(orders)
        last_sig = self._last_signal_ts.get(sym)
        if last_sig is not None and (int(ts) - int(last_sig) < self._signal_cooldown_ms):
            return []
        self._last_signal_ts[sym] = int(ts)
        return list(orders)

    # --------------------- helpers: no-trade windows ---------------------

    @staticmethod
    def _parse_daily_windows(windows: List[str]) -> List[tuple]:
        """
        Преобразует строки "HH:MM-HH:MM" в список кортежей (start_minute, end_minute).
        Окна без склейки, без поддержи 'через полночь' (ожидаем start <= end).
        """
        out: List[tuple] = []
        for w in windows:
            try:
                a, b = str(w).strip().split("-")
                sh, sm = a.split(":")
                eh, em = b.split(":")
                smin = int(sh) * 60 + int(sm)
                emin = int(eh) * 60 + int(em)
                if 0 <= smin <= 1440 and 0 <= emin <= 1440 and smin <= emin:
                    out.append((smin, emin))
            except Exception:
                continue
        return out

    def _in_daily_window(self, ts_ms: int) -> bool:
        if not self._daily_windows_min:
            return False
        mins = int((ts_ms // 60000) % 1440)
        for smin, emin in self._daily_windows_min:
            if smin <= mins < emin:
                return True
        return False

    def _in_funding_buffer(self, ts_ms: int) -> bool:
        buf_min = int(self._no_trade.funding_buffer_min or 0)
        if buf_min <= 0:
            return False
        sec_day = int((ts_ms // 1000) % 86400)
        marks = [0, 8 * 3600, 16 * 3600]
        for m in marks:
            if abs(sec_day - m) <= buf_min * 60:
                return True
        return False

    def _in_custom_window(self, ts_ms: int) -> bool:
        for w in (self._no_trade.custom_ms or []):
            try:
                s = int(w["start_ts_ms"])
                e = int(w["end_ts_ms"])
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Invalid custom window {w}: expected integer 'start_ts_ms' and 'end_ts_ms'"
                ) from exc

            if s >= e:
                raise ValueError(
                    f"Invalid custom window {w}: start_ts_ms ({s}) must be < end_ts_ms ({e})"
                )

            if s <= int(ts_ms) <= e:
                return True

        return False

    def _no_trade_block(self, ts_ms: int) -> bool:
        return self._in_daily_window(ts_ms) or self._in_funding_buffer(ts_ms) or self._in_custom_window(ts_ms)

    # --------------------- main loop ---------------------

    def run(self, df: pd.DataFrame, *, ts_col: str = "ts_ms", symbol_col: str = "symbol", price_col: str = "ref_price") -> List[Dict[str, Any]]:
        if df.empty:
            return []
        need = [ts_col, symbol_col, price_col]
        for c in need:
            if c not in df.columns:
                raise ValueError(f"Отсутствует колонка '{c}' для бэктеста")
        df = df.sort_values([symbol_col, ts_col]).reset_index(drop=True)

        out_reports: List[Dict[str, Any]] = []
        has_hl = ("high" in df.columns) and ("low" in df.columns)
        has_open = "open" in df.columns

        logger = logging.getLogger(__name__)
        skip_cnt = 0

        for _, row in df.iterrows():
            ts = int(row[ts_col])
            sym = str(row[symbol_col]).upper()
            ref = float(row[price_col])

            bar_open = None
            if has_open:
                try:
                    bar_open = float(row["open"])
                except Exception:
                    bar_open = None
            bar_high = None
            bar_low = None
            if has_hl:
                try:
                    bar_high = float(row["high"])
                except Exception:
                    bar_high = None
                try:
                    bar_low = float(row["low"])
                except Exception:
                    bar_low = None

            if self._timing.enforce_closed_bars:
                close_ts = bar_close_ms(ts, self._timing.timeframe_ms)
                if not is_bar_closed(close_ts, clock.now_ms(), self._timing.close_lag_ms):
                    skip_cnt += 1
                    try:
                        logger.info("SKIP_INCOMPLETE_BAR")
                    except Exception:
                        pass
                    try:
                        skipped_incomplete_bars.labels(sym).inc()
                    except Exception:
                        pass
                    continue

            feats: Dict[str, Any] = {}
            for c in df.columns:
                if c in (ts_col, symbol_col, price_col):
                    continue
                feats[c] = row[c]

            allow = self._apply_guards(sym, ts)
            if allow and self._no_trade_block(ts):
                allow = False

            features = {**feats, "ref_price": ref}
            ctx = PolicyCtx(ts=ts, symbol=sym)
            if allow:
                orders = list(self.policy.decide(features, ctx))
            else:
                orders = []

            need_spread = self._dyn.enabled and not self._sim_has_spread_getter
            if self._dyn.enabled or self._sim_has_spread_getter:
                vol_factor = float(self._compute_vol_factor(row, ref=ref, has_hl=has_hl))
            else:
                vol_factor = float("nan")
            if need_spread:
                liquidity = float(self._compute_liquidity(row))
                bid, ask, spread_bps = self._synth_quotes(
                    symbol=sym, ref=ref, vol_factor=vol_factor, liquidity=liquidity
                )
            else:
                liquidity = float("nan")
                bid = None
                ask = None

            orders = self._apply_signal_cooldown(sym, ts, orders)
            orders = self._apply_exchange_rules_to_orders(sym, ref, orders)

            # стандартизированный выход стратегии: OrderIntent[]
            from order_shims import OrderContext, orders_to_order_intents  # локальный импорт во избежание циклов
            _ctx = OrderContext(
                ts_ms=int(ts),
                symbol=str(sym),
                ref_price=float(ref),
                max_position_abs_base=float(self._specs.get(sym).step_size if self._specs.get(sym) else 1.0),  # нижняя оценка; точный объём задаст исполнитель
                tick_size=(self._specs.get(sym).tick_size if self._specs.get(sym) else None),
                price_offset_ticks=0,
                tif="GTC",
                client_tag=None,
                round_qty_fn=None,
            )
            core_order_intents = [it.to_dict() for it in orders_to_order_intents(orders, _ctx)]

            setter = getattr(self.sim, "set_active_symbol", None)
            if callable(setter):
                try:
                    setter(sym)
                except Exception:
                    pass

            rep = self.sim.step(
                ts_ms=ts,
                ref_price=ref,
                bid=bid,
                ask=ask,
                vol_factor=vol_factor,
                liquidity=liquidity,
                orders=orders,
                bar_open=bar_open,
                bar_high=bar_high,
                bar_low=bar_low,
                bar_close=ref,
                bar_timeframe_ms=self.sim.interval_ms,
            )
            out_reports.append({**rep, "symbol": sym, "ts_ms": ts, "core_order_intents": core_order_intents})
        if skip_cnt:
            try:
                logger.info("Skipped %d incomplete bars", skip_cnt)
            except Exception:
                pass

        return out_reports
