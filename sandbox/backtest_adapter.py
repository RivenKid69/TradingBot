# sandbox/backtest_adapter.py
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import clock
from utils_time import floor_to_timeframe, is_bar_closed

from core_contracts import SignalPolicy, PolicyCtx
from core_models import Order, Side
from sandbox.sim_adapter import SimAdapter
from exchange.specs import load_specs, round_price_to_tick


@dataclass
class DynSpreadConfig:
    enabled: bool = True
    base_bps: float = 3.0
    alpha_vol: float = 0.5
    beta_illiquidity: float = 1.0
    vol_mode: str = "hl"                 # "hl" или "ret"
    liq_col: str = "number_of_trades"    # "number_of_trades" или "volume"
    liq_ref: float = 1000.0
    min_bps: float = 1.0
    max_bps: float = 25.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DynSpreadConfig":
        return cls(
            enabled=bool(d.get("enabled", True)),
            base_bps=float(d.get("base_bps", 3.0)),
            alpha_vol=float(d.get("alpha_vol", 0.5)),
            beta_illiquidity=float(d.get("beta_illiquidity", 1.0)),
            vol_mode=str(d.get("vol_mode", "hl")),
            liq_col=str(d.get("liq_col", "number_of_trades")),
            liq_ref=float(d.get("liq_ref", 1000.0)),
            min_bps=float(d.get("min_bps", 1.0)),
            max_bps=float(d.get("max_bps", 25.0)),
        )


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
        self._dyn = DynSpreadConfig.from_dict(dynamic_spread_config or {})
        self._guards = GuardsConfig.from_dict(guards_config or {})
        self._no_trade = NoTradeConfig.from_dict(no_trade_config or {})
        self._timing = TimingConfig.from_dict(timing_config or {})
        self._last_ref_by_symbol: Dict[str, float] = {}

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

    def _compute_vol_factor(self, row: pd.Series, *, ref: float, has_hl: bool) -> float:
        try:
            if self._dyn.vol_mode.lower() == "hl" and has_hl and ("high" in row.index) and ("low" in row.index):
                hi = float(row["high"])
                lo = float(row["low"])
                if ref > 0:
                    return max(0.0, (hi - lo) / float(ref))
            if "ret_1m" in row.index:
                return abs(float(row["ret_1m"]))
        except Exception:
            pass
        sym = str(row.get("symbol", self.sim.symbol)).upper()
        last = self._last_ref_by_symbol.get(sym)
        if last is None or last <= 0 or ref <= 0:
            self._last_ref_by_symbol[sym] = float(ref)
            return 0.0
        v = abs(math.log(float(ref) / float(last)))
        self._last_ref_by_symbol[sym] = float(ref)
        return float(v)

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

    def _synth_quotes(self, *, symbol: str, ref: float, vol_factor: float, liquidity: float) -> (float, float, float):
        base = float(self._dyn.base_bps)
        vol_term = float(self._dyn.alpha_vol) * float(vol_factor) * 10000.0
        illq = 0.0
        if float(self._dyn.liq_ref) > 0 and liquidity == liquidity:
            ratio = max(0.0, (float(self._dyn.liq_ref) - float(liquidity)) / float(self._dyn.liq_ref))
            illq = float(self._dyn.beta_illiquidity) * ratio * base
        spread_bps = base + vol_term + illq
        spread_bps = max(float(self._dyn.min_bps), min(float(self._dyn.max_bps), float(spread_bps)))
        half = float(spread_bps) / 20000.0
        raw_bid = float(ref) * (1.0 - half)
        raw_ask = float(ref) * (1.0 + half)
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
        for o in orders:
            try:
                qty = abs(o.quantity)
                if qty == 0:
                    continue
                side = Side.BUY if str(o.side).upper() == "BUY" else Side.SELL
                po = 0
                if isinstance(o.meta, dict) and "price_offset_ticks" in o.meta:
                    po = int(o.meta.get("price_offset_ticks", 0))
                meta = dict(o.meta)
                meta["price_offset_ticks"] = po
                out.append(replace(o, side=side, quantity=qty, meta=meta))
            except Exception:
                continue
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

        logger = logging.getLogger(__name__)
        skip_cnt = 0
        try:
            from service_signal_runner import skipped_incomplete_bars  # type: ignore
        except Exception:  # pragma: no cover - optional metric
            skipped_incomplete_bars = None

        for _, row in df.iterrows():
            ts = int(row[ts_col])
            sym = str(row[symbol_col]).upper()
            ref = float(row[price_col])

            if self._timing.enforce_closed_bars:
                close_ts = floor_to_timeframe(ts, self._timing.timeframe_ms) + self._timing.timeframe_ms
                if not is_bar_closed(close_ts, clock.now_ms(), self._timing.close_lag_ms):
                    skip_cnt += 1
                    try:
                        logger.info("SKIP_INCOMPLETE_BAR")
                    except Exception:
                        pass
                    if skipped_incomplete_bars is not None:
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

            if self._dyn.enabled:
                vol_factor = float(self._compute_vol_factor(row, ref=ref, has_hl=has_hl))
                liquidity = float(self._compute_liquidity(row))
                bid, ask, spread_bps = self._synth_quotes(symbol=sym, ref=ref, vol_factor=vol_factor, liquidity=liquidity)
            else:
                vol_factor = float("nan")
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

            rep = self.sim.step(
                ts_ms=ts,
                ref_price=ref,
                bid=bid,
                ask=ask,
                vol_factor=vol_factor,
                liquidity=liquidity,
                orders=orders,
            )
            out_reports.append({**rep, "symbol": sym, "ts_ms": ts, "core_order_intents": core_order_intents})
        if skip_cnt:
            try:
                logger.info("Skipped %d incomplete bars", skip_cnt)
            except Exception:
                pass

        return out_reports
