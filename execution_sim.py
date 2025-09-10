# execution_sim.py
from __future__ import annotations

"""
ExecutionSimulator v2

Цели этого переписанного модуля:
1) Ввести ЕДИНУЮ квантизацию цен/количеств и проверку биржевых фильтров Binance:
   - PRICE_FILTER / LOT_SIZE / MIN_NOTIONAL / PERCENT_PRICE_BY_SIDE
   - Квантизация идентична live-адаптеру (см. sim/quantizer.py)
2) Сохранить простой интерфейс очереди с искусственной задержкой:
   - submit(proto, now_ts=None) -> client_order_id
   - pop_ready(now_ts=None, ref_price: float | None = None) -> ExecReport
3) Работать как с внешним LOB (если он передан), так и без него (простая модель):
   - Для MARKET без LOB исполняем по ref_price (если задан) или по last_ref_price.
   - Для LIMIT без LOB исполняем только если есть abs_price; иначе добавляем в new_order_ids (эмуляция размещения).

Примечания по совместимости:
- Тип действия берётся из action_proto.ActionType, если модуль доступен.
- Если action_proto сломан/недоступен, используется локальная «минимальная» замена.

Важно: этот модуль НЕ добавляет комиссии и слиппедж — они будут подключены отдельными шагами.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import math
import time

try:
    import numpy as np
except Exception:  # минимальная замена на случай отсутствия numpy на этапе интеграции
    class _R:
        def __init__(self, seed=0): self.s = seed
        def randint(self, a, b=None, size=None): return a
    class np:  # type: ignore
        @staticmethod
        def random(seed=None): return _R(seed)
        class randomState: ...
        class RandomState:
            def __init__(self, seed=0): self._r = _R(seed)
            def randint(self, a, b=None, size=None): return self._r.randint(a, b, size)

# --- Совместимость с ActionProto/ActionType ---
try:
    from action_proto import ActionType, ActionProto  # type: ignore
except Exception:
    from enum import IntEnum
    @dataclass
    class ActionProto:  # минимально необходимый набор полей
        action_type: int  # 0=HOLD,1=MARKET,2=LIMIT
        volume_frac: float = 0.0
        price_offset_ticks: int = 0
        ttl_steps: int = 0
        abs_price: Optional[float] = None  # опционально, если доступна абсолютная цена лимитки
        tif: str = "GTC"
        client_tag: Optional[str] = None

    class ActionType(IntEnum):
        HOLD = 0
        MARKET = 1
        LIMIT = 2


# --- Импорт квантизатора, комиссий/funding и слиппеджа ---
try:
    from sim.quantizer import Quantizer, load_filters
except Exception:
    Quantizer = None  # type: ignore

try:
    from sim.fees import FeesModel, FundingCalculator, FundingEvent
except Exception:
    FeesModel = None  # type: ignore
    FundingCalculator = None  # type: ignore
    FundingEvent = None  # type: ignore

try:
    from sim.slippage import (
        SlippageConfig,
        estimate_slippage_bps,
        apply_slippage_price,
        compute_spread_bps_from_quotes,
        mid_from_quotes,
    )
except Exception:
    SlippageConfig = None  # type: ignore
    estimate_slippage_bps = None  # type: ignore
    apply_slippage_price = None  # type: ignore
    compute_spread_bps_from_quotes = None  # type: ignore
    mid_from_quotes = None  # type: ignore

# --- Импорт исполнителей ---
try:
    from sim.execution_algos import (
        BaseExecutor,
        MarketChild,
        TakerExecutor,
        TWAPExecutor,
        POVExecutor,
        MarketOpenH1Executor,
        VWAPExecutor,
    )
except Exception:
    try:
        from execution_algos import (
            BaseExecutor,
            MarketChild,
            TakerExecutor,
            TWAPExecutor,
            POVExecutor,
            MarketOpenH1Executor,
            VWAPExecutor,
        )
    except Exception:
        BaseExecutor = None  # type: ignore
        MarketChild = None  # type: ignore
        TakerExecutor = None  # type: ignore
        TWAPExecutor = None  # type: ignore
        POVExecutor = None  # type: ignore
        MarketOpenH1Executor = None  # type: ignore
        VWAPExecutor = None  # type: ignore

# --- Импорт модели латентности ---
try:
    from sim.latency import LatencyModel
except Exception:
    LatencyModel = None  # type: ignore

# --- Импорт менеджера рисков ---
try:
    from sim.risk import RiskManager, RiskEvent
except Exception:
    RiskManager = None  # type: ignore
    RiskEvent = None  # type: ignore

# --- Импорт логгера ---
try:
    from sim.logging import LogWriter, LogConfig
except Exception:
    LogWriter = None  # type: ignore
    LogConfig = None  # type: ignore

@dataclass
class ExecTrade:
    ts: int
    side: str  # "BUY"|"SELL"
    price: float
    qty: float
    notional: float
    liquidity: str  # "taker"|"maker"
    proto_type: int  # см. ActionType
    client_order_id: int
    fee: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    latency_ms: int = 0
    latency_spike: bool = False
    tif: str = "GTC"
    ttl_steps: int = 0


@dataclass
class SimStepReport:
    trades: List[ExecTrade] = field(default_factory=list)
    cancelled_ids: List[int] = field(default_factory=list)
    new_order_ids: List[int] = field(default_factory=list)
    fee_total: float = 0.0
    new_order_pos: List[int] = field(default_factory=list)
    funding_cashflow: float = 0.0
    funding_events: List[FundingEvent] = field(default_factory=list)  # type: ignore
    position_qty: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    equity: float = 0.0
    mark_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    mtm_price: float = 0.0
    risk_events: List[RiskEvent] = field(default_factory=list)  # type: ignore
    risk_paused_until_ms: int = 0
    spread_bps: Optional[float] = None
    vol_factor: Optional[float] = None
    liquidity: Optional[float] = None
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_timeout_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "trades": [t.__dict__ for t in self.trades],
            "cancelled_ids": list(self.cancelled_ids),
            "new_order_ids": list(self.new_order_ids),
            "fee_total": float(self.fee_total),
            "new_order_pos": list(self.new_order_pos),
            "funding_cashflow": float(self.funding_cashflow),
            "funding_events": [fe.__dict__ for fe in self.funding_events],
            "position_qty": float(self.position_qty),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "equity": float(self.equity),
            "mark_price": float(self.mark_price),
            "bid": float(self.bid),
            "ask": float(self.ask),
            "mtm_price": float(self.mtm_price),
            "risk_events": [re.__dict__ for re in self.risk_events],
            "risk_paused_until_ms": int(self.risk_paused_until_ms),
            "spread_bps": float(self.spread_bps) if self.spread_bps is not None else None,
            "vol_factor": float(self.vol_factor) if self.vol_factor is not None else None,
            "liquidity": float(self.liquidity) if self.liquidity is not None else None,
            "latency_p50_ms": float(self.latency_p50_ms),
            "latency_p95_ms": float(self.latency_p95_ms),
            "latency_timeout_ratio": float(self.latency_timeout_ratio),
        }


# Alias for compatibility with older interfaces
ExecReport = SimStepReport


@dataclass
class Pending:
    proto: ActionProto
    client_order_id: int
    remaining_lat: int
    timestamp: int
    lat_ms: int = 0
    timeout: bool = False
    spike: bool = False


class _LatencyQueue:
    def __init__(self, latency_steps: int = 0):
        self.latency_steps = max(0, int(latency_steps))
        self._q: List[Pending] = []

    def push(self, p: Pending) -> None:
        self._q.append(p)

    def pop_ready(self) -> Tuple[List[Pending], List[Pending]]:
        ready: List[Pending] = []
        cancelled: List[Pending] = []
        rest: List[Pending] = []
        for p in self._q:
            if p.remaining_lat <= 0:
                if p.timeout:
                    cancelled.append(p)
                else:
                    ready.append(p)
            else:
                p.remaining_lat -= 1
                rest.append(p)
        self._q = rest
        return ready, cancelled

    def clear(self) -> None:
        self._q.clear()


class ExecutionSimulator:
    """
    Очередь действий + простое исполнение.
    Поддерживает квантизацию Binance-фильтрами.
    """

    def __init__(self, *,
                 symbol: str = "BTCUSDT",
                 latency_steps: int = 0,
                 seed: int = 0,
                 lob: Optional[Any] = None,
                 filters_path: Optional[str] = "data/binance_filters.json",
                 enforce_ppbs: bool = True,
                 strict_filters: bool = True,
                 fees_config: Optional[dict] = None,
                 funding_config: Optional[dict] = None,
                 slippage_config: Optional[dict] = None,
                 execution_config: Optional[dict] = None,
                 execution_profile: Optional[str] = None,
                 execution_params: Optional[dict] = None,
                 latency_config: Optional[dict] = None,
                 pnl_config: Optional[dict] = None,
                 risk_config: Optional[dict] = None,
                 logging_config: Optional[dict] = None,
                 run_config: Any = None):
        self.symbol = str(symbol).upper()
        self.latency_steps = int(max(0, latency_steps))
        self.seed = int(seed)
        try:
            self._rng = np.random.RandomState(seed)  # type: ignore
        except Exception:
            try:
                self._rng = np.RandomState(seed)  # type: ignore
            except Exception:
                self._rng = None  # type: ignore
        self._q = _LatencyQueue(self.latency_steps)
        self._next_cli_id = 1
        self.lob = lob
        self._last_ref_price: Optional[float] = None
        self._next_h1_open_price: Optional[float] = None
        self.run_id: str = str(getattr(run_config, "run_id", "sim") or "sim")
        self.step_ms: int = int(getattr(run_config, "step_ms", 1000)) if run_config is not None else 1000
        if self.step_ms <= 0:
            self.step_ms = 1
        self._cancelled_on_submit: List[int] = []
        self._ttl_orders: List[Tuple[int, int]] = []

        # квантайзер — опционально
        self.quantizer: Optional[Quantizer] = None
        self.enforce_ppbs = bool(enforce_ppbs)
        self.strict_filters = bool(strict_filters)
        try:
            if Quantizer is not None and filters_path:
                filters = load_filters(filters_path)
                if filters:
                    self.quantizer = Quantizer(filters, strict=strict_filters)
        except Exception:
            # не ломаемся, если файл отсутствует; квантизация просто не активна
            self.quantizer = None

        # комиссии и funding
        self.fees = FeesModel.from_dict(fees_config or {}) if FeesModel is not None else None
        self.funding = (
            FundingCalculator(**(funding_config or {"enabled": False}))
            if FundingCalculator is not None
            else None
        )

        # слиппедж
        self.slippage_cfg = (
            SlippageConfig.from_dict(slippage_config or {}) if SlippageConfig is not None else None
        )

        # исполнители
        self._execution_cfg = dict(execution_config or {})
        self.execution_profile = str(execution_profile) if execution_profile is not None else ""
        self.execution_params: dict = dict(execution_params or {})
        self._executor: Optional[BaseExecutor] = None
        self._build_executor()

        # латентность
        self.latency = (
            LatencyModel(**(latency_config or {})) if LatencyModel is not None else None
        )

        # риск-менеджер
        self.risk = (
            RiskManager.from_dict(risk_config or {}) if RiskManager is not None else None
        )

        # состояние позиции и PnL
        self.position_qty: float = 0.0
        self._avg_entry_price: Optional[float] = None
        self.realized_pnl_cum: float = 0.0
        self.fees_cum: float = 0.0
        self.funding_cum: float = 0.0
        self._pnl_mark_to: str = str((pnl_config or {}).get("mark_to", "side")).lower()

        # последний снапшот рынка для оценки spread/vol/liquidity
        self._last_bid: Optional[float] = None
        self._last_ask: Optional[float] = None
        self._last_spread_bps: Optional[float] = None
        self._last_vol_factor: Optional[float] = None
        self._last_liquidity: Optional[float] = None

        # логирование
        self._logger = (
            LogWriter(LogConfig.from_dict(logging_config or {}), run_id=self.run_id)
            if LogWriter is not None
            else None
        )
        self._step_counter: int = 0

        # накопители для VWAP
        self._vwap_pv: float = 0.0
        self._vwap_vol: float = 0.0
        self._vwap_hour: Optional[int] = None
        self._last_hour_vwap: Optional[float] = None

    def set_execution_profile(self, profile: str, params: dict | None = None) -> None:
        """Установить профиль исполнения и параметры."""
        self.execution_profile = str(profile)
        self.execution_params = dict(params or {})
        self._build_executor()

    def set_quantizer(self, q: Quantizer) -> None:
        self.quantizer = q

    def set_symbol(self, symbol: str) -> None:
        self.symbol = str(symbol).upper()

    def set_ref_price(self, price: float) -> None:
        self._last_ref_price = float(price)

    def set_next_open_price(self, price: float) -> None:
        self._next_h1_open_price = float(price)

    def set_market_snapshot(
        self,
        *,
        bid: Optional[float],
        ask: Optional[float],
        spread_bps: Optional[float] = None,
        vol_factor: Optional[float] = None,
        liquidity: Optional[float] = None,
        ts_ms: Optional[int] = None,
        trade_price: Optional[float] = None,
        trade_qty: Optional[float] = None,
    ) -> None:
        """
        Установить последний рыночный снапшот: bid/ask (для вычисления spread и mid),
        vol_factor (например ATR% за бар), liquidity (например rolling_volume_shares).
        """
        self._last_bid = float(bid) if bid is not None else None
        self._last_ask = float(ask) if ask is not None else None
        if spread_bps is not None:
            self._last_spread_bps = float(spread_bps)
        else:
            if compute_spread_bps_from_quotes is not None and self.slippage_cfg is not None:
                self._last_spread_bps = compute_spread_bps_from_quotes(bid=self._last_bid, ask=self._last_ask, cfg=self.slippage_cfg)
            else:
                self._last_spread_bps = None
        self._last_vol_factor = float(vol_factor) if vol_factor is not None else None
        self._last_liquidity = float(liquidity) if liquidity is not None else None
        if self._last_ref_price is None:
            if mid_from_quotes is not None:
                mid = mid_from_quotes(bid=self._last_bid, ask=self._last_ask)
                if mid is not None:
                    self._last_ref_price = float(mid)
        if ts_ms is not None:
            price_tick = trade_price if trade_price is not None else self._last_ref_price
            qty_tick = trade_qty if trade_qty is not None else liquidity
            if price_tick is not None and qty_tick is not None:
                self._vwap_on_tick(int(ts_ms), float(price_tick), float(qty_tick))
    def _build_executor(self) -> None:
        """
        Построить исполнителя согласно self._execution_cfg.
        """
        if TakerExecutor is None:
            self._executor = None
            return
        profile = str(getattr(self, "execution_profile", "")).upper()
        if profile == "MKT_OPEN_NEXT_H1" and MarketOpenH1Executor is not None:
            self._executor = MarketOpenH1Executor()
            return
        cfg = dict(self._execution_cfg or {})
        algo = str(cfg.get("algo", "TAKER")).upper()
        if algo == "TWAP":
            tw = dict(cfg.get("twap", {}))
            parts = int(tw.get("parts", 6))
            interval = int(tw.get("child_interval_s", 600))
            self._executor = TWAPExecutor(parts=parts, child_interval_s=interval)
        elif algo == "VWAP":
            self._executor = VWAPExecutor()
        elif algo == "POV":
            pv = dict(cfg.get("pov", {}))
            part = float(pv.get("participation", 0.10))
            interval = int(pv.get("child_interval_s", 60))
            min_not = float(pv.get("min_child_notional", 20.0))
            self._executor = POVExecutor(participation=part, child_interval_s=interval, min_child_notional=min_not)
        else:
            self._executor = TakerExecutor()

    def _vwap_on_tick(self, ts_ms: int, price: Optional[float], volume: Optional[float]) -> None:
        hour_ms = 3_600_000
        hour = int(ts_ms // hour_ms)
        if self._vwap_hour is None:
            self._vwap_hour = hour
        elif hour != self._vwap_hour:
            if self._vwap_vol > 0.0:
                self._last_hour_vwap = self._vwap_pv / self._vwap_vol
            else:
                self._last_hour_vwap = None
            self._vwap_pv = 0.0
            self._vwap_vol = 0.0
            self._vwap_hour = hour
        if price is not None and volume is not None and volume > 0.0:
            self._vwap_pv += float(price) * float(volume)
            self._vwap_vol += float(volume)

    def _apply_trade_inventory(self, side: str, price: float, qty: float) -> float:
        """
        Обновляет позицию/среднюю цену и возвращает Δреализованного PnL (без учёта комиссии).
        Логика:
          - BUY закрывает шорт (если pos<0) или увеличивает лонг (если pos>=0).
          - SELL закрывает лонг (если pos>0) или увеличивает шорт (если pos<=0).
        """
        realized = 0.0
        q = float(abs(qty))
        px = float(price)
        pos = float(self.position_qty)
        avg = self._avg_entry_price

        if str(side).upper() == "BUY":
            if pos < 0.0:
                close_qty = min(q, -pos)
                if avg is not None:
                     realized += (avg - px) * close_qty
                pos += close_qty
                q_rem = q - close_qty
                if q_rem > 0.0:
                    self.position_qty = q_rem
                    self._avg_entry_price = px
                else:
                    self.position_qty = pos
                    if self.position_qty == 0.0:
                        self._avg_entry_price = None
            else:
                new_pos = pos + q
                if new_pos > 0.0:
                    if pos > 0.0 and avg is not None:
                        self._avg_entry_price = (avg * pos + px * q) / new_pos
                    else:
                        self._avg_entry_price = px
                else:
                    self._avg_entry_price = None
                self.position_qty = new_pos
        else:
            if pos > 0.0:
                close_qty = min(q, pos)
                if avg is not None:
                    realized += (px - avg) * close_qty
                pos -= close_qty
                q_rem = q - close_qty
                if q_rem > 0.0:
                    self.position_qty = -q_rem
                    self._avg_entry_price = px
                else:
                    self.position_qty = pos
                    if self.position_qty == 0.0:
                        self._avg_entry_price = None
            else:
                new_pos = pos - q
                if new_pos < 0.0:
                    if pos < 0.0 and avg is not None:
                        self._avg_entry_price = (avg * (-pos) + px * q) / (-new_pos)
                    else:
                        self._avg_entry_price = px
                else:
                    self._avg_entry_price = None
                self.position_qty = new_pos

        self.realized_pnl_cum += float(realized)
        return float(realized)

    def _mark_price(self, ref: Optional[float], bid: Optional[float], ask: Optional[float]) -> Optional[float]:
        """
        Возвращает цену маркировки в зависимости от режима:
          - "side": long → bid, short → ask, flat → mid/ref
          - "mid": (bid+ask)/2 или ref
          - "bid": bid или ref
          - "ask": ask или ref
        """
        mode = str(self._pnl_mark_to).lower() if hasattr(self, "_pnl_mark_to") else "side"
        b = bid if (bid is not None) else None
        a = ask if (ask is not None) else None
        if mode == "mid":
            if b is not None and a is not None:
                return float((float(b) + float(a)) / 2.0)
            return float(ref) if ref is not None else None
        if mode == "bid":
            return float(b) if b is not None else (float(ref) if ref is not None else None)
        if mode == "ask":
            return float(a) if a is not None else (float(ref) if ref is not None else None)
        # mode == "side"
        if self.position_qty > 0.0:
            return float(b) if b is not None else (float(ref) if ref is not None else None)
        if self.position_qty < 0.0:
            return float(a) if a is not None else (float(ref) if ref is not None else None)
        if b is not None and a is not None:
            return float((float(b) + float(a)) / 2.0)
        return float(ref) if ref is not None else None

    def _unrealized_pnl(self, mark_price: Optional[float]) -> float:
        """
        Возвращает нереализованный PnL относительно средней цены позиции.
        """
        if mark_price is None or self._avg_entry_price is None or self.position_qty == 0.0:
            return 0.0
        mp = float(mark_price)
        ap = float(self._avg_entry_price)
        if self.position_qty > 0.0:
            return float((mp - ap) * self.position_qty)
        else:
            return float((ap - mp) * (-self.position_qty))

    # ---- очередь ----
    def submit(self, proto: ActionProto, now_ts: Optional[int] = None) -> int:
        cid = self._next_cli_id
        self._next_cli_id += 1
        lat_ms = 0
        timeout = False
        spike = False
        remaining = self.latency_steps
        if self.latency is not None:
            try:
                d = self.latency.sample()
                lat_ms = int(d.get("total_ms", 0))
                timeout = bool(d.get("timeout", False))
                spike = bool(d.get("spike", False))
                remaining = int(lat_ms // int(self.step_ms))
            except Exception:
                lat_ms = 0
                timeout = False
                spike = False
                remaining = self.latency_steps
        if timeout:
            self._cancelled_on_submit.append(cid)
            return cid
        self._q.push(Pending(
            proto=proto,
            client_order_id=cid,
            remaining_lat=remaining,
            timestamp=int(now_ts or int(time.time()*1000)),
            lat_ms=int(lat_ms),
            timeout=bool(timeout),
            spike=bool(spike),
        ))
        return cid

    def _ref(self, ref_price: Optional[float]) -> Optional[float]:
        if ref_price is not None:
            self._last_ref_price = float(ref_price)
        return self._last_ref_price

    def _apply_filters_market(self, side: str, qty: float, ref_price: Optional[float]) -> float:
        """
        Применить LOT_SIZE / MIN_NOTIONAL для рыночной заявки.
        Возвращает квантованное qty (может быть 0.0).
        """
        if not self.quantizer:
            return float(qty)
        if ref_price is None:
            # нет цены — не можем проверить minNotional; просто квантуем qty
            return self.quantizer.quantize_qty(self.symbol, qty)
        q = self.quantizer.quantize_qty(self.symbol, qty)
        q = self.quantizer.clamp_notional(self.symbol, ref_price, q)
        return q

    def _apply_filters_limit(self, side: str, price: float, qty: float, ref_price: Optional[float]) -> Tuple[float, float, bool]:
        """
        Применить PRICE_FILTER / LOT_SIZE / MIN_NOTIONAL / PPBS к лимитной заявке.
        Возвращает (price, qty, ok_ppbs).
        """
        if not self.quantizer:
            return float(price), float(qty), True
        p = self.quantizer.quantize_price(self.symbol, price)
        q = self.quantizer.quantize_qty(self.symbol, qty)
        if ref_price is not None:
            q = self.quantizer.clamp_notional(self.symbol, p if p > 0 else ref_price, q)
        ok = True
        if self.enforce_ppbs and ref_price is not None:
            ok = self.quantizer.check_percent_price_by_side(self.symbol, side, p, ref_price)
        return p, q, ok

    # ---- исполнение ----
    def pop_ready(self, now_ts: Optional[int] = None, ref_price: Optional[float] = None) -> ExecReport:
        ready, timed_out = self._q.pop_ready()
        trades: List[ExecTrade] = []
        cancelled_ids: List[int] = list(self._cancelled_on_submit)
        self._cancelled_on_submit = []
        cancelled_ids.extend(int(p.client_order_id) for p in timed_out)
        if self.lob and hasattr(self.lob, "decay_ttl_and_cancel"):
            try:
                expired = self.lob.decay_ttl_and_cancel()
                cancelled_ids.extend(int(x) for x in expired)
            except Exception:
                pass
        else:
            ttl_alive: List[Tuple[int, int]] = []
            for oid, ttl in self._ttl_orders:
                ttl -= 1
                if ttl <= 0:
                    cancelled_ids.append(int(oid))
                    if self.lob and hasattr(self.lob, "remove_order"):
                        try:
                            self.lob.remove_order(int(oid))
                        except Exception:
                            pass
                else:
                    ttl_alive.append((oid, ttl))
            self._ttl_orders = ttl_alive
        new_order_ids: List[int] = []
        new_order_pos: List[int] = []
        fee_total: float = 0.0

        ts = int(now_ts or int(time.time()*1000))
        ref = self._ref(ref_price)
        self._vwap_on_tick(ts, ref, self._last_liquidity)

        for p in ready:
            proto = p.proto
            atype = int(getattr(proto, "action_type", ActionType.HOLD))
            ttl_steps = int(getattr(proto, "ttl_steps", 0))
            tif = str(getattr(proto, "tif", "GTC")).upper()
            # HOLD
            if atype == ActionType.HOLD:
                continue

            # MARKET
            if atype == ActionType.MARKET:
                is_buy = bool(getattr(proto, "volume_frac", 0.0) > 0.0)
                side = "BUY" if is_buy else "SELL"
                qty_raw = abs(float(getattr(proto, "volume_frac", 0.0)))
                qty_total = self._apply_filters_market(side, qty_raw, ref)
                if qty_total <= 0.0:
                    cancelled_ids.append(int(p.client_order_id))
                    continue

                if ref is None or not math.isfinite(ref):
                    cancelled_ids.append(int(p.client_order_id))
                    continue

                # риск: пауза/клампинг размера перед планом
                risk_events_local: List[RiskEvent] = []
                if self.risk is not None:
                    adj_qty = self.risk.pre_trade_adjust(ts_ms=ts, side=side, intended_qty=qty_total, price=ref, position_qty=self.position_qty)
                    risk_events_local.extend(self.risk.pop_events())
                    qty_total = float(adj_qty)
                    if qty_total <= 0.0:
                        cancelled_ids.append(int(p.client_order_id))
                        # накопим события риска
                        # (дальше они будут добавлены к отчёту)
                        continue

                # планирование ребёнков (intra-bar)
                executor = self._executor if self._executor is not None else TakerExecutor()
                snapshot = {
                    "bid": self._last_bid,
                    "ask": self._last_ask,
                    "mid": ( (self._last_bid + self._last_ask) / 2.0 ) if (self._last_bid is not None and self._last_ask is not None) else None,
                    "spread_bps": self._last_spread_bps,
                    "vol_factor": self._last_vol_factor,
                    "liquidity": self._last_liquidity,
                    "ref_price": ref,
                }
                plan = executor.plan_market(now_ts_ms=ts, side=side, target_qty=qty_total, snapshot=snapshot)

                # если план пуст — отклоняем
                if not plan:
                    cancelled_ids.append(int(p.client_order_id))
                    continue

                lat_ms = int(p.lat_ms)
                lat_spike = bool(p.spike)

                for child in plan:
                    ts_fill = int(ts + int(child.ts_offset_ms) + lat_ms)
                    q_child = float(child.qty)
                    if q_child <= 0.0:
                        continue

                    # риск: рейт-лимит на отправку «детей»
                    if self.risk is not None:
                        if not self.risk.can_send_order(ts_fill):
                            # пропускаем этого ребёнка (дросселирование), событие запишем
                            self.risk._emit(ts_fill, "THROTTLE", "order throttled by rate limit", ts_ms=int(ts_fill))
                            # не вызываем on_new_order() в этом случае
                            continue
                        # отметить отправку
                        self.risk.on_new_order(ts_fill)

                    # квантайзер и minNotional для ребёнка
                    if self.quantizer is not None:
                        q_child = self.quantizer.quantize_qty(self.symbol, q_child)
                        q_child = self.quantizer.clamp_notional(self.symbol, ref, q_child)
                        if q_child <= 0.0:
                            continue
                    # базовая котировка
                    if isinstance(executor, VWAPExecutor):
                        self._vwap_on_tick(ts_fill, None, None)
                        base_price = (
                            self._last_hour_vwap
                            if self._last_hour_vwap is not None
                            else ref
                        )
                        filled_price = float(base_price) if base_price is not None else float(ref)
                    elif (
                        str(getattr(self, "execution_profile", "")).upper()
                        == "MKT_OPEN_NEXT_H1"
                        and self._next_h1_open_price is not None
                    ):
                        filled_price = float(self._next_h1_open_price)
                    else:
                        base_price = self._last_ask if side == "BUY" else self._last_bid
                        if base_price is None:
                            base_price = ref
                        filled_price = float(base_price) if base_price is not None else float(ref)

                    # слиппедж на ребёнка
                    slip_bps = 0.0
                    sbps = self._last_spread_bps
                    vf = self._last_vol_factor
                    liq_override = child.liquidity_hint
                    liq = float(liq_override) if (liq_override is not None) else self._last_liquidity
                    cfg_slip = self.execution_params.get("slippage_bps") if isinstance(self.execution_params, dict) else None
                    if cfg_slip is not None:
                        slip_bps = float(cfg_slip)
                        if apply_slippage_price is not None:
                            filled_price = apply_slippage_price(side=side, quote_price=filled_price, slippage_bps=slip_bps)
                    elif self.slippage_cfg is not None and estimate_slippage_bps is not None and apply_slippage_price is not None:
                        slip_bps = estimate_slippage_bps(
                            spread_bps=sbps,
                            size=q_child,
                            liquidity=liq,
                            vol_factor=vf,
                            cfg=self.slippage_cfg,
                        )
                        filled_price = apply_slippage_price(side=side, quote_price=filled_price, slippage_bps=slip_bps)

                    # комиссия
                    fee = 0.0
                    if self.fees is not None:
                        fee = self.fees.compute(side=side, price=filled_price, qty=q_child, liquidity="taker")
                    fee_total += float(fee)
                    self.fees_cum += float(fee)

                    # обновить позицию с расчётом реализованного PnL
                    _ = self._apply_trade_inventory(side=side, price=filled_price, qty=q_child)

                    trades.append(ExecTrade(
                        ts=ts_fill,
                        side=side,
                        price=filled_price,
                        qty=q_child,
                        notional=filled_price * q_child,
                        liquidity="taker",
                        proto_type=atype,
                        client_order_id=p.client_order_id,
                        fee=float(fee),
                        slippage_bps=float(slip_bps),
                        spread_bps=float(sbps if sbps is not None else (self.slippage_cfg.default_spread_bps if self.slippage_cfg is not None else 0.0)),
                        latency_ms=int(p.lat_ms),
                        latency_spike=bool(p.spike),
                        tif=tif,
                        ttl_steps=ttl_steps,
                    ))
                continue
            # Определение направления и базовой цены для прочих типов
            is_buy = bool(getattr(proto, "volume_frac", 0.0) > 0.0)
            side = "BUY" if is_buy else "SELL"
            qty = abs(float(getattr(proto, "volume_frac", 0.0)))
            base_price = self._last_ask if side == "BUY" else self._last_bid
            if base_price is None:
                base_price = ref
            filled_price = float(base_price) if base_price is not None else float(ref)

            # слиппедж
            slip_bps = 0.0
            sbps = self._last_spread_bps
            vf = self._last_vol_factor
            liq = self._last_liquidity
            if (
                self.slippage_cfg is not None
                and estimate_slippage_bps is not None
                and apply_slippage_price is not None
            ):
                slip_bps = estimate_slippage_bps(
                    spread_bps=sbps,
                    size=qty,
                    liquidity=liq,
                    vol_factor=vf,
                    cfg=self.slippage_cfg,
                )
                filled_price = apply_slippage_price(
                    side=side, quote_price=filled_price, slippage_bps=slip_bps
                )

            # комиссия
            fee = 0.0
            if self.fees is not None:
                fee = self.fees.compute(side=side, price=filled_price, qty=qty, liquidity="taker")
            fee_total += float(fee)

            # обновить позицию
            if side == "BUY":
                self.position_qty += float(qty)
            else:
                self.position_qty -= float(qty)

            trades.append(ExecTrade(
                ts=ts,
                side=side,
                price=filled_price,
                qty=qty,
                notional=filled_price * qty,
                liquidity="taker",
                proto_type=atype,
                client_order_id=p.client_order_id,
                fee=float(fee),
                slippage_bps=float(slip_bps),
                spread_bps=float(sbps if sbps is not None else (self.slippage_cfg.default_spread_bps if self.slippage_cfg is not None else 0.0)),
                latency_ms=int(p.lat_ms),
                latency_spike=bool(p.spike),
                tif=tif,
                ttl_steps=ttl_steps,
            ))
            continue

            # LIMIT
            if atype == ActionType.LIMIT:
                is_buy = bool(getattr(proto, "volume_frac", 0.0) > 0.0)
                side = "BUY" if is_buy else "SELL"
                qty_raw = abs(float(getattr(proto, "volume_frac", 0.0)))

                # Определяем лимитную цену
                abs_price = getattr(proto, "abs_price", None)
                if abs_price is None:
                    # нет абсолютной цены в proto — попробуем использовать ref_price как базу
                    if ref is None:
                        # ничего не можем сделать — считаем, что заявка размещена (эмуляция)
                        new_order_ids.append(int(p.client_order_id))
                        new_order_pos.append(0)
                        continue
                    # без знания tickSize в тиках используем abs_price=ref (реальную оффсет-логику добавим позже)
                    abs_price = float(ref)

                price_q, qty_q, ok = self._apply_filters_limit(side, float(abs_price), qty_raw, ref)
                if qty_q <= 0.0 or not ok:
                    cancelled_ids.append(int(p.client_order_id))
                    continue

                filled = False
                liquidity_role = "taker"
                filled_price = float(price_q)
                exec_qty = qty_q
                if self._last_bid is not None or self._last_ask is not None:
                    best_ask = self._last_ask
                    best_bid = self._last_bid
                    if side == "BUY":
                        if best_ask is not None and price_q >= best_ask:
                            filled_price = float(best_ask)
                            liquidity_role = "taker"
                            if self._last_liquidity is not None:
                                exec_qty = min(qty_q, float(self._last_liquidity))
                            filled = exec_qty > 0.0
                        elif best_ask is not None and price_q < best_ask:
                            filled_price = float(price_q)
                            liquidity_role = "maker"
                            filled = True
                    else:  # SELL
                        if best_bid is not None and price_q <= best_bid:
                            filled_price = float(best_bid)
                            liquidity_role = "taker"
                            if self._last_liquidity is not None:
                                exec_qty = min(qty_q, float(self._last_liquidity))
                            filled = exec_qty > 0.0
                        elif best_bid is not None and price_q > best_bid:
                            filled_price = float(price_q)
                            liquidity_role = "maker"
                            filled = True

                if filled and liquidity_role == "taker":
                    if tif == "FOK" and exec_qty + 1e-12 < qty_q:
                        cancelled_ids.append(int(p.client_order_id))
                        continue
                    fee = 0.0
                    if self.fees is not None:
                        fee = self.fees.compute(side=side, price=filled_price, qty=exec_qty, liquidity=liquidity_role)
                    fee_total += float(fee)
                    _ = self._apply_trade_inventory(side=side, price=filled_price, qty=exec_qty)
                    sbps = self._last_spread_bps
                    trades.append(ExecTrade(
                        ts=ts,
                        side=side,
                        price=filled_price,
                        qty=exec_qty,
                        notional=filled_price * exec_qty,
                        liquidity=liquidity_role,
                        proto_type=atype,
                        client_order_id=p.client_order_id,
                        fee=float(fee),
                        slippage_bps=0.0,
                        spread_bps=float(sbps if sbps is not None else (self.slippage_cfg.default_spread_bps if self.slippage_cfg is not None else 0.0)),
                        latency_ms=int(p.lat_ms),
                        latency_spike=bool(p.spike),
                        tif=tif,
                        ttl_steps=ttl_steps,
                    ))
                    if exec_qty + 1e-12 < qty_q:
                        if tif == "IOC":
                            cancelled_ids.append(int(p.client_order_id))
                            continue
                        qty_q = qty_q - exec_qty
                        filled = False
                        liquidity_role = "maker"
                    else:
                        continue

                if filled and liquidity_role == "maker":
                    if tif in ("IOC", "FOK"):
                        cancelled_ids.append(int(p.client_order_id))
                        continue
                    fee = 0.0
                    if self.fees is not None:
                        fee = self.fees.compute(side=side, price=filled_price, qty=qty_q, liquidity=liquidity_role)
                    fee_total += float(fee)
                    _ = self._apply_trade_inventory(side=side, price=filled_price, qty=qty_q)
                    sbps = self._last_spread_bps
                    trades.append(ExecTrade(
                        ts=ts,
                        side=side,
                        price=filled_price,
                        qty=qty_q,
                        notional=filled_price * qty_q,
                        liquidity=liquidity_role,
                        proto_type=atype,
                        client_order_id=p.client_order_id,
                        fee=float(fee),
                        slippage_bps=0.0,
                        spread_bps=float(sbps if sbps is not None else (self.slippage_cfg.default_spread_bps if self.slippage_cfg is not None else 0.0)),
                        latency_ms=int(p.lat_ms),
                        latency_spike=bool(p.spike),
                        tif=tif,
                        ttl_steps=ttl_steps,
                    ))
                    continue

                if tif in ("IOC", "FOK"):
                    cancelled_ids.append(int(p.client_order_id))
                    continue

                if self.lob and hasattr(self.lob, "add_limit_order"):
                    try:
                        oid, qpos = self.lob.add_limit_order(is_buy, float(price_q), float(qty_q), ts, True)
                        if oid:
                            new_order_ids.append(int(oid))
                            new_order_pos.append(int(qpos) if qpos is not None else 0)
                            if ttl_steps > 0:
                                ttl_set = False
                                if hasattr(self.lob, "set_order_ttl"):
                                    try:
                                        ttl_set = bool(self.lob.set_order_ttl(int(oid), int(ttl_steps)))
                                    except Exception:
                                        ttl_set = False
                                if not ttl_set:
                                    self._ttl_orders.append((int(oid), int(ttl_steps)))
                    except Exception:
                        cancelled_ids.append(int(p.client_order_id))
                else:
                    new_order_ids.append(int(p.client_order_id))
                    new_order_pos.append(0)
                    if ttl_steps > 0:
                        self._ttl_orders.append((int(p.client_order_id), int(ttl_steps)))
                continue

            # прочее — no-op
            cancelled_ids.append(int(p.client_order_id))

        # funding: начислить по текущей позиции и актуальной рыночной цене (используем ref как mark)
        funding_cashflow = 0.0
        funding_events_list = []
        if self.funding is not None:
            fc, events = self.funding.accrue(position_qty=self.position_qty, mark_price=ref, now_ts_ms=ts)
            funding_cashflow = float(fc)
            funding_events_list = list(events or [])
            self.funding_cum += float(fc)

        # mark-to-... и PnL
        mark_p = self._mark_price(ref=ref, bid=self._last_bid, ask=self._last_ask)
        unrl = self._unrealized_pnl(mark_p)
        eq = float(self.realized_pnl_cum + unrl - self.fees_cum + self.funding_cum)

        # риск: обновить дневной PnL и возможную паузу
        risk_events_all: List[RiskEvent] = []
        risk_paused_until = 0
        if self.risk is not None:
            try:
                self.risk.on_mark(ts_ms=ts, equity=eq)
                risk_events_all.extend(self.risk.pop_events())
                risk_paused_until = int(self.risk.paused_until_ms)
            except Exception:
                # не рушим исполнение
                pass

        lat_stats = {"p50_ms": 0.0, "p95_ms": 0.0, "timeout_rate": 0.0}
        if self.latency is not None:
            try:
                lat_stats = self.latency.stats()
                self.latency.reset_stats()
            except Exception:
                lat_stats = {"p50_ms": 0.0, "p95_ms": 0.0, "timeout_rate": 0.0}

        report = SimStepReport(
            trades=trades,
            cancelled_ids=cancelled_ids,
            new_order_ids=new_order_ids,
            fee_total=fee_total,
            new_order_pos=new_order_pos,
            funding_cashflow=funding_cashflow,
            funding_events=funding_events_list,  # type: ignore
            position_qty=float(self.position_qty),
            realized_pnl=float(self.realized_pnl_cum),
            unrealized_pnl=float(unrl),
            equity=float(eq),
            mark_price=float(mark_p if mark_p is not None else 0.0),
            bid=float(self._last_bid) if self._last_bid is not None else 0.0,
            ask=float(self._last_ask) if self._last_ask is not None else 0.0,
            mtm_price=float(mark_p if mark_p is not None else 0.0),
            risk_events=risk_events_all,  # type: ignore
            risk_paused_until_ms=int(risk_paused_until),
            spread_bps=self._last_spread_bps,
            vol_factor=self._last_vol_factor,
            liquidity=self._last_liquidity,
            latency_p50_ms=float(lat_stats.get("p50_ms", 0.0)),
            latency_p95_ms=float(lat_stats.get("p95_ms", 0.0)),
            latency_timeout_ratio=float(lat_stats.get("timeout_rate", 0.0)),
        )

# логирование
        try:
            if self._logger is not None:
                self._logger.append(report, symbol=self.symbol, ts_ms=ts)
                self._step_counter += 1
        except Exception:
            # не ломаем симуляцию из-за проблем с логом
            pass

        return report

    # Совместимость с интерфейсами некоторых обёрток
    def run_step(self, *,
                 ts: int,
                 ref_price: float | None,
                 bid: float | None = None,
                 ask: float | None = None,
                 vol_factor: float | None = None,
                 liquidity: float | None = None,
                 trade_price: float | None = None,
                 trade_qty: float | None = None,
                 actions: list[tuple[object, object]] | None = None) -> "ExecReport":
        """
        Универсальный публичный шаг симуляции.
          - Обновляет рыночный снапшот.
          - Обрабатывает список действий: [(ActionType, proto), ...].
          - Возвращает ExecReport с трейдами и PnL-компонентами.
        Примечания:
          - Поддержан тип ActionType.MARKET. Другие типы будут отклонены.
          - proto должен иметь атрибут volume_frac (знак = направление).
        """
        # --- обновить рыночный снапшот ---
        self._last_bid = float(bid) if bid is not None else None
        self._last_ask = float(ask) if ask is not None else None
        self._last_vol_factor = float(vol_factor) if vol_factor is not None else None
        self._last_liquidity = float(liquidity) if liquidity is not None else None
        self._last_spread_bps = None
        try:
            if compute_spread_bps_from_quotes is not None and self.slippage_cfg is not None:
                self._last_spread_bps = compute_spread_bps_from_quotes(
                    bid=self._last_bid,
                    ask=self._last_ask,
                    cfg=self.slippage_cfg,
                )
        except Exception:
            self._last_spread_bps = None
        self._last_ref_price = float(ref_price) if ref_price is not None else None
        price_tick = trade_price if trade_price is not None else self._last_ref_price
        qty_tick = trade_qty if trade_qty is not None else liquidity
        if price_tick is not None and qty_tick is not None:
            self._vwap_on_tick(int(ts), float(price_tick), float(qty_tick))

        # --- инициализация аккамуляторов ---
        trades: list[ExecTrade] = []
        cancelled_ids: list[int] = []
        new_order_ids: list[int] = []
        new_order_pos: list[int] = []
        fee_total: float = 0.0

        # --- обработать действия ---
        acts = list(actions or [])
        for atype, proto in acts:
            # оформление client_order_id
            cli_id = int(self._next_cli_id)
            self._next_cli_id += 1
            new_order_ids.append(cli_id)

            # только MARKET
            if str(getattr(atype, "name", getattr(atype, "__class__", type(atype)))).upper().endswith("MARKET") or str(atype).upper().endswith("MARKET"):
                # определить сторону и величину
                vol = float(getattr(proto, "volume_frac", 0.0))
                is_buy = bool(vol > 0.0)
                side = "BUY" if is_buy else "SELL"
                qty_raw = abs(float(vol))
                ttl_steps = int(getattr(proto, "ttl_steps", 0))
                tif = str(getattr(proto, "tif", "GTC")).upper()
                ref = self._last_ref_price
                if ref is None:
                    cancelled_ids.append(int(cli_id))
                    continue

                # применить фильтры рынка (квантизация/minNotional и т.п. внутри вспом. функции)
                qty_total = self._apply_filters_market(side, qty_raw, ref)
                if qty_total <= 0.0:
                    cancelled_ids.append(int(cli_id))
                    continue

                # риск: корректировка/пауза
                if self.risk is not None:
                    adj_qty = self.risk.pre_trade_adjust(ts_ms=ts, side=side, intended_qty=qty_total, price=ref, position_qty=self.position_qty)
                    qty_total = float(adj_qty)
                    for _e in self.risk.pop_events():
                        # события риска будут добавлены позже через on_mark(); здесь просто очищаем очередь
                        pass
                    if qty_total <= 0.0:
                        cancelled_ids.append(int(cli_id))
                        continue

                # планирование исполнения
                executor = self._executor if self._executor is not None else TakerExecutor()
                snapshot = {
                    "bid": self._last_bid,
                    "ask": self._last_ask,
                    "mid": ((self._last_bid + self._last_ask) / 2.0) if (self._last_bid is not None and self._last_ask is not None) else None,
                    "spread_bps": self._last_spread_bps,
                    "vol_factor": self._last_vol_factor,
                    "liquidity": self._last_liquidity,
                    "ref_price": ref,
                }
                plan = executor.plan_market(now_ts_ms=ts, side=side, target_qty=qty_total, snapshot=snapshot)
                if not plan:
                    cancelled_ids.append(int(cli_id))
                    continue

                # пройтись по детям с учётом латентности/слиппеджа/комиссий/инвентаря/рейт-лимита
                for child in plan:
                    ts_fill = int(ts + int(child.ts_offset_ms))
                    q_child = float(child.qty)
                    if q_child <= 0.0:
                        continue

                    # риск: дросселирование
                    if self.risk is not None:
                        if not self.risk.can_send_order(ts_fill):
                            self.risk._emit(ts_fill, "THROTTLE", "order throttled by rate limit", ts_ms=int(ts_fill))
                            continue
                        self.risk.on_new_order(ts_fill)

                    # квантайзер и minNotional
                    if self.quantizer is not None:
                        q_child = self.quantizer.quantize_qty(self.symbol, q_child)
                        q_child = self.quantizer.clamp_notional(self.symbol, ref, q_child)
                        if q_child <= 0.0:
                            continue

                    # латентность
                    lat_ms = 0
                    lat_spike = False
                    if self.latency is not None:
                        d = self.latency.sample()
                        lat_ms = int(d.get("total_ms", 0))
                        lat_spike = bool(d.get("spike", False))
                        if bool(d.get("timeout", False)):
                            cancelled_ids.append(int(cli_id))
                            continue
                    ts_fill = int(ts_fill + lat_ms)
                    # цена исполнения
                    if isinstance(executor, VWAPExecutor):
                        self._vwap_on_tick(ts_fill, None, None)
                        base_price = (
                            self._last_hour_vwap
                            if self._last_hour_vwap is not None
                            else ref
                        )
                        filled_price = float(base_price) if base_price is not None else float(ref)
                    elif (
                        str(getattr(self, "execution_profile", "")).upper()
                        == "MKT_OPEN_NEXT_H1"
                        and self._next_h1_open_price is not None
                    ):
                        filled_price = float(self._next_h1_open_price)
                    else:
                        base_price = self._last_ask if side == "BUY" else self._last_bid
                        if base_price is None:
                            base_price = ref
                        filled_price = float(base_price) if base_price is not None else float(ref)
                    slip_bps = 0.0
                    sbps = self._last_spread_bps
                    vf = self._last_vol_factor
                    liq = self._last_liquidity
                    if self.slippage_cfg is not None and estimate_slippage_bps is not None and apply_slippage_price is not None:
                        slip_bps = estimate_slippage_bps(
                            spread_bps=sbps,
                            size=q_child,
                            liquidity=liq,
                            vol_factor=vf,
                            cfg=self.slippage_cfg,
                        )
                        filled_price = apply_slippage_price(side=side, quote_price=filled_price, slippage_bps=slip_bps)

                    # комиссия
                    fee = 0.0
                    if self.fees is not None:
                        fee = self.fees.compute(side=side, price=filled_price, qty=q_child, liquidity="taker")
                    fee_total += float(fee)
                    self.fees_cum += float(fee)

                    # инвентарь + реализованный PnL
                    _ = self._apply_trade_inventory(side=side, price=filled_price, qty=q_child)

                    # запись трейда
                    trades.append(ExecTrade(
                        ts=ts_fill,
                        side=side,
                        price=filled_price,
                        qty=q_child,
                        notional=filled_price * q_child,
                        liquidity="taker",
                        proto_type=getattr(atype, "value", 0),
                        client_order_id=int(cli_id),
                        fee=float(fee),
                        slippage_bps=float(slip_bps),
                        spread_bps=float(sbps if sbps is not None else (self.slippage_cfg.default_spread_bps if self.slippage_cfg is not None else 0.0)),
                        latency_ms=int(lat_ms),
                        latency_spike=bool(lat_spike),
                        tif=tif,
                        ttl_steps=ttl_steps,
                    ))
            else:
                # пока другие типы не поддержаны — отменяем
                cancelled_ids.append(int(cli_id))

        # funding начисление
        funding_cashflow = 0.0
        funding_events_list = []
        ref_for_funding = self._last_ref_price
        if self.funding is not None:
            fc, events = self.funding.accrue(position_qty=self.position_qty, mark_price=ref_for_funding, now_ts_ms=ts)
            funding_cashflow = float(fc)
            funding_events_list = list(events or [])
            self.funding_cum += float(fc)

        # PnL/mark
        mark_p = self._mark_price(ref=self._last_ref_price, bid=self._last_bid, ask=self._last_ask)
        unrl = self._unrealized_pnl(mark_p)
        eq = float(self.realized_pnl_cum + unrl - self.fees_cum + self.funding_cum)

        # риск: дневной лосс/пауза + собрать события
        risk_events_all: list[RiskEvent] = []
        risk_paused_until = 0
        if self.risk is not None:
            try:
                self.risk.on_mark(ts_ms=ts, equity=eq)
                risk_events_all.extend(self.risk.pop_events())
                risk_paused_until = int(self.risk.paused_until_ms)
            except Exception:
                pass

        # финальный отчёт
        return ExecReport(
            trades=trades,
            cancelled_ids=cancelled_ids,
            new_order_ids=new_order_ids,
            fee_total=float(fee_total),
            new_order_pos=new_order_pos,
            funding_cashflow=float(funding_cashflow),
            funding_events=funding_events_list,  # type: ignore
            position_qty=float(self.position_qty),
            realized_pnl=float(self.realized_pnl_cum),
            unrealized_pnl=float(unrl),
            equity=float(eq),
            mark_price=float(mark_p if mark_p is not None else 0.0),
            bid=float(self._last_bid) if self._last_bid is not None else 0.0,
            ask=float(self._last_ask) if self._last_ask is not None else 0.0,
            mtm_price=float(mark_p if mark_p is not None else 0.0),
            risk_events=risk_events_all,  # type: ignore
            risk_paused_until_ms=int(risk_paused_until),
            spread_bps=self._last_spread_bps,
            vol_factor=self._last_vol_factor,
            liquidity=self._last_liquidity,
        )
