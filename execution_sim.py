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

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict, Sequence, Mapping, Callable, Deque
import hashlib
import math
import os
import logging
import threading
import random
import json
from clock import now_ms

try:
    from runtime_flags import seasonality_enabled  # type: ignore
except Exception:  # pragma: no cover - fallback if module not found

    def seasonality_enabled(default: bool = True) -> bool:
        return default


from utils.prometheus import Counter, Summary
from config import DataDegradationConfig

try:
    from services.costs import MakerTakerShareSettings
except Exception:  # pragma: no cover - optional dependency in stripped environments
    MakerTakerShareSettings = None  # type: ignore

try:
    from latency_volatility_cache import LatencyVolatilityCache
except Exception:  # pragma: no cover - optional dependency for legacy setups
    LatencyVolatilityCache = None  # type: ignore

try:
    from utils.time import HOUR_MS, HOURS_IN_WEEK, hour_of_week
    from utils_time import (
        get_hourly_multiplier,
        get_liquidity_multiplier,
        load_hourly_seasonality,
        watch_seasonality_file,
    )
except Exception:  # pragma: no cover - fallback when running as standalone file
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parent))
    from utils.time import HOUR_MS, HOURS_IN_WEEK, hour_of_week
    from utils_time import (
        get_hourly_multiplier,
        get_liquidity_multiplier,
        load_hourly_seasonality,
        watch_seasonality_file,
    )

logger = logging.getLogger(__name__)
seasonality_logger = logging.getLogger("seasonality").getChild(__name__)

_SIM_MULT_COUNTER = Counter(
    "sim_hour_of_week_multiplier_total",
    "Simulator liquidity multiplier applications per hour of week",
    ["hour"],
)

_FILLS_CAPPED_COUNT_BASE = Counter(
    "fills_capped_count_base",
    "Number of trades limited by base bar capacity",
    ["symbol", "capacity_reason", "exec_status"],
)

_FILLS_CAPPED_BASE_RATIO = Summary(
    "fills_capped_base_ratio",
    "Fill ratio observed when base bar capacity limits execution",
    ["symbol", "capacity_reason", "exec_status"],
)

try:
    import numpy as np
except Exception:  # минимальная замена на случай отсутствия numpy на этапе интеграции

    class _R:
        def __init__(self, seed=0):
            self.s = seed

        def randint(self, a, b=None, size=None):
            return a

    class np:  # type: ignore
        @staticmethod
        def random(seed=None):
            return _R(seed)

        class randomState:
            pass

        class RandomState:
            def __init__(self, seed=0):
                self._r = _R(seed)

            def randint(self, a, b=None, size=None):
                return self._r.randint(a, b, size)


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
        abs_price: Optional[float] = (
            None  # опционально, если доступна абсолютная цена лимитки
        )
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
    try:
        from slippage import (
            SlippageConfig,  # type: ignore
            estimate_slippage_bps,  # type: ignore
            apply_slippage_price,  # type: ignore
            compute_spread_bps_from_quotes,  # type: ignore
            mid_from_quotes,  # type: ignore
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
        MidOffsetLimitExecutor,
        make_executor,
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
            MidOffsetLimitExecutor,
            make_executor,
        )
    except Exception:
        BaseExecutor = None  # type: ignore
        MarketChild = None  # type: ignore
        TakerExecutor = None  # type: ignore
        TWAPExecutor = None  # type: ignore
        POVExecutor = None  # type: ignore
        MarketOpenH1Executor = None  # type: ignore
        VWAPExecutor = None  # type: ignore
        make_executor = None  # type: ignore

if MarketChild is None:

    @dataclass
    class MarketChild:  # type: ignore[override]
        ts_offset_ms: int
        qty: float
        liquidity_hint: Optional[float] = None


try:
    from adv_store import ADVStore
except Exception:
    ADVStore = None  # type: ignore


if TakerExecutor is None:

    class TakerExecutor:  # type: ignore[override]
        def plan_market(
            self,
            *,
            now_ts_ms: int,
            side: str,
            target_qty: float,
            snapshot: Dict[str, Any],
        ) -> List[MarketChild]:
            q = float(abs(target_qty))
            if q <= 0.0:
                return []
            return [MarketChild(ts_offset_ms=0, qty=q, liquidity_hint=None)]


if MarketOpenH1Executor is None:

    class MarketOpenH1Executor:  # type: ignore[override]
        def plan_market(
            self,
            *,
            now_ts_ms: int,
            side: str,
            target_qty: float,
            snapshot: Dict[str, Any],
        ) -> List[MarketChild]:
            q = float(abs(target_qty))
            if q <= 0.0:
                return []
            next_open = ((now_ts_ms // HOUR_MS) + 1) * HOUR_MS
            offset = int(max(0, next_open - now_ts_ms))
            return [MarketChild(ts_offset_ms=offset, qty=q, liquidity_hint=None)]


if make_executor is None:

    def make_executor(algo: str, cfg: Dict[str, Any] | None = None):  # type: ignore[override]
        a = str(algo).upper()
        cfg = dict(cfg or {})
        if a == "TWAP" and TWAPExecutor is not None:
            tw = dict(cfg.get("twap", {}))
            parts = int(tw.get("parts", 6))
            interval = int(tw.get("child_interval_s", 600))
            return TWAPExecutor(parts=parts, child_interval_s=interval)
        if a == "POV" and POVExecutor is not None:
            pv = dict(cfg.get("pov", {}))
            part = float(pv.get("participation", 0.10))
            interval = int(pv.get("child_interval_s", 60))
            min_not = float(pv.get("min_child_notional", 20.0))
            return POVExecutor(
                participation=part,
                child_interval_s=interval,
                min_child_notional=min_not,
            )
        if a == "VWAP" and VWAPExecutor is not None:
            return VWAPExecutor()
        return TakerExecutor()


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
    liquidity: str  # "taker"|"maker"; may reflect actual fill even if fees use expected share
    proto_type: int  # см. ActionType
    client_order_id: int
    fee: float = 0.0
    slippage_bps: float = 0.0
    spread_bps: float = 0.0
    latency_ms: int = 0
    latency_spike: bool = False
    tif: str = "GTC"
    ttl_steps: int = 0
    status: str = "FILLED"
    used_base_before: float = 0.0
    used_base_after: float = 0.0
    cap_base_per_bar: float = 0.0
    fill_ratio: float = 1.0
    capacity_reason: str = ""
    exec_status: str = "FILLED"


@dataclass
class SimStepReport:
    trades: List[ExecTrade] = field(default_factory=list)
    cancelled_ids: List[int] = field(default_factory=list)
    cancelled_reasons: dict[int, str] = field(default_factory=dict)
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
    execution_profile: str = ""
    latency_timeout_ratio: float = 0.0
    vol_raw: Optional[Dict[str, float]] = None
    cap_base_per_bar: float = 0.0
    used_base_before: float = 0.0
    used_base_after: float = 0.0
    fill_ratio: float = 1.0
    capacity_reason: str = ""
    exec_status: str = ""
    maker_share: float = 0.0
    expected_fee_bps: float = 0.0
    expected_spread_bps: Optional[float] = None
    expected_cost_components: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        trades_payload = []
        for t in self.trades:
            if isinstance(t, ExecTrade):
                trades_payload.append(dict(t.__dict__))
            else:
                trades_payload.append(dict(getattr(t, "__dict__", {})))
        cost_components: Dict[str, Optional[float]] = {}
        if isinstance(self.expected_cost_components, Mapping):
            for key, value in self.expected_cost_components.items():
                name = str(key)
                if value is None:
                    cost_components[name] = None
                    continue
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(num):
                    cost_components[name] = None
                    continue
                cost_components[name] = float(num)
        return {
            "trades": trades_payload,
            "cancelled_ids": list(self.cancelled_ids),
            "cancelled_reasons": {
                int(k): str(v) for k, v in self.cancelled_reasons.items()
            },
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
            "spread_bps": (
                float(self.spread_bps) if self.spread_bps is not None else None
            ),
            "vol_factor": (
                float(self.vol_factor) if self.vol_factor is not None else None
            ),
            "liquidity": float(self.liquidity) if self.liquidity is not None else None,
            "latency_p50_ms": float(self.latency_p50_ms),
            "latency_p95_ms": float(self.latency_p95_ms),
            "latency_timeout_ratio": float(self.latency_timeout_ratio),
            "execution_profile": str(self.execution_profile),
            "vol_raw": (
                {str(k): float(v) for k, v in self.vol_raw.items()}
                if isinstance(self.vol_raw, dict)
                else None
            ),
            "cap_base_per_bar": float(self.cap_base_per_bar),
            "used_base_before": float(self.used_base_before),
            "used_base_after": float(self.used_base_after),
            "fill_ratio": float(self.fill_ratio),
            "capacity_reason": str(self.capacity_reason),
            "exec_status": str(self.exec_status),
            "maker_share": float(self.maker_share),
            "expected_fee_bps": float(self.expected_fee_bps),
            "expected_spread_bps": (
                float(self.expected_spread_bps)
                if self.expected_spread_bps is not None
                else None
            ),
            "expected_cost_components": cost_components,
        }


@dataclass
class _TradeCostResult:
    """Container for detailed trade cost evaluation results."""

    bps: float
    mid: Optional[float]
    base_price: Optional[float]
    inputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    expected_spread_bps: Optional[float] = None
    latency_timeout_ratio: float = 0.0
    execution_profile: str = ""
    vol_raw: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        return {
            "trades": [t.__dict__ for t in self.trades],
            "cancelled_ids": list(self.cancelled_ids),
            "cancelled_reasons": {
                int(k): str(v) for k, v in self.cancelled_reasons.items()
            },
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
            "spread_bps": (
                float(self.spread_bps) if self.spread_bps is not None else None
            ),
            "vol_factor": (
                float(self.vol_factor) if self.vol_factor is not None else None
            ),
            "liquidity": float(self.liquidity) if self.liquidity is not None else None,
            "latency_p50_ms": float(self.latency_p50_ms),
            "latency_p95_ms": float(self.latency_p95_ms),
            "latency_timeout_ratio": float(self.latency_timeout_ratio),
            "execution_profile": str(self.execution_profile),
            "vol_raw": (
                {str(k): float(v) for k, v in self.vol_raw.items()}
                if isinstance(self.vol_raw, dict)
                else None
            ),
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
    delayed: bool = False
    intrabar_latency_ms: Optional[int] = None


@dataclass
class _VolSeriesState:
    values: Deque[float] = field(default_factory=deque)
    sum: float = 0.0
    sum_sq: float = 0.0


class _LatencyQueue:
    def __init__(
        self,
        latency_steps: int = 0,
        *,
        rng: Optional[random.Random] = None,
        drop_prob: float = 0.0,
        dropout_prob: float = 0.0,
        max_delay_steps: int = 0,
    ):
        self.latency_steps = max(0, int(latency_steps))
        self._q: List[Pending] = []
        # Any randomness in the queue (drops / delays) is driven by a dedicated
        # ``random.Random`` instance so that executions are reproducible when a
        # seed is provided.  If no RNG is supplied but probabilistic features are
        # enabled, fall back to a deterministic RNG seeded with ``0``.
        if rng is not None:
            self._rng = rng
        elif drop_prob > 0.0 or dropout_prob > 0.0 or max_delay_steps > 0:
            self._rng = random.Random(0)
        else:
            self._rng = None
        self.drop_prob = float(drop_prob)
        self.dropout_prob = float(dropout_prob)
        self.max_delay_steps = max(0, int(max_delay_steps))
        self.total_cnt = 0
        self.drop_cnt = 0
        self.delay_cnt = 0

    def push(self, p: Pending) -> None:
        self._q.append(p)

    def pop_ready(self) -> Tuple[List[Pending], List[Pending]]:
        ready: List[Pending] = []
        cancelled: List[Pending] = []
        rest: List[Pending] = []
        for p in self._q:
            self.total_cnt += 1
            if self._rng is not None:
                if self._rng.random() < self.drop_prob:
                    self.drop_cnt += 1
                    continue
                if self._rng.random() < self.dropout_prob:
                    delay_steps = self._rng.randint(0, self.max_delay_steps)
                    if delay_steps > 0:
                        p.remaining_lat += delay_steps
                        if not p.delayed:
                            self.delay_cnt += 1
                            p.delayed = True
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
    """Simple order queue with deterministic execution.

    All pseudo-random behaviour inside the simulator (latency dropouts,
    delays, etc.) is driven by RNGs that are explicitly seeded.  Providing the
    same seed and inputs therefore yields identical child order trajectories.
    """

    @staticmethod
    def _plain_mapping(obj: Any) -> Dict[str, Any]:
        """Best-effort conversion of arbitrary objects to a plain mapping."""

        if isinstance(obj, Mapping):
            try:
                return {str(k): v for k, v in obj.items()}
            except Exception:
                try:
                    return dict(obj)
                except Exception:
                    return {}
        if hasattr(obj, "dict"):
            try:
                data = obj.dict(exclude_unset=False)  # type: ignore[call-arg]
            except Exception:
                data = {}
            if isinstance(data, Mapping):
                return ExecutionSimulator._plain_mapping(data)
        if hasattr(obj, "__dict__"):
            try:
                return {
                    str(k): v
                    for k, v in vars(obj).items()
                    if not str(k).startswith("_")
                }
            except Exception:
                return {}
        return {}

    def __init__(
        self,
        *,
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
        liquidity_seasonality: Optional[Sequence[float]] = None,
        spread_seasonality: Optional[Sequence[float]] = None,
        liquidity_seasonality_path: Optional[str] = None,
        liquidity_seasonality_hash: Optional[str] = None,
        liquidity_seasonality_override: Optional[Sequence[float]] = None,
        spread_seasonality_override: Optional[Sequence[float]] = None,
        seasonality_override_path: Optional[str] = None,
        use_seasonality: bool = True,
        seasonality_interpolate: bool = False,
        seasonality_day_only: bool = False,
        seasonality_auto_reload: bool = False,
        data_degradation: Optional[DataDegradationConfig] = None,
        adv_store: Optional["ADVStore"] = None,
        run_config: Any = None,
    ):
        self.symbol = str(symbol).upper()
        self._latency_symbol: Optional[str] = self.symbol
        self.latency_steps = int(max(0, latency_steps))
        self.seed = int(seed)
        # Seed global RNGs for reproducibility of any downstream randomness.
        random.seed(self.seed)
        try:
            np.random.seed(self.seed)
        except Exception:
            pass
        try:
            self._rng = np.random.RandomState(seed)  # type: ignore
        except Exception:
            try:
                self._rng = np.RandomState(seed)  # type: ignore
            except Exception:
                self._rng = None  # type: ignore
        self._next_cli_id = 1
        self._order_seq_counter: int = 0
        self._intrabar_debug_logged: int = 0
        self._trade_cost_debug_logged: int = 0
        self._trade_cost_debug_limit: int = 100

        def _convert_to_plain_mapping(obj: Any) -> Dict[str, Any]:
            if isinstance(obj, Mapping):
                try:
                    return {str(k): v for k, v in obj.items()}
                except Exception:
                    return dict(obj)
            if hasattr(obj, "dict"):
                try:
                    data = obj.dict(exclude_unset=False)  # type: ignore[call-arg]
                except Exception:
                    data = {}
                if isinstance(data, Mapping):
                    return _convert_to_plain_mapping(data)
            if hasattr(obj, "__dict__"):
                try:
                    return {
                        str(k): v
                        for k, v in vars(obj).items()
                        if not str(k).startswith("_")
                    }
                except Exception:
                    return {}
            return {}

        def _safe_float(value: Any, default: float = 0.0) -> float:
            try:
                num = float(value)
            except (TypeError, ValueError):
                return default
            if not math.isfinite(num):
                return default
            return num

        self.lob = lob
        self._last_ref_price: Optional[float] = None
        self._next_h1_open_price: Optional[float] = None
        self._last_bar_open: Optional[float] = None
        self._last_bar_high: Optional[float] = None
        self._last_bar_low: Optional[float] = None
        self._last_bar_close: Optional[float] = None
        self._last_bar_close_ts: Optional[int] = None
        self._intrabar_timeframe_ms: Optional[int] = None
        self.run_config = run_config
        self._run_config = run_config
        self._adv_store: Optional[ADVStore] = None
        self._adv_enabled: bool = False
        self._adv_capacity_fraction: float = 1.0
        self._adv_bars_per_day_override: Optional[float] = None
        self._last_adv_bar_capacity: Optional[float] = None
        self._bar_cap_base_enabled: bool = False
        self._bar_cap_base_frac: Optional[float] = None
        self._bar_cap_base_floor: Optional[float] = None
        self._bar_cap_base_path: Optional[str] = None
        self._bar_cap_base_timeframe_ms: Optional[int] = None
        self._bar_cap_base_cache: Dict[str, float] = {}
        self._bar_cap_base_cache_path: Optional[str] = None
        self._bar_cap_base_loaded_ts: Optional[int] = None
        self._bar_cap_base_meta: Dict[str, Any] = {}
        self._bar_cap_base_warned_symbols: set[str] = set()
        self._used_base_in_bar: dict[str, float] = {}
        self._capacity_bar_ts: Optional[int] = None
        self.run_id: str = str(getattr(run_config, "run_id", "sim") or "sim")
        self.step_ms: int = (
            int(getattr(run_config, "step_ms", 1000))
            if run_config is not None
            else 1000
        )
        if self.step_ms <= 0:
            self.step_ms = 1
        adv_cfg = getattr(run_config, "adv", None) if run_config is not None else None
        adv_enabled_cfg = False
        adv_fraction_cfg: Optional[float] = None
        adv_bars_override_cfg: Optional[float] = None
        if adv_cfg is not None:
            adv_enabled_cfg = bool(getattr(adv_cfg, "enabled", False))
            frac_attr = getattr(adv_cfg, "capacity_fraction", None)
            if frac_attr is None:
                extra_block = getattr(adv_cfg, "extra", None)
                if isinstance(extra_block, Mapping):
                    frac_attr = extra_block.get("capacity_fraction")
            try:
                frac_val = float(frac_attr) if frac_attr is not None else None
            except (TypeError, ValueError):
                frac_val = None
            if frac_val is not None and math.isfinite(frac_val):
                if frac_val < 0.0:
                    frac_val = 0.0
                adv_fraction_cfg = frac_val
            bars_attr = getattr(adv_cfg, "bars_per_day_override", None)
            if bars_attr is None:
                extra_block = getattr(adv_cfg, "extra", None)
                if isinstance(extra_block, Mapping):
                    bars_attr = extra_block.get("bars_per_day_override")
                    if bars_attr is None:
                        bars_attr = extra_block.get("bars_per_day")
            try:
                bars_val = float(bars_attr) if bars_attr is not None else None
            except (TypeError, ValueError):
                bars_val = None
            if bars_val is not None and math.isfinite(bars_val) and bars_val > 0.0:
                adv_bars_override_cfg = bars_val
        adv_store_obj = adv_store
        if adv_enabled_cfg and adv_store_obj is None and ADVStore is not None and adv_cfg is not None:
            try:
                adv_store_obj = ADVStore(adv_cfg)
            except Exception:
                logger.exception("Failed to initialise ADVStore from run_config.adv")
                adv_store_obj = None
        self.set_adv_store(
            adv_store_obj,
            enabled=adv_enabled_cfg,
            capacity_fraction=adv_fraction_cfg,
            bars_per_day_override=adv_bars_override_cfg,
        )
        if data_degradation is None:
            data_degradation = DataDegradationConfig.default()
        self.data_degradation = data_degradation
        self._rng_dd = random.Random(self.data_degradation.seed or self.seed)
        max_delay_steps = 0
        if self.step_ms > 0:
            max_delay_steps = self.data_degradation.max_delay_ms // self.step_ms
        self._q = _LatencyQueue(
            self.latency_steps,
            rng=self._rng_dd,
            drop_prob=self.data_degradation.drop_prob,
            dropout_prob=self.data_degradation.dropout_prob,
            max_delay_steps=max_delay_steps,
        )
        self._stopped = False
        self._cancelled_on_submit: List[int] = []
        self._ttl_orders: List[Tuple[int, int]] = []

        # квантайзер — опционально
        self.quantizer: Optional[Quantizer] = None
        self.enforce_ppbs = bool(enforce_ppbs)
        self.strict_filters = bool(strict_filters)
        try:
            if Quantizer is not None and filters_path:
                max_age = int(os.getenv("TB_FILTER_MAX_AGE_DAYS", "30"))
                fatal = os.getenv("TB_FAIL_ON_STALE_FILTERS") not in (None, "0", "")
                filters, meta = load_filters(
                    filters_path, max_age_days=max_age, fatal=fatal
                )
                if filters:
                    self.quantizer = Quantizer(filters, strict=strict_filters)
                if meta:
                    logger.info("Loaded filter metadata: %s", meta)
        except Exception:
            # не ломаемся, если файл отсутствует; квантизация просто не активна
            self.quantizer = None

        # комиссии и funding
        self.fees = (
            FeesModel.from_dict(fees_config or {}) if FeesModel is not None else None
        )
        self.fees_config_payload: Dict[str, Any] = {}
        self.fees_metadata: Dict[str, Any] = {}
        self.fees_expected_payload: Dict[str, Any] = {}
        self.fees_symbol_fee_table: Dict[str, Any] = {}
        self._maker_taker_share_cfg: Optional[Dict[str, Any]] = None
        self._maker_taker_share_enabled: bool = False
        self._maker_taker_share_mode: Optional[str] = None
        self._expected_fee_bps: float = 0.0
        self._expected_maker_share: float = 0.0

        fees_cfg_payloads: List[Dict[str, Any]] = []
        if fees_config:
            payload = _convert_to_plain_mapping(fees_config)
            if payload:
                fees_cfg_payloads.append(payload)
        if run_config is not None:
            rc_fees = getattr(run_config, "fees", None)
            if rc_fees is not None:
                payload = _convert_to_plain_mapping(rc_fees)
                if payload:
                    fees_cfg_payloads.append(payload)

        share_block: Any = None
        for payload in fees_cfg_payloads:
            block = payload.get("maker_taker_share") if isinstance(payload, Mapping) else None
            if block is not None:
                share_block = block
                break
        if share_block is None and run_config is not None:
            direct_share = getattr(run_config, "maker_taker_share", None)
            if direct_share is not None:
                share_block = direct_share

        maker_fee_bps = 0.0
        taker_fee_bps = 0.0
        if self.fees is not None:
            maker_fee_bps = _safe_float(getattr(self.fees, "maker_bps", 0.0)) * _safe_float(
                getattr(self.fees, "maker_discount_mult", 1.0), 1.0
            )
            taker_fee_bps = _safe_float(getattr(self.fees, "taker_bps", 0.0)) * _safe_float(
                getattr(self.fees, "taker_discount_mult", 1.0), 1.0
            )
        if maker_fee_bps == 0.0 and taker_fee_bps == 0.0 and fees_cfg_payloads:
            base_payload = fees_cfg_payloads[0]
            maker_fee_bps = _safe_float(base_payload.get("maker_bps"), 0.0) * _safe_float(
                base_payload.get("maker_discount_mult", 1.0), 1.0
            )
            taker_fee_bps = _safe_float(base_payload.get("taker_bps"), 0.0) * _safe_float(
                base_payload.get("taker_discount_mult", 1.0), 1.0
            )

        share_payload: Optional[Dict[str, Any]] = None
        if MakerTakerShareSettings is not None and share_block is not None:
            share_cfg = MakerTakerShareSettings.parse(share_block)
            if share_cfg is not None:
                try:
                    share_payload = share_cfg.to_sim_payload(
                        maker_fee_bps, taker_fee_bps
                    )
                except Exception:
                    share_payload = None
        if share_payload is None and isinstance(share_block, Mapping):
            share_payload = _convert_to_plain_mapping(share_block)

        self._initialise_fee_payloads(fees_cfg_payloads, share_payload)

        self._apply_maker_taker_share_payload(share_payload)

        self._slippage_share_enabled: bool = False
        self._slippage_share_default: float = 0.0
        self._slippage_spread_cost_maker_bps: float = 0.0
        self._slippage_spread_cost_taker_bps: float = 0.0

        self.funding = (
            FundingCalculator(**(funding_config or {"enabled": False}))
            if FundingCalculator is not None
            else None
        )

        # слиппедж
        self.slippage_cfg = None
        self._slippage_get_spread: Optional[Callable[..., Any]] = None
        self._slippage_get_trade_cost: Optional[Callable[..., Any]] = None
        if SlippageConfig is not None:
            if slippage_config is None:
                if execution_profile is not None:
                    try:
                        self.slippage_cfg = SlippageConfig.from_dict({})
                    except Exception:
                        self.slippage_cfg = None
            elif isinstance(slippage_config, str):
                try:
                    self.slippage_cfg = SlippageConfig.from_file(slippage_config)
                except Exception:
                    logger.exception("failed to load slippage config from %s", slippage_config)
            elif isinstance(slippage_config, dict):
                self.slippage_cfg = SlippageConfig.from_dict(slippage_config)

        if self.slippage_cfg is not None:
            candidate = getattr(self.slippage_cfg, "get_spread_bps", None)
            if callable(candidate):
                self._slippage_get_spread = candidate
            cost_candidate = getattr(self.slippage_cfg, "get_trade_cost_bps", None)
            if callable(cost_candidate):
                self._slippage_get_trade_cost = cost_candidate

        self._vol_window_size: int = 120
        self._vol_zscore_clip: Optional[float] = None
        self._vol_gamma: float = 0.0
        self._vol_series: Dict[str, _VolSeriesState] = {}
        self._vol_stat_cache: Dict[str, Dict[str, float]] = {}
        self._vol_norm_metric: Optional[str] = None
        self._dyn_spread_enabled: bool = False
        self._dyn_spread_metric_key: Optional[str] = None
        self._dyn_spread_alpha_bps: Optional[float] = None
        self._dyn_spread_beta_coef: Optional[float] = None
        self._dyn_spread_min_bps: Optional[float] = None
        self._dyn_spread_max_bps: Optional[float] = None
        self._dyn_spread_smoothing_alpha: Optional[float] = None
        self._dyn_spread_fallback_bps: Optional[float] = None
        self._dyn_spread_use_volatility: bool = False
        self._dyn_spread_prev_ema: Optional[float] = None
        dyn_cfg_obj: Optional[Any] = None
        if self.slippage_cfg is not None:
            dyn_candidate = None
            getter = getattr(self.slippage_cfg, "get_dynamic_block", None)
            if callable(getter):
                try:
                    dyn_candidate = getter()
                except Exception:
                    dyn_candidate = None
            if dyn_candidate is None:
                dyn_candidate = getattr(self.slippage_cfg, "dynamic", None)
            if dyn_candidate is None:
                dyn_candidate = getattr(self.slippage_cfg, "dynamic_spread", None)
            dyn_cfg_obj = dyn_candidate
        if dyn_cfg_obj is not None:
            if isinstance(dyn_cfg_obj, Mapping):
                metric_attr = dyn_cfg_obj.get("vol_metric")
                window_attr = dyn_cfg_obj.get("vol_window")
                gamma_attr = dyn_cfg_obj.get("gamma")
                clip_attr = dyn_cfg_obj.get("zscore_clip")
            else:
                metric_attr = getattr(dyn_cfg_obj, "vol_metric", None)
                window_attr = getattr(dyn_cfg_obj, "vol_window", None)
                gamma_attr = getattr(dyn_cfg_obj, "gamma", None)
                clip_attr = getattr(dyn_cfg_obj, "zscore_clip", None)
            try:
                metric_key = str(metric_attr).strip().lower() if metric_attr else ""
            except Exception:
                metric_key = ""
            if metric_key:
                self._vol_norm_metric = metric_key
                self._dyn_spread_metric_key = metric_key
            window_val: Optional[int]
            try:
                window_val = int(window_attr) if window_attr is not None else None
            except (TypeError, ValueError):
                window_val = None
            if window_val is not None:
                if window_val < 2:
                    window_val = 2
                self._vol_window_size = window_val
            try:
                gamma_val = float(gamma_attr) if gamma_attr is not None else None
            except (TypeError, ValueError):
                gamma_val = None
            if gamma_val is not None and math.isfinite(gamma_val):
                self._vol_gamma = gamma_val
            try:
                clip_val = float(clip_attr) if clip_attr is not None else None
            except (TypeError, ValueError):
                clip_val = None
            if clip_val is not None and math.isfinite(clip_val):
                if clip_val < 0.0:
                    clip_val = abs(clip_val)
                self._vol_zscore_clip = clip_val
            enabled_attr = None
            if isinstance(dyn_cfg_obj, Mapping):
                enabled_attr = dyn_cfg_obj.get("enabled")
            else:
                enabled_attr = getattr(dyn_cfg_obj, "enabled", None)
            try:
                self._dyn_spread_enabled = bool(enabled_attr)
            except Exception:
                self._dyn_spread_enabled = False

            def _dyn_value(*names: str) -> Any:
                for name in names:
                    if isinstance(dyn_cfg_obj, Mapping):
                        if name in dyn_cfg_obj:
                            return dyn_cfg_obj[name]
                    else:
                        if hasattr(dyn_cfg_obj, name):
                            return getattr(dyn_cfg_obj, name)
                return None

            alpha_val = self._float_or_none(_dyn_value("alpha_bps", "alpha"))
            if alpha_val is not None:
                self._dyn_spread_alpha_bps = alpha_val
            beta_val = self._float_or_none(_dyn_value("beta_coef", "beta"))
            if beta_val is not None:
                self._dyn_spread_beta_coef = beta_val
            min_val = self._float_or_none(_dyn_value("min_spread_bps"))
            if min_val is not None:
                if min_val < 0.0:
                    min_val = 0.0
                self._dyn_spread_min_bps = min_val
            max_val = self._float_or_none(_dyn_value("max_spread_bps"))
            if max_val is not None:
                if max_val < 0.0:
                    max_val = 0.0
                self._dyn_spread_max_bps = max_val
            smooth_val = self._float_or_none(_dyn_value("smoothing_alpha"))
            if smooth_val is not None:
                if smooth_val <= 0.0:
                    smooth_val = None
                elif smooth_val >= 1.0:
                    smooth_val = 1.0
            self._dyn_spread_smoothing_alpha = smooth_val
            fb_val = self._float_or_none(_dyn_value("fallback_spread_bps"))
            if fb_val is not None and fb_val < 0.0:
                fb_val = 0.0
            self._dyn_spread_fallback_bps = fb_val
            self._dyn_spread_use_volatility = bool(_dyn_value("use_volatility"))

        # исполнители
        self._execution_cfg = dict(execution_config or {})
        self.execution_profile = (
            str(execution_profile) if execution_profile is not None else ""
        )
        self.execution_params: dict = dict(execution_params or {})
        limit_cfg = self.execution_params.get("trade_cost_debug_max_logs")
        if limit_cfg is not None:
            try:
                limit_val = int(limit_cfg)
            except (TypeError, ValueError):
                limit_val = self._trade_cost_debug_limit
            else:
                if limit_val < 0:
                    limit_val = 0
            self._trade_cost_debug_limit = int(limit_val)
        self._execution_intrabar_cfg: Dict[str, Any] = {}
        bar_cap_base_cfg: Dict[str, Any] = {}

        def _update_bar_capacity_cfg(candidate: Any) -> None:
            mapping = _convert_to_plain_mapping(candidate)
            if not mapping:
                return
            for raw_key, value in mapping.items():
                try:
                    key = str(raw_key)
                except Exception:
                    continue
                key_lower = key.strip().lower()
                if key_lower == "bar_capacity_base":
                    _update_bar_capacity_cfg(value)
                    continue
                if key_lower == "extra":
                    _update_bar_capacity_cfg(value)
                    continue
                if key_lower == "enabled":
                    bar_cap_base_cfg["enabled"] = value
                    continue
                if key_lower in (
                    "capacity_frac_of_adv_base",
                    "capacity_fraction_of_adv_base",
                    "capacity_frac_of_adv",
                    "capacity_fraction_of_adv",
                ):
                    bar_cap_base_cfg["capacity_frac_of_ADV_base"] = value
                    continue
                if key_lower in (
                    "floor_base",
                    "floor",
                    "adv_floor",
                    "adv_floor_base",
                ):
                    bar_cap_base_cfg["floor_base"] = value
                    continue
                if key_lower in (
                    "adv_base_path",
                    "adv_path",
                    "path",
                    "dataset_path",
                ):
                    bar_cap_base_cfg["adv_base_path"] = value
                    continue
                if key_lower in (
                    "timeframe_ms",
                    "timeframe",
                    "bar_timeframe_ms",
                    "bar_timeframe",
                ):
                    bar_cap_base_cfg["timeframe_ms"] = value

        exec_cfg_sources: List[Any] = []
        if execution_config:
            exec_cfg_sources.append(execution_config)
        if run_config is not None:
            rc_exec = getattr(run_config, "execution", None)
            if rc_exec is not None:
                exec_cfg_sources.append(rc_exec)

        for cfg_src in exec_cfg_sources:
            payload: Dict[str, Any] = {}
            if isinstance(cfg_src, Mapping):
                try:
                    payload = {str(k): v for k, v in cfg_src.items()}
                except Exception:
                    payload = dict(cfg_src)
            elif hasattr(cfg_src, "dict"):
                try:
                    payload = dict(cfg_src.dict(exclude_unset=False))  # type: ignore[call-arg]
                except Exception:
                    payload = {}
            elif hasattr(cfg_src, "__dict__"):
                try:
                    payload = {
                        str(k): v
                        for k, v in vars(cfg_src).items()
                        if not str(k).startswith("_")
                    }
                except Exception:
                    payload = {}
            if payload:
                self._execution_intrabar_cfg.update(payload)
            _update_bar_capacity_cfg(payload)
            _update_bar_capacity_cfg(cfg_src)

        if bar_cap_base_cfg:
            self.set_bar_capacity_base_config(**bar_cap_base_cfg)

        self._intrabar_price_model: Optional[str] = None
        self._intrabar_config_timeframe_ms: Optional[int] = None
        self._intrabar_latency_source: str = "latency"
        self._intrabar_latency_constant_ms: Optional[int] = None
        self._intrabar_log_warnings: bool = False
        self._intrabar_warn_next_log_ms: int = 0
        self._timing_timeframe_ms: Optional[int] = None
        self._intrabar_seed_mode: str = "stable"
        self._intrabar_debug_max_logs: int = 0

        if self._execution_intrabar_cfg:
            mode = self._execution_intrabar_cfg.get("intrabar_price_model")
            if mode is not None:
                try:
                    self._intrabar_price_model = str(mode)
                except Exception:
                    self._intrabar_price_model = None

            tf_cfg = self._execution_intrabar_cfg.get("timeframe_ms")
            try:
                tf_int = int(tf_cfg) if tf_cfg is not None else None
            except (TypeError, ValueError):
                tf_int = None
            if tf_int is not None and tf_int > 0:
                self._intrabar_config_timeframe_ms = tf_int

            lat_src = self._execution_intrabar_cfg.get("use_latency_from")
            if lat_src is not None:
                try:
                    value = str(lat_src).strip().lower()
                except Exception:
                    value = "latency"
                self._intrabar_latency_source = value or "latency"

            const_cfg = self._execution_intrabar_cfg.get("latency_constant_ms")
            try:
                const_int = int(const_cfg) if const_cfg is not None else None
            except (TypeError, ValueError):
                const_int = None
            if const_int is not None and const_int >= 0:
                self._intrabar_latency_constant_ms = const_int

            log_flag_keys = (
                "log_intrabar_warnings",
                "log_intrabar_latency",
                "log_latency_warnings",
            )
            for key in log_flag_keys:
                if key in self._execution_intrabar_cfg:
                    self._intrabar_log_warnings = bool(
                        self._execution_intrabar_cfg.get(key)
                    )
                    if self._intrabar_log_warnings:
                        break

            seed_mode_cfg = None
            for _key in ("intrabar_seed_mode", "intrabar_price_seed_mode", "seed_mode"):
                if _key in self._execution_intrabar_cfg:
                    seed_mode_cfg = self._execution_intrabar_cfg.get(_key)
                    break
            if seed_mode_cfg is not None:
                try:
                    self._intrabar_seed_mode = str(seed_mode_cfg).strip().lower()
                except Exception:
                    self._intrabar_seed_mode = "stable"
                if not self._intrabar_seed_mode:
                    self._intrabar_seed_mode = "stable"

            debug_limit_cfg = None
            for _key in ("intrabar_debug_max_logs", "intrabar_debug_limit", "intrabar_debug_logs"):
                if _key in self._execution_intrabar_cfg:
                    debug_limit_cfg = self._execution_intrabar_cfg.get(_key)
                    break
            if debug_limit_cfg is not None:
                try:
                    dbg_val = int(debug_limit_cfg)
                except (TypeError, ValueError):
                    dbg_val = 0
                if dbg_val < 0:
                    dbg_val = 0
                self._intrabar_debug_max_logs = dbg_val

        if run_config is not None:
            timing_cfg = getattr(run_config, "timing", None)
            if timing_cfg is not None:
                timeframe_val = getattr(timing_cfg, "timeframe_ms", None)
                try:
                    timeframe_int = (
                        int(timeframe_val) if timeframe_val is not None else None
                    )
                except (TypeError, ValueError):
                    timeframe_int = None
                if timeframe_int is not None and timeframe_int > 0:
                    self._timing_timeframe_ms = timeframe_int
        self._executor: Optional[BaseExecutor] = None
        self._build_executor()

        # латентность
        latency_sources: List[Any] = []
        if run_config is not None:
            rc_latency = getattr(run_config, "latency", None)
            if rc_latency:
                latency_sources.append(rc_latency)
        if latency_config:
            latency_sources.append(latency_config)

        latency_kwargs: Dict[str, Any] = {}
        latency_cfg_for_impl: Dict[str, Any] = {}

        def _capture_latency_cfg(payload: Mapping[str, Any]) -> None:
            keys = (
                "use_seasonality",
                "latency_seasonality_path",
                "seasonality_path",
                "refresh_period_days",
                "seasonality_default",
            )
            for key in keys:
                if key in payload and payload[key] is not None:
                    latency_cfg_for_impl[key] = payload[key]

        for src in latency_sources:
            if isinstance(src, dict):
                latency_kwargs.update(src)
                _capture_latency_cfg(src)
            else:
                payload: Dict[str, Any] = {}
                if hasattr(src, "dict"):
                    try:
                        payload = dict(src.dict())  # type: ignore[call-arg]
                    except Exception:
                        payload = {}
                if not payload and hasattr(src, "__dict__"):
                    try:
                        payload = {
                            str(k): v
                            for k, v in vars(src).items()
                            if not str(k).startswith("_")
                        }
                    except Exception:
                        payload = {}
                latency_kwargs.update(payload)
                if payload:
                    _capture_latency_cfg(payload)

        config_keys = (
            "use_seasonality",
            "latency_seasonality_path",
            "seasonality_path",
            "refresh_period_days",
            "seasonality_default",
        )
        for key in config_keys:
            if key in latency_kwargs:
                latency_cfg_for_impl.setdefault(key, latency_kwargs[key])
                latency_kwargs.pop(key, None)

        lat_section = getattr(run_config, "latency", None) if run_config is not None else None
        if lat_section is not None:
            for key in config_keys:
                if key not in latency_cfg_for_impl:
                    value = getattr(lat_section, key, None)
                    if value is not None:
                        latency_cfg_for_impl[key] = value
        if run_config is not None and "latency_seasonality_path" not in latency_cfg_for_impl:
            lat_path_global = getattr(run_config, "latency_seasonality_path", None)
            if lat_path_global is not None:
                latency_cfg_for_impl["latency_seasonality_path"] = lat_path_global

        vol_metric_cfg = str(latency_kwargs.get("vol_metric", "sigma") or "sigma").lower()
        self._latency_vol_metric = vol_metric_cfg if vol_metric_cfg else "sigma"

        vol_window_cfg = latency_kwargs.get("vol_window")
        try:
            vol_window_int = int(vol_window_cfg) if vol_window_cfg is not None else 120
        except (TypeError, ValueError):
            vol_window_int = 120
        if vol_window_int < 2:
            vol_window_int = 2
        latency_kwargs["vol_window"] = vol_window_int
        self._latency_vol_window = vol_window_int

        lat_symbol_cfg = latency_kwargs.get("symbol")
        if lat_symbol_cfg:
            try:
                self._latency_symbol = str(lat_symbol_cfg).upper()
            except Exception:
                pass

        self.volatility_cache = None
        if LatencyVolatilityCache is not None:
            try:
                self.volatility_cache = LatencyVolatilityCache(window=vol_window_int)
            except Exception:
                self.volatility_cache = None
        else:
            self.volatility_cache = None

        self.latency = (
            LatencyModel(**latency_kwargs) if LatencyModel is not None else None
        )

        lat_cfg_payload: Dict[str, Any] = {}
        refresh_cfg = latency_cfg_for_impl.get("refresh_period_days")
        try:
            refresh_days = int(refresh_cfg) if refresh_cfg is not None else None
        except (TypeError, ValueError):
            refresh_days = None
        if refresh_days is not None and refresh_days >= 0:
            lat_cfg_payload["refresh_period_days"] = refresh_days
        lat_path_cfg = latency_cfg_for_impl.get("latency_seasonality_path")
        if not lat_path_cfg:
            lat_path_cfg = latency_cfg_for_impl.get("seasonality_path")
        if lat_path_cfg:
            try:
                lat_cfg_payload["latency_seasonality_path"] = str(lat_path_cfg)
            except Exception:
                pass
        if "use_seasonality" in latency_cfg_for_impl:
            lat_cfg_payload["use_seasonality"] = bool(latency_cfg_for_impl["use_seasonality"])
        if "seasonality_default" in latency_cfg_for_impl:
            lat_cfg_payload["seasonality_default"] = latency_cfg_for_impl["seasonality_default"]
        self.latency_config_payload = lat_cfg_payload

        # риск-менеджер
        self.risk = (
            RiskManager.from_dict(risk_config or {})
            if RiskManager is not None
            else None
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
        self._last_vol_raw: Optional[Dict[str, float]] = None
        self._last_liquidity: Optional[float] = None
        self._snapshot_hour: Optional[int] = None

        # логирование
        self._logger = (
            LogWriter(LogConfig.from_dict(logging_config or {}), run_id=self.run_id)
            if LogWriter is not None
            else None
        )
        self._step_counter: int = 0

        # журнал исполненных трейдов для валидации PnL
        self._trade_log: List[ExecTrade] = []

        # накопители для VWAP
        self._vwap_pv: float = 0.0
        self._vwap_vol: float = 0.0
        self._vwap_hour: Optional[int] = None
        self._last_hour_vwap: Optional[float] = None

        # сезонность ликвидности и спреда по часам недели
        self.use_seasonality = bool(
            getattr(run_config, "use_seasonality", use_seasonality)
            and seasonality_enabled()
        )
        self.seasonality_interpolate = bool(
            getattr(run_config, "seasonality_interpolate", seasonality_interpolate)
        )
        self.seasonality_day_only = bool(
            getattr(run_config, "seasonality_day_only", seasonality_day_only)
        )
        default_len = 7 if self.seasonality_day_only else HOURS_IN_WEEK
        default_seasonality = np.ones(default_len, dtype=float)
        self._liq_seasonality = default_seasonality.copy()
        self._spread_seasonality = default_seasonality.copy()
        self._seasonality_lock = threading.Lock()
        self._seasonality_path: Optional[str] = None
        if self.use_seasonality:
            liq_arr: Optional[Sequence[float]] = liquidity_seasonality
            spread_arr: Optional[Sequence[float]] = spread_seasonality
            path = liquidity_seasonality_path
            if run_config is not None and path is None:
                path = getattr(run_config, "liquidity_seasonality_path", None)
            if path is None:
                path = "configs/liquidity_seasonality.json"
            if path:
                if liq_arr is None:
                    liq_arr = load_hourly_seasonality(path, "liquidity")
                if spread_arr is None:
                    # Prefer the dedicated "spread" field, falling back to generic
                    # "multipliers" for backwards compatibility with older configs.
                    spread_arr = load_hourly_seasonality(path, "spread", "multipliers")

            from utils_time import interpolate_daily_multipliers, daily_from_hourly

            def _prep(arr: Optional[Sequence[float]]) -> Optional[np.ndarray]:
                if arr is None:
                    return None
                arr = np.asarray(arr, dtype=float)
                if self.seasonality_day_only:
                    if arr.size == HOURS_IN_WEEK:
                        arr = daily_from_hourly(arr)
                    elif arr.size != 7:
                        return None
                else:
                    if arr.size == 7:
                        arr = interpolate_daily_multipliers(arr)
                    elif arr.size != HOURS_IN_WEEK:
                        return None
                return arr

            liq_arr = _prep(liq_arr)
            spread_arr = _prep(spread_arr)

            if liq_arr is not None:
                self._liq_seasonality = liq_arr
            else:
                logger.warning(
                    "Liquidity seasonality config %s not found or invalid; using default multipliers of 1.0; "
                    "run scripts/build_hourly_seasonality.py to generate them.",
                    path,
                )
            if spread_arr is not None:
                self._spread_seasonality = spread_arr
            else:
                logger.warning(
                    "Spread seasonality config %s not found or invalid; using default multipliers of 1.0; "
                    "run scripts/build_hourly_seasonality.py to generate them.",
                    path,
                )

            # Apply optional overrides (element-wise multiplication)
            liq_override = liquidity_seasonality_override
            spread_override = spread_seasonality_override
            override_path = seasonality_override_path
            if run_config is not None:
                if liq_override is None:
                    liq_override = getattr(
                        run_config, "liquidity_seasonality_override", None
                    )
                if spread_override is None:
                    spread_override = getattr(
                        run_config, "spread_seasonality_override", None
                    )
                if override_path is None:
                    override_path = getattr(
                        run_config, "liquidity_seasonality_override_path", None
                    ) or getattr(run_config, "seasonality_override_path", None)
            if override_path and (liq_override is None or spread_override is None):
                if liq_override is None:
                    liq_override = load_hourly_seasonality(override_path, "liquidity")
                if spread_override is None:
                    spread_override = load_hourly_seasonality(
                        override_path, "spread", "multipliers"
                    )

            liq_override = _prep(liq_override)
            spread_override = _prep(spread_override)
            if liq_override is not None:
                self._liq_seasonality *= liq_override
            if spread_override is not None:
                self._spread_seasonality *= spread_override

            self._seasonality_path = path
            if seasonality_auto_reload and path:

                def _reload(data: Dict[str, np.ndarray]) -> None:
                    try:
                        self.load_seasonality_multipliers(data)
                        seasonality_logger.info(
                            "Reloaded seasonality multipliers from %s", path
                        )
                    except Exception:
                        seasonality_logger.exception(
                            "Failed to reload seasonality multipliers from %s", path
                        )

                watch_seasonality_file(path, _reload)

        # накопители статистики по сезонности ликвидности
        self._liq_mult_sum: List[float] = [0.0] * HOURS_IN_WEEK
        self._liq_val_sum: List[float] = [0.0] * HOURS_IN_WEEK
        self._liq_count: List[int] = [0] * HOURS_IN_WEEK

    def set_execution_profile(self, profile: str, params: dict | None = None) -> None:
        """Установить профиль исполнения и параметры."""
        self.execution_profile = str(profile).upper()
        self.execution_params = dict(params or {})
        if self.execution_profile == "LIMIT_MID_BPS":
            self.limit_offset_bps = float(
                self.execution_params.get("limit_offset_bps", 0.0)
            )
            self.ttl_steps = int(self.execution_params.get("ttl_steps", 0))
            self.tif = str(self.execution_params.get("tif", "GTC")).upper()
        self._build_executor()

    def set_quantizer(self, q: Quantizer) -> None:
        self.quantizer = q

    def set_symbol(self, symbol: str) -> None:
        self.symbol = str(symbol).upper()
        self._latency_symbol = self.symbol
        self._reset_bar_capacity_base_cache()
        self._last_adv_bar_capacity = None

    def set_ref_price(self, price: float) -> None:
        self._last_ref_price = float(price)

    def set_next_open_price(self, price: float) -> None:
        self._next_h1_open_price = float(price)

    def set_adv_store(
        self,
        store: Optional[ADVStore],
        *,
        enabled: Optional[bool] = None,
        capacity_fraction: Optional[float] = None,
        bars_per_day_override: Optional[float] = None,
    ) -> None:
        """Attach or update ADV runtime configuration."""

        self._adv_store = store
        if store is not None:
            try:
                store.reset_runtime_state()
            except Exception:
                pass
        if capacity_fraction is not None:
            try:
                frac_val = float(capacity_fraction)
            except (TypeError, ValueError):
                frac_val = None
            else:
                if math.isfinite(frac_val):
                    if frac_val < 0.0:
                        frac_val = 0.0
                    self._adv_capacity_fraction = frac_val
        if bars_per_day_override is not None:
            try:
                bpd_val = float(bars_per_day_override)
            except (TypeError, ValueError):
                bpd_val = None
            else:
                if not math.isfinite(bpd_val) or bpd_val <= 0.0:
                    bpd_val = None
                self._adv_bars_per_day_override = bpd_val
        adv_enabled = self._adv_enabled
        if enabled is not None:
            adv_enabled = bool(enabled)
        self._adv_enabled = bool(store) and bool(adv_enabled)
        if store is None and enabled is None:
            self._adv_enabled = False
        if store is None:
            if bars_per_day_override is not None:
                self._adv_bars_per_day_override = None
            elif not self._adv_enabled:
                self._adv_bars_per_day_override = None
        self._last_adv_bar_capacity = None

    def _reset_bar_capacity_base_cache(self) -> None:
        self._bar_cap_base_cache = {}
        self._bar_cap_base_cache_path = None
        self._bar_cap_base_loaded_ts = None
        self._bar_cap_base_meta = {}
        self._bar_cap_base_warned_symbols.clear()

    def set_bar_capacity_base_config(
        self,
        *,
        enabled: Optional[bool] = None,
        capacity_frac_of_ADV_base: Optional[float] = None,
        floor_base: Optional[float] = None,
        adv_base_path: Optional[str] = None,
        timeframe_ms: Optional[int] = None,
    ) -> None:
        reset_cache = False
        config_changed = False

        if adv_base_path is not None:
            try:
                path_str = str(adv_base_path)
            except Exception:
                path_str = ""
            path_str = path_str.strip()
            if path_str:
                path_val = os.path.expanduser(path_str)
            else:
                path_val = None
            if path_val != self._bar_cap_base_path:
                self._bar_cap_base_path = path_val
                reset_cache = True
                config_changed = True

        if capacity_frac_of_ADV_base is not None:
            try:
                frac_val = float(capacity_frac_of_ADV_base)
            except (TypeError, ValueError):
                frac_val = None
            else:
                if not math.isfinite(frac_val):
                    frac_val = None
                elif frac_val < 0.0:
                    frac_val = 0.0
            if frac_val != self._bar_cap_base_frac:
                self._bar_cap_base_frac = frac_val
                config_changed = True

        if floor_base is not None:
            try:
                floor_val = float(floor_base)
            except (TypeError, ValueError):
                floor_val = None
            else:
                if not math.isfinite(floor_val) or floor_val < 0.0:
                    floor_val = None
            if floor_val != self._bar_cap_base_floor:
                self._bar_cap_base_floor = floor_val
                config_changed = True

        if timeframe_ms is not None:
            try:
                timeframe_val = int(timeframe_ms)
            except (TypeError, ValueError):
                timeframe_val = None
            else:
                if timeframe_val <= 0:
                    timeframe_val = None
            if timeframe_val != self._bar_cap_base_timeframe_ms:
                self._bar_cap_base_timeframe_ms = timeframe_val
                config_changed = True

        if enabled is not None:
            flag = bool(enabled)
            if flag != self._bar_cap_base_enabled:
                self._bar_cap_base_enabled = flag
                config_changed = True
                if not flag:
                    reset_cache = True

        if reset_cache or config_changed:
            self._reset_bar_capacity_base_cache()
            self._last_adv_bar_capacity = None

    def has_adv_store(self) -> bool:
        return self._adv_store is not None

    def get_bar_capacity_quote(self, symbol: Optional[str] = None) -> Optional[float]:
        store = self._adv_store
        if store is None:
            return None
        try:
            sym = str(symbol if symbol is not None else self.symbol or "").upper()
        except Exception:
            sym = ""
        if not sym:
            return None
        try:
            quote = store.get_bar_capacity_quote(sym)
        except Exception:
            quote = None
        if quote is None:
            default_q = store.default_quote
            if default_q is None:
                return None
            try:
                quote = float(default_q)
            except (TypeError, ValueError):
                return None
        try:
            val = float(quote)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(val) or val <= 0.0:
            return None
        return val

    def _load_adv_base_dataset(
        self, path: str
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        dataset: Dict[str, float] = {}
        meta: Dict[str, Any] = {}
        if not path:
            return dataset, meta
        try:
            safe_path = str(path)
        except Exception:
            return dataset, meta
        safe_path = safe_path.strip()
        if not safe_path:
            return dataset, meta
        meta["path"] = safe_path
        load_ts = now_ms()
        meta["load_ts"] = load_ts
        try:
            stat_info = os.stat(safe_path)
        except FileNotFoundError:
            logger.warning("ADV base dataset not found: %s", safe_path)
            return dataset, meta
        except OSError as exc:
            logger.warning("Failed to access ADV base dataset %s: %s", safe_path, exc)
            return dataset, meta
        meta["mtime"] = getattr(stat_info, "st_mtime", None)
        meta["size"] = getattr(stat_info, "st_size", None)
        try:
            with open(safe_path, "rb") as fh:
                raw = fh.read()
        except FileNotFoundError:
            logger.warning("ADV base dataset not found: %s", safe_path)
            return dataset, meta
        except OSError as exc:
            logger.warning("Failed to read ADV base dataset %s: %s", safe_path, exc)
            return dataset, meta
        meta["size"] = len(raw)
        if not raw:
            return dataset, meta
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse ADV base dataset %s: %s", safe_path, exc)
            return dataset, meta

        def _as_mapping(obj: Any) -> Dict[str, Any]:
            if isinstance(obj, Mapping):
                try:
                    return {str(k): v for k, v in obj.items()}
                except Exception:
                    return dict(obj)
            return {}

        candidates: List[Dict[str, Any]] = []
        top_mapping = _as_mapping(payload)
        if top_mapping:
            candidates.append(top_mapping)
            for key in ("data", "dataset", "symbols", "per_symbol", "values"):
                sub_mapping = _as_mapping(top_mapping.get(key))
                if sub_mapping:
                    candidates.append(sub_mapping)
        if isinstance(payload, Sequence) and not isinstance(
            payload, (str, bytes, bytearray)
        ):
            for item in payload:
                sub_mapping = _as_mapping(item)
                if sub_mapping:
                    candidates.append(sub_mapping)

        preferred_keys = (
            "adv_base",
            "adv",
            "daily_adv",
            "quote",
            "quote_value",
            "daily_quote",
            "value",
            "capacity",
        )

        for candidate_map in candidates:
            for raw_key, raw_value in candidate_map.items():
                try:
                    symbol_key = str(raw_key).strip().upper()
                except Exception:
                    continue
                if not symbol_key:
                    continue
                numeric_val: Optional[float] = None
                value = raw_value
                if isinstance(value, Mapping):
                    for pref_key in preferred_keys:
                        if pref_key in value:
                            try:
                                numeric_val = float(value[pref_key])
                            except (TypeError, ValueError):
                                numeric_val = None
                            else:
                                if math.isfinite(numeric_val):
                                    break
                                numeric_val = None
                    if numeric_val is None:
                        for nested_val in value.values():
                            try:
                                numeric_val = float(nested_val)
                            except (TypeError, ValueError):
                                continue
                            else:
                                if math.isfinite(numeric_val):
                                    break
                                numeric_val = None
                else:
                    try:
                        numeric_val = float(value)
                    except (TypeError, ValueError):
                        numeric_val = None
                if numeric_val is None:
                    continue
                if not math.isfinite(numeric_val):
                    continue
                if numeric_val < 0.0:
                    numeric_val = 0.0
            dataset[symbol_key] = numeric_val
        return dataset, meta

    def _resolve_cap_base_per_bar(
        self,
        symbol: Optional[str],
        timeframe_ms: Optional[int],
    ) -> Optional[float]:
        if not self._bar_cap_base_enabled:
            return 0.0
        path_val = self._bar_cap_base_path
        if not path_val:
            return 0.0
        try:
            path_key = str(path_val).strip()
        except Exception:
            return 0.0
        if not path_key:
            return 0.0

        now_ts = now_ms()
        cache_path = self._bar_cap_base_cache_path
        loaded_ts = self._bar_cap_base_loaded_ts
        cache_empty = not self._bar_cap_base_cache

        reload_needed = cache_path != path_key
        stat_info = None
        if not reload_needed:
            if loaded_ts is None:
                reload_needed = True
            else:
                try:
                    stat_info = os.stat(path_key)
                except OSError:
                    if cache_empty and now_ts - loaded_ts > 60_000:
                        reload_needed = True
                else:
                    cached_meta = self._bar_cap_base_meta or {}
                    cached_mtime = cached_meta.get("mtime")
                    cached_size = cached_meta.get("size")
                    stat_mtime = getattr(stat_info, "st_mtime", None)
                    stat_size = getattr(stat_info, "st_size", None)
                    if cached_mtime is not None:
                        try:
                            cached_mtime_val = float(cached_mtime)
                        except (TypeError, ValueError):
                            cached_mtime_val = None
                        else:
                            if (
                                cached_mtime_val is not None
                                and stat_mtime is not None
                                and stat_mtime > cached_mtime_val
                            ):
                                reload_needed = True
                    if not reload_needed and cached_size is not None:
                        try:
                            cached_size_val = int(cached_size)
                        except (TypeError, ValueError):
                            cached_size_val = None
                        else:
                            if (
                                cached_size_val is not None
                                and stat_size is not None
                                and stat_size != cached_size_val
                            ):
                                reload_needed = True
                    if (
                        not reload_needed
                        and cache_empty
                        and loaded_ts is not None
                        and now_ts - loaded_ts > 60_000
                    ):
                        reload_needed = True

        if reload_needed:
            dataset, meta = self._load_adv_base_dataset(path_key)
            self._bar_cap_base_cache = dataset
            self._bar_cap_base_meta = meta
            self._bar_cap_base_cache_path = path_key
            load_ts_val = meta.get("load_ts")
            try:
                self._bar_cap_base_loaded_ts = (
                    int(load_ts_val) if load_ts_val is not None else now_ts
                )
            except (TypeError, ValueError):
                self._bar_cap_base_loaded_ts = now_ts
            if dataset:
                self._bar_cap_base_warned_symbols.clear()
            cache_empty = not dataset

        dataset_map = self._bar_cap_base_cache
        try:
            sym_key = str(symbol if symbol is not None else self.symbol or "").upper()
        except Exception:
            sym_key = ""
        if not sym_key:
            return 0.0

        raw_value = dataset_map.get(sym_key)
        base_daily_val: Optional[float]
        if raw_value is not None:
            try:
                base_daily_val = float(raw_value)
            except (TypeError, ValueError):
                base_daily_val = None
            else:
                if not math.isfinite(base_daily_val):
                    base_daily_val = None
                elif base_daily_val < 0.0:
                    base_daily_val = 0.0
        else:
            base_daily_val = None

        floor_daily = self._bar_cap_base_floor
        if base_daily_val is None:
            fallback_val: Optional[float] = None
            if floor_daily is not None:
                try:
                    fallback_val = float(floor_daily)
                except (TypeError, ValueError):
                    fallback_val = None
                else:
                    if not math.isfinite(fallback_val) or fallback_val < 0.0:
                        fallback_val = None
            if fallback_val is None:
                if sym_key and sym_key not in self._bar_cap_base_warned_symbols:
                    logger.warning(
                        "ADV base dataset missing symbol %s and floor is not configured",
                        sym_key,
                    )
                    self._bar_cap_base_warned_symbols.add(sym_key)
                return 0.0
            base_daily_val = fallback_val
            if sym_key and sym_key not in self._bar_cap_base_warned_symbols:
                logger.warning(
                    "ADV base dataset missing symbol %s; using floor %.6g",
                    sym_key,
                    base_daily_val,
                )
                self._bar_cap_base_warned_symbols.add(sym_key)
        else:
            self._bar_cap_base_warned_symbols.discard(sym_key)
            if floor_daily is not None:
                try:
                    floor_val = float(floor_daily)
                except (TypeError, ValueError):
                    floor_val = None
                else:
                    if math.isfinite(floor_val):
                        if floor_val < 0.0:
                            floor_val = 0.0
                        if base_daily_val < floor_val:
                            base_daily_val = floor_val

        frac_val = self._bar_cap_base_frac
        if frac_val is None:
            frac_float = 1.0
        else:
            try:
                frac_float = float(frac_val)
            except (TypeError, ValueError):
                frac_float = 1.0
            else:
                if not math.isfinite(frac_float):
                    frac_float = 1.0
                elif frac_float < 0.0:
                    frac_float = 0.0

        timeframe_candidate: Optional[int]
        if timeframe_ms is not None:
            try:
                timeframe_candidate = int(timeframe_ms)
            except (TypeError, ValueError):
                timeframe_candidate = None
            else:
                if timeframe_candidate <= 0:
                    timeframe_candidate = None
        else:
            timeframe_candidate = None
        if timeframe_candidate is None:
            timeframe_candidate = self._bar_cap_base_timeframe_ms
        if timeframe_candidate is None or timeframe_candidate <= 0:
            timeframe_candidate = self._resolve_intrabar_timeframe(None)
        if timeframe_candidate is None or timeframe_candidate <= 0:
            return 0.0
        try:
            bars_per_day = 86_400_000.0 / float(timeframe_candidate)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(bars_per_day) or bars_per_day <= 0.0:
            return 0.0

        daily_capacity = base_daily_val * frac_float
        capacity_per_bar = daily_capacity / bars_per_day
        if not math.isfinite(capacity_per_bar):
            return 0.0
        if capacity_per_bar < 0.0:
            capacity_per_bar = 0.0
        return capacity_per_bar

    def _reset_bar_capacity_if_needed(self, ts_ms: int) -> float:
        try:
            ts_val = int(ts_ms)
        except (TypeError, ValueError):
            ts_val = int(now_ms())
        timeframe = self._resolve_intrabar_timeframe(ts_val)
        if timeframe <= 0:
            timeframe = 1
        try:
            bar_start = (ts_val // timeframe) * timeframe
        except Exception:
            bar_start = ts_val
        if self._capacity_bar_ts is None or bar_start != self._capacity_bar_ts:
            self._capacity_bar_ts = bar_start
            self._used_base_in_bar = {}
        cap = self._resolve_cap_base_per_bar(self.symbol, timeframe)
        if cap is None:
            return 0.0
        try:
            cap_float = float(cap)
        except (TypeError, ValueError):
            cap_float = 0.0
        if not math.isfinite(cap_float):
            cap_float = 0.0
        if cap_float < 0.0:
            cap_float = 0.0
        return cap_float

    def _record_bar_capacity_metrics(
        self,
        *,
        capacity_reason: str,
        exec_status: str,
        fill_ratio: float,
    ) -> None:
        if not capacity_reason:
            return
        labels = {
            "symbol": str(self.symbol),
            "capacity_reason": str(capacity_reason),
            "exec_status": str(exec_status or ""),
        }
        try:
            ratio_val = float(fill_ratio)
        except (TypeError, ValueError):
            ratio_val = 0.0
        if not math.isfinite(ratio_val):
            return
        if ratio_val < 0.0:
            ratio_val = 0.0
        _FILLS_CAPPED_COUNT_BASE.labels(**labels).inc()
        _FILLS_CAPPED_BASE_RATIO.labels(**labels).observe(ratio_val)

    def _resolve_adv_bars_per_day(self, timeframe_ms: Optional[int]) -> Optional[float]:
        override = self._adv_bars_per_day_override
        if override is not None and override > 0.0 and math.isfinite(override):
            return float(override)
        tf_val: Optional[float]
        if timeframe_ms is None:
            timeframe_guess = self._resolve_intrabar_timeframe(None)
            tf_val = float(timeframe_guess)
        else:
            try:
                tf_val = float(timeframe_ms)
            except (TypeError, ValueError):
                tf_val = None
        if tf_val is None or not math.isfinite(tf_val) or tf_val <= 0.0:
            return None
        bars = 86_400_000.0 / tf_val
        if not math.isfinite(bars) or bars <= 0.0:
            return None
        return bars

    def _adv_bar_capacity(
        self,
        symbol: Optional[str],
        timeframe_ms: Optional[int],
    ) -> Optional[float]:
        base_capacity = self._resolve_cap_base_per_bar(symbol, timeframe_ms)
        if not self._adv_enabled:
            return base_capacity
        store = self._adv_store
        if store is None:
            return base_capacity
        bars_per_day = self._resolve_adv_bars_per_day(timeframe_ms)
        if bars_per_day is None or bars_per_day <= 0.0:
            return base_capacity
        quote = self.get_bar_capacity_quote(symbol)
        if quote is None:
            return base_capacity
        capacity = quote / float(bars_per_day)
        frac = self._adv_capacity_fraction
        try:
            frac_val = float(frac)
        except (TypeError, ValueError):
            frac_val = None
        if frac_val is not None:
            if not math.isfinite(frac_val) or frac_val < 0.0:
                frac_val = None
        if frac_val is not None:
            capacity *= frac_val
        floor_daily = store.floor_quote
        floor_cap: Optional[float] = None
        if floor_daily is not None:
            try:
                floor_val = float(floor_daily)
            except (TypeError, ValueError):
                floor_val = None
            else:
                if math.isfinite(floor_val) and floor_val > 0.0:
                    floor_cap = floor_val / float(bars_per_day)
        if floor_cap is not None:
            capacity = max(capacity, floor_cap)
        if base_capacity is not None:
            try:
                base_val = float(base_capacity)
            except (TypeError, ValueError):
                base_val = None
            else:
                if math.isfinite(base_val):
                    if base_val < 0.0:
                        base_val = 0.0
                    capacity = max(capacity, base_val)
        if not math.isfinite(capacity):
            return base_capacity
        if capacity < 0.0:
            capacity = 0.0
        return capacity

    @staticmethod
    def _combine_liquidity(
        observed: Optional[float], adv_capacity: Optional[float]
    ) -> Optional[float]:
        result: Optional[float] = None
        for value in (observed, adv_capacity):
            if value is None:
                continue
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(val):
                continue
            if val < 0.0:
                val = 0.0
            result = val if result is None else min(result, val)
        return result

    @staticmethod
    def _float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result):
            return None
        return result

    @staticmethod
    def _non_negative_float(value: Any) -> Optional[float]:
        result = ExecutionSimulator._float_or_none(value)
        if result is None:
            return None
        if result < 0.0:
            result = 0.0
        return result

    @staticmethod
    def _resolve_mid_from_inputs(
        bid: Optional[float],
        ask: Optional[float],
        high: Optional[float],
        low: Optional[float],
        close: Optional[float],
        trade: Optional[float],
        ref: Optional[float],
    ) -> Optional[float]:
        if bid is not None and ask is not None:
            mid_val = (bid + ask) * 0.5
        elif high is not None and low is not None:
            mid_val = (high + low) * 0.5
        elif close is not None:
            mid_val = close
        elif trade is not None:
            mid_val = trade
        else:
            mid_val = ref
        mid = ExecutionSimulator._float_or_none(mid_val)
        if mid is None:
            return None
        if mid <= 0.0:
            return None
        return mid

    @staticmethod
    def _compute_bar_range_ratios(
        bar_high: Optional[float],
        bar_low: Optional[float],
        mid_price: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        high_val = ExecutionSimulator._float_or_none(bar_high)
        low_val = ExecutionSimulator._float_or_none(bar_low)
        mid_val = ExecutionSimulator._float_or_none(mid_price)
        if high_val is None or low_val is None or mid_val is None:
            return None, None
        if mid_val <= 0.0:
            return None, None
        price_range = high_val - low_val
        if not math.isfinite(price_range):
            return None, None
        if price_range < 0.0:
            price_range = abs(price_range)
        if price_range < 0.0:
            price_range = 0.0
        ratio = price_range / mid_val if mid_val > 0.0 else None
        if ratio is None or not math.isfinite(ratio):
            return None, None
        if ratio < 0.0:
            ratio = 0.0
        ratio_bps = ratio * 1e4
        return float(ratio), float(ratio_bps)

    def _default_spread_bps(self) -> Optional[float]:
        cfg = self.slippage_cfg
        if cfg is None:
            return None
        try:
            value = float(getattr(cfg, "default_spread_bps"))
        except (AttributeError, TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value

    def _call_spread_getter(self, kwargs: Dict[str, Any]) -> Optional[float]:
        getter = self._slippage_get_spread
        if not callable(getter):
            return None
        attempted = dict(kwargs)
        while True:
            try:
                return getter(**attempted)
            except TypeError as exc:
                message = str(exc)
                removed = False
                for key in list(attempted.keys()):
                    tokens = (f"'{key}'", f'"{key}"')
                    if any(tok in message for tok in tokens) and key in attempted:
                        attempted.pop(key)
                        removed = True
                        break
                if not removed:
                    return None
            except Exception:
                return None
        return None

    def _is_dynamic_trade_cost_enabled(self) -> bool:
        getter = getattr(self, "_slippage_get_trade_cost", None)
        if not callable(getter):
            return False
        cfg = self.slippage_cfg
        if cfg is None:
            return False
        detector = getattr(cfg, "dynamic_trade_cost_enabled", None)
        if callable(detector):
            try:
                if bool(detector()):
                    return True
            except Exception:
                pass
        block: Any = None
        getter_fn = getattr(cfg, "get_dynamic_block", None)
        if callable(getter_fn):
            try:
                block = getter_fn()
            except Exception:
                block = None
        if block is None:
            block = getattr(cfg, "dynamic_spread", None)
        if block is None:
            block = getattr(cfg, "dynamic", None)

        def _flag_enabled(candidate: Any) -> bool:
            if candidate is None:
                return False
            if isinstance(candidate, Mapping):
                flag = candidate.get("enabled")
            else:
                flag = getattr(candidate, "enabled", None)
            try:
                return bool(flag)
            except Exception:
                return False

        dynamic_enabled = False
        if block is not None:
            if isinstance(block, Mapping):
                enabled = block.get("enabled")
            else:
                enabled = getattr(block, "enabled", False)
            try:
                dynamic_enabled = bool(enabled)
            except Exception:
                dynamic_enabled = False
        if dynamic_enabled:
            return True
        tail_cfg = getattr(cfg, "tail_shock", None)
        if _flag_enabled(tail_cfg):
            prob_val = None
            extra = None
            if isinstance(tail_cfg, Mapping):
                prob_val = tail_cfg.get("probability")
                extra = tail_cfg.get("extra")
            else:
                prob_val = getattr(tail_cfg, "probability", None)
                extra = getattr(tail_cfg, "extra", None)
            try:
                prob_float = float(prob_val) if prob_val is not None else 0.0
            except (TypeError, ValueError):
                prob_float = 0.0
            mode = None
            if isinstance(extra, Mapping):
                mode = extra.get("mode")
            if isinstance(mode, str) and mode.strip().lower() == "off":
                prob_float = 0.0
            if prob_float > 0.0:
                return True
        impact_cfg = getattr(cfg, "dynamic_impact", None)
        if _flag_enabled(impact_cfg):
            return True
        adv_cfg = getattr(cfg, "adv", None)
        if _flag_enabled(adv_cfg):
            return True
        return False

    def _call_trade_cost_getter(
        self, kwargs: Dict[str, Any]
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        getter = getattr(self, "_slippage_get_trade_cost", None)
        if not callable(getter):
            return None, {}
        attempted = dict(kwargs)
        while True:
            try:
                result = getter(**attempted)
                return result, dict(attempted)
            except TypeError as exc:
                message = str(exc)
                removed = False
                for key in list(attempted.keys()):
                    tokens = (f"'{key}'", f'"{key}"')
                    if any(tok in message for tok in tokens) and key in attempted:
                        attempted.pop(key)
                        removed = True
                        break
                if not removed:
                    return None, dict(attempted)
            except Exception:
                return None, dict(attempted)
        return None, dict(attempted)

    @staticmethod
    def _trade_cost_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(num):
            return None
        return num

    @staticmethod
    def _trade_cost_non_negative(value: Any, default: float = 0.0) -> float:
        candidate = ExecutionSimulator._trade_cost_float(value)
        if candidate is None:
            candidate = float(default)
        if candidate < 0.0:
            candidate = 0.0
        return float(candidate)

    @staticmethod
    def _trade_cost_share(value: Any, default: float = 0.0) -> float:
        candidate = ExecutionSimulator._trade_cost_float(value)
        if candidate is None:
            candidate = float(default)
        if candidate < 0.0:
            candidate = 0.0
        elif candidate > 1.0:
            candidate = 1.0
        return float(candidate)

    def _initialise_fee_payloads(
        self,
        payloads: Sequence[Mapping[str, Any]],
        share_payload: Optional[Mapping[str, Any]],
    ) -> None:
        """Best-effort ingestion of extended fee payloads at construction time."""

        config_payload: Optional[Mapping[str, Any]] = None
        metadata_payload: Optional[Mapping[str, Any]] = None
        expected_payload: Optional[Mapping[str, Any]] = None

        for payload in payloads:
            if not isinstance(payload, Mapping):
                continue
            if config_payload is None:
                config_payload = self._extract_fee_config_payload(payload)
            if metadata_payload is None:
                metadata_payload = self._extract_fee_metadata(payload)
            if expected_payload is None:
                expected_payload = self._extract_fee_expected(payload)

        self.set_fees_config(config_payload, metadata=metadata_payload, expected_payload=expected_payload)

        if share_payload is not None:
            self.set_fees_config(None, share_payload=share_payload)

    def _extract_fee_config_payload(
        self, payload: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        candidates = (
            payload.get("fees_config_payload"),
            payload.get("config"),
            payload.get("model_payload"),
            payload.get("model"),
        )
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                normalised = ExecutionSimulator._plain_mapping(candidate)
                if normalised:
                    return normalised
        keys = {
            "maker_bps",
            "taker_bps",
            "use_bnb_discount",
            "maker_discount_mult",
            "taker_discount_mult",
            "vip_tier",
            "fee_rounding_step",
            "symbol_fee_table",
        }
        subset: Dict[str, Any] = {}
        for key in keys:
            if key in payload:
                subset[key] = payload[key]
        if subset:
            return ExecutionSimulator._plain_mapping(subset)
        return None

    def _extract_fee_metadata(
        self, payload: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        for key in ("fees_metadata", "metadata", "meta"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                normalised = ExecutionSimulator._plain_mapping(candidate)
                if normalised:
                    return normalised
        return None

    def _extract_fee_expected(
        self, payload: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        for key in ("fees_expected_payload", "expected"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                normalised = ExecutionSimulator._plain_mapping(candidate)
                if normalised:
                    return normalised
        return None

    def _apply_maker_taker_share_payload(
        self, payload: Optional[Mapping[str, Any]]
    ) -> None:
        if payload is None:
            self._maker_taker_share_cfg = None
            self._maker_taker_share_enabled = False
            self._maker_taker_share_mode = None
            self._expected_fee_bps = 0.0
            self._expected_maker_share = 0.0
            return

        mapping = ExecutionSimulator._plain_mapping(payload)
        self._maker_taker_share_cfg = mapping or None
        if not mapping:
            self._maker_taker_share_enabled = False
            self._maker_taker_share_mode = None
            self._expected_fee_bps = 0.0
            self._expected_maker_share = 0.0
            return

        enabled = bool(mapping.get("enabled", self._maker_taker_share_enabled))
        self._maker_taker_share_enabled = enabled
        mode = mapping.get("mode")
        self._maker_taker_share_mode = str(mode) if isinstance(mode, str) else None
        self._refresh_expected_fee_inputs()

    def _refresh_expected_fee_inputs(self) -> None:
        share_payload = (
            self._maker_taker_share_cfg
            if isinstance(self._maker_taker_share_cfg, Mapping)
            else None
        )
        expected_payload = (
            self.fees_expected_payload
            if isinstance(self.fees_expected_payload, Mapping)
            else None
        )

        share_candidates: List[Any] = []
        if share_payload:
            share_candidates.append(share_payload.get("maker_share"))
            share_candidates.append(share_payload.get("maker_share_default"))
        if expected_payload:
            share_candidates.append(expected_payload.get("maker_share"))
            share_candidates.append(expected_payload.get("maker_share_default"))

        maker_share_val: Optional[float] = None
        for candidate in share_candidates:
            share = ExecutionSimulator._trade_cost_float(candidate)
            if share is None:
                continue
            maker_share_val = share
            break
        if maker_share_val is None:
            maker_share_val = ExecutionSimulator._trade_cost_float(self._expected_maker_share)
        if maker_share_val is None:
            maker_share_val = 0.0
        maker_share_val = min(max(float(maker_share_val), 0.0), 1.0)
        self._expected_maker_share = maker_share_val

        fee_candidates: List[Any] = []
        if share_payload:
            fee_candidates.append(share_payload.get("expected_fee_bps"))
        if expected_payload:
            fee_candidates.append(expected_payload.get("expected_fee_bps"))

        maker_fee = None
        taker_fee = None
        if share_payload:
            maker_fee = ExecutionSimulator._trade_cost_float(
                share_payload.get("maker_fee_bps")
            )
            taker_fee = ExecutionSimulator._trade_cost_float(
                share_payload.get("taker_fee_bps")
            )
        if maker_fee is None or taker_fee is None:
            if expected_payload:
                if maker_fee is None:
                    maker_fee = ExecutionSimulator._trade_cost_float(
                        expected_payload.get("maker_fee_bps")
                    )
                if taker_fee is None:
                    taker_fee = ExecutionSimulator._trade_cost_float(
                        expected_payload.get("taker_fee_bps")
                    )
        if maker_fee is not None and taker_fee is not None:
            fee_candidates.append(
                maker_share_val * maker_fee + (1.0 - maker_share_val) * taker_fee
            )

        expected_fee_val: Optional[float] = None
        for candidate in fee_candidates:
            value = ExecutionSimulator._trade_cost_float(candidate)
            if value is None:
                continue
            expected_fee_val = value
            break
        if expected_fee_val is None:
            expected_fee_val = ExecutionSimulator._trade_cost_float(self._expected_fee_bps)
        if expected_fee_val is None:
            expected_fee_val = 0.0
        if not math.isfinite(expected_fee_val) or expected_fee_val < 0.0:
            expected_fee_val = 0.0
        self._expected_fee_bps = float(expected_fee_val)

    @staticmethod
    def _fees_symbol_key(symbol: Optional[str]) -> Optional[str]:
        if not symbol or not isinstance(symbol, str):
            return None
        key = symbol.strip().upper()
        return key or None

    def _resolve_fee_discount_multiplier(
        self,
        symbol: Optional[str],
        is_maker: bool,
        fee_model: Any,
    ) -> float:
        multiplier: Optional[float] = None
        discount_fn = getattr(fee_model, "_discount_multiplier", None)
        if callable(discount_fn):
            try:
                multiplier = float(discount_fn(symbol, is_maker))
            except Exception:
                logger.debug(
                    "fees._discount_multiplier failed for %s/%s",
                    symbol,
                    "maker" if is_maker else "taker",
                    exc_info=True,
                )
                multiplier = None
        if multiplier is None or not math.isfinite(multiplier) or multiplier <= 0.0:
            fallback = self._fallback_discount_multiplier(symbol, is_maker)
            if fallback is not None:
                multiplier = fallback
        if multiplier is None or not math.isfinite(multiplier) or multiplier <= 0.0:
            multiplier = 1.0
        return float(multiplier)

    def _fallback_discount_multiplier(
        self, symbol: Optional[str], is_maker: bool
    ) -> Optional[float]:
        key = "maker_discount_mult" if is_maker else "taker_discount_mult"
        sources: List[Any] = []
        if isinstance(self.fees_expected_payload, Mapping):
            sources.append(self.fees_expected_payload.get(key))
        if isinstance(self.fees_config_payload, Mapping):
            sources.append(self.fees_config_payload.get(key))
        symbol_key = ExecutionSimulator._fees_symbol_key(symbol)
        if symbol_key and isinstance(self.fees_symbol_fee_table, Mapping):
            symbol_cfg = self.fees_symbol_fee_table.get(symbol_key)
            if isinstance(symbol_cfg, Mapping):
                sources.append(symbol_cfg.get(key))
        table_meta = None
        if isinstance(self.fees_metadata, Mapping):
            table_meta = self.fees_metadata.get("table")
        if isinstance(table_meta, Mapping):
            account_overrides = table_meta.get("account_overrides")
            if isinstance(account_overrides, Mapping):
                sources.append(account_overrides.get(key))
        for candidate in sources:
            value = ExecutionSimulator._trade_cost_float(candidate)
            if value is None or value <= 0.0:
                continue
            return float(value)
        return None

    def _fallback_fee_rate(
        self, symbol: Optional[str], is_maker: bool
    ) -> Optional[float]:
        key = "maker_fee_bps" if is_maker else "taker_fee_bps"
        if isinstance(self.fees_expected_payload, Mapping):
            candidate = ExecutionSimulator._trade_cost_float(
                self.fees_expected_payload.get(key)
            )
            if candidate is not None and candidate >= 0.0:
                return float(candidate)
        symbol_key = ExecutionSimulator._fees_symbol_key(symbol)
        if symbol_key and isinstance(self.fees_symbol_fee_table, Mapping):
            symbol_cfg = self.fees_symbol_fee_table.get(symbol_key)
            if isinstance(symbol_cfg, Mapping):
                rate_candidate = ExecutionSimulator._trade_cost_float(
                    symbol_cfg.get("maker_bps" if is_maker else "taker_bps")
                )
                if rate_candidate is not None:
                    discount = self._fallback_discount_multiplier(symbol, is_maker)
                    if discount is None:
                        discount = 1.0
                    return float(max(rate_candidate * discount, 0.0))
        if isinstance(self.fees_config_payload, Mapping):
            rate_candidate = ExecutionSimulator._trade_cost_float(
                self.fees_config_payload.get("maker_bps" if is_maker else "taker_bps")
            )
            if rate_candidate is not None:
                discount = self._fallback_discount_multiplier(symbol, is_maker)
                if discount is None:
                    discount = 1.0
                return float(max(rate_candidate * discount, 0.0))
        expected_default = ExecutionSimulator._trade_cost_float(self._expected_fee_bps)
        if expected_default is not None and expected_default >= 0.0:
            return float(expected_default)
        return None

    def _resolve_fee_rate_bps(
        self,
        *,
        symbol: Optional[str],
        is_maker: bool,
    ) -> Optional[float]:
        fee_model = getattr(self, "fees", None)
        rate_bps: Optional[float] = None
        if fee_model is not None:
            get_bps = getattr(fee_model, "get_fee_bps", None)
            if callable(get_bps):
                try:
                    raw_rate = get_bps(symbol, is_maker)
                except Exception:
                    logger.debug(
                        "fees.get_fee_bps failed for %s/%s",
                        symbol,
                        "maker" if is_maker else "taker",
                        exc_info=True,
                    )
                    raw_rate = None
                rate_bps = ExecutionSimulator._trade_cost_float(raw_rate)
                if rate_bps is not None:
                    discount = self._resolve_fee_discount_multiplier(
                        symbol, is_maker, fee_model
                    )
                    rate_bps = float(rate_bps * discount)
        if rate_bps is None:
            rate_bps = self._fallback_fee_rate(symbol, is_maker)
        if rate_bps is None or not math.isfinite(rate_bps) or rate_bps < 0.0:
            return None
        return float(rate_bps)

    def set_fees_config(
        self,
        config_payload: Optional[Mapping[str, Any]],
        share_payload: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        expected_payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Attach extended fee configuration payloads from :class:`FeesImpl`."""

        if config_payload is not None:
            mapping = ExecutionSimulator._plain_mapping(config_payload)
            if mapping:
                self.fees_config_payload = mapping
                table_payload = mapping.get("symbol_fee_table")
                if isinstance(table_payload, Mapping):
                    normalised: Dict[str, Any] = {}
                    for symbol, data in table_payload.items():
                        if not isinstance(symbol, str) or not isinstance(data, Mapping):
                            continue
                        normalised[symbol.upper()] = ExecutionSimulator._plain_mapping(data)
                    self.fees_symbol_fee_table = normalised
            else:
                logger.debug("Received empty fees config payload; ignoring")

        if metadata is not None:
            mapping = ExecutionSimulator._plain_mapping(metadata)
            if mapping:
                self.fees_metadata = mapping
            else:
                logger.debug("Received empty fees metadata payload; ignoring")

        if expected_payload is not None:
            mapping = ExecutionSimulator._plain_mapping(expected_payload)
            if mapping:
                self.fees_expected_payload = mapping
            else:
                logger.debug("Received empty expected fee payload; ignoring")

        if share_payload is not None:
            self._apply_maker_taker_share_payload(share_payload)
        else:
            # Share settings may change implicitly when expected payload updates.
            self._refresh_expected_fee_inputs()

    def _refresh_slippage_share_info(self) -> None:
        getter = getattr(self, "_slippage_get_maker_taker_share_info", None)
        if not callable(getter):
            return
        try:
            payload = getter()
        except Exception:
            return
        if not isinstance(payload, Mapping):
            return
        enabled = bool(payload.get("enabled", self._slippage_share_enabled))
        maker_share = self._trade_cost_share(
            payload.get("maker_share"), self._slippage_share_default
        )
        maker_cost = self._trade_cost_non_negative(
            payload.get("spread_cost_maker_bps"), self._slippage_spread_cost_maker_bps
        )
        taker_cost = self._trade_cost_non_negative(
            payload.get("spread_cost_taker_bps"), self._slippage_spread_cost_taker_bps
        )
        self._slippage_share_enabled = enabled
        self._slippage_share_default = maker_share
        self._slippage_spread_cost_maker_bps = maker_cost
        self._slippage_spread_cost_taker_bps = taker_cost

    def _blend_expected_spread(
        self,
        *,
        taker_bps: Optional[float],
        maker_bps: Optional[float] = None,
        maker_share: Optional[float] = None,
    ) -> Optional[float]:
        self._refresh_slippage_share_info()
        taker_val = self._trade_cost_float(taker_bps)
        if taker_val is None:
            taker_val = self._trade_cost_float(self._slippage_spread_cost_taker_bps)
        if taker_val is None:
            return None
        taker_val = max(0.0, float(taker_val))
        if not self._slippage_share_enabled:
            return taker_val
        share_val = self._trade_cost_share(maker_share, self._slippage_share_default)
        maker_val = self._trade_cost_non_negative(
            maker_bps, self._slippage_spread_cost_maker_bps
        )
        expected = share_val * maker_val + (1.0 - share_val) * taker_val
        if not math.isfinite(expected):
            expected = taker_val
        if expected < 0.0:
            expected = 0.0
        return float(expected)

    def _trade_cost_expected_bps(self, trade_cost: _TradeCostResult) -> float:
        expected = self._trade_cost_float(getattr(trade_cost, "expected_spread_bps", None))
        if expected is None and trade_cost.metrics:
            expected = self._trade_cost_float(trade_cost.metrics.get("expected_spread_bps"))
        taker = self._trade_cost_float(trade_cost.bps)
        if expected is None:
            expected = taker
        if expected is None:
            return 0.0
        return max(0.0, float(expected))

    def _compute_dynamic_trade_cost_bps(
        self,
        *,
        side: str,
        qty: float,
        spread_bps: Optional[float],
        base_price: Optional[float],
        liquidity: Optional[float],
        vol_factor: Optional[float],
        order_seq: Optional[int],
        bar_close_ts: Optional[int] = None,
    ) -> Optional[_TradeCostResult]:
        if not self._is_dynamic_trade_cost_enabled():
            return None
        qty_val: float
        try:
            qty_val = float(qty)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(qty_val) or qty_val <= 0.0:
            return None
        if base_price is not None:
            try:
                base_val = float(base_price)
            except (TypeError, ValueError):
                base_val = None
            else:
                if not math.isfinite(base_val) or base_val <= 0.0:
                    base_val = None
        else:
            base_val = None
        mid_price = None
        if mid_from_quotes is not None:
            try:
                mid_price = mid_from_quotes(
                    bid=self._last_bid,
                    ask=self._last_ask,
                )
            except Exception:
                mid_price = None
        if mid_price is None:
            mid_price = base_val
        kwargs: Dict[str, Any] = {
            "side": side,
            "qty": qty_val,
        }
        symbol_value = getattr(self, "symbol", None)
        if symbol_value is not None:
            try:
                kwargs["symbol"] = str(symbol_value)
            except Exception:
                pass
        if mid_price is not None and math.isfinite(float(mid_price)):
            try:
                kwargs["mid"] = float(mid_price)
            except (TypeError, ValueError):
                pass
        if spread_bps is not None:
            try:
                sbps_val = float(spread_bps)
            except (TypeError, ValueError):
                sbps_val = None
            else:
                if math.isfinite(sbps_val):
                    kwargs["spread_bps"] = sbps_val
        bar_ts_val = bar_close_ts
        if bar_ts_val is None:
            bar_ts_val = getattr(self, "_last_bar_close_ts", None)
        if bar_ts_val is not None:
            try:
                kwargs["bar_close_ts"] = int(bar_ts_val)
            except (TypeError, ValueError):
                pass
        if order_seq is not None:
            try:
                kwargs["order_seq"] = int(order_seq)
            except (TypeError, ValueError):
                pass
        metrics: Dict[str, Any] = {}
        raw_metrics = getattr(self, "_last_vol_raw", None)
        if isinstance(raw_metrics, Mapping):
            try:
                metrics.update(dict(raw_metrics))
            except Exception:
                metrics = {}
        if vol_factor is not None:
            try:
                vf_val = float(vol_factor)
            except (TypeError, ValueError):
                vf_val = None
            else:
                if math.isfinite(vf_val):
                    metrics["vol_factor"] = vf_val
        if liquidity is not None:
            try:
                liq_val = float(liquidity)
            except (TypeError, ValueError):
                liq_val = None
            else:
                if math.isfinite(liq_val) and liq_val > 0.0:
                    metrics["liquidity"] = liq_val
        if base_val is not None:
            notional = base_val * qty_val
            if math.isfinite(notional) and notional > 0.0:
                metrics["notional"] = notional
            metrics["mid"] = base_val
        if spread_bps is not None:
            try:
                sbps_val = float(spread_bps)
            except (TypeError, ValueError):
                sbps_val = None
            else:
                if math.isfinite(sbps_val):
                    metrics.setdefault("spread_bps", sbps_val)
        metrics_payload: Dict[str, Any] = {}
        for key, value in metrics.items():
            try:
                metrics_payload[key] = float(value)
            except (TypeError, ValueError):
                pass
        if metrics_payload:
            kwargs["vol_metrics"] = dict(metrics_payload)
        result, used_kwargs = self._call_trade_cost_getter(kwargs)
        if result is None:
            return None
        try:
            value = float(result)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        mid_used = used_kwargs.get("mid")
        if mid_used is None:
            mid_used = kwargs.get("mid")
        try:
            mid_out = float(mid_used) if mid_used is not None else None
        except (TypeError, ValueError):
            mid_out = None
        safe_inputs: Dict[str, Any] = {}
        for key, val in used_kwargs.items():
            if key == "vol_metrics":
                continue
            if isinstance(val, (int, float, str, bool)):
                safe_inputs[key] = val
                continue
            try:
                safe_inputs[key] = float(val)
            except (TypeError, ValueError):
                safe_inputs[key] = val
        self._refresh_slippage_share_info()
        metrics_out: Dict[str, Any] = dict(metrics_payload)
        taker_metric = max(0.0, float(value))
        metrics_out["taker_spread_bps"] = taker_metric
        metrics_out["spread_cost_taker_bps"] = taker_metric
        expected_spread_bps: Optional[float] = None
        meta_payload: Optional[Mapping[str, Any]] = None
        meta_getter = getattr(self, "_slippage_consume_trade_cost_meta", None)
        if callable(meta_getter):
            try:
                meta_candidate = meta_getter()
            except Exception:
                meta_candidate = None
            if isinstance(meta_candidate, Mapping):
                meta_payload = dict(meta_candidate)
        if meta_payload:
            share_override = meta_payload.get("maker_share")
            maker_override = meta_payload.get("spread_cost_maker_bps")
            taker_override = meta_payload.get("spread_cost_taker_bps")
            expected_override = meta_payload.get("expected_spread_bps")
            taker_override_val = self._trade_cost_float(taker_override)
            if taker_override_val is not None:
                taker_metric = max(0.0, taker_override_val)
                metrics_out["taker_spread_bps"] = taker_metric
                metrics_out["spread_cost_taker_bps"] = taker_metric
            maker_cost_val = self._trade_cost_non_negative(
                maker_override, self._slippage_spread_cost_maker_bps
            )
            metrics_out["spread_cost_maker_bps"] = maker_cost_val
            share_val = self._trade_cost_share(
                share_override, self._slippage_share_default
            )
            if share_override is not None or self._slippage_share_enabled:
                metrics_out["maker_share"] = share_val
            expected_override_val = self._trade_cost_float(expected_override)
            if expected_override_val is not None:
                expected_spread_bps = max(0.0, expected_override_val)
            else:
                expected_spread_bps = self._blend_expected_spread(
                    taker_bps=taker_metric,
                    maker_bps=maker_override,
                    maker_share=share_override,
                )
        if expected_spread_bps is None:
            expected_spread_bps = self._blend_expected_spread(taker_bps=taker_metric)
        if expected_spread_bps is None:
            expected_spread_bps = taker_metric
        if self._slippage_share_enabled and "maker_share" not in metrics_out:
            metrics_out["maker_share"] = self._slippage_share_default
        if self._slippage_share_enabled and "spread_cost_maker_bps" not in metrics_out:
            metrics_out["spread_cost_maker_bps"] = self._slippage_spread_cost_maker_bps
        metrics_out["expected_spread_bps"] = float(expected_spread_bps)
        return _TradeCostResult(
            bps=float(value),
            mid=mid_out,
            base_price=base_val,
            inputs=safe_inputs,
            metrics=metrics_out,
            expected_spread_bps=float(expected_spread_bps),
        )

    def _apply_trade_cost_price(
        self,
        *,
        side: str,
        pre_slip_price: float,
        trade_cost: _TradeCostResult,
    ) -> float:
        mid_candidate = trade_cost.mid
        mid_val: Optional[float] = None
        if mid_candidate is not None:
            try:
                mid_val = float(mid_candidate)
            except (TypeError, ValueError):
                mid_val = None
        if mid_val is None or not math.isfinite(mid_val) or mid_val <= 0.0:
            mid_candidate = trade_cost.base_price
            if mid_candidate is not None:
                try:
                    mid_val = float(mid_candidate)
                except (TypeError, ValueError):
                    mid_val = None
        if mid_val is None or not math.isfinite(mid_val) or mid_val <= 0.0:
            return float(pre_slip_price)
        expected_bps = self._trade_cost_expected_bps(trade_cost)
        try:
            cost_fraction = float(expected_bps) / 1e4
        except (TypeError, ValueError):
            return float(pre_slip_price)
        if not math.isfinite(cost_fraction):
            return float(pre_slip_price)
        side_key = str(side).upper()
        if side_key == "BUY":
            candidate = mid_val * (1.0 + cost_fraction)
        else:
            candidate = mid_val * (1.0 - cost_fraction)
        if not math.isfinite(candidate) or candidate <= 0.0:
            return float(pre_slip_price)
        return float(candidate)

    def _compute_trade_fee(
        self,
        *,
        side: str,
        price: float,
        qty: float,
        liquidity: str,
    ) -> float:
        side_key = str(side).upper()
        liquidity_key = str(liquidity).lower()
        if liquidity_key not in ("maker", "taker") and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "fee role fallback side=%s liquidity=%s", side_key, liquidity_key
            )

        try:
            price_val = float(price)
            qty_val = float(qty)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(price_val) or not math.isfinite(qty_val):
            return 0.0
        notional = abs(price_val * qty_val)
        if notional <= 0.0:
            return 0.0

        symbol_value: Optional[str] = getattr(self, "symbol", None)
        is_maker = liquidity_key == "maker"
        rate_bps = self._resolve_fee_rate_bps(symbol=symbol_value, is_maker=is_maker)
        fee_model = getattr(self, "fees", None)

        if rate_bps is not None:
            fee_amount = notional * (rate_bps / 1e4)
            round_fn = getattr(fee_model, "_round_fee", None) if fee_model is not None else None
            if callable(round_fn):
                try:
                    fee_amount = round_fn(fee_amount, symbol_value)
                except Exception:
                    logger.debug(
                        "fees._round_fee failed for %s/%s",
                        symbol_value,
                        "maker" if is_maker else "taker",
                        exc_info=True,
                    )
            if not math.isfinite(fee_amount) or fee_amount <= 0.0:
                fee_amount = 0.0
            return float(fee_amount)

        if fee_model is not None:
            compute_fn = getattr(fee_model, "compute", None)
            if callable(compute_fn):
                try:
                    fee_val = compute_fn(
                        side=side,
                        price=price_val,
                        qty=qty_val,
                        liquidity=liquidity,
                        symbol=symbol_value,
                    )
                except Exception:
                    logger.debug(
                        "fees.compute fallback failed for %s/%s",
                        symbol_value,
                        "maker" if is_maker else "taker",
                        exc_info=True,
                    )
                else:
                    fee_num = ExecutionSimulator._trade_cost_float(fee_val)
                    if fee_num is not None and fee_num > 0.0:
                        return float(fee_num)

        fallback_rate = self._fallback_fee_rate(symbol_value, is_maker)
        if fallback_rate is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "fee rate fallback used for %s/%s -> %.6f bps",
                    symbol_value,
                    "maker" if is_maker else "taker",
                    float(fallback_rate),
                )
            fee_amount = notional * (fallback_rate / 1e4)
            if math.isfinite(fee_amount) and fee_amount > 0.0:
                return float(fee_amount)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "fee calculation failed for %s/%s; returning 0",
                symbol_value,
                "maker" if is_maker else "taker",
            )
        return 0.0

    def _expected_cost_snapshot(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Dict[str, Optional[float]]]:
        self._refresh_expected_fee_inputs()
        share_cfg = getattr(self, "_maker_taker_share_cfg", None)
        share_enabled = bool(getattr(self, "_maker_taker_share_enabled", False))
        if isinstance(share_cfg, Mapping):
            share_enabled = bool(share_cfg.get("enabled", share_enabled))

        expected_payload = (
            self.fees_expected_payload
            if isinstance(self.fees_expected_payload, Mapping)
            else None
        )

        share_default = float(getattr(self, "_expected_maker_share", 0.0))
        share_candidate: Any = None
        if isinstance(share_cfg, Mapping):
            share_candidate = share_cfg.get("maker_share")
            if share_candidate is None:
                share_candidate = share_cfg.get("maker_share_default")
        if share_candidate is None and expected_payload is not None:
            share_candidate = expected_payload.get("maker_share")
            if share_candidate is None:
                share_candidate = expected_payload.get("maker_share_default")
        maker_share_val = ExecutionSimulator._trade_cost_float(share_candidate)
        if maker_share_val is None:
            maker_share_val = share_default
        maker_share_out = float(min(max(maker_share_val, 0.0), 1.0))
        if not share_enabled:
            maker_share_out = 0.0

        maker_fee_val = None
        taker_fee_val = None
        if isinstance(share_cfg, Mapping):
            maker_fee_val = ExecutionSimulator._trade_cost_float(
                share_cfg.get("maker_fee_bps")
            )
            taker_fee_val = ExecutionSimulator._trade_cost_float(
                share_cfg.get("taker_fee_bps")
            )
        if expected_payload is not None:
            if maker_fee_val is None:
                maker_fee_val = ExecutionSimulator._trade_cost_float(
                    expected_payload.get("maker_fee_bps")
                )
            if taker_fee_val is None:
                taker_fee_val = ExecutionSimulator._trade_cost_float(
                    expected_payload.get("taker_fee_bps")
                )

        expected_fee_candidates: List[Any] = []
        if isinstance(share_cfg, Mapping):
            expected_fee_candidates.append(share_cfg.get("expected_fee_bps"))
        if expected_payload is not None:
            expected_fee_candidates.append(expected_payload.get("expected_fee_bps"))
        if maker_fee_val is not None and taker_fee_val is not None:
            expected_fee_candidates.append(
                maker_share_out * maker_fee_val
                + (1.0 - maker_share_out) * taker_fee_val
            )
        expected_fee_candidates.append(self._expected_fee_bps)

        expected_fee_out: Optional[float] = None
        for candidate in expected_fee_candidates:
            value = ExecutionSimulator._trade_cost_float(candidate)
            if value is None:
                continue
            if not math.isfinite(value) or value < 0.0:
                continue
            expected_fee_out = float(value)
            break
        if not share_enabled:
            expected_fee_out = 0.0 if expected_fee_out is not None else 0.0

        spread_maker_val: Optional[float] = None
        spread_taker_val: Optional[float] = None
        if isinstance(share_cfg, Mapping):
            spread_maker_val = ExecutionSimulator._trade_cost_float(
                share_cfg.get("spread_cost_maker_bps")
            )
            spread_taker_val = ExecutionSimulator._trade_cost_float(
                share_cfg.get("spread_cost_taker_bps")
            )
        if expected_payload is not None:
            if spread_maker_val is None:
                spread_maker_val = ExecutionSimulator._trade_cost_float(
                    expected_payload.get("spread_cost_maker_bps")
                )
            if spread_taker_val is None:
                spread_taker_val = ExecutionSimulator._trade_cost_float(
                    expected_payload.get("spread_cost_taker_bps")
                )
        expected_spread_val = self._blend_expected_spread(
            taker_bps=spread_taker_val,
            maker_bps=spread_maker_val,
            maker_share=maker_share_out if share_enabled else None,
        )
        if expected_spread_val is None:
            expected_spread_val = self._blend_expected_spread(
                taker_bps=self._last_spread_bps,
                maker_share=maker_share_out if share_enabled else None,
            )
        expected_spread_out: Optional[float]
        if expected_spread_val is None:
            expected_spread_out = None
        else:
            try:
                spread_num = float(expected_spread_val)
            except (TypeError, ValueError):
                expected_spread_out = None
            else:
                if math.isfinite(spread_num):
                    expected_spread_out = max(0.0, spread_num)
                else:
                    expected_spread_out = None
        if not share_enabled:
            expected_spread_out = None

        components: Dict[str, Optional[float]] = {}
        if share_enabled and expected_fee_out is not None:
            components["fee_bps"] = float(expected_fee_out)
        else:
            components["fee_bps"] = 0.0 if share_enabled else 0.0
        if expected_spread_out is not None:
            components["spread_bps"] = float(expected_spread_out)
        if maker_fee_val is not None:
            components.setdefault("maker_fee_bps", float(max(maker_fee_val, 0.0)))
        if taker_fee_val is not None:
            components.setdefault("taker_fee_bps", float(max(taker_fee_val, 0.0)))
        total_val = 0.0
        total_count = 0
        for name, value in components.items():
            if value is None:
                continue
            if name in {"maker_fee_bps", "taker_fee_bps"}:
                continue
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(num):
                continue
            total_val += num
            total_count += 1
        if total_count:
            components["total_bps"] = float(total_val)
        return maker_share_out, expected_fee_out, expected_spread_out, components

    def _log_trade_cost_debug(
        self,
        *,
        context: str,
        side: str,
        qty: float,
        pre_price: float,
        final_price: float,
        trade_cost: _TradeCostResult,
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        limit = int(self._trade_cost_debug_limit)
        if limit > 0 and self._trade_cost_debug_logged >= limit:
            return
        metrics = getattr(trade_cost, "metrics", None)
        if isinstance(metrics, dict):
            _, fee_bps, spread_bps, components = self._expected_cost_snapshot()
            if fee_bps is not None and "fee_bps" not in metrics:
                try:
                    metrics["fee_bps"] = float(fee_bps)
                except (TypeError, ValueError):
                    metrics.setdefault("fee_bps", None)
            if spread_bps is not None and "spread_bps" not in metrics:
                try:
                    metrics["spread_bps"] = float(spread_bps)
                except (TypeError, ValueError):
                    metrics.setdefault("spread_bps", None)
            if components and "expected_cost_components" not in metrics:
                metrics["expected_cost_components"] = dict(components)
        try:
            qty_val = float(qty)
        except (TypeError, ValueError):
            qty_val = 0.0
        payload: Dict[str, Any] = {
            "ctx": context,
            "side": side,
            "qty": qty_val,
            "cost_bps": float(trade_cost.bps),
            "mid": trade_cost.mid,
            "base_price": trade_cost.base_price,
            "pre_price": float(pre_price),
            "final_price": float(final_price),
            "inputs": trade_cost.inputs,
        }
        payload["expected_bps"] = self._trade_cost_expected_bps(trade_cost)
        if trade_cost.metrics:
            payload["metrics"] = trade_cost.metrics
        logger.debug("trade_cost %s", payload)
        self._trade_cost_debug_logged += 1

    def _compute_effective_spread_bps(
        self,
        *,
        base_spread_bps: Optional[float],
        ts_ms: Optional[int],
        vol_factor: Optional[float],
        range_ratio_bps: Optional[float] = None,
    ) -> Optional[float]:
        base_val = self._float_or_none(base_spread_bps)
        if base_val is not None and base_val < 0.0:
            base_val = None
        default_val = self._float_or_none(self._default_spread_bps())
        if base_val is None:
            base_val = default_val
        if base_val is None:
            return None

        bid = self._float_or_none(self._last_bid)
        ask = self._float_or_none(self._last_ask)
        bar_high = self._float_or_none(self._last_bar_high)
        bar_low = self._float_or_none(self._last_bar_low)
        bar_close = self._float_or_none(self._last_bar_close)
        ref_price = self._float_or_none(self._last_ref_price)

        mid_price = self._resolve_mid_from_inputs(
            bid,
            ask,
            bar_high,
            bar_low,
            bar_close,
            None,
            ref_price,
        )
        if mid_price is None:
            fallback = default_val if default_val is not None else base_val
            if fallback is None:
                return None
            return float(fallback)

        vol_metrics = None
        if isinstance(self._last_vol_raw, Mapping):
            try:
                vol_metrics = dict(self._last_vol_raw)
            except Exception:
                vol_metrics = None

        range_hint = self._non_negative_float(range_ratio_bps)
        _, bar_range_bps = self._compute_bar_range_ratios(bar_high, bar_low, mid_price)
        if range_hint is None:
            range_hint = self._non_negative_float(bar_range_bps)

        if self._dyn_spread_enabled:
            dynamic_val = self._compute_dynamic_spread_override(
                base_spread_bps=base_val,
                default_spread_bps=default_val,
                bar_high=bar_high,
                bar_low=bar_low,
                mid_price=mid_price,
                vol_metrics=vol_metrics,
                vol_factor=vol_factor,
                range_ratio_bps_hint=range_hint,
            )
            if dynamic_val is not None:
                base_val = dynamic_val

        getter_kwargs: Dict[str, Any] = {
            "symbol": self.symbol,
            "ts_ms": ts_ms,
            "base_spread_bps": base_val,
            "vol_factor": vol_factor,
            "mid_price": mid_price,
        }
        if vol_factor is None:
            getter_kwargs.pop("vol_factor")
        # Всегда передаём ts_ms даже если None — совместимо с существующим интерфейсом.
        if ts_ms is None:
            getter_kwargs["ts_ms"] = ts_ms
        if bar_high is not None:
            getter_kwargs["bar_high"] = bar_high
        if bar_low is not None:
            getter_kwargs["bar_low"] = bar_low
        if vol_metrics is not None:
            getter_kwargs["vol_metrics"] = vol_metrics

        result = self._call_spread_getter(getter_kwargs)
        if result is not None:
            try:
                candidate = float(result)
            except (TypeError, ValueError):
                candidate = None
            else:
                if math.isfinite(candidate) and candidate >= 0.0:
                    base_val = candidate
        return float(base_val)

    def _compute_dynamic_spread_override(
        self,
        *,
        base_spread_bps: float,
        default_spread_bps: Optional[float],
        bar_high: Optional[float],
        bar_low: Optional[float],
        mid_price: Optional[float],
        vol_metrics: Optional[Mapping[str, Any]],
        vol_factor: Optional[float],
        range_ratio_bps_hint: Optional[float],
    ) -> Optional[float]:
        if not self._dyn_spread_enabled:
            return None

        fallback_value = self._non_negative_float(self._dyn_spread_fallback_bps)
        if fallback_value is None:
            fallback_value = self._non_negative_float(default_spread_bps)
        if fallback_value is None:
            fallback_value = max(float(base_spread_bps), 0.0)

        metric_key = (self._dyn_spread_metric_key or "range_ratio_bps").strip().lower()
        if not metric_key:
            metric_key = "range_ratio_bps"

        def _coerce_value(source_key: str, raw: Any) -> Optional[float]:
            candidate = self._non_negative_float(raw)
            if candidate is None:
                return None
            if metric_key in {"range_ratio", "range"} and source_key == "range_ratio_bps":
                return candidate / 1e4
            if metric_key == "range_ratio_bps" and source_key in {"range_ratio", "range"}:
                return candidate * 1e4
            return candidate

        metric_value: Optional[float] = None
        if vol_metrics is not None and metric_key in vol_metrics:
            metric_value = _coerce_value(metric_key, vol_metrics.get(metric_key))

        if metric_value is None:
            metric_value = _coerce_value("range_ratio_bps", range_ratio_bps_hint)

        if metric_value is None and vol_metrics is not None:
            for alias in ("range_ratio_bps", "range_ratio", "range"):
                if alias == metric_key:
                    continue
                if alias not in vol_metrics:
                    continue
                metric_value = _coerce_value(alias, vol_metrics.get(alias))
                if metric_value is not None:
                    break

        if metric_value is None:
            range_ratio, range_ratio_bps = self._compute_bar_range_ratios(
                bar_high, bar_low, mid_price
            )
            metric_value = _coerce_value("range_ratio", range_ratio)
            if metric_value is None:
                metric_value = _coerce_value("range_ratio_bps", range_ratio_bps)

        if metric_value is None and self._dyn_spread_use_volatility:
            metric_value = self._non_negative_float(vol_factor)

        if metric_value is None:
            return float(fallback_value)

        alpha = self._float_or_none(self._dyn_spread_alpha_bps)
        if alpha is None:
            alpha = self._float_or_none(default_spread_bps)
        if alpha is None:
            alpha = float(base_spread_bps)
        beta = self._float_or_none(self._dyn_spread_beta_coef)
        if beta is None:
            beta = 0.0

        candidate = alpha + beta * metric_value
        if not math.isfinite(candidate):
            return float(fallback_value)

        min_bps = self._float_or_none(self._dyn_spread_min_bps)
        if min_bps is not None:
            candidate = max(min_bps, candidate)
        max_bps = self._float_or_none(self._dyn_spread_max_bps)
        if max_bps is not None:
            candidate = min(max_bps, candidate)

        smoothing = self._dyn_spread_smoothing_alpha
        if smoothing is not None:
            prev = self._dyn_spread_prev_ema
            if prev is None or not math.isfinite(prev):
                ema_val = candidate
            else:
                ema_val = smoothing * candidate + (1.0 - smoothing) * prev
            self._dyn_spread_prev_ema = float(ema_val)
            candidate = ema_val
        else:
            self._dyn_spread_prev_ema = None

        if not math.isfinite(candidate):
            return float(fallback_value)
        if candidate < 0.0:
            candidate = 0.0
        return float(candidate)

    def _report_spread_bps(self, spread_bps: Optional[float]) -> float:
        if spread_bps is not None:
            try:
                value = float(spread_bps)
            except (TypeError, ValueError):
                value = None
            else:
                if math.isfinite(value):
                    return value
        if self._adv_enabled:
            adv_cap = self._last_adv_bar_capacity
            if adv_cap is None:
                adv_cap = self._adv_bar_capacity(self.symbol, None)
                if adv_cap is not None:
                    adv_cap = max(0.0, float(adv_cap))
                    self._last_adv_bar_capacity = adv_cap
            if adv_cap is not None:
                self._last_liquidity = self._combine_liquidity(
                    self._last_liquidity, adv_cap
                )
        default = self._default_spread_bps()
        if default is not None:
            return float(default)
        return 0.0

    def set_market_snapshot(
        self,
        *,
        bid: Optional[float],
        ask: Optional[float],
        spread_bps: Optional[float] = None,
        vol_factor: Optional[float] = None,
        vol_raw: Optional[Mapping[str, float]] = None,
        liquidity: Optional[float] = None,
        ts_ms: Optional[int] = None,
        trade_price: Optional[float] = None,
        trade_qty: Optional[float] = None,
        bar_open: Optional[float] = None,
        bar_high: Optional[float] = None,
        bar_low: Optional[float] = None,
        bar_close: Optional[float] = None,
    ) -> None:
        """
        Установить последний рыночный снапшот: bid/ask (для вычисления spread и mid),
        vol_factor (например ATR% за бар), liquidity (например rolling_volume_shares).
        """
        self._last_bid = float(bid) if bid is not None else None
        self._last_ask = float(ask) if ask is not None else None

        liq_mult = 1.0
        spread_mult = 1.0
        how: Optional[int] = None
        if ts_ms is None:
            if self.use_seasonality:
                logger.warning("ts_ms is None; seasonality multipliers not applied")
        else:
            # Convert UTC milliseconds to hour-of-week (0=Mon 00:00 UTC) via the
            # shared helper: (ts_ms // HOUR_MS + 72) % 168.  This keeps hour
            # indexing consistent across modules.
            how = hour_of_week(int(ts_ms))
            if self.use_seasonality:
                liq_mult = get_liquidity_multiplier(
                    int(ts_ms),
                    self._liq_seasonality,
                    interpolate=self.seasonality_interpolate,
                )
                spread_mult = get_hourly_multiplier(
                    int(ts_ms),
                    self._spread_seasonality,
                    interpolate=self.seasonality_interpolate,
                )

        sbps: Optional[float]
        if spread_bps is not None:
            try:
                sbps = float(spread_bps)
            except (TypeError, ValueError):
                sbps = None
        elif (
            compute_spread_bps_from_quotes is not None
            and self.slippage_cfg is not None
        ):
            sbps = compute_spread_bps_from_quotes(
                bid=self._last_bid, ask=self._last_ask, cfg=self.slippage_cfg
            )
        else:
            sbps = None

        bar_open_val = self._float_or_none(bar_open)
        bar_high_val = self._float_or_none(bar_high)
        bar_low_val = self._float_or_none(bar_low)
        close_candidate = bar_close
        if close_candidate is None:
            close_candidate = trade_price
        bar_close_val = self._float_or_none(close_candidate)
        trade_val = self._float_or_none(trade_price)
        mid_for_range = self._resolve_mid_from_inputs(
            self._last_bid,
            self._last_ask,
            bar_high_val,
            bar_low_val,
            bar_close_val,
            trade_val,
            self._last_ref_price,
        )
        range_ratio, range_ratio_bps = self._compute_bar_range_ratios(
            bar_high_val, bar_low_val, mid_for_range
        )

        vf_val: Optional[float]
        if vol_factor is not None:
            try:
                vf_val = float(vol_factor)
            except (TypeError, ValueError):
                vf_val = None
            else:
                if not math.isfinite(vf_val):
                    vf_val = None
        else:
            vf_val = None
        self._last_vol_factor = vf_val

        metrics = self._normalize_vol_metrics(
            vol_raw,
            computed_range_ratio=range_ratio,
            computed_range_ratio_bps=range_ratio_bps,
        )
        self._last_vol_raw = metrics

        range_ratio_bps_val: Optional[float] = None
        if metrics is not None:
            range_ratio_bps_val = self._non_negative_float(
                metrics.get("range_ratio_bps")
            )
            if range_ratio_bps_val is None:
                range_ratio_hint = self._non_negative_float(metrics.get("range_ratio"))
                if range_ratio_hint is not None:
                    range_ratio_bps_val = range_ratio_hint * 1e4
        if range_ratio_bps_val is None:
            range_ratio_bps_val = self._non_negative_float(range_ratio_bps)

        self._last_bar_open = bar_open_val
        self._last_bar_high = bar_high_val
        self._last_bar_low = bar_low_val
        self._last_bar_close = bar_close_val

        effective_spread = self._compute_effective_spread_bps(
            base_spread_bps=sbps,
            ts_ms=ts_ms,
            vol_factor=self._last_vol_factor,
            range_ratio_bps=range_ratio_bps_val,
        )
        if effective_spread is not None:
            try:
                mult = float(spread_mult)
            except (TypeError, ValueError):
                mult = 1.0
            if not math.isfinite(mult):
                mult = 1.0
            self._last_spread_bps = float(effective_spread) * mult
        else:
            self._last_spread_bps = None
        liq_val: Optional[float]
        if liquidity is None:
            liq_val = None
        else:
            try:
                liq_val = float(liquidity)
            except (TypeError, ValueError):
                liq_val = None
            else:
                if not math.isfinite(liq_val):
                    liq_val = None
                elif liq_val < 0.0:
                    liq_val = 0.0
        if liq_val is not None:
            liq_val *= liq_mult
        timeframe_guess = self._resolve_intrabar_timeframe(ts_ms)
        adv_capacity = self._adv_bar_capacity(self.symbol, timeframe_guess)
        adv_capacity_scaled: Optional[float]
        if adv_capacity is None:
            adv_capacity_scaled = None
        else:
            adv_capacity_scaled = max(0.0, float(adv_capacity))
            try:
                mult_val = float(liq_mult)
            except (TypeError, ValueError):
                mult_val = 1.0
            if not math.isfinite(mult_val) or mult_val <= 0.0:
                mult_val = 1.0
            adv_capacity_scaled *= mult_val
        self._last_adv_bar_capacity = adv_capacity_scaled
        self._last_liquidity = self._combine_liquidity(liq_val, adv_capacity_scaled)
        if ts_ms is not None and self._last_vol_factor is not None:
            ts_val = int(ts_ms)
            cache_value = self._select_cache_value(metrics, self._last_vol_factor)
            cache_updated = False
            if cache_value is not None:
                cache_updated = self._feed_latency_cache(ts_val, cache_value)
            self._update_latency_volatility(
                ts_val,
                self._last_vol_factor,
                metrics,
                cache_already_updated=cache_updated,
            )
        if seasonality_logger.isEnabledFor(logging.DEBUG) and ts_ms is not None:
            seasonality_logger.debug(
                "snapshot h%03d mult=%.3f liquidity=%s",
                how,
                liq_mult,
                self._last_liquidity,
            )
        if self.use_seasonality and how is not None and 0 <= how < HOURS_IN_WEEK:
            self._liq_mult_sum[how] += liq_mult
            if self._last_liquidity is not None:
                self._liq_val_sum[how] += self._last_liquidity
            self._liq_count[how] += 1
            _SIM_MULT_COUNTER.labels(hour=how).inc()
        if self._last_ref_price is None:
            if mid_from_quotes is not None:
                mid = mid_from_quotes(bid=self._last_bid, ask=self._last_ask)
                if mid is not None:
                    self._last_ref_price = float(mid)
        if ts_ms is not None:
            hour = int(ts_ms // HOUR_MS)
            if self._snapshot_hour is None:
                self._snapshot_hour = hour
            elif hour != self._snapshot_hour:
                if trade_price is not None:
                    self._next_h1_open_price = float(trade_price)
                self._snapshot_hour = hour
            price_tick = (
                trade_price if trade_price is not None else self._last_ref_price
            )
            qty_tick = trade_qty if trade_qty is not None else liquidity
            if price_tick is not None and qty_tick is not None:
                self._vwap_on_tick(int(ts_ms), float(price_tick), float(qty_tick))

    def set_intrabar_timeframe_ms(self, timeframe_ms: Optional[int]) -> None:
        """Сохранить продолжительность бара для intrabar-логики."""
        if timeframe_ms is None:
            self._intrabar_timeframe_ms = None
            return
        try:
            value = int(timeframe_ms)
        except (TypeError, ValueError):
            return
        if value <= 0:
            return
        self._intrabar_timeframe_ms = value

    def _resolve_intrabar_timeframe(self, ts_ms: Optional[int] = None) -> int:
        """Определить длину бара для intrabar-расчётов."""

        timeframe: Optional[int] = self._intrabar_timeframe_ms
        if timeframe is None and self._intrabar_config_timeframe_ms is not None:
            timeframe = int(self._intrabar_config_timeframe_ms)
        if timeframe is None and self._timing_timeframe_ms is not None:
            timeframe = int(self._timing_timeframe_ms)
        if timeframe is None:
            base = int(self.step_ms) if self.step_ms > 0 else 0
            timeframe = base if base > 0 else 1
        if timeframe <= 0:
            timeframe = 1
        return int(timeframe)

    def _intrabar_latency_ms(
        self,
        latency_sample: Any,
        child_offset_ms: Optional[int] = None,
    ) -> int:
        """Выбрать источник латентности для intrabar-модели."""

        mode = (self._intrabar_latency_source or "latency").strip().lower()

        sample_val: Optional[float]
        if isinstance(latency_sample, Mapping):
            sample_val = latency_sample.get("total_ms")  # type: ignore[index]
        else:
            sample_val = latency_sample
        try:
            sample_ms = int(round(float(sample_val))) if sample_val is not None else 0
        except (TypeError, ValueError):
            sample_ms = 0
        if not math.isfinite(float(sample_ms)) or sample_ms < 0:
            sample_ms = 0

        offset_ms = 0
        if child_offset_ms is not None:
            try:
                offset_ms = int(child_offset_ms)
            except (TypeError, ValueError):
                offset_ms = 0
            if offset_ms < 0:
                offset_ms = 0

        const_ms = self._intrabar_latency_constant_ms
        if mode == "constant" and const_ms is not None:
            return int(const_ms)

        child_modes = {"child", "child_offset", "schedule", "plan"}
        sum_modes = {
            "child_plus_latency",
            "child+latency",
            "schedule_plus_latency",
            "latency_plus_child",
            "child_latency_sum",
            "latency+child",
        }

        if mode in child_modes:
            if child_offset_ms is not None:
                return offset_ms
            return sample_ms
        if mode in sum_modes:
            return offset_ms + sample_ms

        if mode == "constant" and const_ms is None and child_offset_ms is not None:
            return offset_ms

        return sample_ms

    def _intrabar_time_fraction(
        self, latency_ms: Optional[int | float], timeframe_ms: Optional[int | float]
    ) -> float:
        """Рассчитать положение внутри бара по латентности."""

        try:
            lat_val = float(latency_ms) if latency_ms is not None else 0.0
        except (TypeError, ValueError):
            lat_val = 0.0
        if not math.isfinite(lat_val) or lat_val < 0.0:
            lat_val = 0.0

        try:
            tf_val = float(timeframe_ms) if timeframe_ms is not None else 0.0
        except (TypeError, ValueError):
            tf_val = 0.0
        if not math.isfinite(tf_val) or tf_val <= 0.0:
            tf_val = float(self._resolve_intrabar_timeframe(None))
        if tf_val <= 0.0:
            tf_val = 1.0

        ratio = lat_val / tf_val
        if ratio > 1.0 and self._intrabar_log_warnings:
            now = now_ms()
            if now >= self._intrabar_warn_next_log_ms:
                logger.warning(
                    "intrabar latency %.0f ms exceeds timeframe %.0f ms (mode=%s, source=%s)",
                    lat_val,
                    tf_val,
                    self._intrabar_price_model,
                    self._intrabar_latency_source,
                )
                throttle = max(int(tf_val), 1)
                self._intrabar_warn_next_log_ms = now + throttle

        if ratio < 0.0:
            ratio = 0.0
        if ratio > 1.0:
            ratio = 1.0
        return float(ratio)

    def _next_order_seq(self) -> int:
        """Return a monotonically increasing child order identifier."""

        self._order_seq_counter += 1
        return self._order_seq_counter

    def _reset_intrabar_debug_counter(self) -> None:
        """Reset per-step intrabar debug logging counter."""

        self._intrabar_debug_logged = 0
        self._trade_cost_debug_logged = 0

    def _intrabar_atr_hint(self) -> Optional[float]:
        """Extract an ATR/volatility hint from the latest metrics."""

        hint = 0.0
        metrics = self._last_vol_raw
        if isinstance(metrics, Mapping):
            preferred_keys = (
                "atr",
                "atr_abs",
                "atr_value",
                "atr_price",
                "atr_quote",
                "atr_usd",
                "atr_last",
            )
            for key in preferred_keys:
                if key not in metrics:
                    continue
                try:
                    val = float(metrics.get(key))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(val):
                    continue
                val = abs(val)
                if val > hint:
                    hint = val
        if hint <= 0.0 and self._last_vol_factor is not None:
            try:
                ref_price = float(self._last_ref_price) if self._last_ref_price is not None else None
            except (TypeError, ValueError):
                ref_price = None
            if ref_price is not None and math.isfinite(ref_price) and ref_price > 0.0:
                try:
                    percent = float(self._last_vol_factor)
                except (TypeError, ValueError):
                    percent = 0.0
                if math.isfinite(percent):
                    derived = abs(ref_price * percent / 100.0)
                    if derived > hint:
                        hint = derived
        return hint if hint > 0.0 else None

    def _intrabar_rng_seed(
        self,
        *,
        bar_ts: Optional[int],
        side: str,
        order_seq: int,
    ) -> int:
        """Derive a deterministic RNG seed for intrabar price sampling."""

        try:
            ts_val = int(bar_ts) if bar_ts is not None else 0
        except (TypeError, ValueError):
            ts_val = 0
        side_val = str(side).upper()
        key = f"{self.symbol}|{ts_val}|{side_val}|{int(order_seq)}|{int(self.seed)}"
        mode = (self._intrabar_seed_mode or "stable").strip().lower()
        if mode in {"off", "none"}:
            return int(self.seed)
        if mode in {"python", "hash"}:
            return hash(key)
        if mode in {"xor", "mix"}:
            return hash(key) ^ int(self.seed)
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _clip_to_bar_range(self, price: float) -> tuple[float, bool]:
        """Clip ``price`` to the latest bar range when available."""

        try:
            low = float(self._last_bar_low) if self._last_bar_low is not None else None
        except (TypeError, ValueError):
            low = None
        try:
            high = float(self._last_bar_high) if self._last_bar_high is not None else None
        except (TypeError, ValueError):
            high = None
        if low is None or high is None or not math.isfinite(low) or not math.isfinite(high):
            return float(price), False
        if high < low:
            low, high = high, low
        clipped = float(price)
        if clipped < low:
            return low, True
        if clipped > high:
            return high, True
        return clipped, False

    def _compute_intrabar_price(
        self,
        *,
        side: str,
        time_fraction: float,
        fallback_price: Optional[float],
        bar_ts: Optional[int],
        order_seq: Optional[int] = None,
    ) -> tuple[Optional[float], bool, float]:
        """Return intrabar reference price with clipping information."""

        mode_raw = self._intrabar_price_model
        try:
            frac = float(time_fraction)
        except (TypeError, ValueError):
            frac = 0.0
        if not math.isfinite(frac):
            frac = 0.0
        if frac < 0.0:
            frac = 0.0
        if frac > 1.0:
            frac = 1.0

        fallback: Optional[float]
        if fallback_price is None:
            fallback = None
        else:
            try:
                fb = float(fallback_price)
            except (TypeError, ValueError):
                fallback = None
            else:
                fallback = fb if math.isfinite(fb) else None

        if not mode_raw:
            return fallback, False, frac
        mode = str(mode_raw).strip().lower()
        if not mode or mode in {"book", "none", "default"}:
            return fallback, False, frac
        if mode in {"off", "disabled"}:
            return fallback, False, frac

        def _finite(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            try:
                val = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(val):
                return None
            return val

        open_p = _finite(self._last_bar_open)
        high_p = _finite(self._last_bar_high)
        low_p = _finite(self._last_bar_low)
        close_p = _finite(self._last_bar_close)

        linear_value: Optional[float] = None
        if open_p is not None and close_p is not None:
            linear_value = open_p + (close_p - open_p) * frac

        if mode in {"open"}:
            price = open_p if open_p is not None else fallback
            return price, False, frac
        if mode in {"close"}:
            price = close_p if close_p is not None else fallback
            return price, False, frac
        if mode in {"high"}:
            price = high_p if high_p is not None else fallback
            return price, False, frac
        if mode in {"low"}:
            price = low_p if low_p is not None else fallback
            return price, False, frac
        if mode in {"mid"}:
            bid = _finite(self._last_bid)
            ask = _finite(self._last_ask)
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
            else:
                mid = None
            price = mid
            if price is None:
                price = linear_value
            if price is None:
                price = fallback
            if price is None:
                return None, False, frac
            clipped, clipped_flag = self._clip_to_bar_range(price)
            return clipped, clipped_flag, frac

        linear_modes = {
            "linear",
            "open_close_linear",
            "open-close-linear",
            "oc_linear",
            "linear_oc",
        }
        if mode in linear_modes and linear_value is not None:
            clipped, clipped_flag = self._clip_to_bar_range(linear_value)
            return clipped, clipped_flag, frac

        ohlc_modes = {"ohlc", "ohlc-linear", "ohlc_linear"}
        if mode in ohlc_modes and open_p is not None and close_p is not None:
            side_u = str(side).upper()
            extreme = high_p if side_u == "BUY" else low_p
            if extreme is None:
                extreme = high_p if high_p is not None else low_p
            if extreme is None:
                price = linear_value if linear_value is not None else fallback
                if price is None:
                    return None, False, frac
                clipped, clipped_flag = self._clip_to_bar_range(price)
                return clipped, clipped_flag, frac
            if frac <= 0.5:
                price = open_p + (extreme - open_p) * (frac / 0.5)
            else:
                price = extreme + (close_p - extreme) * ((frac - 0.5) / 0.5)
            clipped, clipped_flag = self._clip_to_bar_range(price)
            return clipped, clipped_flag, frac

        if mode in {"bridge", "brownian", "brownian_bridge"}:
            if linear_value is None:
                if fallback is None:
                    return None, False, frac
                return fallback, False, frac
            sigma = 0.0
            if high_p is not None and low_p is not None:
                sigma = abs(high_p - low_p)
            atr_hint = self._intrabar_atr_hint()
            if atr_hint is not None:
                sigma = max(sigma, float(atr_hint))
            if sigma <= 0.0 or frac <= 0.0 or frac >= 1.0:
                clipped, clipped_flag = self._clip_to_bar_range(linear_value)
                return clipped, clipped_flag, frac
            seq = int(order_seq) if order_seq is not None else int(self._order_seq_counter)
            rng_seed = self._intrabar_rng_seed(
                bar_ts=bar_ts if bar_ts is not None else self._last_bar_close_ts,
                side=side,
                order_seq=seq,
            )
            rng = random.Random(rng_seed)
            std = float(sigma) * math.sqrt(frac * (1.0 - frac))
            noise = rng.gauss(0.0, std)
            price = linear_value + noise
            clipped, clipped_flag = self._clip_to_bar_range(price)
            return clipped, clipped_flag, frac

        # fallback to previous behaviour without additional logic
        if linear_value is not None:
            clipped, clipped_flag = self._clip_to_bar_range(linear_value)
            return clipped, clipped_flag, frac
        return fallback, False, frac

    def _intrabar_reference_price(self, side: str, time_fraction: float) -> Optional[float]:
        """Вернуть референсную цену внутри бара по выбранной модели."""

        price, _, _ = self._compute_intrabar_price(
            side=side,
            time_fraction=time_fraction,
            fallback_price=None,
            bar_ts=self._last_bar_close_ts,
            order_seq=None,
        )
        return price

    def get_hourly_liquidity_stats(self) -> dict:
        """Return averaged liquidity multiplier/value per hour of week."""
        avg_mult = [
            self._liq_mult_sum[i] / self._liq_count[i] if self._liq_count[i] else 0.0
            for i in range(HOURS_IN_WEEK)
        ]
        avg_liq = [
            self._liq_val_sum[i] / self._liq_count[i] if self._liq_count[i] else 0.0
            for i in range(HOURS_IN_WEEK)
        ]
        return {
            "multiplier": avg_mult,
            "liquidity": avg_liq,
            "count": list(self._liq_count),
        }

    def reset_hourly_liquidity_stats(self) -> None:
        """Reset accumulated liquidity seasonality statistics."""
        self._liq_mult_sum = [0.0] * HOURS_IN_WEEK
        self._liq_val_sum = [0.0] * HOURS_IN_WEEK
        self._liq_count = [0] * HOURS_IN_WEEK

    def get_hourly_seasonality_stats(self) -> dict:
        """Return combined hourly liquidity and latency statistics."""
        result = {"liquidity": self.get_hourly_liquidity_stats(), "latency": None}
        if self.latency is not None and hasattr(self.latency, "hourly_stats"):
            try:
                result["latency"] = self.latency.hourly_stats()  # type: ignore[attr-defined]
            except Exception:
                result["latency"] = None
        return result

    def dump_seasonality_multipliers(self) -> Dict[str, List[float]]:
        """Return current liquidity and spread seasonality multipliers.

        The result is a JSON-serializable mapping with two keys:
        ``liquidity`` and ``spread``. Each contains a list of floats representing
        multipliers either for each hour of week (length 168) or for each day
        of week (length 7) when ``seasonality_day_only`` is enabled.
        """

        return {
            "liquidity": self._liq_seasonality.tolist(),
            "spread": self._spread_seasonality.tolist(),
        }

    def load_seasonality_multipliers(self, data: Dict[str, Sequence[float]]) -> None:
        """Load liquidity and spread seasonality multipliers from ``data``.

        ``data`` should be a mapping with optional ``liquidity`` and ``spread``
        keys, each mapping to a sequence of 168 floats (or 7 if
        ``seasonality_day_only`` is enabled). Missing keys are ignored. Raises
        ``ValueError`` if provided arrays have invalid length.
        """

        liq = data.get("liquidity")
        spread = data.get("spread")

        expected = 7 if self.seasonality_day_only else HOURS_IN_WEEK

        new_liq = self._liq_seasonality
        new_spr = self._spread_seasonality

        if liq is not None:
            arr = np.asarray(liq, dtype=float)
            if arr.size != expected:
                raise ValueError(f"liquidity multipliers must have length {expected}")
            new_liq = arr.copy()

        if spread is not None:
            arr = np.asarray(spread, dtype=float)
            if arr.size != expected:
                raise ValueError(f"spread multipliers must have length {expected}")
            new_spr = arr.copy()

        with self._seasonality_lock:
            self._liq_seasonality = new_liq
            self._spread_seasonality = new_spr

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
        if profile == "VWAP_CURRENT_H1" and VWAPExecutor is not None:
            self._executor = VWAPExecutor()
            return
        cfg = dict(self._execution_cfg or {})
        algo = str(cfg.get("algo", "TAKER")).upper()
        if make_executor is not None:
            self._executor = make_executor(algo, cfg)
        else:
            self._executor = TakerExecutor()

    def _vwap_on_tick(
        self, ts_ms: int, price: Optional[float], volume: Optional[float]
    ) -> None:
        hour = int(ts_ms // HOUR_MS)
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

    def _update_vol_series(
        self, metric: str, value: float
    ) -> Optional[Dict[str, float]]:
        if not metric:
            return None
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(val):
            return None
        name = str(metric).lower()
        window = self._vol_window_size if self._vol_window_size >= 2 else 2
        state = self._vol_series.get(name)
        if state is None:
            state = _VolSeriesState(values=deque(maxlen=window))
            self._vol_series[name] = state
        dq = state.values
        if dq.maxlen != window:
            dq = deque(dq, maxlen=window)
            state.values = dq
            state.sum = 0.0
            state.sum_sq = 0.0
            for existing in dq:
                state.sum += existing
                state.sum_sq += existing * existing
        if dq.maxlen is not None and len(dq) == dq.maxlen:
            old = dq.popleft()
            state.sum -= old
            state.sum_sq -= old * old
        dq.append(val)
        state.sum += val
        state.sum_sq += val * val
        count = len(dq)
        if count <= 0:
            return None
        mean = state.sum / count
        variance = max(state.sum_sq / count - mean * mean, 0.0)
        std = math.sqrt(variance)
        z_raw = 0.0
        if count >= 2 and std > 0.0:
            z_raw = (val - mean) / std
        clip = self._vol_zscore_clip
        if clip is not None and math.isfinite(clip):
            clip_abs = abs(float(clip))
            zscore = max(-clip_abs, min(clip_abs, z_raw))
        else:
            zscore = z_raw
        gamma_val = self._vol_gamma if math.isfinite(self._vol_gamma) else 0.0
        multiplier: Optional[float] = None
        if gamma_val != 0.0:
            mult_candidate = 1.0 + gamma_val * zscore
            if math.isfinite(mult_candidate):
                multiplier = max(0.0, mult_candidate)
        stats: Dict[str, float] = {
            "mean": float(mean),
            "std": float(std),
            "zscore_raw": float(z_raw),
            "zscore": float(zscore),
            "count": float(count),
        }
        if multiplier is not None:
            stats["multiplier"] = float(multiplier)
        self._vol_stat_cache[name] = stats
        return stats

    def _normalize_vol_metrics(
        self,
        metrics: Optional[Mapping[str, Any]],
        *,
        computed_range_ratio: Optional[float] = None,
        computed_range_ratio_bps: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        if not metrics:
            return None
        try:
            items = metrics.items()  # type: ignore[attr-defined]
        except AttributeError:
            try:
                items = dict(metrics).items()  # type: ignore[arg-type]
            except Exception:
                return None
        normalized: Dict[str, float] = {}
        for key, raw_val in items:
            if key is None:
                continue
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(val):
                continue
            normalized[str(key).lower()] = val
        if not normalized:
            return None
        if "atr_pct" in normalized:
            atr_val = normalized["atr_pct"]
            normalized.setdefault("atr", atr_val)
            normalized.setdefault("atr/price", atr_val)
        elif "atr" in normalized:
            atr_val = normalized["atr"]
            normalized.setdefault("atr_pct", atr_val)
            normalized.setdefault("atr/price", atr_val)
        range_val = normalized.get("range")
        if range_val is not None:
            range_clean = max(0.0, float(range_val))
            normalized["range"] = range_clean
            normalized.setdefault("range_ratio", range_clean)
            normalized.setdefault("range_ratio_bps", range_clean * 1e4)
        elif "range_ratio_bps" in normalized:
            ratio_bps = max(0.0, float(normalized["range_ratio_bps"]))
            normalized["range_ratio_bps"] = ratio_bps
            ratio = ratio_bps / 1e4 if ratio_bps is not None else 0.0
            normalized.setdefault("range_ratio", ratio)
            normalized.setdefault("range", ratio)
        ratio_hint = self._float_or_none(computed_range_ratio)
        if ratio_hint is not None:
            if ratio_hint < 0.0:
                ratio_hint = 0.0
        ratio_bps_hint = self._float_or_none(computed_range_ratio_bps)
        if ratio_bps_hint is not None:
            if ratio_bps_hint < 0.0:
                ratio_bps_hint = 0.0
        if ratio_hint is not None and ratio_bps_hint is None:
            ratio_bps_hint = ratio_hint * 1e4
        if ratio_bps_hint is not None and ratio_hint is None:
            ratio_hint = ratio_bps_hint / 1e4
        if ratio_hint is not None:
            normalized.setdefault("range", ratio_hint)
            normalized.setdefault("range_ratio", ratio_hint)
        if ratio_bps_hint is not None:
            normalized.setdefault("range_ratio_bps", ratio_bps_hint)
        stat_targets = set()
        if self._vol_norm_metric and self._vol_norm_metric in normalized:
            stat_targets.add(self._vol_norm_metric)
        for candidate in ("sigma", "atr", "atr_pct", "range", "range_ratio_bps"):
            if candidate in normalized:
                stat_targets.add(candidate)
        for metric_key in stat_targets:
            value = normalized.get(metric_key)
            if value is None or not math.isfinite(value):
                continue
            stats = self._update_vol_series(metric_key, value)
            if not stats:
                continue
            base = metric_key.replace("/", "_")
            try:
                normalized[f"{base}_mean"] = float(stats["mean"])
                normalized[f"{base}_std"] = float(stats["std"])
                normalized[f"{base}_zscore_raw"] = float(stats["zscore_raw"])
                normalized[f"{base}_zscore"] = float(stats["zscore"])
                normalized[f"{base}_count"] = float(stats.get("count", 0.0))
            except (KeyError, TypeError, ValueError):
                pass
            multiplier = stats.get("multiplier")
            if multiplier is not None and math.isfinite(multiplier):
                normalized[f"{base}_gamma_mult"] = float(multiplier)
        return normalized or None

    def _select_cache_value(
        self,
        metrics: Optional[Mapping[str, float]],
        fallback: Optional[float],
    ) -> Optional[float]:
        metric_name = str(getattr(self, "_latency_vol_metric", "sigma") or "sigma").lower()
        if metrics:
            aliases: Tuple[str, ...]
            if metric_name in {"atr", "atr_pct", "atr/price"}:
                aliases = ("atr_pct", "atr", "atr/price")
            else:
                aliases = (metric_name,)
            for alias in aliases:
                val = metrics.get(alias)
                if val is None:
                    continue
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    return v
        if fallback is None:
            return None
        try:
            fb = float(fallback)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(fb):
            return None
        return fb

    def _feed_latency_cache(self, ts: int, value: Optional[float]) -> bool:
        cache = getattr(self, "volatility_cache", None)
        if cache is None:
            return False
        if value is None:
            return False
        try:
            val = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(val):
            return False
        symbol = getattr(self, "_latency_symbol", None) or getattr(self, "symbol", None)
        if not symbol:
            return False
        sym = str(symbol).upper()
        try:
            cache.update_latency_factor(symbol=sym, ts_ms=int(ts), value=val)
            return True
        except TypeError:
            try:
                cache.update_latency_factor(sym, int(ts), val)  # type: ignore[misc]
                return True
            except Exception:
                return False
        except Exception:
            return False

    def _update_latency_volatility(
        self,
        ts_ms: int,
        value: float,
        raw_metrics: Optional[Mapping[str, float]] = None,
        *,
        cache_already_updated: bool = False,
    ) -> None:
        try:
            ts = int(ts_ms)
            val = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(val):
            return
        metrics = (
            raw_metrics
            if raw_metrics is None or isinstance(raw_metrics, dict)
            else self._normalize_vol_metrics(raw_metrics)
        )
        if metrics is not None and not isinstance(metrics, dict):
            metrics = self._normalize_vol_metrics(metrics)
        cache_value = self._select_cache_value(metrics, val)

        latency = getattr(self, "latency", None)
        if latency is not None:
            updater = getattr(latency, "update_volatility", None)
            if callable(updater):
                symbol = getattr(self, "_latency_symbol", None) or getattr(self, "symbol", None)
                try:
                    updater(symbol, ts, val)
                except TypeError:
                    try:
                        updater(symbol, ts, value)  # type: ignore[arg-type]
                    except TypeError:
                        try:
                            updater(ts, val)
                        except Exception:
                            pass
                except Exception:
                    pass

        if not cache_already_updated and cache_value is not None:
            self._feed_latency_cache(ts, cache_value)

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

    def _mark_price(
        self, ref: Optional[float], bid: Optional[float], ask: Optional[float]
    ) -> Optional[float]:
        """Возвращает цену маркировки позиции.

        Лонги маркируются по bid, шорты — по ask. При отсутствии позиции
        используется mid (если доступны обе стороны) либо ref_price.
        """
        b = bid if bid is not None else None
        a = ask if ask is not None else None
        if self.position_qty > 0.0:
            return (
                float(b) if b is not None else (float(ref) if ref is not None else None)
            )
        if self.position_qty < 0.0:
            return (
                float(a) if a is not None else (float(ref) if ref is not None else None)
            )
        if b is not None and a is not None:
            return float((float(b) + float(a)) / 2.0)
        return float(ref) if ref is not None else None

    def _unrealized_pnl(self, mark_price: Optional[float]) -> float:
        """
        Возвращает нереализованный PnL относительно средней цены позиции.
        """
        if (
            mark_price is None
            or self._avg_entry_price is None
            or self.position_qty == 0.0
        ):
            return 0.0
        mp = float(mark_price)
        ap = float(self._avg_entry_price)
        if self.position_qty > 0.0:
            return float((mp - ap) * self.position_qty)
        else:
            return float((ap - mp) * (-self.position_qty))

    def _recompute_pnl_from_log(
        self, mark_price: Optional[float]
    ) -> tuple[float, float]:
        """Пере вычислить реализованный и нереализованный PnL из журнала трейдов."""
        pos = 0.0
        avg: Optional[float] = None
        realized = 0.0
        for t in self._trade_log:
            q = float(abs(t.qty))
            px = float(t.price)
            if str(t.side).upper() == "BUY":
                if pos < 0.0:
                    close_qty = min(q, -pos)
                    if avg is not None:
                        realized += (avg - px) * close_qty
                    pos += close_qty
                    q_rem = q - close_qty
                    if q_rem > 0.0:
                        pos = q_rem
                        avg = px
                    else:
                        if pos == 0.0:
                            avg = None
                else:
                    new_pos = pos + q
                    if new_pos > 0.0:
                        if pos > 0.0 and avg is not None:
                            avg = (avg * pos + px * q) / new_pos
                        else:
                            avg = px
                    else:
                        avg = None
                    pos = new_pos
            else:
                if pos > 0.0:
                    close_qty = min(q, pos)
                    if avg is not None:
                        realized += (px - avg) * close_qty
                    pos -= close_qty
                    q_rem = q - close_qty
                    if q_rem > 0.0:
                        pos = -q_rem
                        avg = px
                    else:
                        if pos == 0.0:
                            avg = None
                else:
                    new_pos = pos - q
                    if new_pos < 0.0:
                        if pos < 0.0 and avg is not None:
                            avg = (avg * (-pos) + px * q) / (-new_pos)
                        else:
                            avg = px
                    else:
                        avg = None
                    pos = new_pos
        unrl = 0.0
        if mark_price is not None and avg is not None and pos != 0.0:
            if pos > 0.0:
                unrl = (mark_price - avg) * pos
            else:
                unrl = (avg - mark_price) * (-pos)
        return float(realized), float(unrl)

    # ---- очередь ----
    def submit(self, proto: ActionProto, now_ts: Optional[int] = None) -> int:
        cid = self._next_cli_id
        self._next_cli_id += 1
        lat_ms = 0
        timeout = False
        spike = False
        remaining = self.latency_steps
        latency_payload: Any = 0
        if self.latency is not None:
            try:
                ts = int(now_ts) if now_ts is not None else 0
                try:
                    d = self.latency.sample(ts)
                except TypeError:  # fallback for models without ts_ms
                    d = self.latency.sample()  # type: ignore[call-arg]
                lat_ms = int(d.get("total_ms", 0))
                timeout = bool(d.get("timeout", False))
                spike = bool(d.get("spike", False))
                remaining = int(lat_ms // int(self.step_ms))
                latency_payload = d
            except Exception:
                lat_ms = 0
                timeout = False
                spike = False
                remaining = self.latency_steps
                latency_payload = 0
        else:
            latency_payload = lat_ms
        intrabar_lat_ms = self._intrabar_latency_ms(latency_payload)
        if timeout:
            self._cancelled_on_submit.append(cid)
            return cid
        self._q.push(
            Pending(
                proto=proto,
                client_order_id=cid,
                remaining_lat=remaining,
                timestamp=int(now_ts or now_ms()),
                lat_ms=int(lat_ms),
                timeout=bool(timeout),
                spike=bool(spike),
                intrabar_latency_ms=int(max(0, intrabar_lat_ms)),
            )
        )
        return cid

    def _ref(self, ref_price: Optional[float]) -> Optional[float]:
        if ref_price is not None:
            self._last_ref_price = float(ref_price)
        return self._last_ref_price

    def _apply_filters_market(
        self, side: str, qty: float, ref_price: Optional[float]
    ) -> float:
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

    def _apply_filters_limit(
        self, side: str, price: float, qty: float, ref_price: Optional[float]
    ) -> Tuple[float, float, bool]:
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
            ok = self.quantizer.check_percent_price_by_side(
                self.symbol, side, p, ref_price
            )
        return p, q, ok

    def _build_limit_action(self, side: str, qty: float) -> Optional[ActionProto]:
        """Build a LIMIT ActionProto around the mid price."""
        if MidOffsetLimitExecutor is None:
            return None
        try:
            mid = None
            if self._last_bid is not None and self._last_ask is not None:
                mid = (self._last_bid + self._last_ask) / 2.0
            elif self._last_ref_price is not None:
                mid = float(self._last_ref_price)
            if mid is None or qty <= 0.0:
                return None
            execu = MidOffsetLimitExecutor(
                offset_bps=float(self.execution_params.get("limit_offset_bps", 0.0)),
                ttl_steps=int(self.execution_params.get("ttl_steps", 0)),
                tif=str(self.execution_params.get("tif", "GTC")),
            )
            snap = {"mid": float(mid)}
            built = execu.build_action(side=str(side), qty=float(qty), snapshot=snap)
            if built is None:
                return None
            if isinstance(built, ActionProto):
                ap = built
            else:
                ap = ActionProto(
                    action_type=ActionType.LIMIT,
                    volume_frac=float(built.get("volume_frac", 0.0)),
                    tif=str(built.get("tif", "GTC")),
                )
                object.__setattr__(ap, "ttl_steps", int(built.get("ttl_steps", 0)))
                object.__setattr__(ap, "abs_price", built.get("abs_price"))
            return ap
        except Exception:
            return None

    # ---- исполнение ----
    def pop_ready(
        self, now_ts: Optional[int] = None, ref_price: Optional[float] = None
    ) -> ExecReport:
        """Execute all actions whose latency has elapsed.

        The execution path is completely deterministic: child order timing and
        quantity depend solely on the provided timestamps, the latency queue and
        the seeded RNGs.  No new randomness is introduced here, making repeated
        runs with identical inputs reproducible.
        """
        try:
            self._last_bar_close_ts = int(now_ts) if now_ts is not None else self._last_bar_close_ts
        except (TypeError, ValueError):
            pass
        self._reset_intrabar_debug_counter()
        ready, timed_out = self._q.pop_ready()
        trades: List[ExecTrade] = []
        cancelled_ids: List[int] = list(self._cancelled_on_submit)
        cancelled_reasons: dict[int, str] = {int(cid): "OTHER" for cid in cancelled_ids}
        self._cancelled_on_submit = []

        def _cancel(cid: int | str, reason: str = "OTHER") -> None:
            cid_i = int(cid)
            cancelled_ids.append(cid_i)
            cancelled_reasons[cid_i] = reason

        for p in timed_out:
            _cancel(p.client_order_id, "OTHER")
        if self.lob and hasattr(self.lob, "decay_ttl_and_cancel"):
            try:
                expired = self.lob.decay_ttl_and_cancel()
                for x in expired:
                    _cancel(x, "TTL")
            except Exception:
                pass
        else:
            ttl_alive: List[Tuple[int, int]] = []
            for oid, ttl in self._ttl_orders:
                ttl -= 1
                if ttl <= 0:
                    _cancel(oid, "TTL")
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
        risk_events_buffer: List[RiskEvent] = []  # type: ignore[var-annotated]

        ts = int(now_ts or now_ms())
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
                parent_latency_src: Any = (
                    p.intrabar_latency_ms
                    if p.intrabar_latency_ms is not None
                    else p.lat_ms
                )
                parent_latency = self._intrabar_latency_ms(parent_latency_src)
                parent_timeframe = self._resolve_intrabar_timeframe(ts)
                parent_fraction = self._intrabar_time_fraction(
                    parent_latency, parent_timeframe
                )
                ref_market: Optional[float] = None
                intrabar_parent_ref = self._intrabar_reference_price(
                    side, parent_fraction
                )
                if intrabar_parent_ref is not None and math.isfinite(
                    float(intrabar_parent_ref)
                ):
                    ref_market = float(intrabar_parent_ref)
                else:
                    try:
                        ref_market = float(ref) if ref is not None else None
                    except (TypeError, ValueError):
                        ref_market = None
                if ref_market is None or not math.isfinite(ref_market):
                    _cancel(p.client_order_id)
                    continue

                qty_total = self._apply_filters_market(side, qty_raw, ref_market)
                if qty_total <= 0.0:
                    _cancel(p.client_order_id)
                    continue

                # риск: пауза/клампинг размера перед планом
                risk_events_local: List[RiskEvent] = []
                if self.risk is not None:
                    portfolio_total = None
                    try:
                        ref_val = float(ref_market)
                    except (TypeError, ValueError):
                        ref_val = None
                    else:
                        if math.isfinite(ref_val) and ref_val > 0.0:
                            portfolio_total = abs(float(self.position_qty)) * ref_val
                    adj_qty = self.risk.pre_trade_adjust(
                        ts_ms=ts,
                        side=side,
                        intended_qty=qty_total,
                        price=ref_market,
                        position_qty=self.position_qty,
                        total_notional=portfolio_total,
                    )
                    risk_events_local.extend(self.risk.pop_events())
                    if risk_events_local:
                        risk_events_buffer.extend(risk_events_local)
                qty_total = float(adj_qty)
                if qty_total <= 0.0:
                    _cancel(p.client_order_id)
                    # накопим события риска
                    # (дальше они будут добавлены к отчёту)
                    continue

                cap_base_per_bar = self._reset_bar_capacity_if_needed(ts)
                cap_enforced = bool(
                    self._bar_cap_base_enabled and cap_base_per_bar > 0.0
                )
                symbol_key = str(self.symbol).upper() if self.symbol is not None else ""
                used_base_now = max(
                    0.0, float(self._used_base_in_bar.get(symbol_key, 0.0))
                )
                if cap_enforced:
                    remaining_total = max(0.0, cap_base_per_bar - used_base_now)
                    if remaining_total <= 0.0:
                        cid_int = int(p.client_order_id)
                        if cid_int not in cancelled_ids:
                            _cancel(p.client_order_id, "BAR_CAPACITY_BASE")
                        else:
                            cancelled_reasons[cid_int] = "BAR_CAPACITY_BASE"
                        cap_val = float(cap_base_per_bar)
                        capacity_reason = "BAR_CAPACITY_BASE"
                        exec_status = "REJECTED_BY_CAPACITY"
                        trade = ExecTrade(
                            ts=ts,
                            side=side,
                            price=float(ref_market),
                            qty=0.0,
                            notional=0.0,
                            liquidity="taker",
                            proto_type=atype,
                            client_order_id=int(p.client_order_id),
                            slippage_bps=0.0,
                            spread_bps=self._report_spread_bps(self._last_spread_bps),
                            latency_ms=int(p.lat_ms),
                            latency_spike=bool(p.spike),
                            tif=tif,
                            ttl_steps=ttl_steps,
                            status="CANCELED",
                            used_base_before=used_base_now,
                            used_base_after=used_base_now,
                            cap_base_per_bar=cap_val,
                            fill_ratio=0.0,
                            capacity_reason=capacity_reason,
                            exec_status=exec_status,
                        )
                        trades.append(trade)
                        self._trade_log.append(trade)
                        self._record_bar_capacity_metrics(
                            capacity_reason=capacity_reason,
                            exec_status=exec_status,
                            fill_ratio=0.0,
                        )
                        continue
                    qty_total = min(qty_total, remaining_total)
                    if qty_total <= 0.0:
                        cid_int = int(p.client_order_id)
                        if cid_int not in cancelled_ids:
                            _cancel(p.client_order_id, "BAR_CAPACITY_BASE")
                        else:
                            cancelled_reasons[cid_int] = "BAR_CAPACITY_BASE"
                        cap_val = float(cap_base_per_bar)
                        capacity_reason = "BAR_CAPACITY_BASE"
                        exec_status = "REJECTED_BY_CAPACITY"
                        trade = ExecTrade(
                            ts=ts,
                            side=side,
                            price=float(ref_market),
                            qty=0.0,
                            notional=0.0,
                            liquidity="taker",
                            proto_type=atype,
                            client_order_id=int(p.client_order_id),
                            slippage_bps=0.0,
                            spread_bps=self._report_spread_bps(self._last_spread_bps),
                            latency_ms=int(p.lat_ms),
                            latency_spike=bool(p.spike),
                            tif=tif,
                            ttl_steps=ttl_steps,
                            status="CANCELED",
                            used_base_before=used_base_now,
                            used_base_after=used_base_now,
                            cap_base_per_bar=cap_val,
                            fill_ratio=0.0,
                            capacity_reason=capacity_reason,
                            exec_status=exec_status,
                        )
                        trades.append(trade)
                        self._trade_log.append(trade)
                        self._record_bar_capacity_metrics(
                            capacity_reason=capacity_reason,
                            exec_status=exec_status,
                            fill_ratio=0.0,
                        )
                        continue
                else:
                    cap_base_per_bar = 0.0

                # планирование ребёнков (intra-bar)
                notional = abs(qty_total) * float(ref_market)
                executor = self._executor
                if executor is None:
                    thr = float(
                        self._execution_cfg.get("notional_threshold", float("inf"))
                    )
                    if notional > thr:
                        algo = str(
                            self._execution_cfg.get("large_order_algo", "TWAP")
                        ).upper()
                        executor = make_executor(algo, self._execution_cfg)
                    else:
                        executor = TakerExecutor()
                snapshot = {
                    "bid": self._last_bid,
                    "ask": self._last_ask,
                    "mid": (
                        ((self._last_bid + self._last_ask) / 2.0)
                        if (self._last_bid is not None and self._last_ask is not None)
                        else None
                    ),
                    "spread_bps": self._last_spread_bps,
                    "vol_factor": self._last_vol_factor,
                    "liquidity": self._last_liquidity,
                    "ref_price": ref_market,
                }
                plan = executor.plan_market(
                    now_ts_ms=ts, side=side, target_qty=qty_total, snapshot=snapshot
                )

                # если план пуст — отклоняем
                if not plan:
                    _cancel(p.client_order_id)
                    continue

                lat_ms = int(p.lat_ms)
                _ = bool(p.spike)  # spike flag unused, kept for diagnostics

                for child in plan:
                    child_offset = int(getattr(child, "ts_offset_ms", 0))
                    base_ts = int(ts + child_offset)
                    ts_fill = int(base_ts + lat_ms)
                    q_child = float(child.qty)
                    if q_child <= 0.0:
                        continue
                    hint_raw = getattr(child, "liquidity_hint", None)
                    if hint_raw is not None:
                        try:
                            hint_val = float(hint_raw)
                        except (TypeError, ValueError):
                            hint_val = None
                        else:
                            if math.isfinite(hint_val):
                                if hint_val < 0.0:
                                    hint_val = 0.0
                                q_child = min(q_child, hint_val)
                    last_liq_cap = self._last_liquidity
                    if last_liq_cap is not None:
                        try:
                            cap_val = float(last_liq_cap)
                        except (TypeError, ValueError):
                            cap_val = None
                        else:
                            if math.isfinite(cap_val):
                                if cap_val < 0.0:
                                    cap_val = 0.0
                                q_child = min(q_child, cap_val)
                    if q_child <= 0.0:
                        continue
                    order_qty_base = max(0.0, float(q_child))

                    child_latency = self._intrabar_latency_ms(
                        p.lat_ms, child_offset_ms=child_offset
                    )
                    child_timeframe = self._resolve_intrabar_timeframe(base_ts)
                    child_fraction = self._intrabar_time_fraction(
                        child_latency, child_timeframe
                    )
                    order_seq = self._next_order_seq()
                    intrabar_child_price, intrabar_clipped, intrabar_frac = (
                        self._compute_intrabar_price(
                            side=side,
                            time_fraction=child_fraction,
                            fallback_price=ref_market,
                            bar_ts=self._last_bar_close_ts,
                            order_seq=order_seq,
                        )
                    )
                    ref_child_price = ref_market
                    if intrabar_child_price is not None and math.isfinite(
                        float(intrabar_child_price)
                    ):
                        ref_child_price = float(intrabar_child_price)
                    intrabar_base_price = ref_child_price

                    # риск: рейт-лимит на отправку «детей»
                    if self.risk is not None:
                        if not self.risk.can_send_order(ts_fill):
                            # пропускаем этого ребёнка (дросселирование), событие запишем
                            self.risk._emit(
                                ts_fill,
                                "THROTTLE",
                                "order throttled by rate limit",
                                ts_ms=int(ts_fill),
                            )
                            # не вызываем on_new_order() в этом случае
                            continue
                        # отметить отправку
                        self.risk.on_new_order(ts_fill)

                    # квантайзер и minNotional для ребёнка
                    used_base_before_child = max(
                        0.0, float(self._used_base_in_bar.get(symbol_key, 0.0))
                    )
                    if cap_enforced:
                        remaining_base = max(
                            0.0, cap_base_per_bar - used_base_before_child
                        )
                        fill_qty_base = min(q_child, remaining_base)
                    else:
                        remaining_base = float("inf")
                        fill_qty_base = q_child

                    if self.quantizer is not None:
                        fill_qty_base = self.quantizer.quantize_qty(
                            self.symbol, fill_qty_base
                        )
                        if cap_enforced and fill_qty_base > remaining_base + 1e-12:
                            alt_qty = self.quantizer.quantize_qty(
                                self.symbol, max(0.0, remaining_base)
                            )
                            if alt_qty <= remaining_base + 1e-12:
                                fill_qty_base = alt_qty
                            else:
                                fill_qty_base = 0.0
                        if fill_qty_base > 0.0:
                            fill_qty_base = self.quantizer.clamp_notional(
                                self.symbol, ref_child_price, fill_qty_base
                            )
                            if cap_enforced and fill_qty_base > remaining_base + 1e-12:
                                alt_qty = self.quantizer.quantize_qty(
                                    self.symbol, max(0.0, remaining_base)
                                )
                                if alt_qty > 0.0:
                                    alt_qty = self.quantizer.clamp_notional(
                                        self.symbol, ref_child_price, alt_qty
                                    )
                                if alt_qty <= remaining_base + 1e-12:
                                    fill_qty_base = alt_qty
                                else:
                                    fill_qty_base = 0.0
                    elif cap_enforced:
                        fill_qty_base = min(fill_qty_base, remaining_base)

                        if cap_enforced and fill_qty_base <= 0.0:
                            ts_zero = int(base_ts + lat_ms)
                            cid_int = int(p.client_order_id)
                            if cid_int not in cancelled_ids:
                                _cancel(p.client_order_id, "BAR_CAPACITY_BASE")
                            else:
                                cancelled_reasons[cid_int] = "BAR_CAPACITY_BASE"
                            used_before = used_base_before_child
                            capacity_reason = "BAR_CAPACITY_BASE"
                            exec_status = "REJECTED_BY_CAPACITY"
                            cap_val = float(cap_base_per_bar)
                            trade = ExecTrade(
                                ts=ts_zero,
                                side=side,
                                price=float(ref_child_price),
                                qty=0.0,
                            notional=0.0,
                            liquidity="taker",
                            proto_type=atype,
                            client_order_id=int(p.client_order_id),
                            slippage_bps=0.0,
                            spread_bps=self._report_spread_bps(self._last_spread_bps),
                            latency_ms=int(p.lat_ms),
                            latency_spike=bool(p.spike),
                            tif=tif,
                                ttl_steps=ttl_steps,
                                status="CANCELED",
                                used_base_before=used_before,
                                used_base_after=used_before,
                                cap_base_per_bar=cap_val,
                                fill_ratio=0.0,
                                capacity_reason=capacity_reason,
                                exec_status=exec_status,
                            )
                            trades.append(trade)
                            self._trade_log.append(trade)
                            self._record_bar_capacity_metrics(
                                capacity_reason=capacity_reason,
                                exec_status=exec_status,
                                fill_ratio=0.0,
                            )
                            continue

                        q_child = float(fill_qty_base)
                        if q_child <= 0.0:
                            continue

                        cap_val = float(cap_base_per_bar) if cap_enforced else 0.0
                        if order_qty_base > 0.0:
                            fill_ratio = max(0.0, min(1.0, q_child / order_qty_base))
                        else:
                            fill_ratio = 1.0 if q_child > 0.0 else 0.0
                        capacity_reason = ""
                        exec_status = "FILLED"
                        if cap_enforced and order_qty_base > 0.0 and q_child + 1e-12 < order_qty_base:
                            exec_status = "PARTIAL"
                            capacity_reason = "BAR_CAPACITY_BASE"

                        used_base_after_child = max(0.0, used_base_before_child + q_child)
                        self._used_base_in_bar[symbol_key] = used_base_after_child

                        # базовая котировка
                    if VWAPExecutor is not None and isinstance(executor, VWAPExecutor):
                        self._vwap_on_tick(ts_fill, None, None)
                        base_price = (
                            self._last_hour_vwap
                            if self._last_hour_vwap is not None
                            else ref_child_price
                        )
                        filled_price = (
                            float(base_price)
                            if base_price is not None
                            else float(ref_child_price)
                        )
                    elif (
                        str(getattr(self, "execution_profile", "")).upper()
                        == "MKT_OPEN_NEXT_H1"
                    ):
                        if self._next_h1_open_price is not None:
                            filled_price = float(self._next_h1_open_price)
                        else:
                            filled_price = float(ref_child_price)
                            if ref_child_price is not None:
                                self._next_h1_open_price = float(ref_child_price)
                    else:
                        if side == "BUY":
                            base_price = (
                                self._last_ask
                                if self._last_ask is not None
                                else ref_child_price
                            )
                        else:
                            base_price = (
                                self._last_bid
                                if self._last_bid is not None
                                else ref_child_price
                            )
                        filled_price = (
                            float(base_price)
                            if base_price is not None
                            else float(ref_child_price)
                        )

                    # слиппедж на ребёнка
                    slip_bps = 0.0
                    sbps = self._last_spread_bps
                    vf = self._last_vol_factor
                    liq_override = child.liquidity_hint
                    liq = (
                        float(liq_override)
                        if (liq_override is not None)
                        else self._last_liquidity
                    )
                    pre_slip_price = float(filled_price)
                    cfg_slip = (
                        self.execution_params.get("slippage_bps")
                        if isinstance(self.execution_params, dict)
                        else None
                    )
                    trade_cost: Optional[_TradeCostResult] = None
                    fallback_slip: Optional[float] = None
                    if cfg_slip is not None:
                        try:
                            fallback_slip = float(cfg_slip)
                        except (TypeError, ValueError):
                            fallback_slip = None
                    elif self.slippage_cfg is not None:
                        trade_cost = self._compute_dynamic_trade_cost_bps(
                            side=side,
                            qty=q_child,
                            spread_bps=sbps,
                            base_price=pre_slip_price,
                            liquidity=liq,
                            vol_factor=vf,
                            order_seq=order_seq,
                            bar_close_ts=getattr(self, "_last_bar_close_ts", None),
                        )
                        if trade_cost is None and estimate_slippage_bps is not None:
                            fallback_slip = estimate_slippage_bps(
                                spread_bps=sbps,
                                size=q_child,
                                liquidity=liq,
                                vol_factor=vf,
                                cfg=self.slippage_cfg,
                            )
                    if trade_cost is not None:
                        slip_bps = self._trade_cost_expected_bps(trade_cost)
                        new_price = self._apply_trade_cost_price(
                            side=side,
                            pre_slip_price=pre_slip_price,
                            trade_cost=trade_cost,
                        )
                        filled_price = new_price
                        self._log_trade_cost_debug(
                            context="child",
                            side=side,
                            qty=q_child,
                            pre_price=pre_slip_price,
                            final_price=filled_price,
                            trade_cost=trade_cost,
                        )
                    elif fallback_slip is not None:
                        blended = self._blend_expected_spread(
                            taker_bps=fallback_slip
                        )
                        if blended is None:
                            slip_bps = 0.0
                        else:
                            slip_bps = float(blended)
                        if apply_slippage_price is not None:
                            filled_price = apply_slippage_price(
                                side=side,
                                quote_price=pre_slip_price,
                                slippage_bps=slip_bps,
                            )

                    filled_price, final_clip = self._clip_to_bar_range(filled_price)

                    if logger.isEnabledFor(logging.DEBUG):
                        limit = int(self._intrabar_debug_max_logs)
                        if limit <= 0 or self._intrabar_debug_logged < limit:
                            logger.debug(
                                "intrabar fill lat=%sms t=%.4f price=%.6f base_clip=%s final_clip=%s final=%.6f seq=%s",
                                int(lat_ms),
                                float(intrabar_frac),
                                float(intrabar_base_price)
                                if intrabar_base_price is not None
                                else float(ref_child_price),
                                bool(intrabar_clipped),
                                bool(final_clip),
                                float(filled_price),
                                int(order_seq),
                            )
                            self._intrabar_debug_logged += 1

                    # комиссия
                    fee = self._compute_trade_fee(
                        side=side,
                        price=filled_price,
                        qty=q_child,
                        liquidity="taker",
                    )
                    fee_total += float(fee)
                    self.fees_cum += float(fee)

                    # обновить позицию с расчётом реализованного PnL
                    _ = self._apply_trade_inventory(
                        side=side, price=filled_price, qty=q_child
                    )

                    trade = ExecTrade(
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
                        spread_bps=self._report_spread_bps(sbps),
                        latency_ms=int(p.lat_ms),
                        latency_spike=bool(p.spike),
                        tif=tif,
                        ttl_steps=ttl_steps,
                        used_base_before=used_base_before_child,
                        used_base_after=used_base_after_child,
                        cap_base_per_bar=cap_val,
                        fill_ratio=float(fill_ratio),
                        capacity_reason=capacity_reason,
                        exec_status=exec_status,
                    )
                    trades.append(trade)
                    self._trade_log.append(trade)
                    if capacity_reason:
                        self._record_bar_capacity_metrics(
                            capacity_reason=capacity_reason,
                            exec_status=exec_status,
                            fill_ratio=float(fill_ratio),
                        )
                continue
            # Определение направления и базовой цены для прочих типов
            is_buy = bool(getattr(proto, "volume_frac", 0.0) > 0.0)
            side = "BUY" if is_buy else "SELL"
            qty = abs(float(getattr(proto, "volume_frac", 0.0)))
            last_liq_cap = self._last_liquidity
            if last_liq_cap is not None:
                try:
                    cap_val = float(last_liq_cap)
                except (TypeError, ValueError):
                    cap_val = None
                else:
                    if math.isfinite(cap_val):
                        if cap_val < 0.0:
                            cap_val = 0.0
                        qty = min(qty, cap_val)
                        if qty <= 0.0:
                            _cancel(p.client_order_id)
                            continue
            if side == "BUY":
                base_price = self._last_ask if self._last_ask is not None else ref
            else:
                base_price = self._last_bid if self._last_bid is not None else ref
            filled_price = float(base_price) if base_price is not None else float(ref)

            # слиппедж
            slip_bps = 0.0
            sbps = self._last_spread_bps
            vf = self._last_vol_factor
            liq = self._last_liquidity
            pre_slip_price = float(filled_price)
            trade_cost: Optional[_TradeCostResult] = None
            fallback_slip: Optional[float] = None
            if self.slippage_cfg is not None:
                trade_cost = self._compute_dynamic_trade_cost_bps(
                    side=side,
                    qty=qty,
                    spread_bps=sbps,
                    base_price=pre_slip_price,
                    liquidity=liq,
                    vol_factor=vf,
                    order_seq=None,
                    bar_close_ts=getattr(self, "_last_bar_close_ts", None),
                )
                if trade_cost is None and estimate_slippage_bps is not None:
                    fallback_slip = estimate_slippage_bps(
                        spread_bps=sbps,
                        size=qty,
                        liquidity=liq,
                        vol_factor=vf,
                        cfg=self.slippage_cfg,
                    )
            if trade_cost is not None:
                slip_bps = self._trade_cost_expected_bps(trade_cost)
                filled_price = self._apply_trade_cost_price(
                    side=side,
                    pre_slip_price=pre_slip_price,
                    trade_cost=trade_cost,
                )
                self._log_trade_cost_debug(
                    context="market",
                    side=side,
                    qty=qty,
                    pre_price=pre_slip_price,
                    final_price=filled_price,
                    trade_cost=trade_cost,
                )
            elif fallback_slip is not None:
                blended = self._blend_expected_spread(taker_bps=fallback_slip)
                if blended is None:
                    slip_bps = 0.0
                else:
                    slip_bps = float(blended)
                if apply_slippage_price is not None:
                    filled_price = apply_slippage_price(
                        side=side, quote_price=pre_slip_price, slippage_bps=slip_bps
                    )

            # комиссия
            fee = self._compute_trade_fee(
                side=side, price=filled_price, qty=qty, liquidity="taker"
            )
            fee_total += float(fee)
            self.fees_cum += float(fee)

            # обновить позицию с расчётом реализованного PnL
            _ = self._apply_trade_inventory(side=side, price=filled_price, qty=qty)

            trade = ExecTrade(
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
                spread_bps=self._report_spread_bps(sbps),
                latency_ms=int(p.lat_ms),
                latency_spike=bool(p.spike),
                tif=tif,
                ttl_steps=ttl_steps,
            )
            trades.append(trade)
            self._trade_log.append(trade)
            continue

            # LIMIT
            if atype == ActionType.LIMIT:
                is_buy = bool(getattr(proto, "volume_frac", 0.0) > 0.0)
                side = "BUY" if is_buy else "SELL"
                qty_raw = abs(float(getattr(proto, "volume_frac", 0.0)))
                limit_latency_src: Any = (
                    p.intrabar_latency_ms
                    if p.intrabar_latency_ms is not None
                    else p.lat_ms
                )
                limit_latency = self._intrabar_latency_ms(limit_latency_src)
                limit_timeframe = self._resolve_intrabar_timeframe(ts)
                limit_fraction = self._intrabar_time_fraction(
                    limit_latency, limit_timeframe
                )
                order_seq = self._next_order_seq()
                limit_intrabar_price, limit_intrabar_clipped, limit_intrabar_frac = (
                    self._compute_intrabar_price(
                        side=side,
                        time_fraction=limit_fraction,
                        fallback_price=ref,
                        bar_ts=self._last_bar_close_ts,
                        order_seq=order_seq,
                    )
                )
                ref_limit: Optional[float] = None
                if limit_intrabar_price is not None and math.isfinite(
                    float(limit_intrabar_price)
                ):
                    ref_limit = float(limit_intrabar_price)
                else:
                    try:
                        ref_limit = float(ref) if ref is not None else None
                    except (TypeError, ValueError):
                        ref_limit = None
                intrabar_base_price = ref_limit
                if intrabar_base_price is None and ref is not None:
                    try:
                        intrabar_base_price = float(ref)
                    except (TypeError, ValueError):
                        intrabar_base_price = None

                # Определяем лимитную цену
                abs_price = getattr(proto, "abs_price", None)
                if abs_price is None:
                    # нет абсолютной цены в proto — попробуем использовать ref_price как базу
                    if ref_limit is None:
                        # ничего не можем сделать — считаем, что заявка размещена (эмуляция)
                        new_order_ids.append(int(p.client_order_id))
                        new_order_pos.append(0)
                        continue
                    # без знания tickSize в тиках используем abs_price=ref (реальную оффсет-логику добавим позже)
                    abs_price = float(ref_limit)

                price_q, qty_q, ok = self._apply_filters_limit(
                    side, float(abs_price), qty_raw, ref_limit
                )
                if qty_q <= 0.0 or not ok:
                    _cancel(p.client_order_id)
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
                        _cancel(p.client_order_id, "FOK")
                        continue
                    filled_price, final_clip = self._clip_to_bar_range(filled_price)
                    if logger.isEnabledFor(logging.DEBUG):
                        limit = int(self._intrabar_debug_max_logs)
                        if limit <= 0 or self._intrabar_debug_logged < limit:
                            logger.debug(
                                "intrabar fill lat=%sms t=%.4f price=%.6f base_clip=%s final_clip=%s final=%.6f seq=%s",
                                int(limit_latency),
                                float(limit_intrabar_frac),
                                float(intrabar_base_price)
                                if intrabar_base_price is not None
                                else float(price_q),
                                bool(limit_intrabar_clipped),
                                bool(final_clip),
                                float(filled_price),
                                int(order_seq),
                            )
                            self._intrabar_debug_logged += 1
                    fee = self._compute_trade_fee(
                        side=side,
                        price=filled_price,
                        qty=exec_qty,
                        liquidity=liquidity_role,
                    )
                    fee_total += float(fee)
                    _ = self._apply_trade_inventory(
                        side=side, price=filled_price, qty=exec_qty
                    )
                    sbps = self._last_spread_bps
                    trade = ExecTrade(
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
                        spread_bps=self._report_spread_bps(sbps),
                        latency_ms=int(p.lat_ms),
                        latency_spike=bool(p.spike),
                        tif=tif,
                        ttl_steps=ttl_steps,
                    )
                    trades.append(trade)
                    self._trade_log.append(trade)
                    if exec_qty + 1e-12 < qty_q:
                        if tif == "IOC":
                            _cancel(p.client_order_id, "IOC")
                            continue
                        qty_q = qty_q - exec_qty
                        filled = False
                        liquidity_role = "maker"
                    else:
                        continue

                if filled and liquidity_role == "maker":
                    if tif in ("IOC", "FOK"):
                        _cancel(p.client_order_id, tif)
                        continue
                    filled_price, final_clip = self._clip_to_bar_range(filled_price)
                    if logger.isEnabledFor(logging.DEBUG):
                        limit = int(self._intrabar_debug_max_logs)
                        if limit <= 0 or self._intrabar_debug_logged < limit:
                            logger.debug(
                                "intrabar fill lat=%sms t=%.4f price=%.6f base_clip=%s final_clip=%s final=%.6f seq=%s",
                                int(limit_latency),
                                float(limit_intrabar_frac),
                                float(intrabar_base_price)
                                if intrabar_base_price is not None
                                else float(price_q),
                                bool(limit_intrabar_clipped),
                                bool(final_clip),
                                float(filled_price),
                                int(order_seq),
                            )
                            self._intrabar_debug_logged += 1
                    fee = self._compute_trade_fee(
                        side=side,
                        price=filled_price,
                        qty=qty_q,
                        liquidity=liquidity_role,
                    )
                    fee_total += float(fee)
                    _ = self._apply_trade_inventory(
                        side=side, price=filled_price, qty=qty_q
                    )
                    sbps = self._last_spread_bps
                    trade = ExecTrade(
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
                        spread_bps=self._report_spread_bps(sbps),
                        latency_ms=int(p.lat_ms),
                        latency_spike=bool(p.spike),
                        tif=tif,
                        ttl_steps=ttl_steps,
                    )
                    trades.append(trade)
                    self._trade_log.append(trade)
                    continue

                if tif in ("IOC", "FOK"):
                    _cancel(p.client_order_id, tif)
                    continue

                if self.lob and hasattr(self.lob, "add_limit_order"):
                    try:
                        oid, qpos = self.lob.add_limit_order(
                            is_buy, float(price_q), float(qty_q), ts, True
                        )
                        if oid:
                            new_order_ids.append(int(oid))
                            new_order_pos.append(int(qpos) if qpos is not None else 0)
                            if ttl_steps > 0:
                                ttl_set = False
                                if hasattr(self.lob, "set_order_ttl"):
                                    try:
                                        ttl_set = bool(
                                            self.lob.set_order_ttl(
                                                int(oid), int(ttl_steps)
                                            )
                                        )
                                    except Exception:
                                        ttl_set = False
                                if not ttl_set:
                                    self._ttl_orders.append((int(oid), int(ttl_steps)))
                    except Exception:
                        _cancel(p.client_order_id)
                else:
                    new_order_ids.append(int(p.client_order_id))
                    new_order_pos.append(0)
                    if ttl_steps > 0:
                        self._ttl_orders.append(
                            (int(p.client_order_id), int(ttl_steps))
                        )
                continue

            # прочее — no-op
            _cancel(p.client_order_id)

        # funding: начислить по текущей позиции и актуальной рыночной цене (используем ref как mark)
        funding_cashflow = 0.0
        funding_events_list = []
        if self.funding is not None:
            fc, events = self.funding.accrue(
                position_qty=self.position_qty, mark_price=ref, now_ts_ms=ts
            )
            funding_cashflow = float(fc)
            funding_events_list = list(events or [])
            self.funding_cum += float(fc)

        # mark-to-... и PnL
        mark_p = self._mark_price(ref=ref, bid=self._last_bid, ask=self._last_ask)
        unrl = self._unrealized_pnl(mark_p)
        eq = float(self.realized_pnl_cum + unrl - self.fees_cum + self.funding_cum)
        if self._trade_log:
            r_chk, u_chk = self._recompute_pnl_from_log(mark_p)
            assert abs((self.realized_pnl_cum + unrl) - (r_chk + u_chk)) < 1e-6

        # риск: обновить дневной PnL и возможную паузу
        risk_events_all: List[RiskEvent] = []
        if risk_events_buffer:
            risk_events_all.extend(risk_events_buffer)
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
            cancelled_reasons=cancelled_reasons,
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
            execution_profile=str(getattr(self, "execution_profile", "")),
            vol_raw=self._last_vol_raw,
        )

        if trades:
            last_trade = trades[-1]
            report.cap_base_per_bar = float(
                getattr(last_trade, "cap_base_per_bar", 0.0)
            )
            report.used_base_before = float(
                getattr(last_trade, "used_base_before", 0.0)
            )
            report.used_base_after = float(
                getattr(last_trade, "used_base_after", 0.0)
            )
            report.fill_ratio = float(getattr(last_trade, "fill_ratio", 0.0))
            report.capacity_reason = str(
                getattr(last_trade, "capacity_reason", "") or ""
            )
            report.exec_status = str(getattr(last_trade, "exec_status", "") or "")
        else:
            cap_val = locals().get("cap_base_per_bar", 0.0)  # type: ignore[arg-type]
            try:
                report.cap_base_per_bar = float(cap_val)
            except (TypeError, ValueError):
                report.cap_base_per_bar = 0.0
            report.used_base_before = 0.0
            report.used_base_after = 0.0
            report.fill_ratio = 0.0
            report.capacity_reason = ""
            report.exec_status = ""

        maker_share_val, expected_fee_bps, expected_spread_bps, cost_components = (
            self._expected_cost_snapshot()
        )
        if maker_share_val is not None:
            try:
                report.maker_share = float(maker_share_val)
            except (TypeError, ValueError):
                report.maker_share = 0.0
        else:
            report.maker_share = 0.0
        if expected_fee_bps is not None:
            try:
                report.expected_fee_bps = float(expected_fee_bps)
            except (TypeError, ValueError):
                report.expected_fee_bps = 0.0
        else:
            report.expected_fee_bps = 0.0
        report.expected_spread_bps = (
            float(expected_spread_bps)
            if expected_spread_bps is not None
            else None
        )
        report.expected_cost_components = dict(cost_components)

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
    def run_step(
        self,
        *,
        ts: int,
        ref_price: float | None,
        bid: float | None = None,
        ask: float | None = None,
        vol_factor: float | None = None,
        vol_raw: Optional[Mapping[str, float]] = None,
        liquidity: float | None = None,
        trade_price: float | None = None,
        trade_qty: float | None = None,
        bar_open: float | None = None,
        bar_high: float | None = None,
        bar_low: float | None = None,
        bar_close: float | None = None,
        bar_timeframe_ms: int | None = None,
        actions: list[tuple[object, object]] | None = None,
    ) -> "ExecReport":
        """
        Универсальный публичный шаг симуляции.
          - Обновляет рыночный снапшот.
          - Обрабатывает список действий: [(ActionType, proto), ...].
          - Возвращает ExecReport с трейдами и PnL-компонентами.
        Примечания:
          - Поддержан тип ActionType.MARKET. Другие типы будут отклонены.
          - proto должен иметь атрибут volume_frac (знак = направление).
        """
        try:
            self._last_bar_close_ts = int(ts)
        except (TypeError, ValueError):
            self._last_bar_close_ts = None
        self._reset_intrabar_debug_counter()
        # --- обновить рыночный снапшот ---
        self._last_bid = float(bid) if bid is not None else None
        self._last_ask = float(ask) if ask is not None else None
        if vol_factor is not None:
            try:
                vf_val = float(vol_factor)
            except (TypeError, ValueError):
                vf_val = None
            else:
                if not math.isfinite(vf_val):
                    vf_val = None
        else:
            vf_val = None
        self._last_vol_factor = vf_val
        bar_open_val = self._float_or_none(bar_open)
        bar_high_val = self._float_or_none(bar_high)
        bar_low_val = self._float_or_none(bar_low)
        ref_val = self._float_or_none(ref_price)
        close_candidate = bar_close if bar_close is not None else ref_val
        bar_close_val = self._float_or_none(close_candidate)
        trade_val = self._float_or_none(trade_price)
        mid_for_range = self._resolve_mid_from_inputs(
            self._last_bid,
            self._last_ask,
            bar_high_val,
            bar_low_val,
            bar_close_val,
            trade_val,
            self._last_ref_price,
        )
        range_ratio, range_ratio_bps = self._compute_bar_range_ratios(
            bar_high_val, bar_low_val, mid_for_range
        )
        metrics = self._normalize_vol_metrics(
            vol_raw,
            computed_range_ratio=range_ratio,
            computed_range_ratio_bps=range_ratio_bps,
        )
        self._last_vol_raw = metrics
        range_ratio_bps_val: Optional[float] = None
        if metrics is not None:
            range_ratio_bps_val = self._non_negative_float(
                metrics.get("range_ratio_bps")
            )
            if range_ratio_bps_val is None:
                ratio_hint = self._non_negative_float(metrics.get("range_ratio"))
                if ratio_hint is not None:
                    range_ratio_bps_val = ratio_hint * 1e4
        if range_ratio_bps_val is None:
            range_ratio_bps_val = self._non_negative_float(range_ratio_bps)
        if liquidity is None:
            incoming_liq = None
        else:
            try:
                incoming_liq = float(liquidity)
            except (TypeError, ValueError):
                incoming_liq = None
            else:
                if not math.isfinite(incoming_liq):
                    incoming_liq = None
                elif incoming_liq < 0.0:
                    incoming_liq = 0.0
        timeframe_hint = bar_timeframe_ms
        if timeframe_hint is None:
            timeframe_hint = self._resolve_intrabar_timeframe(ts)
        adv_capacity = self._adv_bar_capacity(self.symbol, timeframe_hint)
        adv_capacity_norm = (
            max(0.0, float(adv_capacity)) if adv_capacity is not None else None
        )
        self._last_adv_bar_capacity = adv_capacity_norm
        self._last_liquidity = self._combine_liquidity(incoming_liq, adv_capacity_norm)
        self._last_ref_price = ref_val
        self._last_bar_open = bar_open_val
        self._last_bar_high = bar_high_val
        self._last_bar_low = bar_low_val
        self._last_bar_close = bar_close_val
        base_spread: Optional[float] = None
        if (
            compute_spread_bps_from_quotes is not None
            and self.slippage_cfg is not None
        ):
            try:
                base_spread = compute_spread_bps_from_quotes(
                    bid=self._last_bid,
                    ask=self._last_ask,
                    cfg=self.slippage_cfg,
                )
            except Exception:
                base_spread = None
        self._last_spread_bps = self._compute_effective_spread_bps(
            base_spread_bps=base_spread,
            ts_ms=ts,
            vol_factor=self._last_vol_factor,
            range_ratio_bps=range_ratio_bps_val,
        )
        if bar_timeframe_ms is not None:
            self.set_intrabar_timeframe_ms(bar_timeframe_ms)
        price_tick = trade_price if trade_price is not None else self._last_ref_price
        qty_tick = trade_qty if trade_qty is not None else liquidity
        if price_tick is not None and qty_tick is not None:
            self._vwap_on_tick(int(ts), float(price_tick), float(qty_tick))
        if self._last_vol_factor is not None:
            self._update_latency_volatility(int(ts), self._last_vol_factor, metrics)

        # --- инициализация аккамуляторов ---
        trades: list[ExecTrade] = []
        cancelled_ids: list[int] = []
        cancelled_reasons: dict[int, str] = {}
        new_order_ids: list[int] = []
        new_order_pos: list[int] = []
        fee_total: float = 0.0

        # --- обработать действия ---
        acts = list(actions or [])
        if str(getattr(self, "execution_profile", "")).upper() == "LIMIT_MID_BPS":
            for atype, proto in acts:
                if str(
                    getattr(atype, "name", getattr(atype, "__class__", type(atype)))
                ).upper().endswith("MARKET") or str(atype).upper().endswith("MARKET"):
                    vol = float(getattr(proto, "volume_frac", 0.0))
                    side = "BUY" if vol > 0.0 else "SELL"
                    qty = abs(vol)
                    built = self._build_limit_action(side, qty)
                    if built is not None:
                        self.submit(built, now_ts=ts)
            return self.pop_ready(now_ts=ts, ref_price=ref_price)

        def _cancel(cid: int | str, reason: str = "OTHER") -> None:
            cid_i = int(cid)
            cancelled_ids.append(cid_i)
            cancelled_reasons[cid_i] = reason

        for atype, proto in acts:
            # оформление client_order_id
            cli_id = int(self._next_cli_id)
            self._next_cli_id += 1
            new_order_ids.append(cli_id)

            # только MARKET
            if str(
                getattr(atype, "name", getattr(atype, "__class__", type(atype)))
            ).upper().endswith("MARKET") or str(atype).upper().endswith("MARKET"):
                # определить сторону и величину
                vol = float(getattr(proto, "volume_frac", 0.0))
                is_buy = bool(vol > 0.0)
                side = "BUY" if is_buy else "SELL"
                qty_raw = abs(float(vol))
                ttl_steps = int(getattr(proto, "ttl_steps", 0))
                tif = str(getattr(proto, "tif", "GTC")).upper()
                ref_parent = self._last_ref_price
                parent_latency = self._intrabar_latency_ms(0)
                parent_timeframe = self._resolve_intrabar_timeframe(ts)
                parent_fraction = self._intrabar_time_fraction(
                    parent_latency, parent_timeframe
                )
                ref_market: Optional[float] = None
                intrabar_parent_ref = self._intrabar_reference_price(
                    side, parent_fraction
                )
                if intrabar_parent_ref is not None and math.isfinite(
                    float(intrabar_parent_ref)
                ):
                    ref_market = float(intrabar_parent_ref)
                else:
                    try:
                        ref_market = (
                            float(ref_parent) if ref_parent is not None else None
                        )
                    except (TypeError, ValueError):
                        ref_market = None
                if ref_market is None or not math.isfinite(ref_market):
                    _cancel(cli_id)
                    continue

                # применить фильтры рынка (квантизация/minNotional и т.п. внутри вспом. функции)
                qty_total = self._apply_filters_market(side, qty_raw, ref_market)
                if qty_total <= 0.0:
                    _cancel(cli_id)
                    continue

                # риск: корректировка/пауза
                if self.risk is not None:
                    portfolio_total = None
                    if ref_market is not None:
                        try:
                            ref_val = float(ref_market)
                        except (TypeError, ValueError):
                            ref_val = None
                        else:
                            if math.isfinite(ref_val) and ref_val > 0.0:
                                portfolio_total = abs(float(self.position_qty)) * ref_val
                    adj_qty = self.risk.pre_trade_adjust(
                        ts_ms=ts,
                        side=side,
                        intended_qty=qty_total,
                        price=ref_market,
                        position_qty=self.position_qty,
                        total_notional=portfolio_total,
                    )
                    qty_total = float(adj_qty)
                    for _e in self.risk.pop_events():
                        # события риска будут добавлены позже через on_mark(); здесь просто очищаем очередь
                        pass
                    if qty_total <= 0.0:
                        _cancel(cli_id)
                        continue

                cap_base_per_bar = self._reset_bar_capacity_if_needed(ts)
                cap_enforced = bool(
                    self._bar_cap_base_enabled and cap_base_per_bar > 0.0
                )
                symbol_key = str(self.symbol).upper() if self.symbol is not None else ""
                used_base_now = max(
                    0.0, float(self._used_base_in_bar.get(symbol_key, 0.0))
                )
                if cap_enforced:
                    remaining_total = max(0.0, cap_base_per_bar - used_base_now)
                    if remaining_total <= 0.0:
                        if cli_id not in cancelled_ids:
                            _cancel(cli_id, "BAR_CAPACITY_BASE")
                        else:
                            cancelled_reasons[int(cli_id)] = "BAR_CAPACITY_BASE"
                        capacity_reason = "BAR_CAPACITY_BASE"
                        exec_status = "REJECTED_BY_CAPACITY"
                        cap_val = float(cap_base_per_bar)
                        trade = ExecTrade(
                            ts=ts,
                            side=side,
                            price=float(ref_market),
                            qty=0.0,
                            notional=0.0,
                            liquidity="taker",
                            proto_type=getattr(atype, "value", 0),
                            client_order_id=int(cli_id),
                            slippage_bps=0.0,
                            spread_bps=self._report_spread_bps(self._last_spread_bps),
                            latency_ms=0,
                            latency_spike=False,
                            tif=tif,
                            ttl_steps=ttl_steps,
                            status="CANCELED",
                            used_base_before=used_base_now,
                            used_base_after=used_base_now,
                            cap_base_per_bar=cap_val,
                            fill_ratio=0.0,
                            capacity_reason=capacity_reason,
                            exec_status=exec_status,
                        )
                        trades.append(trade)
                        self._trade_log.append(trade)
                        self._record_bar_capacity_metrics(
                            capacity_reason=capacity_reason,
                            exec_status=exec_status,
                            fill_ratio=0.0,
                        )
                        continue
                    qty_total = min(qty_total, remaining_total)
                    if qty_total <= 0.0:
                        if cli_id not in cancelled_ids:
                            _cancel(cli_id, "BAR_CAPACITY_BASE")
                        else:
                            cancelled_reasons[int(cli_id)] = "BAR_CAPACITY_BASE"
                        capacity_reason = "BAR_CAPACITY_BASE"
                        exec_status = "REJECTED_BY_CAPACITY"
                        cap_val = float(cap_base_per_bar)
                        trade = ExecTrade(
                            ts=ts,
                            side=side,
                            price=float(ref_market),
                            qty=0.0,
                            notional=0.0,
                            liquidity="taker",
                            proto_type=getattr(atype, "value", 0),
                            client_order_id=int(cli_id),
                            slippage_bps=0.0,
                            spread_bps=self._report_spread_bps(self._last_spread_bps),
                            latency_ms=0,
                            latency_spike=False,
                            tif=tif,
                            ttl_steps=ttl_steps,
                            status="CANCELED",
                            used_base_before=used_base_now,
                            used_base_after=used_base_now,
                            cap_base_per_bar=cap_val,
                            fill_ratio=0.0,
                            capacity_reason=capacity_reason,
                            exec_status=exec_status,
                        )
                        trades.append(trade)
                        self._trade_log.append(trade)
                        self._record_bar_capacity_metrics(
                            capacity_reason=capacity_reason,
                            exec_status=exec_status,
                            fill_ratio=0.0,
                        )
                        continue
                else:
                    cap_base_per_bar = 0.0

                # планирование исполнения
                executor = (
                    self._executor if self._executor is not None else TakerExecutor()
                )
                snapshot = {
                    "bid": self._last_bid,
                    "ask": self._last_ask,
                    "mid": (
                        ((self._last_bid + self._last_ask) / 2.0)
                        if (self._last_bid is not None and self._last_ask is not None)
                        else None
                    ),
                    "spread_bps": self._last_spread_bps,
                    "vol_factor": self._last_vol_factor,
                    "liquidity": self._last_liquidity,
                    "ref_price": ref_market,
                }
                plan = executor.plan_market(
                    now_ts_ms=ts, side=side, target_qty=qty_total, snapshot=snapshot
                )
                if not plan:
                    _cancel(cli_id)
                    continue

                # пройтись по детям с учётом латентности/слиппеджа/комиссий/инвентаря/рейт-лимита
                for child in plan:
                    child_offset = int(getattr(child, "ts_offset_ms", 0))
                    base_ts = int(ts + child_offset)
                    ts_fill = int(base_ts)
                    q_child = float(child.qty)
                    if q_child <= 0.0:
                        continue
                    hint_raw = getattr(child, "liquidity_hint", None)
                    if hint_raw is not None:
                        try:
                            hint_val = float(hint_raw)
                        except (TypeError, ValueError):
                            hint_val = None
                        else:
                            if math.isfinite(hint_val):
                                if hint_val < 0.0:
                                    hint_val = 0.0
                                q_child = min(q_child, hint_val)
                    last_liq_cap = self._last_liquidity
                    if last_liq_cap is not None:
                        try:
                            cap_val = float(last_liq_cap)
                        except (TypeError, ValueError):
                            cap_val = None
                        else:
                            if math.isfinite(cap_val):
                                if cap_val < 0.0:
                                    cap_val = 0.0
                                q_child = min(q_child, cap_val)
                    if q_child <= 0.0:
                        continue
                    order_qty_base = max(0.0, float(q_child))

                    # риск: дросселирование
                    if self.risk is not None:
                        if not self.risk.can_send_order(ts_fill):
                            self.risk._emit(
                                ts_fill,
                                "THROTTLE",
                                "order throttled by rate limit",
                                ts_ms=int(ts_fill),
                            )
                            continue
                        self.risk.on_new_order(ts_fill)

                    # латентность и intrabar-референс
                    lat_ms = 0
                    lat_spike = False
                    latency_payload: Any = lat_ms
                    if self.latency is not None:
                        try:
                            d = self.latency.sample(int(ts_fill))
                        except TypeError:  # fallback for non-seasonal models
                            d = self.latency.sample()  # type: ignore[call-arg]
                        lat_ms = int(d.get("total_ms", 0))
                        lat_spike = bool(d.get("spike", False))
                        if bool(d.get("timeout", False)):
                            _cancel(cli_id)
                            continue
                        latency_payload = d
                    child_latency = self._intrabar_latency_ms(
                        latency_payload, child_offset_ms=child_offset
                    )
                    child_timeframe = self._resolve_intrabar_timeframe(base_ts)
                    child_fraction = self._intrabar_time_fraction(
                        child_latency, child_timeframe
                    )
                    order_seq = self._next_order_seq()
                    intrabar_child_price, intrabar_clipped, intrabar_frac = (
                        self._compute_intrabar_price(
                            side=side,
                            time_fraction=child_fraction,
                            fallback_price=ref_market,
                            bar_ts=self._last_bar_close_ts,
                            order_seq=order_seq,
                        )
                    )
                    ref_child_price = ref_market
                    if intrabar_child_price is not None and math.isfinite(
                        float(intrabar_child_price)
                    ):
                        ref_child_price = float(intrabar_child_price)
                    intrabar_base_price = ref_child_price

                    # квантайзер и minNotional
                    used_base_before_child = max(
                        0.0, float(self._used_base_in_bar.get(symbol_key, 0.0))
                    )
                    if cap_enforced:
                        remaining_base = max(
                            0.0, cap_base_per_bar - used_base_before_child
                        )
                        fill_qty_base = min(q_child, remaining_base)
                    else:
                        remaining_base = float("inf")
                        fill_qty_base = q_child

                    if self.quantizer is not None:
                        fill_qty_base = self.quantizer.quantize_qty(
                            self.symbol, fill_qty_base
                        )
                        if cap_enforced and fill_qty_base > remaining_base + 1e-12:
                            alt_qty = self.quantizer.quantize_qty(
                                self.symbol, max(0.0, remaining_base)
                            )
                            if alt_qty <= remaining_base + 1e-12:
                                fill_qty_base = alt_qty
                            else:
                                fill_qty_base = 0.0
                        if fill_qty_base > 0.0:
                            fill_qty_base = self.quantizer.clamp_notional(
                                self.symbol, ref_child_price, fill_qty_base
                            )
                            if cap_enforced and fill_qty_base > remaining_base + 1e-12:
                                alt_qty = self.quantizer.quantize_qty(
                                    self.symbol, max(0.0, remaining_base)
                                )
                                if alt_qty > 0.0:
                                    alt_qty = self.quantizer.clamp_notional(
                                        self.symbol, ref_child_price, alt_qty
                                    )
                                if alt_qty <= remaining_base + 1e-12:
                                    fill_qty_base = alt_qty
                                else:
                                    fill_qty_base = 0.0
                    elif cap_enforced:
                        fill_qty_base = min(fill_qty_base, remaining_base)

                    if cap_enforced and fill_qty_base <= 0.0:
                        ts_zero = int(base_ts + lat_ms)
                        if cli_id not in cancelled_ids:
                            _cancel(cli_id, "BAR_CAPACITY_BASE")
                        else:
                            cancelled_reasons[int(cli_id)] = "BAR_CAPACITY_BASE"
                        used_before = used_base_before_child
                        capacity_reason = "BAR_CAPACITY_BASE"
                        exec_status = "REJECTED_BY_CAPACITY"
                        cap_val = float(cap_base_per_bar)
                        trade = ExecTrade(
                            ts=ts_zero,
                            side=side,
                            price=float(ref_child_price),
                            qty=0.0,
                            notional=0.0,
                            liquidity="taker",
                            proto_type=getattr(atype, "value", 0),
                            client_order_id=int(cli_id),
                            slippage_bps=0.0,
                            spread_bps=self._report_spread_bps(self._last_spread_bps),
                            latency_ms=int(lat_ms),
                            latency_spike=bool(lat_spike),
                            tif=tif,
                            ttl_steps=ttl_steps,
                            status="CANCELED",
                            used_base_before=used_before,
                            used_base_after=used_before,
                            cap_base_per_bar=cap_val,
                            fill_ratio=0.0,
                            capacity_reason=capacity_reason,
                            exec_status=exec_status,
                        )
                        trades.append(trade)
                        self._trade_log.append(trade)
                        self._record_bar_capacity_metrics(
                            capacity_reason=capacity_reason,
                            exec_status=exec_status,
                            fill_ratio=0.0,
                        )
                        continue

                    q_child = float(fill_qty_base)
                    if q_child <= 0.0:
                        continue

                    cap_val = float(cap_base_per_bar) if cap_enforced else 0.0
                    if order_qty_base > 0.0:
                        fill_ratio = max(0.0, min(1.0, q_child / order_qty_base))
                    else:
                        fill_ratio = 1.0 if q_child > 0.0 else 0.0
                    capacity_reason = ""
                    exec_status = "FILLED"
                    if cap_enforced and order_qty_base > 0.0 and q_child + 1e-12 < order_qty_base:
                        exec_status = "PARTIAL"
                        capacity_reason = "BAR_CAPACITY_BASE"

                    used_base_after_child = max(0.0, used_base_before_child + q_child)
                    self._used_base_in_bar[symbol_key] = used_base_after_child

                    ts_fill = int(base_ts + lat_ms)
                    # цена исполнения
                    if VWAPExecutor is not None and isinstance(executor, VWAPExecutor):
                        self._vwap_on_tick(ts_fill, None, None)
                        base_price = (
                            self._last_hour_vwap
                            if self._last_hour_vwap is not None
                            else ref_child_price
                        )
                        filled_price = (
                            float(base_price)
                            if base_price is not None
                            else float(ref_child_price)
                        )
                    elif (
                        str(getattr(self, "execution_profile", "")).upper()
                        == "MKT_OPEN_NEXT_H1"
                        and self._next_h1_open_price is not None
                    ):
                        filled_price = float(self._next_h1_open_price)
                    else:
                        if side == "BUY":
                            base_price = (
                                self._last_ask
                                if self._last_ask is not None
                                else ref_child_price
                            )
                        else:
                            base_price = (
                                self._last_bid
                                if self._last_bid is not None
                                else ref_child_price
                            )
                        filled_price = (
                            float(base_price)
                            if base_price is not None
                            else float(ref_child_price)
                        )
                    slip_bps = 0.0
                    sbps = self._last_spread_bps
                    vf = self._last_vol_factor
                    liq = self._last_liquidity
                    pre_slip_price = float(filled_price)
                    trade_cost: Optional[_TradeCostResult] = None
                    fallback_slip: Optional[float] = None
                    if self.slippage_cfg is not None:
                        trade_cost = self._compute_dynamic_trade_cost_bps(
                            side=side,
                            qty=q_child,
                            spread_bps=sbps,
                            base_price=pre_slip_price,
                            liquidity=liq,
                            vol_factor=vf,
                            order_seq=order_seq,
                            bar_close_ts=getattr(self, "_last_bar_close_ts", None),
                        )
                        if trade_cost is None and estimate_slippage_bps is not None:
                            fallback_slip = estimate_slippage_bps(
                                spread_bps=sbps,
                                size=q_child,
                                liquidity=liq,
                                vol_factor=vf,
                                cfg=self.slippage_cfg,
                            )
                    if trade_cost is not None:
                        slip_bps = self._trade_cost_expected_bps(trade_cost)
                        new_price = self._apply_trade_cost_price(
                            side=side,
                            pre_slip_price=pre_slip_price,
                            trade_cost=trade_cost,
                        )
                        filled_price = new_price
                        self._log_trade_cost_debug(
                            context="algo_child",
                            side=side,
                            qty=q_child,
                            pre_price=pre_slip_price,
                            final_price=filled_price,
                            trade_cost=trade_cost,
                        )
                    elif fallback_slip is not None:
                        blended = self._blend_expected_spread(
                            taker_bps=fallback_slip
                        )
                        if blended is None:
                            slip_bps = 0.0
                        else:
                            slip_bps = float(blended)
                        if apply_slippage_price is not None:
                            filled_price = apply_slippage_price(
                                side=side,
                                quote_price=pre_slip_price,
                                slippage_bps=slip_bps,
                            )

                    filled_price, final_clip = self._clip_to_bar_range(filled_price)

                    if logger.isEnabledFor(logging.DEBUG):
                        limit = int(self._intrabar_debug_max_logs)
                        if limit <= 0 or self._intrabar_debug_logged < limit:
                            logger.debug(
                                "intrabar fill lat=%sms t=%.4f price=%.6f base_clip=%s final_clip=%s final=%.6f seq=%s",
                                int(lat_ms),
                                float(intrabar_frac),
                                float(intrabar_base_price)
                                if intrabar_base_price is not None
                                else float(ref_child_price),
                                bool(intrabar_clipped),
                                bool(final_clip),
                                float(filled_price),
                                int(order_seq),
                            )
                            self._intrabar_debug_logged += 1

                    # комиссия
                    fee = self._compute_trade_fee(
                        side=side,
                        price=filled_price,
                        qty=q_child,
                        liquidity="taker",
                    )
                    fee_total += float(fee)
                    self.fees_cum += float(fee)

                    # инвентарь + реализованный PnL
                    _ = self._apply_trade_inventory(
                        side=side, price=filled_price, qty=q_child
                    )

                    # запись трейда
                    trade = ExecTrade(
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
                        spread_bps=self._report_spread_bps(sbps),
                        latency_ms=int(lat_ms),
                        latency_spike=bool(lat_spike),
                        tif=tif,
                        ttl_steps=ttl_steps,
                        used_base_before=used_base_before_child,
                        used_base_after=used_base_after_child,
                        cap_base_per_bar=cap_val,
                        fill_ratio=float(fill_ratio),
                        capacity_reason=capacity_reason,
                        exec_status=exec_status,
                    )
                    trades.append(trade)
                    self._trade_log.append(trade)
                    if capacity_reason:
                        self._record_bar_capacity_metrics(
                            capacity_reason=capacity_reason,
                            exec_status=exec_status,
                            fill_ratio=float(fill_ratio),
                        )
            else:
                # пока другие типы не поддержаны — отменяем
                _cancel(cli_id)

        # funding начисление
        funding_cashflow = 0.0
        funding_events_list = []
        ref_for_funding = self._last_ref_price
        if self.funding is not None:
            fc, events = self.funding.accrue(
                position_qty=self.position_qty, mark_price=ref_for_funding, now_ts_ms=ts
            )
            funding_cashflow = float(fc)
            funding_events_list = list(events or [])
            self.funding_cum += float(fc)

        # PnL/mark
        mark_p = self._mark_price(
            ref=self._last_ref_price, bid=self._last_bid, ask=self._last_ask
        )
        unrl = self._unrealized_pnl(mark_p)
        eq = float(self.realized_pnl_cum + unrl - self.fees_cum + self.funding_cum)
        if self._trade_log:
            r_chk, u_chk = self._recompute_pnl_from_log(mark_p)
            assert abs((self.realized_pnl_cum + unrl) - (r_chk + u_chk)) < 1e-6

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
            cancelled_reasons=cancelled_reasons,
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
            execution_profile=str(getattr(self, "execution_profile", "")),
        )

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        total = getattr(self._q, "total_cnt", 0)
        if total:
            logger.info(
                "LatencyQueue degradation: drop=%0.2f%% (%d/%d), delay=%0.2f%% (%d/%d)",
                self._q.drop_cnt / total * 100.0,
                self._q.drop_cnt,
                total,
                self._q.delay_cnt / total * 100.0,
                self._q.delay_cnt,
                total,
            )
        else:
            logger.info("LatencyQueue degradation: no events processed")

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
