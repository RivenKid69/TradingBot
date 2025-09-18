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
from typing import List, Optional, Tuple, Any, Dict, Sequence, Mapping
import math
import os
import logging
import threading
import random
from clock import now_ms

try:
    from runtime_flags import seasonality_enabled  # type: ignore
except Exception:  # pragma: no cover - fallback if module not found

    def seasonality_enabled(default: bool = True) -> bool:
        return default


from utils.prometheus import Counter
from config import DataDegradationConfig

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
        self.lob = lob
        self._last_ref_price: Optional[float] = None
        self._next_h1_open_price: Optional[float] = None
        self.run_config = run_config
        self._run_config = run_config
        self.run_id: str = str(getattr(run_config, "run_id", "sim") or "sim")
        self.step_ms: int = (
            int(getattr(run_config, "step_ms", 1000))
            if run_config is not None
            else 1000
        )
        if self.step_ms <= 0:
            self.step_ms = 1
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
        self.funding = (
            FundingCalculator(**(funding_config or {"enabled": False}))
            if FundingCalculator is not None
            else None
        )

        # слиппедж
        self.slippage_cfg = None
        if SlippageConfig is not None and slippage_config is not None:
            if isinstance(slippage_config, str):
                try:
                    self.slippage_cfg = SlippageConfig.from_file(slippage_config)
                except Exception:
                    logger.exception("failed to load slippage config from %s", slippage_config)
            elif isinstance(slippage_config, dict):
                self.slippage_cfg = SlippageConfig.from_dict(slippage_config)

        # исполнители
        self._execution_cfg = dict(execution_config or {})
        self.execution_profile = (
            str(execution_profile) if execution_profile is not None else ""
        )
        self.execution_params: dict = dict(execution_params or {})
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
        vol_raw: Optional[Mapping[str, float]] = None,
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
            sbps = float(spread_bps)
        else:
            if (
                compute_spread_bps_from_quotes is not None
                and self.slippage_cfg is not None
            ):
                sbps = compute_spread_bps_from_quotes(
                    bid=self._last_bid, ask=self._last_ask, cfg=self.slippage_cfg
                )
            else:
                sbps = None
        self._last_spread_bps = sbps * spread_mult if sbps is not None else None
        self._last_vol_factor = float(vol_factor) if vol_factor is not None else None
        metrics = self._normalize_vol_metrics(vol_raw)
        self._last_vol_raw = metrics
        liq_val = float(liquidity) if liquidity is not None else None
        self._last_liquidity = liq_val * liq_mult if liq_val is not None else None
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

    def _normalize_vol_metrics(
        self, metrics: Optional[Mapping[str, Any]]
    ) -> Optional[Dict[str, float]]:
        if not metrics:
            return None
        normalized: Dict[str, float] = {}
        for key, raw_val in metrics.items():
            if key is None:
                continue
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(val):
                continue
            normalized[str(key).lower()] = val
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
            except Exception:
                lat_ms = 0
                timeout = False
                spike = False
                remaining = self.latency_steps
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
                qty_total = self._apply_filters_market(side, qty_raw, ref)
                if qty_total <= 0.0:
                    _cancel(p.client_order_id)
                    continue

                if ref is None or not math.isfinite(ref):
                    _cancel(p.client_order_id)
                    continue

                # риск: пауза/клампинг размера перед планом
                risk_events_local: List[RiskEvent] = []
                if self.risk is not None:
                    portfolio_total = None
                    if ref is not None:
                        try:
                            ref_val = float(ref)
                        except (TypeError, ValueError):
                            ref_val = None
                        else:
                            if math.isfinite(ref_val) and ref_val > 0.0:
                                portfolio_total = abs(float(self.position_qty)) * ref_val
                    adj_qty = self.risk.pre_trade_adjust(
                        ts_ms=ts,
                        side=side,
                        intended_qty=qty_total,
                        price=ref,
                        position_qty=self.position_qty,
                        total_notional=portfolio_total,
                    )
                    risk_events_local.extend(self.risk.pop_events())
                    qty_total = float(adj_qty)
                    if qty_total <= 0.0:
                        _cancel(p.client_order_id)
                        # накопим события риска
                        # (дальше они будут добавлены к отчёту)
                        continue

                # планирование ребёнков (intra-bar)
                notional = abs(qty_total) * float(ref)
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
                    "ref_price": ref,
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
                    ts_fill = int(ts + int(child.ts_offset_ms) + lat_ms)
                    q_child = float(child.qty)
                    if q_child <= 0.0:
                        continue

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
                    if self.quantizer is not None:
                        q_child = self.quantizer.quantize_qty(self.symbol, q_child)
                        q_child = self.quantizer.clamp_notional(
                            self.symbol, ref, q_child
                        )
                        if q_child <= 0.0:
                            continue
                    # базовая котировка
                    if VWAPExecutor is not None and isinstance(executor, VWAPExecutor):
                        self._vwap_on_tick(ts_fill, None, None)
                        base_price = (
                            self._last_hour_vwap
                            if self._last_hour_vwap is not None
                            else ref
                        )
                        filled_price = (
                            float(base_price) if base_price is not None else float(ref)
                        )
                    elif (
                        str(getattr(self, "execution_profile", "")).upper()
                        == "MKT_OPEN_NEXT_H1"
                    ):
                        if self._next_h1_open_price is not None:
                            filled_price = float(self._next_h1_open_price)
                        else:
                            filled_price = float(ref)
                            if ref is not None:
                                self._next_h1_open_price = float(ref)
                    else:
                        if side == "BUY":
                            base_price = (
                                self._last_ask if self._last_ask is not None else ref
                            )
                        else:
                            base_price = (
                                self._last_bid if self._last_bid is not None else ref
                            )
                        filled_price = (
                            float(base_price) if base_price is not None else float(ref)
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
                    cfg_slip = (
                        self.execution_params.get("slippage_bps")
                        if isinstance(self.execution_params, dict)
                        else None
                    )
                    if cfg_slip is not None:
                        slip_bps = float(cfg_slip)
                        if apply_slippage_price is not None:
                            filled_price = apply_slippage_price(
                                side=side,
                                quote_price=filled_price,
                                slippage_bps=slip_bps,
                            )
                    elif (
                        self.slippage_cfg is not None
                        and estimate_slippage_bps is not None
                        and apply_slippage_price is not None
                    ):
                        slip_bps = estimate_slippage_bps(
                            spread_bps=sbps,
                            size=q_child,
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
                        fee = self.fees.compute(
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
                        spread_bps=float(
                            sbps
                            if sbps is not None
                            else (
                                self.slippage_cfg.default_spread_bps
                                if self.slippage_cfg is not None
                                else 0.0
                            )
                        ),
                        latency_ms=int(p.lat_ms),
                        latency_spike=bool(p.spike),
                        tif=tif,
                        ttl_steps=ttl_steps,
                    )
                    trades.append(trade)
                    self._trade_log.append(trade)
                continue
            # Определение направления и базовой цены для прочих типов
            is_buy = bool(getattr(proto, "volume_frac", 0.0) > 0.0)
            side = "BUY" if is_buy else "SELL"
            qty = abs(float(getattr(proto, "volume_frac", 0.0)))
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
                fee = self.fees.compute(
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
                spread_bps=float(
                    sbps
                    if sbps is not None
                    else (
                        self.slippage_cfg.default_spread_bps
                        if self.slippage_cfg is not None
                        else 0.0
                    )
                ),
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

                price_q, qty_q, ok = self._apply_filters_limit(
                    side, float(abs_price), qty_raw, ref
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
                    fee = 0.0
                    if self.fees is not None:
                        fee = self.fees.compute(
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
                        spread_bps=float(
                            sbps
                            if sbps is not None
                            else (
                                self.slippage_cfg.default_spread_bps
                                if self.slippage_cfg is not None
                                else 0.0
                            )
                        ),
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
                    fee = 0.0
                    if self.fees is not None:
                        fee = self.fees.compute(
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
                        spread_bps=float(
                            sbps
                            if sbps is not None
                            else (
                                self.slippage_cfg.default_spread_bps
                                if self.slippage_cfg is not None
                                else 0.0
                            )
                        ),
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
        # --- обновить рыночный снапшот ---
        self._last_bid = float(bid) if bid is not None else None
        self._last_ask = float(ask) if ask is not None else None
        self._last_vol_factor = float(vol_factor) if vol_factor is not None else None
        metrics = self._normalize_vol_metrics(vol_raw)
        self._last_vol_raw = metrics
        self._last_liquidity = float(liquidity) if liquidity is not None else None
        self._last_spread_bps = None
        try:
            if (
                compute_spread_bps_from_quotes is not None
                and self.slippage_cfg is not None
            ):
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
                ref = self._last_ref_price
                if ref is None:
                    _cancel(cli_id)
                    continue

                # применить фильтры рынка (квантизация/minNotional и т.п. внутри вспом. функции)
                qty_total = self._apply_filters_market(side, qty_raw, ref)
                if qty_total <= 0.0:
                    _cancel(cli_id)
                    continue

                # риск: корректировка/пауза
                if self.risk is not None:
                    portfolio_total = None
                    if ref is not None:
                        try:
                            ref_val = float(ref)
                        except (TypeError, ValueError):
                            ref_val = None
                        else:
                            if math.isfinite(ref_val) and ref_val > 0.0:
                                portfolio_total = abs(float(self.position_qty)) * ref_val
                    adj_qty = self.risk.pre_trade_adjust(
                        ts_ms=ts,
                        side=side,
                        intended_qty=qty_total,
                        price=ref,
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
                    "ref_price": ref,
                }
                plan = executor.plan_market(
                    now_ts_ms=ts, side=side, target_qty=qty_total, snapshot=snapshot
                )
                if not plan:
                    _cancel(cli_id)
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
                            self.risk._emit(
                                ts_fill,
                                "THROTTLE",
                                "order throttled by rate limit",
                                ts_ms=int(ts_fill),
                            )
                            continue
                        self.risk.on_new_order(ts_fill)

                    # квантайзер и minNotional
                    if self.quantizer is not None:
                        q_child = self.quantizer.quantize_qty(self.symbol, q_child)
                        q_child = self.quantizer.clamp_notional(
                            self.symbol, ref, q_child
                        )
                        if q_child <= 0.0:
                            continue

                    # латентность
                    lat_ms = 0
                    lat_spike = False
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
                    ts_fill = int(ts_fill + lat_ms)
                    # цена исполнения
                    if VWAPExecutor is not None and isinstance(executor, VWAPExecutor):
                        self._vwap_on_tick(ts_fill, None, None)
                        base_price = (
                            self._last_hour_vwap
                            if self._last_hour_vwap is not None
                            else ref
                        )
                        filled_price = (
                            float(base_price) if base_price is not None else float(ref)
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
                                self._last_ask if self._last_ask is not None else ref
                            )
                        else:
                            base_price = (
                                self._last_bid if self._last_bid is not None else ref
                            )
                        filled_price = (
                            float(base_price) if base_price is not None else float(ref)
                        )
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
                            size=q_child,
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
                        fee = self.fees.compute(
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
                        spread_bps=float(
                            sbps
                            if sbps is not None
                            else (
                                self.slippage_cfg.default_spread_bps
                                if self.slippage_cfg is not None
                                else 0.0
                            )
                        ),
                        latency_ms=int(lat_ms),
                        latency_spike=bool(lat_spike),
                        tif=tif,
                        ttl_steps=ttl_steps,
                    )
                    trades.append(trade)
                    self._trade_log.append(trade)
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
