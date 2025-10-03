# -*- coding: utf-8 -*-
"""
services/service_signal_runner.py
Онлайн-оркестратор: MarketDataSource -> FeaturePipe -> Strategy -> RiskGuards -> TradeExecutor.
Не содержит бизнес-логики, только склейка компонентов.

Пример использования через конфиг:

```python
from core_config import CommonRunConfig
from service_signal_runner import from_config

cfg = CommonRunConfig(...)
for report in from_config(cfg):
    print(report)
```
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import (
    Any,
    Dict,
    Optional,
    Sequence,
    Iterator,
    Protocol,
    Callable,
    Mapping,
    Tuple,
    TYPE_CHECKING,
)
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
import os
import logging
import math
import threading
import signal
from pathlib import Path
from collections import OrderedDict, deque, defaultdict
import time
from datetime import datetime, timezone
import shlex
import subprocess
import weakref

import json
import yaml
import numpy as np
import clock
from services import monitoring, ops_kill_switch
from services.monitoring import (
    skipped_incomplete_bars,
    pipeline_stage_drop_count,
    MonitoringAggregator,
)
from services.alerts import AlertManager
from pipeline import (
    check_ttl,
    closed_bar_guard,
    policy_decide,
    apply_risk,
    Stage,
    Reason,
    PipelineResult,
    PipelineConfig,
    PipelineStageConfig,
)
from services.signal_bus import (
    build_envelope as build_signal_envelope,
    log_drop,
    publish_signal as publish_signal_envelope,
    signal_id as envelope_signal_id,
)
from services.event_bus import EventBus
from services.shutdown import ShutdownManager
from services.signal_csv_writer import SignalCSVWriter
from adapters.binance_spot_private import reconcile_state

# ``SimAdapter`` pulls in heavy execution-simulator dependencies.  Import it lazily to
# allow bar-mode setups to operate without the microstructure stack being available at
# runtime.
if TYPE_CHECKING:  # pragma: no cover - typing helper
    from sandbox.sim_adapter import SimAdapter  # type: ignore
else:  # pragma: no cover - runtime placeholder
    SimAdapter = Any  # type: ignore
from impl_bar_executor import BarExecutor
from core_models import Bar, Tick
from core_contracts import FeaturePipe, SignalPolicy
from services.utils_config import (
    snapshot_config,
)  # снапшот конфига (Фаза 3)  # noqa: F401
from services.utils_app import atomic_write_with_retry
from api.spot_signals import SpotSignalEnvelope
from core_config import (
    CommonRunConfig,
    ClockSyncConfig,
    ThrottleConfig,
    OpsKillSwitchConfig,
    MonitoringConfig,
    StateConfig,
    ExecutionProfile,
    load_timing_profiles,
    resolve_execution_timing,
    SpotTurnoverCaps,
)
from no_trade import (
    _parse_daily_windows_min,
    _in_daily_window,
    _in_funding_buffer,
    _in_custom_window,
)
from no_trade_config import (
    NoTradeConfig,
    DEFAULT_NO_TRADE_STATE_PATH,
    DynamicGuardConfig,
)
from dynamic_no_trade_guard import DynamicNoTradeGuard
from risk_guard import PortfolioLimitGuard, PortfolioLimitConfig
from utils import TokenBucket
import di_registry
import ws_dedup_state as signal_bus
from services import state_storage
from strategies.base import SignalPosition
from utils.time import daily_reset_key


class RiskGuards(Protocol):
    def apply(
        self, ts_ms: int, symbol: str, decisions: Sequence[Any]
    ) -> tuple[Sequence[Any], str | None]: ...


@dataclass
class SignalQualityConfig:
    enabled: bool = False
    sigma_window: int = 0
    sigma_threshold: float = 0.0
    vol_median_window: int = 0
    vol_floor_frac: float = 0.0
    log_reason: bool | str = ""


@dataclass
class _EntryLimiterConfig:
    limit: int | None = None
    reset_hour: int = 0


@dataclass
class CooldownSettings:
    """Configuration for per-symbol cooldown handling."""

    bars: int = 0
    large_trade_threshold: float = 0.0
    small_delta_threshold: float = 0.0

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    @property
    def enabled(self) -> bool:
        return self.bars > 0 and self.large_trade_threshold > 0.0

    @property
    def suppress_small_deltas(self) -> bool:
        return self.enabled and self.small_delta_threshold > 0.0

    @classmethod
    def from_mapping(cls, data: Any) -> "CooldownSettings":
        if isinstance(data, CooldownSettings):
            return data
        if not isinstance(data, MappingABC):
            return cls()

        def _pick_int(*keys: str) -> int:
            for key in keys:
                if key not in data:
                    continue
                parsed = cls._coerce_int(data.get(key))
                if parsed is not None and parsed > 0:
                    return parsed
            return 0

        def _pick_float(*keys: str) -> float:
            for key in keys:
                if key not in data:
                    continue
                parsed = cls._coerce_float(data.get(key))
                if parsed is not None and parsed > 0.0:
                    return parsed
            return 0.0

        bars = _pick_int("cooldown_bars", "bars")
        large_trade = _pick_float(
            "cooldown_large_trade",
            "large_trade_threshold",
            "large_trade",
            "turnover_cap",
        )
        small_delta = _pick_float(
            "cooldown_small_delta",
            "small_delta_threshold",
            "small_delta",
            "min_rebalance_step",
        )
        return cls(bars=bars, large_trade_threshold=large_trade, small_delta_threshold=small_delta)


class DailyEntryLimiter:
    """Limit the number of entry transitions within a trading day."""

    def __init__(self, limit: Any = None, reset_hour: Any = 0) -> None:
        self._limit: int | None = self.normalize_limit(limit)
        self._reset_hour: int = self.normalize_hour(reset_hour)
        self._state: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def normalize_limit(value: Any) -> int | None:
        if value is None:
            return None
        try:
            limit = int(value)
        except (TypeError, ValueError):
            return None
        if limit <= 0:
            return None
        return limit

    @staticmethod
    def normalize_hour(value: Any) -> int:
        if value is None:
            return 0
        try:
            hour = int(value)
        except (TypeError, ValueError):
            return 0
        return hour % 24

    @property
    def enabled(self) -> bool:
        return self._limit is not None

    @property
    def limit(self) -> int | None:
        return self._limit

    @property
    def reset_hour(self) -> int:
        return self._reset_hour

    def allow(self, symbol: str, ts_ms: int, *, entry_steps: Any = 1) -> bool:
        if not self.enabled:
            return True
        try:
            steps = int(entry_steps)
        except (TypeError, ValueError):
            steps = 1
        if steps <= 0:
            steps = 1
        try:
            ts_val = int(ts_ms)
        except (TypeError, ValueError):
            ts_val = 0
        key = self._normalize_symbol(symbol)
        day_key = daily_reset_key(ts_val, self._reset_hour)
        state = self._state.get(key)
        if state is None:
            state = {"count": 0, "day_key": day_key}
            self._state[key] = state
        elif state.get("day_key") != day_key:
            state["day_key"] = day_key
            state["count"] = 0
        if self._limit is None:
            return True
        try:
            current = int(state.get("count", 0))
        except (TypeError, ValueError):
            current = 0
        if current + steps > self._limit:
            return False
        state["count"] = current + steps
        return True

    def snapshot(self, symbol: str | None = None) -> Dict[str, Any]:
        key = self._normalize_symbol(symbol) if symbol is not None else None
        state = self._state.get(key, {}) if key is not None else {}
        try:
            count = int(state.get("count", 0))
        except (TypeError, ValueError):
            count = 0
        day_key = state.get("day_key") if isinstance(state, dict) else None
        return {
            "limit": self._limit,
            "entries_today": count,
            "day_key": day_key,
            "reset_hour": self._reset_hour,
        }

    def export_state(self) -> Dict[str, Dict[str, Any]]:
        if not self.enabled:
            return {}
        exported: Dict[str, Dict[str, Any]] = {}
        for symbol, state in self._state.items():
            if not isinstance(state, dict):
                continue
            try:
                count = int(state.get("count", 0))
            except (TypeError, ValueError):
                count = 0
            day_key = state.get("day_key")
            exported[symbol] = {
                "count": max(0, count),
                "day_key": str(day_key) if day_key is not None else None,
            }
        return exported

    def restore_state(self, state: Mapping[str, Any] | None) -> None:
        if not self.enabled:
            self._state.clear()
            return
        self._state.clear()
        if not state:
            return
        for symbol, payload in state.items():
            if not isinstance(payload, Mapping):
                continue
            try:
                count = int(payload.get("count", 0))
            except (TypeError, ValueError):
                count = 0
            if count < 0:
                count = 0
            day_key = payload.get("day_key")
            if day_key is not None:
                try:
                    day_key = str(day_key)
                except Exception:
                    day_key = None
            key = self._normalize_symbol(symbol)
            self._state[key] = {"count": count, "day_key": day_key}

    @staticmethod
    def _normalize_symbol(symbol: Any) -> str:
        if symbol is None:
            return ""
        try:
            value = str(symbol)
        except Exception:
            return ""
        value = value.strip()
        return value.upper()


class MixedQuoteError(RuntimeError):
    """Raised when configured symbols reference multiple quote assets."""


class _ScheduleNoTradeChecker:
    """Evaluate maintenance-based no-trade windows for individual bars."""

    def __init__(self, cfg: NoTradeConfig | None) -> None:
        self._enabled = False
        self._daily_windows: list[tuple[int, int]] = []
        self._funding_buffer_min = 0
        self._custom_windows: list[dict[str, int]] = []

        if cfg is None:
            return

        maintenance_cfg = getattr(cfg, "maintenance", None)
        if maintenance_cfg is not None:
            daily = list(getattr(maintenance_cfg, "daily_utc", []) or [])
            buffer_min = getattr(maintenance_cfg, "funding_buffer_min", None)
            custom = list(getattr(maintenance_cfg, "custom_ms", []) or [])
        else:
            daily = list(getattr(cfg, "daily_utc", []) or [])
            buffer_min = getattr(cfg, "funding_buffer_min", 0)
            custom = list(getattr(cfg, "custom_ms", []) or [])

        if not daily:
            daily = list(getattr(cfg, "daily_utc", []) or [])
        if buffer_min is None:
            buffer_min = getattr(cfg, "funding_buffer_min", 0)
        if not custom:
            custom = list(getattr(cfg, "custom_ms", []) or [])

        try:
            self._daily_windows = _parse_daily_windows_min(daily)
        except Exception:
            self._daily_windows = []
        try:
            self._funding_buffer_min = int(buffer_min or 0)
        except Exception:
            self._funding_buffer_min = 0
        try:
            self._custom_windows = list(custom or [])
        except Exception:
            self._custom_windows = []

        self._enabled = bool(
            self._daily_windows
            or self._funding_buffer_min > 0
            or self._custom_windows
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def evaluate(self, ts_ms: int, symbol: str | None = None) -> tuple[bool, str | None]:
        """Return whether ``ts_ms`` falls inside a maintenance window."""

        if not self._enabled:
            return False, None

        ts_arr = np.asarray([int(ts_ms)], dtype=np.int64)

        daily_hit = False
        if self._daily_windows:
            try:
                daily_hit = bool(_in_daily_window(ts_arr, self._daily_windows)[0])
            except Exception:
                daily_hit = False

        funding_hit = False
        if self._funding_buffer_min > 0:
            try:
                funding_hit = bool(
                    _in_funding_buffer(ts_arr, self._funding_buffer_min)[0]
                )
            except Exception:
                funding_hit = False

        custom_hit = False
        if self._custom_windows:
            try:
                custom_hit = bool(_in_custom_window(ts_arr, self._custom_windows)[0])
            except Exception:
                custom_hit = False

        if not (daily_hit or funding_hit or custom_hit):
            return False, None

        if custom_hit:
            return True, "MAINTENANCE_CUSTOM"
        if daily_hit:
            return True, "MAINTENANCE_DAILY"
        if funding_hit:
            return True, "MAINTENANCE_FUNDING"
        return True, "WINDOW"


def _format_signal_quality_log(payload: Mapping[str, Any]) -> str:
    """Render a structured log line for signal quality drops."""

    ordered_keys = (
        "stage",
        "reason",
        "symbol",
        "bar_close_at",
        "bar_close_ms",
        "sigma_threshold",
        "vol_floor_frac",
        "bar_volume",
        "detail",
        "current_sigma",
        "vol_median",
        "window_ready",
    )

    parts: list[str] = ["DROP"]
    for key in ordered_keys:
        if key not in payload:
            continue
        value = payload[key]
        if value is None:
            continue
        if isinstance(value, float):
            value_str = f"{value:.6g}"
        else:
            value_str = str(value)
        parts.append(f"{key}={value_str}")
    return " ".join(parts)


@dataclass
class SignalRunnerConfig:
    snapshot_config_path: Optional[str] = None
    artifacts_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    marker_path: Optional[str] = None
    run_id: Optional[str] = None
    snapshot_metrics_json: Optional[str] = None
    snapshot_metrics_csv: Optional[str] = None
    snapshot_metrics_sec: int = 60
    kill_switch_ops: OpsKillSwitchConfig = field(default_factory=OpsKillSwitchConfig)
    state: StateConfig = field(default_factory=StateConfig)
    portfolio_limits: Dict[str, Any] = field(default_factory=dict)


# обратная совместимость
RunnerConfig = SignalRunnerConfig


def _parse_pipeline_config(data: Mapping[str, Any]) -> PipelineConfig:
    """Parse raw mapping into :class:`PipelineConfig`."""

    enabled = bool((data or {}).get("enabled", True))
    stages_cfg = (data or {}).get("stages", {}) or {}
    stages: Dict[str, PipelineStageConfig] = {}
    for name, st in stages_cfg.items():
        if isinstance(st, MappingABC):
            st_enabled = bool(st.get("enabled", True))
            params = {k: v for k, v in st.items() if k != "enabled"}
        else:
            st_enabled = bool(st)
            params = {}
        stages[name] = PipelineStageConfig(enabled=st_enabled, params=params)
    return PipelineConfig(enabled=enabled, stages=stages)


def _extract_bar_initial_weights_from_state(state: Any) -> Dict[str, float] | None:
    try:
        exposure_state_raw = getattr(state, "exposure_state", {}) or {}
    except Exception:
        return None
    if not isinstance(exposure_state_raw, MappingABC):
        return None
    raw_weights = exposure_state_raw.get("weights", {}) or {}
    if not isinstance(raw_weights, MappingABC):
        return None
    sanitized: Dict[str, float] = {}
    for symbol, weight in raw_weights.items():
        sym = str(symbol).upper()
        if not sym:
            continue
        try:
            weight_val = float(weight)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(weight_val):
            continue
        if math.isclose(weight_val, 0.0, rel_tol=0.0, abs_tol=1e-12):
            continue
        sanitized[sym] = weight_val
    return sanitized or None


class _Worker:
    _TURNOVER_KEYS = ("turnover_usd", "turnover", "notional_usd")
    _ADV_KEYS = ("adv_quote", "adv_usd")

    def __init__(
        self,
        fp: FeaturePipe,
        policy: SignalPolicy,
        logger: logging.Logger,
        executor: Any,
        guards: Optional[RiskGuards] = None,
        safe_mode_fn: Callable[[], bool] | None = None,
        *,
        enforce_closed_bars: bool,
        close_lag_ms: int = 0,
        ws_dedup_enabled: bool = False,
        ws_dedup_log_skips: bool = False,
        ws_dedup_timeframe_ms: int = 0,
        bar_timeframe_ms: int | None = None,
        throttle_cfg: ThrottleConfig | None = None,
        no_trade_cfg: NoTradeConfig | None = None,
        pipeline_cfg: PipelineConfig | None = None,
        signal_quality_cfg: SignalQualityConfig | None = None,
        zero_signal_alert: int = 0,
        state_enabled: bool = False,
        rest_candidates: Sequence[Any] | None = None,
        monitoring: MonitoringAggregator | None = None,
        monitoring_agg: MonitoringAggregator | None = None,
        worker_id: str | None = None,
        status_callback: Callable[[str, Dict[str, Any]], None] | None = None,
        execution_mode: str = "bar",
        portfolio_equity: float | None = None,
        max_total_weight: float | None = None,
        idempotency_cache_size: int = 1024,
        cooldown_settings: CooldownSettings | Mapping[str, Any] | None = None,
        signal_dispatcher: Callable[[Any], None] | None = None,
    ) -> None:
        self._fp = fp
        self._policy = policy
        self._logger = logger
        self._executor = executor
        bar_price_field = getattr(self._executor, "bar_price_field", "close")
        try:
            bar_price_field_str = str(bar_price_field)
        except Exception:
            bar_price_field_str = "close"
        bar_price_field_str = bar_price_field_str.strip() or "close"
        self._bar_price_field = bar_price_field_str
        self._guards = guards
        self._safe_mode_fn = safe_mode_fn or (lambda: False)
        self._enforce_closed_bars = enforce_closed_bars
        try:
            self._close_lag_ms = max(0, int(close_lag_ms))
        except (TypeError, ValueError):
            self._close_lag_ms = 0
        self._ws_dedup_enabled = ws_dedup_enabled
        self._ws_dedup_log_skips = ws_dedup_log_skips
        self._ws_dedup_timeframe_ms = self._coerce_timeframe_ms(ws_dedup_timeframe_ms)
        bar_timeframe_candidate: Any = bar_timeframe_ms
        if bar_timeframe_candidate is None:
            bar_timeframe_candidate = ws_dedup_timeframe_ms
        self._bar_timeframe_ms = self._coerce_timeframe_ms(bar_timeframe_candidate)
        if self._bar_timeframe_ms <= 0:
            fallback_candidates = (
                getattr(fp, "timeframe_ms", None),
                getattr(getattr(fp, "execution", None), "timeframe_ms", None),
                getattr(policy, "timeframe_ms", None),
            )
            for candidate in fallback_candidates:
                fallback_ms = self._coerce_timeframe_ms(candidate)
                if fallback_ms > 0:
                    self._bar_timeframe_ms = fallback_ms
                    break
        if self._bar_timeframe_ms <= 0:
            self._bar_timeframe_ms = self._ws_dedup_timeframe_ms
        self._last_invalid_ttl_timeframe: Any | None = None
        self._throttle_cfg: ThrottleConfig | None = None
        self._throttle_lock = threading.RLock()
        self._no_trade_cfg = no_trade_cfg
        self._pipeline_cfg = pipeline_cfg
        self._signal_quality_cfg = signal_quality_cfg or SignalQualityConfig()
        self._zero_signal_alert = int(zero_signal_alert)
        self._zero_signal_streak = 0
        self._state_enabled = bool(state_enabled)
        self._worker_id = worker_id or f"worker-{id(self):x}"
        self._status_callback = status_callback
        self._last_status_payload: Dict[str, Any] = {"id": self._worker_id}
        self._positions: Dict[str, float] = {}
        self._pending_exposure: Dict[int, Dict[str, Any]] = {}
        self._exposure_state: Dict[str, Any] = {
            "positions": {},
            "pending": {},
            "updated_at_ms": 0,
            "total_notional": 0.0,
            "daily_turnover": {
                "day_key": None,
                "per_symbol": {},
                "portfolio": {"day_key": None, "used_usd": 0.0},
            },
        }
        self._monitoring: MonitoringAggregator | None = (
            monitoring if monitoring is not None else monitoring_agg
        )
        execution_mode_normalized = str(execution_mode or "bar").lower()
        if execution_mode_normalized not in {"order", "bar"}:
            execution_mode_normalized = "bar"
        self._execution_mode = execution_mode_normalized
        try:
            cache_size = int(idempotency_cache_size)
        except (TypeError, ValueError):
            cache_size = 0
        if cache_size <= 0:
            cache_size = 1024
        self._idempotency_cache_size = cache_size
        self._idempotency_cache: OrderedDict[str, None] = OrderedDict()
        if self._monitoring is not None:
            try:
                self._monitoring.set_execution_mode(self._execution_mode)
            except Exception:
                pass
        self._portfolio_guard: PortfolioLimitGuard | None = None
        self._global_bucket = None
        self._symbol_bucket_factory = None
        self._symbol_buckets = None
        self._queue = None
        self._signal_dispatcher: Callable[[Any], None] | None = signal_dispatcher
        try:
            self._portfolio_equity = (
                float(portfolio_equity) if portfolio_equity is not None else None
            )
        except (TypeError, ValueError):
            self._portfolio_equity = None
        try:
            weight_cap = (
                float(max_total_weight) if max_total_weight is not None else None
            )
        except (TypeError, ValueError):
            weight_cap = None
        if weight_cap is not None:
            if not math.isfinite(weight_cap) or weight_cap <= 0.0:
                weight_cap = None
        self._max_total_weight: float | None = weight_cap
        self._weights: Dict[str, float] = {}
        self._pending_weight: Dict[int, Dict[str, Any]] = {}
        self._pending_weight_refs: Dict[int, Any] = {}
        self._symbol_equity: Dict[str, float] = {}
        entry_cfg = self._resolve_entry_limiter_config(executor)
        self._entry_limiter = DailyEntryLimiter(
            entry_cfg.limit, entry_cfg.reset_hour
        )
        self._entry_lock = threading.Lock()
        if self._entry_limiter.enabled:
            try:
                self._logger.info(
                    "daily entry limiter enabled: limit=%s reset_hour=%s",
                    self._entry_limiter.limit,
                    self._entry_limiter.reset_hour,
                )
            except Exception:
                pass
        try:
            self._daily_reset_hour = int(entry_cfg.reset_hour)
        except Exception:
            self._daily_reset_hour = 0
        self._daily_symbol_turnover: Dict[str, Dict[str, Any]] = {}
        self._daily_portfolio_turnover: Dict[str, Any] = {"day": None, "total": 0.0}
        self._daily_turnover_caps = self._resolve_daily_turnover_caps(executor)
        self._daily_day_key: str | None = None
        self._cooldown_settings = CooldownSettings.from_mapping(cooldown_settings)
        self._symbol_cooldowns: Dict[str, int] = {}
        self._symbol_cooldown_set_ts: Dict[str, int] = {}
        self._configure_throttle(throttle_cfg)
        self._last_bar_ts: Dict[str, int] = {}
        self._last_bar_snapshot: Dict[str, Any] = {}
        self._dynamic_guard: DynamicNoTradeGuard | None = None
        dyn_cfg: DynamicGuardConfig | None = None
        if self._no_trade_cfg is not None:
            structured = getattr(self._no_trade_cfg, "dynamic", None)
            if structured is not None:
                dyn_enabled = getattr(structured, "enabled", None)
                guard_cfg = getattr(structured, "guard", None)
                if (
                    guard_cfg is not None
                    and getattr(guard_cfg, "enable", False)
                    and (dyn_enabled is None or bool(dyn_enabled))
                ):
                    dyn_cfg = guard_cfg
            if dyn_cfg is None:
                legacy = getattr(self._no_trade_cfg, "dynamic_guard", None)
                if legacy is not None and getattr(legacy, "enable", False):
                    dyn_cfg = legacy
        if dyn_cfg is not None:
            self._dynamic_guard = DynamicNoTradeGuard(dyn_cfg)
        self._schedule_checker = _ScheduleNoTradeChecker(self._no_trade_cfg)
        try:
            ttl_attr = getattr(fp, "spread_ttl_ms", None)
            if ttl_attr is None:
                ttl_attr = getattr(fp, "_spread_ttl_ms", None)
            self._spread_ttl_ms = max(0, int(ttl_attr or 0))
        except Exception:
            self._spread_ttl_ms = 0
        cache_ttl = max(self._spread_ttl_ms, int(ws_dedup_timeframe_ms))
        if cache_ttl <= 0:
            cache_ttl = 60_000
        self._spread_cache_max_ms = cache_ttl
        self._spread_injections: Dict[str, Dict[str, float | int | None]] = {}
        self._last_prices: Dict[str, float] = {}
        self._rest_helper = self._resolve_rest_helper(
            rest_candidates or ()
        )
        self._rest_backoff_until: Dict[str, float] = {}
        self._rest_backoff_step: Dict[str, float] = {}
        self._dispatch_stack: list[Any] = []
        self._order_meta_sidecar: "weakref.WeakKeyDictionary[Any, Dict[str, Any]]" = (
            weakref.WeakKeyDictionary()
        )
        self._order_meta_fallback: dict[int, tuple[weakref.ReferenceType[Any] | None, Dict[str, Any]]] = {}
        if self._state_enabled:
            self._seed_last_prices_from_state()
            self._load_exposure_state()
        self._update_queue_metrics(update_status=False)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _configure_throttle(self, throttle_cfg: ThrottleConfig | None) -> None:
        with self._throttle_lock:
            self._throttle_cfg = throttle_cfg
            if throttle_cfg is not None and throttle_cfg.enabled:
                self._global_bucket = TokenBucket(
                    rps=throttle_cfg.global_.rps, burst=throttle_cfg.global_.burst
                )
                self._symbol_bucket_factory = lambda: TokenBucket(
                    rps=throttle_cfg.symbol.rps, burst=throttle_cfg.symbol.burst
                )
                self._symbol_buckets = defaultdict(self._symbol_bucket_factory)
                max_items = max(0, int(throttle_cfg.queue.max_items))
                self._queue = deque(maxlen=max_items if max_items > 0 else None)
            else:
                self._global_bucket = None
                self._symbol_bucket_factory = None
                self._symbol_buckets = None
                self._queue = None
        self._update_queue_metrics(update_status=False)

    def update_throttle_config(self, throttle_cfg: ThrottleConfig | None) -> None:
        self._configure_throttle(throttle_cfg)

    def update_pipeline_config(self, pipeline_cfg: PipelineConfig | None) -> None:
        self._pipeline_cfg = pipeline_cfg

    def update_ws_settings(
        self,
        *,
        enabled: bool | None = None,
        log_skips: bool | None = None,
        timeframe_ms: int | None = None,
    ) -> None:
        if enabled is not None:
            self._ws_dedup_enabled = bool(enabled)
        if log_skips is not None:
            self._ws_dedup_log_skips = bool(log_skips)
        if timeframe_ms is not None:
            dedup_timeframe = self._coerce_timeframe_ms(timeframe_ms)
            self._ws_dedup_timeframe_ms = dedup_timeframe
            if dedup_timeframe > 0:
                self._bar_timeframe_ms = dedup_timeframe
            self._last_invalid_ttl_timeframe = None

    def _queue_snapshot(self) -> tuple[int, int]:
        with self._throttle_lock:
            if self._queue is None:
                return 0, 0
            maxlen = self._queue.maxlen or 0
            return len(self._queue), int(maxlen)

    def _cooldown_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {"global": False, "symbols": [], "count": 0}
        with self._throttle_lock:
            now = time.monotonic()
            global_active = False
            if self._global_bucket is not None:
                try:
                    self._global_bucket.consume(0.0, now=now)
                except Exception:
                    pass
                global_active = self._global_bucket.tokens < 1.0
            symbols_active: list[str] = []
            if self._symbol_buckets is not None:
                for sym, bucket in self._symbol_buckets.items():
                    try:
                        bucket.consume(0.0, now=now)
                    except Exception:
                        continue
                    if bucket.tokens < 1.0:
                        try:
                            sym_key = str(sym)
                        except Exception:
                            continue
                        if sym_key:
                            symbols_active.append(sym_key)
        symbols_unique = sorted({sym for sym in symbols_active if sym})
        snapshot["global"] = bool(global_active)
        snapshot["symbols"] = symbols_unique
        snapshot["count"] = int(len(symbols_unique) + (1 if global_active else 0))
        return snapshot

    @staticmethod
    def _coerce_timeframe_ms(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        if parsed <= 0:
            return 0
        return parsed

    def _symbol_cooldown_snapshot(self) -> Dict[str, Any]:
        settings = self._cooldown_settings
        if not settings.enabled:
            return {}
        active: Dict[str, int] = {}
        for sym, remaining in self._symbol_cooldowns.items():
            try:
                remaining_int = int(remaining)
            except (TypeError, ValueError):
                continue
            if remaining_int <= 0:
                continue
            sym_key = str(sym).upper()
            if not sym_key:
                continue
            active[sym_key] = remaining_int
        if not active:
            return {}
        payload: Dict[str, Any] = {
            "count": len(active),
            "bars": int(settings.bars),
            "remaining": dict(sorted(active.items())),
            "large_trade_threshold": float(settings.large_trade_threshold),
        }
        if settings.small_delta_threshold > 0.0:
            payload["small_delta_threshold"] = float(settings.small_delta_threshold)
        return payload

    def _update_queue_metrics(
        self,
        size: int | None = None,
        max_size: int | None = None,
        *,
        payload_override: Dict[str, Any] | None = None,
        update_status: bool = True,
    ) -> None:
        if size is None or max_size is None:
            queue_size, queue_max = self._queue_snapshot()
        else:
            try:
                queue_size = int(size)
            except Exception:
                queue_size = 0
            try:
                queue_max = int(max_size)
            except Exception:
                queue_max = 0
        queue_size = max(0, queue_size)
        queue_max = max(0, queue_max)
        try:
            monitoring.queue_len.set(queue_size)
        except Exception:
            pass
        runtime_monitoring = self._monitoring
        if runtime_monitoring is not None:
            try:
                runtime_monitoring.update_queue_depth(queue_size, queue_max)
            except Exception:
                pass
            try:
                runtime_monitoring.update_cooldowns(self._cooldown_snapshot())
            except Exception:
                pass
        if payload_override is not None:
            payload = dict(payload_override)
        elif self._last_status_payload:
            payload = dict(self._last_status_payload)
        else:
            payload = {"id": self._worker_id}
        daily_snapshot = self._daily_turnover_snapshot()
        if daily_snapshot:
            payload["daily_turnover"] = daily_snapshot
        cooldown_payload = self._symbol_cooldown_snapshot()
        if cooldown_payload:
            payload["symbol_cooldowns"] = cooldown_payload
        else:
            payload.pop("symbol_cooldowns", None)
        payload["queue"] = {"size": queue_size, "max": queue_max}
        payload["updated_at_ms"] = clock.now_ms()
        self._last_status_payload = payload
        if update_status and self._status_callback is not None:
            try:
                self._status_callback(self._worker_id, payload)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Cooldown helpers
    # ------------------------------------------------------------------
    def _cooldown_remaining(self, symbol: str) -> int:
        try:
            remaining = int(self._symbol_cooldowns.get(symbol, 0))
        except (TypeError, ValueError):
            return 0
        return max(0, remaining)

    def _maybe_start_symbol_cooldown(
        self, symbol: str, executed_delta: float, *, bar_close_ms: int | None = None
    ) -> None:
        settings = self._cooldown_settings
        if not settings.enabled:
            return
        try:
            magnitude = abs(float(executed_delta))
        except (TypeError, ValueError):
            return
        if not math.isfinite(magnitude) or magnitude < settings.large_trade_threshold:
            return
        sym = str(symbol).upper()
        if not sym:
            return
        bars = int(settings.bars)
        if bars <= 0:
            return
        remaining = self._cooldown_remaining(sym)
        new_value = max(remaining, bars)
        self._symbol_cooldowns[sym] = new_value
        if bar_close_ms is not None:
            try:
                self._symbol_cooldown_set_ts[sym] = int(bar_close_ms)
            except (TypeError, ValueError):
                self._symbol_cooldown_set_ts[sym] = int(clock.now_ms())
        else:
            self._symbol_cooldown_set_ts[sym] = int(clock.now_ms())
        self._update_queue_metrics()
        try:
            self._logger.info(
                "COOLDOWN_TRIGGER %s",
                {
                    "symbol": sym,
                    "delta": float(magnitude),
                    "threshold": float(settings.large_trade_threshold),
                    "bars": bars,
                    "remaining": new_value,
                },
            )
        except Exception:
            pass

    def _advance_symbol_cooldown(
        self, symbol: str, bar_ts: int, *, new_bar: bool
    ) -> None:
        if not new_bar or not self._cooldown_settings.enabled:
            return
        sym = str(symbol).upper()
        if not sym:
            return
        remaining = self._cooldown_remaining(sym)
        if remaining <= 0:
            self._symbol_cooldowns.pop(sym, None)
            self._symbol_cooldown_set_ts.pop(sym, None)
            return
        last_set = self._symbol_cooldown_set_ts.get(sym)
        try:
            last_set_int = int(last_set) if last_set is not None else None
        except (TypeError, ValueError):
            last_set_int = None
        try:
            bar_ts_int = int(bar_ts)
        except (TypeError, ValueError):
            bar_ts_int = 0
        if last_set_int is not None and bar_ts_int == last_set_int:
            return
        new_value = remaining - 1
        if new_value <= 0:
            self._symbol_cooldowns.pop(sym, None)
            self._symbol_cooldown_set_ts.pop(sym, None)
            try:
                self._logger.info(
                    "COOLDOWN_RELEASE %s",
                    {"symbol": sym, "bar_ts": bar_ts_int},
                )
            except Exception:
                pass
        else:
            self._symbol_cooldowns[sym] = new_value
        self._update_queue_metrics()

    def _current_order_delta(self, symbol: str, order: Any) -> float | None:
        if self._execution_mode != "bar":
            return None
        payload = self._extract_signal_payload(order)
        if not payload:
            return None
        _, delta, _ = self._resolve_weight_targets(symbol, payload)
        return delta

    def _maybe_drop_due_to_cooldown(
        self, symbol: str, order: Any, bar_ts_ms: int
    ) -> bool:
        settings = self._cooldown_settings
        if not settings.suppress_small_deltas:
            return False
        sym = str(symbol).upper()
        if not sym:
            return False
        remaining = self._cooldown_remaining(sym)
        if remaining <= 0:
            return False
        delta = self._current_order_delta(sym, order)
        if delta is None:
            return False
        try:
            magnitude = abs(float(delta))
        except (TypeError, ValueError):
            return False
        threshold = settings.small_delta_threshold
        if not math.isfinite(magnitude) or magnitude >= threshold:
            return False
        try:
            self._logger.info(
                "COOLDOWN_SUPPRESS %s",
                {
                    "symbol": sym,
                    "delta": float(magnitude),
                    "threshold": float(threshold),
                    "remaining": remaining,
                    "bar_ts": int(bar_ts_ms),
                },
            )
        except Exception:
            pass
        return True

    def _publish_status(
        self,
        *,
        symbol: str,
        last_bar_ts: int | None,
        emitted: int,
        duplicates: int,
    ) -> None:
        if self._status_callback is None:
            return
        queue_size, queue_max = self._queue_snapshot()
        payload = {
            "id": self._worker_id,
            "last_bar_symbol": symbol,
            "last_bar_ts": last_bar_ts,
            "emitted": emitted,
            "duplicates": duplicates,
            "safe_mode": bool(self._safe_mode_fn()),
        }
        self._update_queue_metrics(
            queue_size,
            queue_max,
            payload_override=payload,
            update_status=False,
        )
        try:
            self._status_callback(self._worker_id, self._last_status_payload)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Monitoring helpers
    # ------------------------------------------------------------------
    def _extract_monitoring_snapshot(self, source: Any) -> Mapping[str, Any] | None:
        if source is None:
            return None
        queue: deque[Any] = deque([source])
        visited: set[int] = set()
        callable_attrs = (
            "get_monitoring_snapshot",
            "monitoring_snapshot",
            "get_monitoring_metrics",
            "monitoring_metrics",
            "get_monitoring_stats",
            "monitoring_stats",
            "get_stats",
            "stats",
            "snapshot",
            "get_snapshot",
        )
        value_attrs = (
            "monitoring",
            "metrics",
            "state",
            "stats",
            "latest_stats",
        )
        while queue:
            obj = queue.popleft()
            if obj is None:
                continue
            if isinstance(obj, MappingABC):
                return obj
            ident = id(obj)
            if ident in visited:
                continue
            visited.add(ident)
            if isinstance(obj, SequenceABC) and not isinstance(
                obj, (str, bytes, bytearray)
            ):
                for item in obj:
                    if item is not None:
                        queue.append(item)
                continue
            mapping_view = None
            if hasattr(obj, "__dict__"):
                try:
                    mapping_view = vars(obj)
                except Exception:
                    mapping_view = None
                if isinstance(mapping_view, MappingABC):
                    queue.append(mapping_view)
            for name in callable_attrs:
                candidate = getattr(obj, name, None)
                if callable(candidate):
                    try:
                        result = candidate()
                    except Exception:
                        continue
                    if result is not None:
                        queue.append(result)
                elif candidate is not None:
                    queue.append(candidate)
            for name in value_attrs:
                if hasattr(obj, name):
                    queue.append(getattr(obj, name))
        return None

    @staticmethod
    def _find_in_mapping(data: Mapping[str, Any], keys: Sequence[str]) -> Any:
        queue: deque[Any] = deque([data])
        visited: set[int] = set()
        while queue:
            item = queue.popleft()
            if isinstance(item, SequenceABC) and not isinstance(
                item, (str, bytes, bytearray)
            ):
                for elem in item:
                    if elem is not None:
                        queue.append(elem)
                continue
            if isinstance(item, MappingABC):
                mapping_item: Mapping[str, Any] = item
            elif hasattr(item, "__dict__"):
                try:
                    mapping_item = vars(item)
                except Exception:
                    mapping_item = None
                if not isinstance(mapping_item, MappingABC):
                    continue
            else:
                continue
            item = mapping_item
            ident = id(item)
            if ident in visited:
                continue
            visited.add(ident)
            for key in keys:
                if key in item:
                    value = item[key]
                    if value is not None:
                        return value
            for value in item.values():
                if isinstance(value, MappingABC):
                    queue.append(value)
                elif isinstance(value, SequenceABC) and not isinstance(
                    value, (str, bytes, bytearray)
                ):
                    for elem in value:
                        if isinstance(elem, MappingABC):
                            queue.append(elem)
        return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_adv_quote(
        self,
        order: Any,
        payload: Mapping[str, Any] | None,
        meta: Mapping[str, Any] | None,
    ) -> float | None:
        candidates: list[Any] = []

        def _collect(value: Any) -> None:
            if value is not None:
                candidates.append(value)

        adv_keys = {str(key).lower() for key in self._ADV_KEYS}

        def _collect_from_mapping(candidate: Any) -> None:
            if candidate is None:
                return
            queue: deque[Any] = deque([candidate])
            visited: set[int] = set()
            while queue:
                item = queue.popleft()
                if isinstance(item, MappingABC):
                    mapping_item: Mapping[str, Any] = item
                elif hasattr(item, "__dict__"):
                    try:
                        mapping_view = vars(item)
                    except Exception:
                        mapping_view = None
                    if not isinstance(mapping_view, MappingABC):
                        continue
                    mapping_item = mapping_view
                elif isinstance(item, SequenceABC) and not isinstance(item, (str, bytes, bytearray)):
                    for elem in item:
                        if elem is not None:
                            queue.append(elem)
                    continue
                else:
                    continue
                ident = id(mapping_item)
                if ident in visited:
                    continue
                visited.add(ident)
                for key, value in mapping_item.items():
                    key_normalized = str(key).lower()
                    if key_normalized in adv_keys and value is not None:
                        _collect(value)
                    if value is None:
                        continue
                    if isinstance(value, MappingABC) or (
                        isinstance(value, SequenceABC)
                        and not isinstance(value, (str, bytes, bytearray))
                    ) or hasattr(value, "__dict__"):
                        queue.append(value)

        if isinstance(payload, MappingABC):
            _collect_from_mapping(payload)

        if isinstance(meta, MappingABC):
            _collect_from_mapping(meta)

        if order is not None:
            for key in self._ADV_KEYS:
                _collect(getattr(order, key, None))
            raw_meta = getattr(order, "meta", None)
            if raw_meta is not None and raw_meta is not meta:
                raw_meta_mapping: Mapping[str, Any] | None
                if isinstance(raw_meta, MappingABC):
                    raw_meta_mapping = raw_meta
                else:
                    raw_meta_mapping = self._materialize_mapping(raw_meta)
                _collect_from_mapping(raw_meta_mapping)

        for value in candidates:
            candidate = self._coerce_float(value)
            if candidate is None or not math.isfinite(candidate) or candidate <= 0.0:
                continue
            return float(candidate)
        return None

    def _extract_fill_metrics(
        self, snapshot: Mapping[str, Any] | None
    ) -> tuple[float | None, float | None]:
        if not snapshot:
            return None, None
        requested_keys = (
            "requested",
            "requested_qty",
            "requested_quantity",
            "requested_volume",
            "requested_notional",
            "requested_base",
            "qty_requested",
            "fill_requested",
        )
        filled_keys = (
            "filled",
            "filled_qty",
            "filled_quantity",
            "filled_volume",
            "filled_notional",
            "filled_base",
            "qty_filled",
            "fill_filled",
        )
        requested = self._coerce_float(
            self._find_in_mapping(snapshot, requested_keys)
        )
        filled = self._coerce_float(self._find_in_mapping(snapshot, filled_keys))
        if requested is None or filled is None:
            ratio = self._coerce_float(
                self._find_in_mapping(
                    snapshot,
                    (
                        "fill_ratio",
                        "fillRatio",
                        "fill_rate",
                        "fillRate",
                    ),
                )
            )
            if ratio is not None:
                requested = 1.0
                filled = ratio
        return requested, filled

    def _extract_bar_execution_metrics(
        self,
        snapshot: Mapping[str, Any] | None,
        *,
        expected_bar_ts: int | None = None,
        total_signals: int | None = None,
    ) -> Dict[str, Any] | None:
        if not snapshot or self._execution_mode != "bar":
            return None
        expected_count: int | None = None
        if total_signals is not None:
            try:
                expected_count = int(total_signals)
            except (TypeError, ValueError):
                expected_count = None
            else:
                if expected_count <= 0:
                    return None

        def _coerce_int(value: Any) -> int | None:
            if value is None:
                return None
            if isinstance(value, bool):
                return 1 if value else 0
            try:
                return int(value)
            except (TypeError, ValueError):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return None

        decision_raw = snapshot.get("decision")
        decision = self._materialize_mapping(decision_raw)
        has_decision_data = False
        if decision:
            has_decision_data = True
        elif decision_raw is not None:
            try:
                has_decision_data = bool(decision_raw)
            except Exception:
                has_decision_data = True
        if not has_decision_data and snapshot is not None:
            nested_decision = self._find_in_mapping(snapshot, ("decision",))
            if isinstance(nested_decision, MappingABC):
                decision = self._materialize_mapping(nested_decision)
                has_decision_data = bool(decision)
            elif nested_decision is not None:
                try:
                    has_decision_data = bool(nested_decision)
                except Exception:
                    has_decision_data = True
        if not has_decision_data and snapshot is not None:
            alt_decision = snapshot.get("bar_execution")
            if isinstance(alt_decision, MappingABC):
                decision = self._materialize_mapping(alt_decision)
                has_decision_data = bool(decision)
        turnover = self._coerce_float(
            self._find_in_mapping(decision, ("turnover_usd", "turnover", "notional_usd"))
        )
        if turnover is None:
            turnover = self._coerce_float(snapshot.get("turnover_usd"))
        if turnover is None and not has_decision_data:
            return None
        if turnover is None:
            turnover = 0.0
        act_now_val = self._find_in_mapping(decision, ("act_now", "execute_now"))
        if act_now_val is None:
            act_now_val = snapshot.get("act_now")
        if isinstance(act_now_val, str):
            act_now_flag = act_now_val.strip().lower() in {"1", "true", "yes", "y"}
        elif act_now_val is None:
            act_now_flag = False
        else:
            try:
                act_now_flag = bool(act_now_val)
            except Exception:
                act_now_flag = False
        normalized_val = self._find_in_mapping(decision, ("normalized",))
        if normalized_val is None:
            normalized_val = snapshot.get("normalized")
        if isinstance(normalized_val, str):
            normalized_flag = normalized_val.strip().lower() in {"1", "true", "yes", "y"}
        elif normalized_val is None:
            normalized_flag = False
        else:
            try:
                normalized_flag = bool(normalized_val)
            except Exception:
                normalized_flag = False
        cap_value = self._coerce_float(snapshot.get("cap_usd"))
        if cap_value is None:
            cap_value = self._coerce_float(
                self._find_in_mapping(decision, ("cap_usd", "cap_quote", "daily_notional_cap"))
            )
        if cap_value is None:
            cap_value = self._coerce_float(
                self._find_in_mapping(snapshot, ("cap_usd", "cap_quote", "daily_notional_cap"))
            )
        if cap_value is not None and cap_value <= 0.0:
            cap_value = None
        adv_value = self._coerce_float(snapshot.get("adv_quote"))
        if adv_value is None:
            adv_value = self._coerce_float(
                self._find_in_mapping(decision, ("adv_quote",))
            )
        if adv_value is None:
            adv_value = self._coerce_float(self._find_in_mapping(snapshot, ("adv_quote",)))
        if adv_value is not None and adv_value <= 0.0:
            adv_value = None
        impact_mode_val = self._find_in_mapping(decision, ("impact_mode",))
        if impact_mode_val is None and snapshot is not None:
            impact_mode_val = snapshot.get("impact_mode")
        impact_mode: str | None
        if impact_mode_val is None:
            impact_mode = None
        else:
            try:
                impact_mode = str(impact_mode_val)
            except Exception:
                impact_mode = None
        normalization_details = self._materialize_mapping(
            self._find_in_mapping(decision, ("normalization",))
        )
        if not normalization_details:
            normalization_details = self._materialize_mapping(snapshot.get("normalization"))
        reason_val = self._find_in_mapping(decision, ("reason", "skip_reason"))
        if reason_val is None and snapshot is not None:
            reason_val = snapshot.get("reason") or snapshot.get("skip_reason")
        reason_label: str | None = None
        if isinstance(reason_val, str):
            candidate = reason_val.strip()
            reason_label = candidate if candidate else None
        elif reason_val is not None:
            try:
                reason_label = str(reason_val)
            except Exception:
                reason_label = None
        decisions_raw = None
        if decision is not None:
            decisions_raw = self._find_in_mapping(
                decision,
                (
                    "decisions",
                    "decision_count",
                    "count",
                    "emissions",
                    "emitted",
                    "signals",
                    "orders",
                ),
            )
        if decisions_raw is None:
            decisions_raw = self._find_in_mapping(
                snapshot,
                (
                    "decisions",
                    "decision_count",
                    "count",
                    "emissions",
                    "emitted",
                    "signals",
                    "orders",
                ),
            )
        decisions_count = _coerce_int(decisions_raw)
        bar_ts: int | None = None
        if snapshot is not None:
            ts_value = snapshot.get("bar_ts")
            if ts_value is None:
                ts_value = self._find_in_mapping(snapshot, ("bar_ts",))
            if ts_value is None and decision is not None:
                ts_value = self._find_in_mapping(decision, ("bar_ts",))
            if ts_value is not None:
                try:
                    bar_ts = int(ts_value)
                except Exception:
                    try:
                        bar_ts = int(float(ts_value))
                    except Exception:
                        bar_ts = None

        if expected_bar_ts is not None:
            try:
                expected_ts = int(expected_bar_ts)
            except (TypeError, ValueError):
                expected_ts = None
            else:
                if bar_ts is None or bar_ts != expected_ts:
                    return None

        metrics: Dict[str, Any] = {
            "decisions": 1,
            "act_now": 1 if act_now_flag else 0,
            "turnover_usd": float(turnover),
            "cap_usd": cap_value,
            "normalized": bool(normalized_flag),
        }
        if decisions_count is not None:
            if expected_count is not None and decisions_count <= 0:
                return None
            metrics["decisions"] = max(decisions_count, 0)
        elif expected_count is not None:
            metrics["decisions"] = max(expected_count, 0)
        if adv_value is not None:
            metrics["adv_quote"] = adv_value
        if bar_ts is not None:
            metrics["bar_ts"] = bar_ts
        if impact_mode:
            metrics["impact_mode"] = impact_mode
        if reason_label:
            metrics["reason"] = reason_label
        cost_estimate = self._coerce_float(
            self._find_in_mapping(
                decision,
                ("modeled_cost_bps", "cost_bps", "expected_cost_bps"),
            )
        )
        if cost_estimate is None and snapshot is not None:
            cost_estimate = self._coerce_float(
                self._find_in_mapping(snapshot, ("modeled_cost_bps", "cost_bps"))
            )
        realized_cost = self._coerce_float(
            self._find_in_mapping(
                decision,
                ("realized_slippage_bps", "realized_cost_bps"),
            )
        )
        if realized_cost is None and snapshot is not None:
            realized_cost = self._coerce_float(
                self._find_in_mapping(
                    snapshot,
                    ("realized_slippage_bps", "realized_cost_bps"),
                )
            )
        bias_val = self._coerce_float(
            self._find_in_mapping(decision, ("cost_bias_bps",))
        )
        if bias_val is None and snapshot is not None:
            bias_val = self._coerce_float(
                self._find_in_mapping(snapshot, ("cost_bias_bps",))
            )
        if bias_val is None and realized_cost is not None and cost_estimate is not None:
            bias_val = realized_cost - cost_estimate
        if cost_estimate is not None:
            metrics["modeled_cost_bps"] = cost_estimate
        if realized_cost is not None:
            metrics["realized_slippage_bps"] = realized_cost
        if bias_val is not None:
            metrics["cost_bias_bps"] = bias_val
        if normalization_details:
            metrics["normalization"] = normalization_details
        return metrics

    def _extract_daily_pnl(self, snapshot: Mapping[str, Any] | None) -> float | None:
        if not snapshot:
            return None
        pnl = self._find_in_mapping(
            snapshot,
            (
                "daily_pnl",
                "dailyPnl",
                "dailyPnL",
                "pnl",
                "realized_pnl",
            ),
        )
        return self._coerce_float(pnl)

    # ------------------------------------------------------------------
    # Dynamic guard helpers
    # ------------------------------------------------------------------
    def _resolve_rest_helper(self, candidates: Sequence[Any]) -> Any | None:
        seen: set[int] = set()
        queue: deque[Any] = deque()
        for base in (self._fp, self._policy, self._executor, self._guards):
            if base is not None:
                queue.append(base)
        for obj in candidates:
            if obj is not None:
                queue.append(obj)
        attr_names = (
            "rest_helper",
            "rest_client",
            "rest",
            "client",
            "public",
            "api",
            "source",
            "adapter",
            "market_data",
        )
        while queue:
            obj = queue.popleft()
            ident = id(obj)
            if ident in seen:
                continue
            seen.add(ident)
            getter = getattr(obj, "get_last_price", None)
            if callable(getter):
                return obj
            for name in attr_names:
                try:
                    attr = getattr(obj, name)
                except Exception:
                    attr = None
                if attr is None or id(attr) in seen:
                    continue
                queue.append(attr)
        return None

    def _seed_last_prices_from_state(self) -> None:
        try:
            st = state_storage.get_state()
        except Exception:
            return
        raw = getattr(st, "last_prices", {}) or {}
        if not isinstance(raw, MappingABC):
            return
        for sym, value in raw.items():
            symbol = str(sym).upper()
            price = self._coerce_price(value)
            if symbol and price is not None:
                self._last_prices[symbol] = price

    def _load_exposure_state(self) -> None:
        try:
            st = state_storage.get_state()
        except Exception:
            st = None
        self._positions = {}
        self._pending_exposure = {}
        self._daily_symbol_turnover = {}
        self._daily_portfolio_turnover = {"day": None, "total": 0.0}
        self._daily_day_key = None
        stored_state: Mapping[str, Any] = {}
        if st is not None:
            raw_state = getattr(st, "exposure_state", {}) or {}
            if isinstance(raw_state, MappingABC):
                stored_state = dict(raw_state)

        if st is not None:
            raw_positions = getattr(st, "positions", {}) or {}
            if isinstance(raw_positions, MappingABC):
                for symbol, payload in raw_positions.items():
                    sym = str(symbol).upper()
                    if not sym:
                        continue
                    qty_val: float | None
                    try:
                        qty_val = float(payload)
                    except (TypeError, ValueError):
                        if hasattr(payload, "qty"):
                            try:
                                qty_val = float(getattr(payload, "qty"))
                            except Exception:
                                qty_val = None
                        else:
                            qty_val = None
                    if qty_val is None or math.isclose(qty_val, 0.0, rel_tol=0.0, abs_tol=1e-12):
                        continue
                    self._positions[sym] = qty_val

        raw_positions = (
            stored_state.get("positions", {}) if isinstance(stored_state, MappingABC) else {}
        )
        if isinstance(raw_positions, MappingABC):
            for symbol, qty in raw_positions.items():
                sym = str(symbol).upper()
                if not sym or sym in self._positions:
                    continue
                try:
                    qty_val = float(qty)
                except (TypeError, ValueError):
                    continue
                if math.isclose(qty_val, 0.0, rel_tol=0.0, abs_tol=1e-12):
                    continue
                self._positions[sym] = qty_val

        if self._execution_mode == "bar":
            self._weights.clear()
            if isinstance(stored_state, MappingABC):
                raw_weights = stored_state.get("weights", {})
                if isinstance(raw_weights, MappingABC):
                    for symbol, weight in raw_weights.items():
                        sym = str(symbol).upper()
                        if not sym:
                            continue
                        try:
                            weight_val = float(weight)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(weight_val):
                            continue
                        if math.isclose(weight_val, 0.0, rel_tol=0.0, abs_tol=1e-12):
                            continue
                        self._weights[sym] = weight_val
                        self._apply_weight_to_positions(sym, weight_val)
            if not self._weights and self._positions and self._portfolio_equity:
                equity = float(self._portfolio_equity)
                if math.isfinite(equity) and equity > 0.0:
                    for symbol, qty in list(self._positions.items()):
                        price = self._last_prices.get(symbol)
                        if price is None or price <= 0.0:
                            continue
                        try:
                            qty_val = float(qty)
                        except (TypeError, ValueError):
                            continue
                        weight_guess = qty_val * float(price) / equity
                        weight_clamped = self._clamp_weight(weight_guess)
                        if math.isclose(weight_clamped, 0.0, rel_tol=0.0, abs_tol=1e-12):
                            continue
                        self._weights[symbol] = weight_clamped
                        self._apply_weight_to_positions(symbol, weight_clamped)

        pending_summary: Dict[str, Dict[str, float]] = {}
        if isinstance(stored_state, MappingABC):
            raw_pending = stored_state.get("pending", {})
            if isinstance(raw_pending, MappingABC):
                for symbol, legs in raw_pending.items():
                    sym = str(symbol).upper()
                    if not sym or not isinstance(legs, MappingABC):
                        continue
                    sanitized_legs: Dict[str, float] = {}
                    for leg, delta in legs.items():
                        leg_name = str(leg)
                        try:
                            delta_val = float(delta)
                        except (TypeError, ValueError):
                            continue
                        if not math.isfinite(delta_val) or math.isclose(
                            delta_val, 0.0, rel_tol=0.0, abs_tol=1e-12
                        ):
                            continue
                        sanitized_legs[leg_name] = delta_val
                        key = f"state:{sym}:{leg_name}"
                        self._pending_exposure[key] = {
                            "symbol": sym,
                            "leg": leg_name,
                            "delta": delta_val,
                            "source": "state",
                        }
                    if sanitized_legs:
                        pending_summary[sym] = sanitized_legs
        total_notional = 0.0
        updated_at_ms = 0
        if isinstance(stored_state, MappingABC):
            try:
                total_notional = float(stored_state.get("total_notional", 0.0) or 0.0)
            except (TypeError, ValueError):
                total_notional = 0.0
            try:
                updated_at_ms = int(stored_state.get("updated_at_ms", 0) or 0)
            except (TypeError, ValueError):
                updated_at_ms = 0
        if st is not None and not total_notional:
            try:
                total_notional = float(getattr(st, "total_notional", 0.0) or 0.0)
            except (TypeError, ValueError):
                total_notional = 0.0
        daily_state_raw = {}
        if isinstance(stored_state, MappingABC):
            daily_state_raw = stored_state.get("daily_turnover", {})
        if isinstance(daily_state_raw, MappingABC):
            self._restore_daily_turnover_state(daily_state_raw)
        self._exposure_state = {
            "positions": dict(self._positions),
            "pending": pending_summary,
            "updated_at_ms": updated_at_ms,
            "total_notional": total_notional,
            "daily_turnover": self._daily_turnover_state_payload(),
        }

    @staticmethod
    def _coerce_price(value: Any) -> float | None:
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(price) or price <= 0.0:
            return None
        return price

    def _sanitize_snapshot_value(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._sanitize_snapshot_value(asdict(value))
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, MappingABC):
            sanitized: Dict[str, Any] = {}
            for key, val in value.items():
                if callable(val):
                    continue
                sanitized[str(key)] = self._sanitize_snapshot_value(val)
            return sanitized
        if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
            return [self._sanitize_snapshot_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            return self._sanitize_snapshot_value(vars(value))
        except Exception:
            return str(value)

    def _build_config_snapshot(self) -> Dict[str, Any]:
        try:
            raw = asdict(self.cfg)
        except Exception:
            try:
                raw = dict(vars(self.cfg))
            except Exception:
                return {}
        snapshot = self._sanitize_snapshot_value(raw)
        return snapshot if isinstance(snapshot, dict) else {}

    def _read_git_hash(self) -> str | None:
        if self._git_hash is not None:
            return self._git_hash
        try:
            output = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            self._git_hash = output.decode("utf-8").strip() or None
        except Exception:
            self._git_hash = None
        return self._git_hash

    def _persist_run_metadata(self) -> None:
        if self._metadata_persisted or not self.cfg.state.enabled:
            return
        updates: Dict[str, Any] = {}
        snapshot = self._build_config_snapshot()
        if snapshot:
            updates["config_snapshot"] = snapshot
        git_hash = self._read_git_hash()
        if git_hash:
            updates["git_hash"] = git_hash
        if updates:
            try:
                state_storage.update_state(**updates)
            except Exception:
                self.logger.exception("failed to persist run metadata")
                return
        self._metadata_persisted = True

    def _set_last_price(self, symbol: str, price: Any) -> None:
        sym = str(symbol).upper()
        if not sym:
            return
        val = self._coerce_price(price)
        if val is None:
            return
        prev = self._last_prices.get(sym)
        if prev is not None and math.isclose(prev, val, rel_tol=1e-9, abs_tol=1e-12):
            return
        self._last_prices[sym] = val
        if self._execution_mode == "bar" and sym in self._weights:
            self._apply_weight_to_positions(sym, self._weights.get(sym, 0.0))
        if self._state_enabled:
            try:
                state_storage.update_state(last_prices=dict(self._last_prices))
            except Exception:
                pass

    def _maybe_fetch_rest_price(self, symbol: str) -> None:
        sym = str(symbol).upper()
        if not sym:
            return
        if sym in self._last_prices:
            return
        if self._rest_helper is None:
            return
        now = time.monotonic()
        next_allowed = self._rest_backoff_until.get(sym, 0.0)
        if now < next_allowed:
            return
        step = self._rest_backoff_step.get(sym, 1.0)
        try:
            price = self._rest_helper.get_last_price(sym)
        except Exception:
            self._rest_backoff_until[sym] = now + step
            self._rest_backoff_step[sym] = min(step * 2.0, 300.0)
            return
        coerced = self._coerce_price(price)
        if coerced is None:
            self._rest_backoff_until[sym] = now + step
            self._rest_backoff_step[sym] = min(step * 2.0, 300.0)
            return
        self._set_last_price(sym, coerced)
        self._rest_backoff_until[sym] = now + 0.0
        self._rest_backoff_step[sym] = 1.0

    @staticmethod
    def _coerce_quantity(order: Any) -> float | None:
        try:
            qty = getattr(order, "quantity")
        except Exception:
            return None
        if qty is None:
            return None
        try:
            value = float(qty)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value

    @staticmethod
    def _order_side(order: Any) -> str:
        side = getattr(order, "side", "")
        if hasattr(side, "value"):
            try:
                side = side.value
            except Exception:
                side = ""
        return str(side or "").upper()

    @staticmethod
    def _clamp_weight(value: float) -> float:
        try:
            weight = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(weight):
            return 0.0
        if weight < 0.0:
            return 0.0
        if weight > 1.0:
            return 1.0
        return weight

    @staticmethod
    def _materialize_mapping(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, MappingABC):
            return dict(value)
        for attr in ("model_dump", "dict"):
            getter = getattr(value, attr, None)
            if callable(getter):
                try:
                    result = getter()
                except Exception:
                    continue
                if isinstance(result, MappingABC):
                    return dict(result)
        attrs = getattr(value, "__dict__", None)
        if isinstance(attrs, MappingABC):
            return {
                k: v
                for k, v in attrs.items()
                if not callable(v)
            }
        return {}

    def _extract_signal_payload(self, order: Any) -> Dict[str, Any]:
        meta = getattr(order, "meta", None)

        def _materialize_candidate(candidate: Any) -> Dict[str, Any]:
            if isinstance(candidate, SpotSignalEnvelope):
                return self._materialize_mapping(candidate.payload)
            payload_map = self._materialize_mapping(candidate)
            if payload_map:
                return payload_map
            if isinstance(candidate, MappingABC):
                return dict(candidate)
            return {}

        sidecar_meta = self._order_meta_sidecar_lookup(order)
        if isinstance(sidecar_meta, MappingABC):
            sidecar_map = dict(sidecar_meta)
            payload_val = sidecar_map.get("payload")
            if payload_val is not None:
                resolved = _materialize_candidate(payload_val)
                if resolved:
                    return resolved
            rebalance_val = sidecar_map.get("rebalance")
            if rebalance_val is not None:
                resolved = _materialize_candidate(rebalance_val)
                if resolved:
                    return resolved
            signal_payload_val = sidecar_map.get("signal_payload")
            if signal_payload_val is not None:
                resolved = _materialize_candidate(signal_payload_val)
                if resolved:
                    return resolved

        if isinstance(meta, SpotSignalEnvelope):
            return _materialize_candidate(meta)

        meta_map: Dict[str, Any] | None = None
        if meta is not None:
            meta_map = self._materialize_mapping(meta)
            if not meta_map and isinstance(meta, MappingABC):
                meta_map = dict(meta)
        elif isinstance(meta, MappingABC):
            meta_map = dict(meta)

        if meta_map:
            payload_val = meta_map.get("payload")
            if payload_val is not None:
                return _materialize_candidate(payload_val)
            rebalance_val = meta_map.get("rebalance")
            if rebalance_val is not None:
                return _materialize_candidate(rebalance_val)
            signal_payload_val = meta_map.get("signal_payload")
            if signal_payload_val is not None:
                return _materialize_candidate(signal_payload_val)

        if meta is not None and not isinstance(meta, MappingABC):
            getter = getattr(meta, "get", None)
            if callable(getter):
                payload_val = getter("payload")
                if payload_val is not None:
                    return _materialize_candidate(payload_val)
                rebalance_val = getter("rebalance")
                if rebalance_val is not None:
                    return _materialize_candidate(rebalance_val)
                signal_payload_val = getter("signal_payload")
                if signal_payload_val is not None:
                    return _materialize_candidate(signal_payload_val)

        payload_attr = getattr(order, "payload", None)
        if payload_attr is not None:
            return _materialize_candidate(payload_attr)

        if isinstance(meta, MappingABC):
            rebalance_payload = meta.get("rebalance")
            if isinstance(rebalance_payload, MappingABC):
                return dict(rebalance_payload)

        return {}

    @staticmethod
    def _coerce_mapping(value: Any) -> Dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, MappingABC):
            return dict(value)
        return None

    @classmethod
    def _scale_turnover_mapping(
        cls, mapping: Dict[str, Any] | None, scale: float
    ) -> Dict[str, Any] | None:
        if mapping is None:
            return None
        if not math.isfinite(scale) or scale < 0.0:
            return mapping
        for key in cls._TURNOVER_KEYS:
            if key not in mapping:
                continue
            try:
                raw_val = float(mapping[key])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(raw_val):
                continue
            mapping[key] = raw_val * scale
        return mapping

    @staticmethod
    def _scale_normalization_metadata(
        mapping: Dict[str, Any] | None, scale: float
    ) -> Dict[str, Any] | None:
        if mapping is None:
            return None
        if not math.isfinite(scale) or scale <= 0.0 or scale >= 1.0:
            return mapping
        if "factor" in mapping:
            try:
                factor_val = float(mapping.get("factor"))
            except (TypeError, ValueError):
                factor_val = None
            if factor_val is not None and math.isfinite(factor_val):
                mapping["factor"] = max(0.0, factor_val * scale)
        if "available_delta" in mapping:
            try:
                available_delta = float(mapping.get("available_delta"))
            except (TypeError, ValueError):
                available_delta = None
            if available_delta is not None and math.isfinite(available_delta):
                mapping["available_delta"] = max(0.0, available_delta * scale)
        return mapping

    def _order_meta_sidecar_lookup(self, order_obj: Any) -> Dict[str, Any] | None:
        try:
            sidecar_meta = self._order_meta_sidecar.get(order_obj)
        except Exception:
            sidecar_meta = None
        if isinstance(sidecar_meta, dict):
            return sidecar_meta
        key = id(order_obj)
        record = self._order_meta_fallback.get(key)
        if record is None:
            return None
        ref, mapping = record
        if ref is not None:
            obj = ref()
            if obj is None:
                self._order_meta_fallback.pop(key, None)
                return None
            if obj is not order_obj:
                return None
        return mapping

    def _attach_order_meta_sidecar(
        self, order_obj: Any, meta_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            self._order_meta_sidecar[order_obj] = meta_dict
            return meta_dict
        except TypeError:
            key = id(order_obj)
            try:
                self_ref = weakref.ref(self)

                def _callback(ref: weakref.ReferenceType[Any]) -> None:
                    inst = self_ref()
                    if inst is not None:
                        inst._order_meta_fallback.pop(key, None)

                obj_ref: weakref.ReferenceType[Any] | None = weakref.ref(order_obj, _callback)
            except TypeError:
                obj_ref = None
            if obj_ref is None:
                self._order_meta_fallback[key] = (None, meta_dict)
            else:
                self._order_meta_fallback[key] = (obj_ref, meta_dict)
            return meta_dict
        except Exception:
            return meta_dict

    def _lookup_order_meta(self, order_obj: Any) -> Mapping[str, Any] | None:
        meta_val = getattr(order_obj, "meta", None)
        if isinstance(meta_val, MappingABC):
            return meta_val
        return self._order_meta_sidecar_lookup(order_obj)

    def _get_order_meta(self, order_obj: Any) -> Mapping[str, Any] | None:
        meta_val = self._lookup_order_meta(order_obj)
        if isinstance(meta_val, MappingABC):
            return meta_val
        return None

    def _ensure_order_meta(self, order_obj: Any) -> Dict[str, Any]:
        existing_meta = self._lookup_order_meta(order_obj)
        if isinstance(existing_meta, dict):
            return existing_meta
        if isinstance(existing_meta, MappingABC):
            meta_dict = dict(existing_meta)
        else:
            meta_dict = {}
        attached = False
        try:
            setattr(order_obj, "meta", meta_dict)
            attached = True
        except Exception:
            attached = False
        if not attached:
            meta_dict = self._attach_order_meta_sidecar(order_obj, meta_dict)
        return meta_dict

    def _store_created_ts(self, order_obj: Any, created_ts_ms: int) -> None:
        stored = False
        try:
            setattr(order_obj, "created_ts_ms", created_ts_ms)
            stored = True
        except Exception:
            stored = False
        if not stored:
            meta_map = self._ensure_order_meta(order_obj)
            try:
                meta_map["created_ts_ms"] = created_ts_ms
            except Exception:
                pass

    def _extract_created_ts_ms(self, order_obj: Any) -> int:
        candidates: list[Any] = [getattr(order_obj, "created_ts_ms", None)]
        meta_map = self._get_order_meta(order_obj)
        if isinstance(meta_map, MappingABC):
            candidates.append(meta_map.get("created_ts_ms"))
            candidates.append(meta_map.get("now_ms"))
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                parsed = int(candidate)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
        return 0

    def _resolve_weight_targets(
        self,
        symbol: str,
        payload: Mapping[str, Any],
        current_weight: float | None = None,
    ) -> tuple[float, float, str | None]:
        if current_weight is None:
            base_weight = self._weights.get(symbol, 0.0)
        else:
            base_weight = current_weight
        try:
            current = float(base_weight)
        except (TypeError, ValueError):
            current = 0.0
        if not math.isfinite(current):
            current = 0.0
        target = current
        reason: str | None = None
        tol = 1e-9

        if "target_weight" in payload or "target" in payload or "weight" in payload:
            raw = payload.get("target_weight")
            if raw is None:
                raw = payload.get("target")
            if raw is None:
                raw = payload.get("weight")
            try:
                requested = float(raw)
            except (TypeError, ValueError):
                requested = current
            if math.isfinite(requested):
                if requested < -tol or requested > 1.0 + tol:
                    reason = "target_weight_out_of_bounds"
                    try:
                        self._logger.warning(
                            "Rejecting target weight for %s: requested=%s outside [0, 1]",
                            symbol,
                            raw,
                        )
                    except Exception:
                        pass
                else:
                    target = self._clamp_weight(requested)
        elif "delta_weight" in payload or "delta" in payload:
            raw_delta = payload.get("delta_weight")
            if raw_delta is None:
                raw_delta = payload.get("delta")
            try:
                delta_val = float(raw_delta)
            except (TypeError, ValueError):
                delta_val = 0.0
            if math.isfinite(delta_val):
                desired = current + delta_val
                if desired < -tol or desired > 1.0 + tol:
                    reason = "delta_weight_out_of_bounds"
                    try:
                        self._logger.warning(
                            "Rejecting delta weight for %s: current=%s delta=%s => %s outside [0, 1]",
                            symbol,
                            current,
                            raw_delta,
                            desired,
                        )
                    except Exception:
                        pass
                else:
                    target = self._clamp_weight(desired)
        if reason is not None:
            target = current
        delta = target - current
        return target, delta, reason

    def _resolve_economics(
        self, payload: Mapping[str, Any], meta: Mapping[str, Any] | None
    ) -> Dict[str, Any]:
        economics = payload.get("economics")
        econ_map: Dict[str, Any] = {}
        if isinstance(economics, MappingABC):
            econ_map = dict(economics)
        else:
            candidate = None
            if isinstance(meta, MappingABC):
                candidate = meta.get("economics") or meta.get("decision")
            if candidate is None:
                candidate = economics
            econ_map = self._materialize_mapping(candidate)
        def _get_float(key: str, default: float = 0.0) -> float:
            try:
                value = float(econ_map.get(key, payload.get(key, default)))
            except (TypeError, ValueError):
                value = default
            if not math.isfinite(value):
                return default
            return value

        edge = _get_float("edge_bps", 0.0)
        cost = _get_float("cost_bps", 0.0)
        net = _get_float("net_bps", edge - cost)
        turnover = _get_float("turnover_usd", 0.0)
        impact = _get_float("impact", 0.0)
        impact_mode_val = econ_map.get("impact_mode") or payload.get("impact_mode")
        impact_mode: str | None
        if impact_mode_val is None:
            impact_mode = "model"
        else:
            try:
                impact_mode = str(impact_mode_val)
            except Exception:
                impact_mode = "model"
            else:
                normalized = impact_mode.strip()
                impact_mode = normalized if normalized else "model"
        act_now = econ_map.get("act_now")
        if isinstance(act_now, str):
            act_now = act_now.strip().lower() in {"1", "true", "yes"}
        act_flag = bool(act_now) if act_now is not None else bool(payload.get("act_now", True))
        return {
            "edge_bps": edge,
            "cost_bps": cost,
            "net_bps": net,
            "turnover_usd": max(0.0, turnover),
            "act_now": act_flag,
            "impact": max(0.0, impact),
            "impact_mode": impact_mode,
        }

    @staticmethod
    def _coerce_timestamp_ms(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        elif hasattr(value, "to_pydatetime"):
            try:
                dt = value.to_pydatetime()
            except Exception:
                dt = None
            if dt is None:
                return None
            if not isinstance(dt, datetime):
                return None
        else:
            dt = None

        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return int(round(dt.timestamp() * 1000.0))

        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            if not math.isfinite(value):
                return None
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return int(text)
            except ValueError:
                try:
                    return int(float(text))
                except ValueError:
                    normalized = text
                    if text.endswith("Z") or text.endswith("z"):
                        normalized = text[:-1] + "+00:00"
                    try:
                        dt = datetime.fromisoformat(normalized)
                    except ValueError:
                        return None
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    return int(round(dt.timestamp() * 1000.0))
        timestamp = getattr(value, "timestamp", None)
        if callable(timestamp):
            try:
                ts = float(timestamp())
            except Exception:
                return None
            if not math.isfinite(ts):
                return None
            return int(round(ts * 1000.0))
        return None

    def _build_envelope_payload(
        self, order: Any, symbol: str
    ) -> tuple[Dict[str, Any], int | None, float | None]:
        payload = self._extract_signal_payload(order)
        meta_raw = getattr(order, "meta", None)
        meta = meta_raw if isinstance(meta_raw, MappingABC) else self._materialize_mapping(meta_raw)
        envelope_payload = dict(payload)
        economics = self._resolve_economics(envelope_payload, meta)
        adv_quote = self._extract_adv_quote(order, envelope_payload, meta)
        if adv_quote is not None:
            economics["adv_quote"] = adv_quote
        envelope_payload["economics"] = economics
        valid_until_ms: int | None = None

        def _extract_valid_until(mapping: Mapping[str, Any]) -> int | None:
            for key in ("valid_until_ms", "valid_until"):
                if key not in mapping:
                    continue
                ms = self._coerce_timestamp_ms(mapping.get(key))
                if ms is not None:
                    return ms
            return None

        if isinstance(envelope_payload, MappingABC):
            valid_until_ms = _extract_valid_until(envelope_payload)
        if valid_until_ms is None and isinstance(meta, MappingABC):
            valid_until_ms = _extract_valid_until(meta)
        if valid_until_ms is not None:
            envelope_payload["valid_until_ms"] = valid_until_ms
        kind = envelope_payload.get("kind")
        if not isinstance(kind, str):
            if "delta_weight" in envelope_payload or "delta" in envelope_payload:
                kind = "delta_weight"
            else:
                kind = "target_weight"
        kind = str(kind)
        if kind not in {"target_weight", "delta_weight"}:
            kind = "target_weight"
        envelope_payload["kind"] = kind
        symbol_key = str(symbol).upper()
        requested_target = envelope_payload.get("target_weight")
        requested_delta = envelope_payload.get("delta_weight")
        resolved_target, resolved_delta, weight_reason = self._resolve_weight_targets(
            symbol_key, envelope_payload
        )
        if weight_reason:
            envelope_payload.setdefault("reject_reason", weight_reason)
            if requested_target is not None:
                envelope_payload.setdefault("requested_target_weight", requested_target)
            if requested_delta is not None:
                envelope_payload.setdefault("requested_delta_weight", requested_delta)
        if kind == "target_weight":
            envelope_payload["target_weight"] = resolved_target
        else:
            envelope_payload["delta_weight"] = resolved_delta
        return envelope_payload, valid_until_ms, adv_quote

    def _build_drop_envelope(
        self, order: Any, symbol: str, bar_close_ms: int
    ) -> Any | None:
        payload, payload_valid_until_ms, adv_quote = self._build_envelope_payload(
            order, symbol
        )
        try:
            meta_map = self._ensure_order_meta(order)
        except Exception:
            meta_map = None
        if adv_quote is not None and isinstance(meta_map, MappingABC):
            try:
                meta_map["adv_quote"] = adv_quote
            except Exception:
                pass
        created_ts = self._extract_created_ts_ms(order)
        try:
            base_bar_close_ms = int(bar_close_ms)
        except (TypeError, ValueError):
            base_bar_close_ms = 0
        expires_at_ms = max(created_ts, base_bar_close_ms)
        if self._ws_dedup_timeframe_ms > 0 and created_ts > 0:
            expires_at_ms = max(
                expires_at_ms, created_ts + self._ws_dedup_timeframe_ms
            )
        ttl_timeframe_ms = self._resolve_ttl_timeframe_ms(log_if_invalid=False)
        if ttl_timeframe_ms is not None and ttl_timeframe_ms > 0:
            try:
                _, ttl_expires_at_ms, _ = check_ttl(
                    bar_close_ms=base_bar_close_ms,
                    now_ms=clock.now_ms(),
                    timeframe_ms=ttl_timeframe_ms,
                )
            except Exception:
                ttl_expires_at_ms = None
            if ttl_expires_at_ms is not None:
                expires_at_ms = min(expires_at_ms, ttl_expires_at_ms)
        if payload_valid_until_ms is not None:
            try:
                payload_expires_ms = int(payload_valid_until_ms)
            except (TypeError, ValueError):
                payload_expires_ms = None
            if payload_expires_ms is not None:
                expires_at_ms = min(expires_at_ms, payload_expires_ms)
        expires_at_ms = max(expires_at_ms, base_bar_close_ms)
        try:
            return build_signal_envelope(
                symbol=str(symbol),
                bar_close_ms=base_bar_close_ms,
                expires_at_ms=int(expires_at_ms),
                payload=payload,
            )
        except Exception:
            return None

    def _log_drop_envelope(
        self, order: Any, symbol: str, bar_close_ms: int, reason: str
    ) -> None:
        try:
            envelope = self._build_drop_envelope(order, symbol, bar_close_ms)
        except Exception:
            envelope = None
        if envelope is None:
            return
        try:
            log_drop(envelope, reason)
        except Exception:
            pass

    def _persist_exposure_state(self, ts_ms: int | None = None) -> None:
        timestamp = int(ts_ms if ts_ms is not None else clock.now_ms())
        positions_snapshot: Dict[str, float] = {}
        for symbol, qty in self._positions.items():
            if math.isclose(qty, 0.0, rel_tol=0.0, abs_tol=1e-12):
                continue
            positions_snapshot[symbol] = float(qty)
        pending_summary: Dict[str, Dict[str, float]] = {}
        for data in self._pending_exposure.values():
            symbol = str(data.get("symbol", "")).upper()
            if not symbol:
                continue
            leg = str(data.get("leg", "unknown") or "unknown")
            try:
                delta = float(data.get("delta", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            summary = pending_summary.setdefault(symbol, {})
            summary[leg] = summary.get(leg, 0.0) + delta
        weight_snapshot: Dict[str, float] = {}
        if self._execution_mode == "bar":
            for symbol, weight in self._weights.items():
                try:
                    weight_val = float(weight)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(weight_val):
                    continue
                if math.isclose(weight_val, 0.0, rel_tol=0.0, abs_tol=1e-12):
                    continue
                weight_snapshot[symbol] = weight_val

        total_notional = 0.0
        if self._execution_mode == "bar":
            equity = self._portfolio_equity
            if equity is not None and equity > 0.0:
                total_notional = sum(abs(val) for val in weight_snapshot.values()) * float(equity)
            if total_notional <= 0.0:
                symbol_total = 0.0
                for symbol, weight in weight_snapshot.items():
                    equity_override = self._symbol_equity.get(symbol)
                    if equity_override is None or equity_override <= 0.0:
                        continue
                    try:
                        weight_abs = abs(float(weight))
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(weight_abs):
                        continue
                    symbol_total += weight_abs * float(equity_override)
                if symbol_total > 0.0:
                    total_notional = symbol_total
        if total_notional <= 0.0:
            for symbol, qty in positions_snapshot.items():
                price = self._last_prices.get(symbol)
                if price is None:
                    continue
                try:
                    total_notional += abs(float(qty)) * float(price)
                except Exception:
                    continue
        self._exposure_state = {
            "positions": positions_snapshot,
            "pending": pending_summary,
            "updated_at_ms": timestamp,
            "total_notional": total_notional,
        }
        daily_state = self._daily_turnover_state_payload()
        self._exposure_state["daily_turnover"] = daily_state
        if self._execution_mode == "bar":
            self._exposure_state["weights"] = weight_snapshot
        if self._state_enabled:
            try:
                state_storage.update_state(
                    exposure_state=dict(self._exposure_state),
                    total_notional=float(total_notional),
                    daily_turnover=daily_state,
                )
            except Exception:
                pass

    def set_portfolio_guard(self, guard: PortfolioLimitGuard | None) -> None:
        self._portfolio_guard = guard

    def _portfolio_positions_snapshot(self) -> Dict[str, float]:
        snapshot: Dict[str, float] = {}
        for symbol, qty in self._positions.items():
            try:
                value = float(qty)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12):
                continue
            snapshot[str(symbol).upper()] = value
        return snapshot

    def _portfolio_total_notional(self) -> float:
        if self._execution_mode == "bar":
            equity = self._portfolio_equity
            if equity is not None and equity > 0.0:
                total = 0.0
                for weight in self._weights.values():
                    try:
                        total += abs(float(weight))
                    except (TypeError, ValueError):
                        continue
                if total > 0.0:
                    return float(equity) * total
            symbol_total = 0.0
            for symbol, weight in self._weights.items():
                equity_override = self._symbol_equity.get(str(symbol).upper())
                if equity_override is None or equity_override <= 0.0:
                    continue
                try:
                    weight_abs = abs(float(weight))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(weight_abs):
                    continue
                symbol_total += weight_abs * float(equity_override)
            if symbol_total > 0.0:
                return symbol_total
        try:
            value = float(self._exposure_state.get("total_notional", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(value) or value < 0.0:
            return 0.0
        return value

    def _portfolio_last_price(self, symbol: str) -> float | None:
        sym = str(symbol or "").upper()
        price = self._last_prices.get(sym)
        if price is None:
            return None
        try:
            value = float(price)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        return value

    def _apply_weight_to_positions(
        self, symbol: str, weight: float, *, equity_override: float | None = None
    ) -> None:
        sym = str(symbol or "").upper()
        if not sym:
            return
        try:
            weight_val = float(weight)
        except (TypeError, ValueError):
            return
        if not math.isfinite(weight_val):
            return
        if math.isclose(weight_val, 0.0, rel_tol=0.0, abs_tol=1e-12):
            self._positions.pop(sym, None)
            return
        equity = None
        if equity_override is not None and equity_override > 0.0:
            equity = float(equity_override)
        elif sym in self._symbol_equity:
            cached_equity = self._symbol_equity.get(sym)
            if cached_equity is not None and cached_equity > 0.0:
                equity = float(cached_equity)
        if equity is None:
            base_equity = self._portfolio_equity
            if base_equity is not None and base_equity > 0.0:
                equity = float(base_equity)
        price = self._last_prices.get(sym)
        qty: float | None = None
        if equity is not None and equity > 0.0 and price is not None and price > 0.0:
            qty = (weight_val * float(equity)) / float(price)
        if qty is None:
            # Without reliable equity and price information we can't derive a
            # notional-based quantity.  Drop any cached exposure to avoid
            # reporting fictitious positions.
            self._positions.pop(sym, None)
            return
        if math.isclose(qty, 0.0, rel_tol=0.0, abs_tol=1e-12):
            self._positions.pop(sym, None)
        else:
            self._positions[sym] = qty

    def _normalize_weight_targets(
        self, orders: Sequence[Any]
    ) -> tuple[list[Any], bool]:
        if not orders or self._execution_mode != "bar":
            return list(orders), False
        cap = self._max_total_weight
        if cap is None or cap <= 0.0:
            return list(orders), False

        orders_list = list(orders)
        scalable_infos: list[Dict[str, Any]] = []
        scalable_by_symbol: Dict[str, Dict[str, Any]] = {}
        normalized_by_symbol: Dict[str, float] = {}
        seen_symbols: set[str] = set()
        running_weights: Dict[str, float] = {}

        def _coerce_bool(value: Any) -> bool:
            if isinstance(value, str):
                candidate = value.strip().lower()
                if candidate in {"", "0", "false", "no", "off"}:
                    return False
                if candidate in {"1", "true", "yes", "on"}:
                    return True
            return bool(value)

        for order in orders_list:
            symbol = str(getattr(order, "symbol", "") or "").upper()
            payload = self._extract_signal_payload(order)
            target, _, weight_reason = self._resolve_weight_targets(symbol, payload)
            if weight_reason:
                payload.setdefault("reject_reason", weight_reason)
            meta_map = self._ensure_order_meta(order)
            normalized_source: Any = None
            if "normalized" in payload:
                normalized_source = payload.get("normalized")
            elif "normalized" in meta_map:
                normalized_source = meta_map.get("normalized")
            normalized_flag = _coerce_bool(normalized_source)
            if "normalized" in payload or normalized_source is not None:
                payload["normalized"] = normalized_flag
            if "normalized" in meta_map or normalized_source is not None:
                meta_map["normalized"] = normalized_flag
            if symbol in running_weights:
                current_weight = running_weights[symbol]
            else:
                try:
                    current_weight = float(self._weights.get(symbol, 0.0))
                except (TypeError, ValueError):
                    current_weight = 0.0
                if not math.isfinite(current_weight) or current_weight < 0.0:
                    current_weight = 0.0
                running_weights[symbol] = current_weight
            try:
                target_weight = float(target)
            except (TypeError, ValueError):
                target_weight = 0.0
            if not math.isfinite(target_weight):
                target_weight = 0.0
            if target_weight < 0.0:
                target_weight = 0.0
            delta = target_weight - current_weight
            if not math.isfinite(delta):
                delta = 0.0
            info = {
                "order": order,
                "symbol": symbol,
                "payload": payload,
                "current": current_weight,
                "target": target_weight,
                "requested": target_weight,
                "delta": delta,
                "normalized": bool(normalized_flag),
            }
            if symbol:
                seen_symbols.add(symbol)
            if normalized_flag:
                if symbol:
                    post_weight = max(0.0, target_weight)
                    normalized_by_symbol[symbol] = post_weight
                    running_weights[symbol] = post_weight
            elif symbol:
                scalable_infos.append(info)
                post_weight = max(0.0, target_weight)
                aggregate = scalable_by_symbol.get(symbol)
                if aggregate is None:
                    scalable_by_symbol[symbol] = {
                        "current": current_weight,
                        "target": post_weight,
                        "requested": post_weight,
                        "delta_total": delta,
                        "positive_delta_total": delta if delta > 0.0 else 0.0,
                        "negative_delta_total": delta if delta < 0.0 else 0.0,
                    }
                else:
                    existing_current = aggregate.get("current", 0.0)
                    try:
                        existing_current_val = float(existing_current)
                    except (TypeError, ValueError):
                        existing_current_val = None
                    if existing_current_val is None or not math.isfinite(existing_current_val):
                        aggregate["current"] = current_weight
                    aggregate["target"] = post_weight
                    aggregate["requested"] = post_weight
                    try:
                        prev_delta_total = float(aggregate.get("delta_total", 0.0))
                    except (TypeError, ValueError):
                        prev_delta_total = 0.0
                    aggregate["delta_total"] = prev_delta_total + delta
                    if delta > 0.0:
                        try:
                            prev_positive = float(aggregate.get("positive_delta_total", 0.0))
                        except (TypeError, ValueError):
                            prev_positive = 0.0
                        aggregate["positive_delta_total"] = prev_positive + delta
                    elif delta < 0.0:
                        try:
                            prev_negative = float(aggregate.get("negative_delta_total", 0.0))
                        except (TypeError, ValueError):
                            prev_negative = 0.0
                        aggregate["negative_delta_total"] = prev_negative + delta
                running_weights[symbol] = post_weight

        if not scalable_infos:
            return orders_list, False

        normalized_total = sum(normalized_by_symbol.values())
        requested_total = sum(
            max(0.0, float(aggregate.get("requested", 0.0)))
            for aggregate in scalable_by_symbol.values()
        )

        current_total = 0.0
        delta_total = 0.0
        positive_delta_total = 0.0
        negative_delta_total = 0.0
        for aggregate in scalable_by_symbol.values():
            try:
                current_weight = float(aggregate.get("current", 0.0))
            except (TypeError, ValueError):
                current_weight = 0.0
            if not math.isfinite(current_weight) or current_weight < 0.0:
                current_weight = 0.0
            current_total += current_weight

            agg_delta: float | None
            try:
                agg_delta = float(aggregate.get("delta_total", 0.0))
            except (TypeError, ValueError):
                agg_delta = None
            if agg_delta is None or not math.isfinite(agg_delta):
                try:
                    target_weight = float(aggregate.get("target", current_weight))
                except (TypeError, ValueError):
                    target_weight = current_weight
                if not math.isfinite(target_weight):
                    target_weight = current_weight
                agg_delta = target_weight - current_weight
                if not math.isfinite(agg_delta):
                    agg_delta = 0.0
            delta_total += agg_delta

            try:
                pos_total = float(aggregate.get("positive_delta_total", 0.0))
            except (TypeError, ValueError):
                pos_total = 0.0
            if not math.isfinite(pos_total) or pos_total < 0.0:
                pos_total = max(0.0, agg_delta)
            positive_delta_total += max(0.0, pos_total)

            try:
                neg_total = float(aggregate.get("negative_delta_total", 0.0))
            except (TypeError, ValueError):
                neg_total = 0.0
            if not math.isfinite(neg_total) or neg_total > 0.0:
                neg_total = min(0.0, agg_delta)
            negative_delta_total += min(0.0, neg_total)

        existing_total = 0.0
        for sym, weight in self._weights.items():
            if sym in seen_symbols:
                continue
            try:
                weight_val = float(weight)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(weight_val) or weight_val <= 0.0:
                continue
            existing_total += weight_val

        available = float(cap) - existing_total - normalized_total
        desired_total = current_total + delta_total
        if available >= desired_total and available >= 0.0:
            return orders_list, False
        if positive_delta_total <= 0.0:
            return orders_list, False
        available = max(0.0, available)
        base_after_sells = max(0.0, current_total + negative_delta_total)
        space_for_delta = max(0.0, available - base_after_sells)
        if space_for_delta <= 0.0:
            factor = 0.0
        else:
            factor = min(1.0, max(0.0, space_for_delta / positive_delta_total))
        if math.isclose(factor, 1.0, rel_tol=1e-9):
            return orders_list, False

        normalization_payload: Dict[str, Any] = {
            "factor": factor,
            "cap": float(cap),
            "existing_total": existing_total,
            "normalized_total": normalized_total,
            "requested_total": requested_total,
            "available_total": available,
            "current_total": current_total,
            "delta_total": delta_total,
            "desired_total": desired_total,
        }
        normalization_payload["available_delta"] = space_for_delta
        normalization_payload["delta_positive_total"] = positive_delta_total
        normalization_payload["delta_negative_total"] = negative_delta_total
        equity = self._portfolio_equity
        if equity is not None and math.isfinite(equity):
            normalization_payload["cap_usd"] = float(cap) * float(equity)

        changed = False
        scaled_current: Dict[str, float] = {}
        for info in scalable_infos:
            symbol_init = info.get("symbol")
            if not symbol_init:
                continue
            if symbol_init in scaled_current:
                continue
            try:
                current_weight = float(info.get("current", 0.0))
            except (TypeError, ValueError):
                current_weight = 0.0
            if not math.isfinite(current_weight) or current_weight < 0.0:
                current_weight = 0.0
            scaled_current[symbol_init] = current_weight

        for info in scalable_infos:
            symbol = info["symbol"]
            if not symbol:
                continue
            order = info["order"]
            try:
                current_weight = float(
                    scaled_current.get(symbol, info.get("current", 0.0))
                )
            except (TypeError, ValueError):
                current_weight = 0.0
            if not math.isfinite(current_weight) or current_weight < 0.0:
                current_weight = 0.0
            try:
                original_target = float(info["target"])
            except (TypeError, ValueError):
                original_target = current_weight
            if not math.isfinite(original_target):
                original_target = current_weight
            try:
                requested_delta = float(info.get("delta", 0.0))
            except (TypeError, ValueError):
                requested_delta = 0.0
            if not math.isfinite(requested_delta):
                requested_delta = 0.0
            if requested_delta < 0.0:
                new_target = self._clamp_weight(original_target)
            elif requested_delta <= 0.0:
                new_target = self._clamp_weight(current_weight)
            else:
                new_target = self._clamp_weight(current_weight + requested_delta * factor)
            new_delta = new_target - current_weight
            order_factor = factor if requested_delta > 0.0 else 1.0
            scaled_current[symbol] = new_target
            payload_map = dict(info["payload"])
            payload_map["normalized"] = True
            payload_map["normalization"] = dict(normalization_payload)
            payload_map["target_weight"] = new_target
            payload_map["target"] = new_target
            payload_map["weight"] = new_target
            payload_map["delta_weight"] = new_delta
            payload_map["delta"] = new_delta
            self._scale_turnover_mapping(payload_map, order_factor)
            economics_map = self._coerce_mapping(payload_map.get("economics"))
            if economics_map is not None:
                payload_map["economics"] = (
                    self._scale_turnover_mapping(economics_map, order_factor)
                    or economics_map
                )
            decision_map = self._coerce_mapping(payload_map.get("decision"))
            if decision_map is not None:
                payload_map["decision"] = decision_map
                self._scale_turnover_mapping(decision_map, order_factor)
                decision_econ = self._coerce_mapping(decision_map.get("economics"))
                if decision_econ is not None:
                    decision_map["economics"] = (
                        self._scale_turnover_mapping(decision_econ, order_factor)
                        or decision_econ
                    )
            info["payload"] = payload_map
            info["target"] = new_target
            meta_map = self._ensure_order_meta(order)
            meta_map["payload"] = payload_map
            meta_map["normalized"] = True
            meta_map["normalization"] = dict(normalization_payload)
            meta_economics = self._coerce_mapping(meta_map.get("economics"))
            if meta_economics is not None:
                meta_map["economics"] = (
                    self._scale_turnover_mapping(meta_economics, order_factor)
                    or meta_economics
                )
            meta_decision = self._coerce_mapping(meta_map.get("decision"))
            if meta_decision is not None:
                self._scale_turnover_mapping(meta_decision, order_factor)
                decision_econ = self._coerce_mapping(meta_decision.get("economics"))
                if decision_econ is not None:
                    meta_decision["economics"] = (
                        self._scale_turnover_mapping(decision_econ, order_factor)
                        or decision_econ
                    )
                meta_map["decision"] = meta_decision
            self._pending_weight[id(order)] = {
                "symbol": symbol,
                "target_weight": new_target,
                "delta_weight": new_delta,
                "normalized": True,
                "factor": factor,
                "normalization": dict(normalization_payload),
            }
            try:
                self._pending_weight_refs[id(order)] = weakref.ref(order)
            except TypeError:
                self._pending_weight_refs[id(order)] = order
            changed = True

        if changed:
            try:
                log_payload = {
                    "cap": float(cap),
                    "existing_total": existing_total,
                    "normalized_total": normalized_total,
                    "requested_total": requested_total,
                    "available_total": available,
                    "current_total": current_total,
                    "delta_total": delta_total,
                    "desired_total": current_total + delta_total,
                    "available_delta": normalization_payload["available_delta"],
                    "factor": factor,
                    "symbols": {
                        symbol: data.get("requested", 0.0)
                        for symbol, data in scalable_by_symbol.items()
                    },
                }
                if equity is not None and math.isfinite(equity):
                    log_payload["cap_usd"] = float(cap) * float(equity)
                self._logger.info("WEIGHT_CAP_NORMALIZED %s", log_payload)
            except Exception:
                pass

        return orders_list, changed

    def _register_pending_exposure(self, order: Any, ts_ms: int) -> bool:
        if self._execution_mode == "bar":
            return False
        key = id(order)
        if key in self._pending_exposure:
            return False
        symbol = str(getattr(order, "symbol", "") or "").upper()
        if not symbol:
            return False
        qty = self._coerce_quantity(order)
        if qty is None or math.isclose(qty, 0.0, rel_tol=0.0, abs_tol=1e-12):
            return False
        side = self._order_side(order)
        if side == "BUY":
            delta = qty
        elif side == "SELL":
            delta = -qty
        else:
            return False
        prev = self._positions.get(symbol, 0.0)
        new = prev + delta
        if math.isclose(new, 0.0, rel_tol=0.0, abs_tol=1e-12):
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = new
        leg = self._signal_leg(order) or ("exit" if delta < 0.0 else "entry")
        self._pending_exposure[key] = {
            "symbol": symbol,
            "delta": float(delta),
            "leg": leg,
            "ts_ms": int(ts_ms),
        }
        return True

    def _stage_exposure_adjustments(
        self, orders: Sequence[Any], ts_ms: int
    ) -> None:
        if not orders:
            return
        if self._execution_mode == "bar":
            return
        exit_orders: list[Any] = []
        entry_orders: list[Any] = []
        other_orders: list[Any] = []
        for order in orders:
            leg = self._signal_leg(order)
            if leg == "exit":
                exit_orders.append(order)
            elif leg == "entry":
                entry_orders.append(order)
            else:
                other_orders.append(order)
        changed = False
        for order in (*exit_orders, *entry_orders, *other_orders):
            if self._register_pending_exposure(order, ts_ms):
                changed = True
        if changed:
            self._persist_exposure_state(ts_ms)

    def _rollback_exposure(self, order: Any) -> None:
        if self._execution_mode == "bar":
            self._pending_weight.pop(id(order), None)
            self._pending_weight_refs.pop(id(order), None)
            return
        key = id(order)
        data = self._pending_exposure.pop(key, None)
        if not data:
            return
        symbol = str(data.get("symbol", "")).upper()
        try:
            delta = float(data.get("delta", 0.0) or 0.0)
        except (TypeError, ValueError):
            delta = 0.0
        if symbol:
            prev = self._positions.get(symbol, 0.0)
            new = prev - delta
            if math.isclose(new, 0.0, rel_tol=0.0, abs_tol=1e-12):
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = new
        self._persist_exposure_state(clock.now_ms())

    def _commit_exposure(self, order: Any, *, bar_close_ms: int | None = None) -> None:
        if self._execution_mode == "bar":
            symbol = str(getattr(order, "symbol", "") or "").upper()
            if not symbol:
                return
            meta_map = self._ensure_order_meta(order)
            payload = self._extract_signal_payload(order)
            equity_override = self._resolve_order_equity(order, payload, symbol)
            target_weight, delta_weight, _ = self._resolve_weight_targets(
                symbol, payload
            )
            exec_meta = self._coerce_mapping(meta_map.get("_bar_execution"))
            try:
                self._pending_weight_refs.setdefault(id(order), weakref.ref(order))
            except TypeError:
                self._pending_weight_refs[id(order)] = order
            pending = self._pending_weight.get(id(order))
            if not isinstance(pending, dict):
                pending = {"symbol": symbol}
                self._pending_weight[id(order)] = pending
            pending.setdefault("symbol", symbol)
            if "target_weight" not in pending:
                pending["target_weight"] = target_weight
            if "delta_weight" not in pending:
                pending["delta_weight"] = delta_weight
            if bar_close_ms is not None:
                try:
                    pending["bar_close_ms"] = int(bar_close_ms)
                except (TypeError, ValueError, OverflowError):
                    pass
            if not exec_meta:
                return
            executed_flag: bool | None = None
            executed_turnover: float | None = None
            executed_target: float | None = None
            executed_delta: float | None = None
            if exec_meta:
                executed_turnover = self._coerce_float(exec_meta.get("turnover_usd"))
                executed_target = self._coerce_float(exec_meta.get("target_weight"))
                executed_delta = self._coerce_float(exec_meta.get("delta_weight"))

                def _interpret_bool(value: Any) -> bool | None:
                    if value is None:
                        return None
                    if isinstance(value, bool):
                        return value
                    if isinstance(value, str):
                        candidate = value.strip().lower()
                        if candidate in {"", "0", "false", "no", "off"}:
                            return False
                        if candidate in {"1", "true", "yes", "on"}:
                            return True
                        return None
                    if isinstance(value, (int, float)):
                        return bool(value)
                    return bool(value)

                executed_flag = _interpret_bool(exec_meta.get("filled"))
                if executed_flag is None:
                    executed_flag = _interpret_bool(exec_meta.get("executed"))
                if executed_flag is None:
                    instructions_payload = exec_meta.get("instructions")
                    executed_flag = bool(instructions_payload)
                if executed_flag is None and executed_turnover is not None:
                    executed_flag = executed_turnover > 0.0
            executed = executed_flag if executed_flag is not None else True
            prev_weight = self._weights.get(symbol, 0.0)
            self._pending_weight.pop(id(order), None)
            self._pending_weight_refs.pop(id(order), None)
            if not executed:
                meta_map.pop("_daily_turnover_usd", None)
                return
            applied_target = (
                self._clamp_weight(executed_target)
                if executed_target is not None
                else target_weight
            )
            if math.isclose(applied_target, 0.0, rel_tol=0.0, abs_tol=1e-12):
                self._weights.pop(symbol, None)
            else:
                self._weights[symbol] = applied_target
            applied_delta = (
                executed_delta
                if executed_delta is not None
                else applied_target - prev_weight
            )
            if equity_override is not None and equity_override > 0.0:
                self._symbol_equity[symbol] = float(equity_override)
            self._apply_weight_to_positions(
                symbol, applied_target, equity_override=equity_override
            )
            turnover_to_record: float | None = executed_turnover
            if turnover_to_record is not None:
                if turnover_to_record <= 0.0:
                    meta_map.pop("_daily_turnover_usd", None)
                else:
                    meta_map["_daily_turnover_usd"] = float(turnover_to_record)
            decision_meta = self._coerce_mapping(meta_map.get("decision"))
            if decision_meta is not None and turnover_to_record is not None:
                decision_meta["turnover_usd"] = float(max(0.0, turnover_to_record))
                meta_map["decision"] = decision_meta
            self._record_daily_turnover_execution(order)
            self._persist_exposure_state(clock.now_ms())
            self._maybe_start_symbol_cooldown(
                symbol,
                float(applied_delta),
                bar_close_ms=bar_close_ms,
            )
            return
        key = id(order)
        if key not in self._pending_exposure:
            return
        self._pending_exposure.pop(key, None)
        self._persist_exposure_state(clock.now_ms())

    def _rollback_pending_exposure_for_symbol(
        self, symbol: str, *, legs: set[str] | None = None
    ) -> None:
        sym = str(symbol).upper()
        if not sym:
            return
        keys: list[int] = []
        for key, data in list(self._pending_exposure.items()):
            if str(data.get("symbol", "")).upper() != sym:
                continue
            leg = str(data.get("leg", "unknown") or "unknown")
            if legs is not None and leg not in legs:
                continue
            keys.append(key)
        if not keys:
            return
        for key in keys:
            data = self._pending_exposure.pop(key, None)
            if not data:
                continue
            try:
                delta = float(data.get("delta", 0.0) or 0.0)
            except (TypeError, ValueError):
                delta = 0.0
            prev = self._positions.get(sym, 0.0)
            new = prev - delta
            if math.isclose(new, 0.0, rel_tol=0.0, abs_tol=1e-12):
                self._positions.pop(sym, None)
            else:
                self._positions[sym] = new
        self._persist_exposure_state(clock.now_ms())

    @staticmethod
    def _extract_entry_config_candidate(obj: Any) -> tuple[int | None, int | None]:
        if obj is None:
            return None, None
        limit: Any = None
        reset: Any = None
        if isinstance(obj, Mapping):
            limit = obj.get("max_entries_per_day")
            reset = obj.get("daily_reset_utc_hour")
        else:
            if hasattr(obj, "max_entries_per_day"):
                limit = getattr(obj, "max_entries_per_day")
            if hasattr(obj, "daily_reset_utc_hour"):
                reset = getattr(obj, "daily_reset_utc_hour")
        if limit is None and reset is None:
            return None, None
        return (
            DailyEntryLimiter.normalize_limit(limit),
            DailyEntryLimiter.normalize_hour(reset),
        )

    def _resolve_entry_limiter_config(self, executor: Any) -> _EntryLimiterConfig:
        cfg = _EntryLimiterConfig()
        seen: set[int] = set()
        stack: list[Any] = [executor]
        while stack:
            obj = stack.pop()
            if obj is None:
                continue
            obj_id = id(obj)
            if obj_id in seen:
                continue
            seen.add(obj_id)
            for candidate in (
                obj,
                getattr(obj, "cfg", None),
                getattr(obj, "config", None),
            ):
                limit_val, reset_val = self._extract_entry_config_candidate(candidate)
                if limit_val is not None and cfg.limit is None:
                    cfg.limit = limit_val
                if reset_val is not None:
                    cfg.reset_hour = reset_val
                if cfg.limit is not None:
                    break
            if cfg.limit is not None:
                break
            for name in ("risk", "_risk", "manager", "_manager", "sim", "_sim"):
                try:
                    attr_val = getattr(obj, name)
                except Exception:
                    attr_val = None
                if attr_val is None:
                    continue
                if id(attr_val) in seen:
                    continue
                stack.append(attr_val)
        return cfg

    def _resolve_daily_turnover_caps(self, executor: Any) -> Any:
        caps = getattr(executor, "turnover_caps", None)
        if caps is None:
            return None
        if isinstance(caps, SpotTurnoverCaps):
            try:
                return caps.resolve()
            except Exception:
                return None
        if isinstance(caps, MappingABC):
            try:
                parsed = SpotTurnoverCaps.parse_obj(caps)
            except Exception:
                parsed = None
            if parsed is not None:
                try:
                    return parsed.resolve()
                except Exception:
                    return None
        resolve = getattr(caps, "resolve", None)
        if callable(resolve):
            try:
                return resolve()
            except Exception:
                return None
        return None

    def _daily_limit_values(self) -> tuple[Optional[float], Optional[float]]:
        caps = self._daily_turnover_caps
        if caps is None:
            return None, None
        equity = self._portfolio_equity
        try:
            equity_val = float(equity) if equity is not None else 0.0
        except (TypeError, ValueError):
            equity_val = 0.0
        symbol_limit: Optional[float]
        portfolio_limit: Optional[float]
        try:
            symbol_limit = caps.per_symbol.daily_limit_for_equity(equity_val)
        except Exception:
            symbol_limit = None
        try:
            portfolio_limit = caps.portfolio.daily_limit_for_equity(equity_val)
        except Exception:
            portfolio_limit = None
        return symbol_limit, portfolio_limit

    def _current_day_key(self, ts_ms: int) -> str:
        try:
            ts_val = int(ts_ms)
        except (TypeError, ValueError):
            ts_val = 0
        return daily_reset_key(ts_val, self._daily_reset_hour)

    def _daily_symbol_tracker(self, symbol: str, day_key: str) -> Dict[str, Any]:
        tracker = self._daily_symbol_turnover.setdefault(
            symbol, {"day": day_key, "total": 0.0}
        )
        if tracker.get("day") != day_key:
            tracker["day"] = day_key
            tracker["total"] = 0.0
        return tracker

    def _daily_portfolio_tracker(self, day_key: str) -> Dict[str, Any]:
        tracker = self._daily_portfolio_turnover
        if tracker.get("day") != day_key:
            tracker["day"] = day_key
            tracker["total"] = 0.0
        return tracker

    def _daily_turnover_state_payload(self) -> Dict[str, Any]:
        per_symbol: Dict[str, Dict[str, Any]] = {}
        for sym, tracker in self._daily_symbol_turnover.items():
            if not sym:
                continue
            day = tracker.get("day")
            if day is not None:
                try:
                    day = str(day)
                except Exception:
                    day = None
            try:
                used = float(tracker.get("total", 0.0) or 0.0)
            except (TypeError, ValueError):
                used = 0.0
            per_symbol[sym] = {"day_key": day, "used_usd": max(0.0, used)}
        portfolio_day = self._daily_portfolio_turnover.get("day")
        if portfolio_day is not None:
            try:
                portfolio_day = str(portfolio_day)
            except Exception:
                portfolio_day = None
        try:
            portfolio_used = float(
                self._daily_portfolio_turnover.get("total", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            portfolio_used = 0.0
        return {
            "day_key": self._daily_day_key,
            "per_symbol": per_symbol,
            "portfolio": {
                "day_key": portfolio_day,
                "used_usd": max(0.0, portfolio_used),
            },
        }

    def _daily_turnover_snapshot(self) -> Dict[str, Any]:
        symbol_limit, portfolio_limit = self._daily_limit_values()
        snapshot: Dict[str, Any] = {
            "day_key": self._daily_day_key,
            "per_symbol": {},
            "portfolio": {},
        }
        portfolio_state = self._daily_turnover_state_payload().get("portfolio", {})
        portfolio_day = portfolio_state.get("day_key")
        portfolio_used = float(portfolio_state.get("used_usd", 0.0) or 0.0)
        remaining_portfolio: Optional[float] = None
        if portfolio_limit is not None:
            if portfolio_day is None or portfolio_day == self._daily_day_key:
                remaining_portfolio = max(0.0, float(portfolio_limit) - portfolio_used)
            else:
                remaining_portfolio = float(portfolio_limit)
        snapshot["portfolio"] = {
            "day_key": portfolio_day,
            "used_usd": portfolio_used,
            "limit_usd": portfolio_limit,
            "remaining_usd": remaining_portfolio,
        }
        per_symbol: Dict[str, Dict[str, Any]] = {}
        for sym in sorted(self._daily_symbol_turnover.keys()):
            tracker = self._daily_symbol_turnover.get(sym, {})
            tracker_day = tracker.get("day")
            if tracker_day is not None:
                try:
                    tracker_day = str(tracker_day)
                except Exception:
                    tracker_day = None
            try:
                used_val = float(tracker.get("total", 0.0) or 0.0)
            except (TypeError, ValueError):
                used_val = 0.0
            remaining_symbol: Optional[float] = None
            if symbol_limit is not None:
                if tracker_day is None or tracker_day == self._daily_day_key:
                    remaining_symbol = max(0.0, float(symbol_limit) - used_val)
                else:
                    remaining_symbol = float(symbol_limit)
            per_symbol[sym] = {
                "day_key": tracker_day,
                "used_usd": used_val,
                "limit_usd": symbol_limit,
                "remaining_usd": remaining_symbol,
            }
        snapshot["per_symbol"] = per_symbol
        return snapshot

    def _restore_daily_turnover_state(self, payload: Mapping[str, Any]) -> None:
        day_key = payload.get("day_key")
        if day_key is not None:
            try:
                self._daily_day_key = str(day_key)
            except Exception:
                self._daily_day_key = None
        per_symbol = payload.get("per_symbol")
        if isinstance(per_symbol, MappingABC):
            for sym, data in per_symbol.items():
                try:
                    sym_key = str(sym).upper()
                except Exception:
                    sym_key = ""
                if not sym_key or not isinstance(data, MappingABC):
                    continue
                tracker_day = data.get("day_key")
                if tracker_day is not None:
                    try:
                        tracker_day = str(tracker_day)
                    except Exception:
                        tracker_day = None
                try:
                    used_val = float(data.get("used_usd", 0.0) or 0.0)
                except (TypeError, ValueError):
                    used_val = 0.0
                self._daily_symbol_turnover[sym_key] = {
                    "day": tracker_day,
                    "total": max(0.0, used_val),
                }
        portfolio = payload.get("portfolio")
        if isinstance(portfolio, MappingABC):
            tracker_day = portfolio.get("day_key")
            if tracker_day is not None:
                try:
                    tracker_day = str(tracker_day)
                except Exception:
                    tracker_day = None
            try:
                used_val = float(portfolio.get("used_usd", 0.0) or 0.0)
            except (TypeError, ValueError):
                used_val = 0.0
            self._daily_portfolio_turnover = {
                "day": tracker_day,
                "total": max(0.0, used_val),
            }

    def _extract_order_turnover(self, order: Any) -> float:
        meta = getattr(order, "meta", None)
        mappings: list[Mapping[str, Any]] = []
        if isinstance(meta, MappingABC):
            mappings.append(meta)
            decision = meta.get("decision")
            if isinstance(decision, MappingABC):
                mappings.append(decision)
        payload = self._extract_signal_payload(order)
        if payload:
            mappings.append(payload)
        turnover_keys = ("turnover_usd", "turnover", "notional_usd")
        for mapping in mappings:
            if not isinstance(mapping, MappingABC):
                continue
            for key in turnover_keys:
                if key not in mapping:
                    continue
                parsed = self._coerce_float(mapping.get(key))
                if parsed is not None:
                    return max(0.0, parsed)
            economics = mapping.get("economics")
            if isinstance(economics, MappingABC):
                for key in turnover_keys:
                    if key not in economics:
                        continue
                    parsed = self._coerce_float(economics.get(key))
                    if parsed is not None:
                        return max(0.0, parsed)
                nested_value = self._find_in_mapping(economics, turnover_keys)
                if nested_value is not None:
                    parsed = self._coerce_float(nested_value)
                    if parsed is not None:
                        return max(0.0, parsed)
        for mapping in mappings:
            if not isinstance(mapping, MappingABC):
                continue
            nested_value = self._find_in_mapping(mapping, turnover_keys)
            if nested_value is None:
                continue
            parsed = self._coerce_float(nested_value)
            if parsed is not None:
                return max(0.0, parsed)
        return 0.0

    def _resolve_order_equity(
        self,
        order: Any | None,
        payload: Mapping[str, Any] | None,
        symbol: str | None = None,
    ) -> float | None:
        cached_equity: float | None = None
        if symbol:
            try:
                sym_key = str(symbol).upper()
            except Exception:
                sym_key = ""
            if sym_key:
                cached = self._symbol_equity.get(sym_key)
                if cached is not None and cached > 0.0:
                    cached_equity = float(cached)
        mappings: list[Mapping[str, Any]] = []
        meta = getattr(order, "meta", None) if order is not None else None
        if isinstance(meta, MappingABC):
            mappings.append(meta)
            decision = meta.get("decision")
            if isinstance(decision, MappingABC):
                mappings.append(decision)
        elif meta is not None:
            materialized = self._materialize_mapping(meta)
            if materialized:
                mappings.append(materialized)
                decision = materialized.get("decision")
                if isinstance(decision, MappingABC):
                    mappings.append(decision)
        if isinstance(payload, MappingABC) and payload:
            mappings.append(payload)

        for mapping in mappings:
            if not isinstance(mapping, MappingABC):
                continue
            candidate = mapping.get("equity_usd")
            if candidate is None:
                candidate = self._find_in_mapping(mapping, ("equity_usd",))
            parsed = self._coerce_float(candidate)
            if parsed is None:
                continue
            if not math.isfinite(parsed) or parsed <= 0.0:
                continue
            return float(parsed)
        equity = self._portfolio_equity
        if equity is not None:
            try:
                equity_val = float(equity)
            except (TypeError, ValueError):
                equity_val = None
            else:
                if math.isfinite(equity_val) and equity_val > 0.0:
                    return equity_val
        return cached_equity

    def _mutate_order_payload(self, order: Any, updates: Mapping[str, Any]) -> None:
        if not updates:
            return
        meta = self._ensure_order_meta(order)
        payload = meta.get("payload")
        if isinstance(payload, MappingABC):
            payload_dict = dict(payload)
        else:
            payload_dict = {}
        payload_dict.update(updates)
        meta["payload"] = payload_dict

    def _annotate_order_daily(
        self,
        order: Any,
        *,
        requested_usd: float,
        executed_usd: float,
        day_key: str,
        clamped: bool,
        headroom_before: Optional[float],
    ) -> None:
        meta = self._ensure_order_meta(order)
        executed_val = float(max(0.0, executed_usd))
        requested_val = float(max(0.0, requested_usd))
        meta["_daily_turnover_usd"] = executed_val
        meta["_daily_turnover_day"] = day_key
        daily_info = meta.get("daily_turnover")
        if isinstance(daily_info, MappingABC):
            info_dict = dict(daily_info)
        else:
            info_dict = {}
        info_dict.update(
            {
                "requested_usd": requested_val,
                "executed_usd": executed_val,
                "day_key": day_key,
                "clamped": bool(clamped),
            }
        )
        if headroom_before is not None:
            info_dict["headroom_before_usd"] = float(max(0.0, headroom_before))
        meta["daily_turnover"] = info_dict
        decision = meta.get("decision")
        if isinstance(decision, MappingABC):
            decision_dict = dict(decision)
        else:
            decision_dict = {}
        decision_dict["turnover_usd"] = executed_val
        if clamped:
            decision_dict["daily_turnover_clamped"] = True
        meta["decision"] = decision_dict

    def _scale_order_for_turnover(
        self,
        order: Any,
        symbol: str,
        scale: float,
        *,
        current_weight: float | None = None,
    ) -> bool:
        if scale <= 0.0:
            return False
        if scale >= 1.0:
            return True
        payload = self._extract_signal_payload(order)
        if current_weight is None:
            base_weight = self._weights.get(symbol, 0.0)
        else:
            base_weight = current_weight
        try:
            current = float(base_weight)
        except (TypeError, ValueError):
            current = 0.0
        if not math.isfinite(current):
            current = 0.0
        target_weight, delta_weight, _ = self._resolve_weight_targets(
            symbol, payload, current_weight=current
        )
        if math.isclose(delta_weight, 0.0, rel_tol=0.0, abs_tol=1e-12):
            return False
        new_delta = delta_weight * scale
        new_target = current + new_delta
        if not math.isfinite(new_target):
            return False
        if new_target < 0.0:
            new_target = 0.0
        elif new_target > 1.0:
            new_target = 1.0
        updates: Dict[str, Any] = {}
        for key in ("target_weight", "target", "weight"):
            if key in payload:
                updates[key] = new_target
        for key in ("delta_weight", "delta"):
            if key in payload:
                updates[key] = new_delta
        if not updates:
            updates["target_weight"] = new_target
        self._mutate_order_payload(order, updates)
        meta_map = self._ensure_order_meta(order)
        payload_map = self._coerce_mapping(meta_map.get("payload"))
        if payload_map is None:
            payload_map = {}
            meta_map["payload"] = payload_map
        self._scale_turnover_mapping(payload_map, scale)
        normalization_payload = self._coerce_mapping(payload_map.get("normalization"))
        if normalization_payload is not None:
            payload_map["normalization"] = (
                self._scale_normalization_metadata(normalization_payload, scale)
                or normalization_payload
            )
        economics_map = self._coerce_mapping(payload_map.get("economics"))
        if economics_map is not None:
            payload_map["economics"] = (
                self._scale_turnover_mapping(economics_map, scale) or economics_map
            )
        decision_map = self._coerce_mapping(payload_map.get("decision"))
        if decision_map is not None:
            payload_map["decision"] = decision_map
            self._scale_turnover_mapping(decision_map, scale)
            decision_norm = self._coerce_mapping(decision_map.get("normalization"))
            if decision_norm is not None:
                decision_map["normalization"] = (
                    self._scale_normalization_metadata(decision_norm, scale)
                    or decision_norm
                )
            decision_econ = self._coerce_mapping(decision_map.get("economics"))
            if decision_econ is not None:
                decision_map["economics"] = (
                    self._scale_turnover_mapping(decision_econ, scale)
                    or decision_econ
                )
        meta_economics = self._coerce_mapping(meta_map.get("economics"))
        if meta_economics is not None:
            meta_map["economics"] = (
                self._scale_turnover_mapping(meta_economics, scale) or meta_economics
            )
        meta_decision = self._coerce_mapping(meta_map.get("decision"))
        if meta_decision is not None:
            self._scale_turnover_mapping(meta_decision, scale)
            meta_decision_norm = self._coerce_mapping(meta_decision.get("normalization"))
            if meta_decision_norm is not None:
                meta_decision["normalization"] = (
                    self._scale_normalization_metadata(meta_decision_norm, scale)
                    or meta_decision_norm
                )
            decision_econ = self._coerce_mapping(meta_decision.get("economics"))
            if decision_econ is not None:
                meta_decision["economics"] = (
                    self._scale_turnover_mapping(decision_econ, scale)
                    or decision_econ
                )
            meta_map["decision"] = meta_decision
        meta_normalization = self._coerce_mapping(meta_map.get("normalization"))
        if meta_normalization is not None:
            meta_map["normalization"] = (
                self._scale_normalization_metadata(meta_normalization, scale)
                or meta_normalization
            )
        pending = self._pending_weight.get(id(order))
        if isinstance(pending, dict):
            pending["target_weight"] = new_target
            pending["delta_weight"] = new_delta
            if "factor" in pending:
                try:
                    pending_factor = float(pending.get("factor"))
                except (TypeError, ValueError):
                    pending_factor = None
                if pending_factor is not None and math.isfinite(pending_factor):
                    pending["factor"] = pending_factor * scale
            pending_norm = self._coerce_mapping(pending.get("normalization"))
            if pending_norm is not None:
                pending["normalization"] = (
                    self._scale_normalization_metadata(pending_norm, scale)
                    or pending_norm
                )
        return True

    def _apply_daily_turnover_limits(
        self, orders: Sequence[Any], symbol: str, ts_ms: int
    ) -> list[Any]:
        if self._execution_mode != "bar":
            return list(orders)
        sym = str(symbol).upper()
        if not sym:
            return list(orders)
        day_key = self._current_day_key(ts_ms)
        self._daily_day_key = day_key
        symbol_tracker = self._daily_symbol_tracker(sym, day_key)
        portfolio_tracker = self._daily_portfolio_tracker(day_key)
        symbol_limit, portfolio_limit = self._daily_limit_values()
        try:
            symbol_used = float(symbol_tracker.get("total", 0.0) or 0.0)
        except (TypeError, ValueError):
            symbol_used = 0.0
        try:
            portfolio_used = float(portfolio_tracker.get("total", 0.0) or 0.0)
        except (TypeError, ValueError):
            portfolio_used = 0.0
        try:
            working_weight = float(self._weights.get(sym, 0.0))
        except (TypeError, ValueError):
            working_weight = 0.0
        if not math.isfinite(working_weight):
            working_weight = 0.0
        adjusted: list[Any] = []
        for order in orders:
            order_id = id(order)
            payload = self._extract_signal_payload(order)
            target_weight, delta_weight, _ = self._resolve_weight_targets(
                sym, payload, current_weight=working_weight
            )
            requested = self._extract_order_turnover(order)
            requested_non_finite = not math.isfinite(requested)
            if requested_non_finite:
                requested = 0.0
            if requested <= 0.0 and not requested_non_finite:
                delta = abs(float(delta_weight))
                if delta > 0.0:
                    equity = self._resolve_order_equity(order, payload, sym)
                    if equity is not None and equity > 0.0:
                        requested = delta * float(equity)
                        if not math.isfinite(requested):
                            requested = 0.0
            headroom_candidates: list[float] = []
            if symbol_limit is not None:
                headroom_candidates.append(max(0.0, float(symbol_limit) - symbol_used))
            if portfolio_limit is not None:
                headroom_candidates.append(max(0.0, float(portfolio_limit) - portfolio_used))
            headroom = min(headroom_candidates) if headroom_candidates else None
            if headroom is not None and headroom <= 1e-9:
                self._pending_weight.pop(order_id, None)
                self._pending_weight_refs.pop(order_id, None)
                try:
                    self._logger.info(
                        "DAILY_TURNOVER_DEFER %s",
                        {
                            "symbol": sym,
                            "requested_usd": requested,
                            "day_key": day_key,
                            "headroom_usd": headroom,
                        },
                    )
                except Exception:
                    pass
                continue
            executed = requested
            if not math.isfinite(executed):
                executed = 0.0
            clamped = False
            if headroom is not None and requested > headroom + 1e-9:
                if not self._scale_order_for_turnover(
                    order, sym, headroom / requested, current_weight=working_weight
                ):
                    self._pending_weight.pop(order_id, None)
                    self._pending_weight_refs.pop(order_id, None)
                    try:
                        self._logger.info(
                            "DAILY_TURNOVER_DEFER %s",
                            {
                                "symbol": sym,
                                "requested_usd": requested,
                                "day_key": day_key,
                                "headroom_usd": headroom,
                                "reason": "scale_failed",
                            },
                        )
                    except Exception:
                        pass
                    continue
                executed = headroom
                clamped = True
                payload = self._extract_signal_payload(order)
                target_weight, delta_weight, _ = self._resolve_weight_targets(
                    sym, payload, current_weight=working_weight
                )
                if not math.isfinite(executed):
                    executed = 0.0
                try:
                    self._logger.info(
                        "DAILY_TURNOVER_CLAMP %s",
                        {
                            "symbol": sym,
                            "requested_usd": requested,
                            "executed_usd": executed,
                            "day_key": day_key,
                        },
                    )
                except Exception:
                    pass
            symbol_used += executed
            portfolio_used += executed
            self._annotate_order_daily(
                order,
                requested_usd=requested,
                executed_usd=executed,
                day_key=day_key,
                clamped=clamped,
                headroom_before=headroom,
            )
            adjusted.append(order)
            working_weight = target_weight
        return adjusted

    def _replay_pending_weight_entries(
        self,
        order: Any | None = None,
        *,
        exclude: set[int] | None = None,
    ) -> bool:
        if self._execution_mode != "bar":
            return False
        exclude_ids: set[int] = set(exclude or ())
        if order is not None:
            candidate_ids = [id(order)]
        else:
            candidate_ids = [
                key for key in list(self._pending_weight_refs.keys()) if key not in exclude_ids
            ]
        processed = False
        for key in candidate_ids:
            if key in exclude_ids:
                continue
            order_obj: Any | None
            if order is not None and key == id(order):
                order_obj = order
            else:
                ref = self._pending_weight_refs.get(key)
                if ref is None:
                    self._pending_weight.pop(key, None)
                    continue
                order_obj = ref() if callable(ref) else ref
            if order_obj is None:
                self._pending_weight.pop(key, None)
                self._pending_weight_refs.pop(key, None)
                continue
            meta_map = self._ensure_order_meta(order_obj)
            exec_meta = self._coerce_mapping(meta_map.get("_bar_execution"))
            if not exec_meta:
                continue
            pending_payload = self._pending_weight.get(key)
            bar_close_ms: int | None = None
            if isinstance(pending_payload, dict):
                bar_close_candidate = pending_payload.get("bar_close_ms")
                try:
                    if bar_close_candidate is not None:
                        bar_close_ms = int(bar_close_candidate)
                except (TypeError, ValueError, OverflowError):
                    bar_close_ms = None
            self._commit_exposure(order_obj, bar_close_ms=bar_close_ms)
            processed = True
        return processed

    def _record_daily_turnover_execution(self, order: Any) -> None:
        if self._execution_mode != "bar":
            return
        meta = getattr(order, "meta", None)
        if not isinstance(meta, MappingABC):
            return
        exec_meta = self._coerce_mapping(meta.get("_bar_execution"))
        executed_from_exec = None
        if exec_meta:
            executed_from_exec = self._coerce_float(exec_meta.get("turnover_usd"))
        executed = self._coerce_float(meta.get("_daily_turnover_usd"))
        if executed_from_exec is not None:
            if executed_from_exec <= 0.0:
                try:
                    meta.pop("_daily_turnover_usd", None)
                except Exception:
                    pass
                return
            executed = executed_from_exec
            meta["_daily_turnover_usd"] = executed_from_exec
        if executed is None or executed <= 0.0:
            return
        day_key_val = meta.get("_daily_turnover_day")
        if isinstance(day_key_val, str):
            day_key = day_key_val
        else:
            try:
                day_key = str(day_key_val)
            except Exception:
                day_key = None
        if not day_key:
            day_key = self._daily_day_key or self._current_day_key(clock.now_ms())
        symbol = str(getattr(order, "symbol", "") or "").upper()
        if not symbol:
            return
        tracker = self._daily_symbol_tracker(symbol, day_key)
        try:
            tracker_total = float(tracker.get("total", 0.0) or 0.0)
        except (TypeError, ValueError):
            tracker_total = 0.0
        tracker["total"] = tracker_total + float(executed)
        portfolio_tracker = self._daily_portfolio_tracker(day_key)
        try:
            portfolio_total = float(portfolio_tracker.get("total", 0.0) or 0.0)
        except (TypeError, ValueError):
            portfolio_total = 0.0
        portfolio_tracker["total"] = portfolio_total + float(executed)
        self._daily_day_key = day_key
        try:
            meta.pop("_daily_turnover_usd", None)
            meta.pop("_daily_turnover_day", None)
        except Exception:
            pass
        if exec_meta:
            self._replay_pending_weight_entries(exclude={id(order)})

    @staticmethod
    def _signal_leg(order: Any) -> str:
        meta = getattr(order, "meta", None)
        value: Any = None
        if isinstance(meta, Mapping):
            value = meta.get("signal_leg")
        elif meta is not None:
            getter = getattr(meta, "get", None)
            if callable(getter):
                try:
                    value = getter("signal_leg")
                except Exception:
                    value = None
        return str(value or "").lower()

    @staticmethod
    def _format_signal_state(state: Any) -> str:
        if isinstance(state, SignalPosition):
            return state.value
        if state is None:
            return ""
        return str(state)

    def _remove_entry_orders(
        self, orders: Sequence[Any], symbol: str
    ) -> tuple[list[Any], list[Any]]:
        remaining: list[Any] = []
        removed: list[Any] = []
        for order in orders:
            try:
                order_symbol = getattr(order, "symbol", None)
            except Exception:
                order_symbol = None
            if str(order_symbol or "") != symbol:
                remaining.append(order)
                continue
            leg = self._signal_leg(order)
            if leg == "entry":
                removed.append(order)
                continue
            remaining.append(order)
        return remaining, removed

    def _handle_entry_limit_refusal(
        self,
        symbol: str,
        ts_ms: int,
        transition: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        *,
        entry_steps: int,
        removed_count: int,
        removed_orders: Sequence[Any] | None = None,
    ) -> None:
        prev_state = transition.get("prev")
        new_state = transition.get("new")
        revert_target: Any = prev_state
        if (
            prev_state not in (None, SignalPosition.FLAT)
            and new_state not in (None, SignalPosition.FLAT)
            and self._format_signal_state(prev_state)
            != self._format_signal_state(new_state)
        ):
            # For reversals we keep the exit leg to flatten the position but block the
            # subsequent entry, thus the signal state is rolled back to flat.
            revert_target = SignalPosition.FLAT
        if revert_target is None:
            revert_target = SignalPosition.FLAT
        try:
            self._policy.revert_signal_state(symbol, revert_target)
        except Exception:
            pass
        payload = {
            "symbol": symbol,
            "ts_ms": int(ts_ms),
            "prev": self._format_signal_state(prev_state),
            "new": self._format_signal_state(new_state),
            "revert_to": self._format_signal_state(revert_target),
            "entry_steps": int(entry_steps),
            "removed_orders": int(removed_count),
        }
        try:
            for key in ("limit", "entries_today", "day_key", "reset_hour"):
                if key in snapshot:
                    payload[key] = snapshot[key]
        except Exception:
            pass
        try:
            self._logger.info("ENTRY_LIMIT %s", payload)
        except Exception:
            pass
        try:
            monitoring.entry_limiter_block_count.labels(symbol).inc()
        except Exception:
            pass
        try:
            pipeline_stage_drop_count.labels(
                symbol,
                Stage.RISK.name,
                "ENTRY_LIMIT",
            ).inc()
        except Exception:
            pass
        if removed_orders:
            for order in removed_orders:
                try:
                    self._rollback_exposure(order)
                except Exception:
                    continue
        self._rollback_pending_exposure_for_symbol(symbol, legs={"entry"})

    def _apply_entry_limiter(
        self,
        orders: Sequence[Any],
        transitions: Sequence[Mapping[str, Any]],
        *,
        default_symbol: str,
        ts_ms: int,
    ) -> list[Any]:
        if not getattr(self._entry_limiter, "enabled", False):
            return list(orders)
        filtered = list(orders)
        for transition in transitions:
            try:
                raw_steps = transition.get("entry_steps", 0)
            except AttributeError:
                raw_steps = 0
            try:
                steps = int(raw_steps)
            except (TypeError, ValueError):
                steps = 0
            if steps <= 0:
                continue
            symbol = str(transition.get("symbol") or default_symbol or "")
            if not symbol:
                continue
            with self._entry_lock:
                allowed = self._entry_limiter.allow(
                    symbol, ts_ms, entry_steps=steps
                )
                snapshot = (
                    self._entry_limiter.snapshot(symbol) if not allowed else {}
                )
                if (
                    allowed
                    and steps > 0
                    and self._state_enabled
                    and self._entry_limiter.enabled
                ):
                    try:
                        state_storage.update_state(
                            entry_limits=self._entry_limiter.export_state()
                        )
                    except Exception:
                        pass
            if allowed:
                continue
            filtered, removed = self._remove_entry_orders(filtered, symbol)
            self._handle_entry_limit_refusal(
                symbol,
                ts_ms,
                transition,
                snapshot,
                entry_steps=steps,
                removed_count=len(removed),
                removed_orders=removed,
            )
        return filtered

    def prewarm_dynamic_guard(
        self,
        history: Mapping[str, Sequence[Bar]]
        | Sequence[Bar]
        | Sequence[tuple[str, Sequence[Bar]]]
        | None,
    ) -> None:
        """Seed the dynamic guard with historical bars if available."""

        if self._dynamic_guard is None or not history:
            return

        grouped: Dict[str, list[tuple[Bar, float | None]]] = {}

        def _coerce_spread(value: object) -> float | None:
            try:
                spread_val = float(value)
            except Exception:
                return None
            if not math.isfinite(spread_val) or spread_val <= 0.0:
                return None
            return spread_val

        def _infer_spread_from_bar(bar: Bar) -> float | None:
            for attr, multiplier in (("spread_bps", 1.0), ("half_spread_bps", 2.0)):
                raw = getattr(bar, attr, None)
                if raw is None:
                    continue
                try:
                    val = float(raw)
                except Exception:
                    continue
                if math.isfinite(val) and val > 0.0:
                    return val * multiplier
            bid = getattr(bar, "bid", None) or getattr(bar, "bid_price", None)
            ask = getattr(bar, "ask", None) or getattr(bar, "ask_price", None)
            try:
                if bid is not None and ask is not None:
                    bid_val = float(bid)
                    ask_val = float(ask)
                    if (
                        math.isfinite(bid_val)
                        and math.isfinite(ask_val)
                        and bid_val > 0.0
                        and ask_val > 0.0
                    ):
                        mid = (bid_val + ask_val) * 0.5
                        if mid > 0.0:
                            spread = (ask_val - bid_val) / mid * 10000.0
                            if math.isfinite(spread) and spread > 0.0:
                                return spread
            except Exception:
                pass
            try:
                high_val = float(getattr(bar, "high", float("nan")))
                low_val = float(getattr(bar, "low", float("nan")))
            except Exception:
                return None
            if not (math.isfinite(high_val) and math.isfinite(low_val)):
                return None
            span = high_val - low_val
            if span <= 0.0:
                return None
            mid = (high_val + low_val) * 0.5
            if mid == 0.0 or not math.isfinite(mid):
                return None
            spread = span / abs(mid) * 10000.0
            if spread <= 0.0 or not math.isfinite(spread):
                return None
            return spread

        def _append(sym: str, bar: Bar, spread: float | None) -> None:
            grouped.setdefault(sym, []).append((bar, spread))

        try:
            if isinstance(history, MappingABC):
                iterable = history.items()
            elif isinstance(history, SequenceABC) and not isinstance(history, (str, bytes)):
                iterable = history  # type: ignore[assignment]
            else:
                iterable = list(history)  # type: ignore[arg-type]

            for item in iterable:
                if isinstance(item, tuple) and len(item) == 2:
                    symbol, bars = item
                    sym = str(symbol).upper()
                    seq = list(bars or [])
                elif isinstance(item, Bar):
                    sym = str(getattr(item, "symbol", "")).upper()
                    if not sym:
                        continue
                    seq = [item]
                else:
                    seq = []
                    sym = ""
                    if isinstance(item, MappingABC):
                        keys = {str(k).lower() for k in item.keys()}
                        if {"ts", "symbol", "open", "high", "low", "close"}.issubset(keys):
                            try:
                                bar_obj = Bar.from_dict(item)  # type: ignore[arg-type]
                            except Exception:
                                bar_obj = None
                            if bar_obj is not None:
                                sym = str(getattr(bar_obj, "symbol", "")).upper()
                                if sym:
                                    spread_val = _coerce_spread(
                                        item.get("spread_bps")
                                        if isinstance(item, MappingABC)
                                        else None
                                    )
                                    if spread_val is None:
                                        half_val = _coerce_spread(
                                            item.get("half_spread_bps")
                                            if isinstance(item, MappingABC)
                                            else None
                                        )
                                        if half_val is not None:
                                            spread_val = half_val * 2.0
                                    if spread_val is None:
                                        spread_val = _infer_spread_from_bar(bar_obj)
                                    _append(sym, bar_obj, spread_val)
                                continue
                        for symbol, bars in item.items():
                            sym = str(symbol).upper()
                            seq = list(bars or [])
                            if not seq:
                                continue
                            for entry in seq:
                                if isinstance(entry, Bar):
                                    spread_val = _infer_spread_from_bar(entry)
                                    _append(sym, entry, spread_val)
                                elif (
                                    isinstance(entry, tuple)
                                    and len(entry) == 2
                                    and isinstance(entry[0], Bar)
                                ):
                                    spread_val = _coerce_spread(entry[1])
                                    if spread_val is None:
                                        spread_val = _infer_spread_from_bar(entry[0])
                                    _append(sym, entry[0], spread_val)
                        continue
                if not sym or not seq:
                    continue
                for entry in seq:
                    bar_obj = None
                    spread_val: float | None = None
                    if isinstance(entry, Bar):
                        bar_obj = entry
                        spread_val = _infer_spread_from_bar(entry)
                    elif (
                        isinstance(entry, tuple)
                        and len(entry) == 2
                        and isinstance(entry[0], Bar)
                    ):
                        bar_obj = entry[0]
                        spread_val = _coerce_spread(entry[1])
                        if spread_val is None:
                            spread_val = _infer_spread_from_bar(bar_obj)
                    elif (
                        isinstance(entry, tuple)
                        and len(entry) == 2
                        and isinstance(entry[1], Bar)
                    ):
                        bar_obj = entry[1]
                        spread_val = _coerce_spread(entry[0])
                        if spread_val is None:
                            spread_val = _infer_spread_from_bar(bar_obj)
                    elif isinstance(entry, MappingABC):
                        bar_candidate = entry.get("bar")
                        if isinstance(bar_candidate, Bar):
                            bar_obj = bar_candidate
                        else:
                            keys = {str(k).lower() for k in entry.keys()}
                            if {"ts", "symbol", "open", "high", "low", "close"}.issubset(keys):
                                try:
                                    bar_obj = Bar.from_dict(entry)  # type: ignore[arg-type]
                                except Exception:
                                    bar_obj = None
                        if bar_obj is not None:
                            spread_val = _coerce_spread(entry.get("spread_bps"))
                            if spread_val is None:
                                half_val = _coerce_spread(entry.get("half_spread_bps"))
                                if half_val is not None:
                                    spread_val = half_val * 2.0
                            if spread_val is None:
                                spread_val = _infer_spread_from_bar(bar_obj)
                    if bar_obj is None:
                        continue
                    _append(sym, bar_obj, spread_val)
        except Exception:
            return

        if not grouped:
            return

        counts: Dict[str, int] = {}
        for sym, entries in grouped.items():
            if not entries:
                continue
            try:
                self._dynamic_guard.prewarm(sym, entries)
            except Exception:
                continue
            counts[sym] = len(entries)

        if counts:
            try:
                ordered = {k: counts[k] for k in sorted(counts)}
                self._logger.info("dynamic guard prewarm complete %s", ordered)
            except Exception:
                pass

    def _acquire_tokens(self, symbol: str) -> tuple[bool, str | None]:
        with self._throttle_lock:
            if self._global_bucket is None:
                return True, None
            now = time.monotonic()
            if not self._global_bucket.consume(now=now):
                return False, "THROTTLED_GLOBAL"
            sb_factory = self._symbol_bucket_factory
            if sb_factory is None:
                return True, None
            sb = self._symbol_buckets[symbol]
            if not sb.consume(now=now):
                self._global_bucket.tokens = min(
                    self._global_bucket.tokens + 1.0, self._global_bucket.burst
                )
                return False, "THROTTLED_SYMBOL"
            return True, None

    def _refund_tokens(self, symbol: str) -> None:
        with self._throttle_lock:
            if self._global_bucket is not None:
                self._global_bucket.tokens = min(
                    self._global_bucket.tokens + 1.0, self._global_bucket.burst
                )
            if self._symbol_buckets is not None:
                sb = self._symbol_buckets[symbol]
                sb.tokens = min(sb.tokens + 1.0, sb.burst)

    def _current_bar_volume(self, bar: Bar) -> float | None:
        for attr in ("volume_quote", "volume_base", "trades"):
            value = getattr(bar, attr, None)
            if value is None:
                continue
            try:
                return abs(float(value))
            except Exception:
                continue
        return None

    def _extract_spread_bps(self, bar: Bar, snapshot: Any | None = None) -> float | None:
        """Best-effort extraction of spread information from ``bar``."""

        symbol = str(getattr(bar, "symbol", "")).upper()
        now = clock.now_ms()
        metrics_snapshot = snapshot
        if metrics_snapshot is None:
            metrics_getter = getattr(self._fp, "get_market_metrics", None)
            if callable(metrics_getter):
                try:
                    metrics_snapshot = metrics_getter(symbol)
                except Exception:
                    metrics_snapshot = None
        if metrics_snapshot is not None:
            spread = getattr(metrics_snapshot, "spread_bps", None)
            if spread is not None and math.isfinite(float(spread)) and float(spread) > 0:
                expiry = getattr(metrics_snapshot, "spread_valid_until", None)
                try:
                    expiry_int = int(expiry) if expiry is not None else None
                except Exception:
                    expiry_int = None
                if expiry_int is None and self._spread_ttl_ms > 0:
                    spread_ts = getattr(metrics_snapshot, "spread_ts", None)
                    try:
                        spread_ts_int = int(spread_ts)
                    except Exception:
                        spread_ts_int = None
                    if spread_ts_int is not None:
                        expiry_int = spread_ts_int + self._spread_ttl_ms
                if expiry_int is None or now <= expiry_int:
                    return float(spread)
        injected = self._spread_injections.get(symbol)
        if injected is not None:
            try:
                spread_bps = float(injected.get("spread", float("nan")))
            except Exception:
                spread_bps = float("nan")
            expiry_raw = injected.get("expiry")
            try:
                expiry_int = int(expiry_raw) if expiry_raw is not None else None
            except Exception:
                expiry_int = None
            valid = True
            if expiry_int is not None and now > expiry_int:
                valid = False
            if valid and math.isfinite(spread_bps) and spread_bps > 0.0:
                return spread_bps
            tick_ts_raw = injected.get("ts")
            try:
                tick_ts_int = int(tick_ts_raw) if tick_ts_raw is not None else None
            except Exception:
                tick_ts_int = None
            max_age_ms = self._spread_cache_max_ms
            if tick_ts_int is not None:
                age = max(0, now - tick_ts_int)
                if age > max_age_ms * 4:
                    self._spread_injections.pop(symbol, None)
            else:
                self._spread_injections.pop(symbol, None)

        for attr, multiplier in (("spread_bps", 1.0), ("half_spread_bps", 2.0)):
            raw = getattr(bar, attr, None)
            if raw is None:
                continue
            try:
                val = float(raw)
            except Exception:
                continue
            if math.isfinite(val):
                return val * multiplier

        bid = getattr(bar, "bid", None)
        ask = getattr(bar, "ask", None)
        if bid is None or ask is None:
            bid = getattr(bar, "bid_price", None)
            ask = getattr(bar, "ask_price", None)
        try:
            bid_val = float(bid) if bid is not None else float("nan")
            ask_val = float(ask) if ask is not None else float("nan")
        except Exception:
            bid_val = float("nan")
            ask_val = float("nan")
        if math.isfinite(bid_val) and math.isfinite(ask_val) and bid_val > 0 and ask_val > 0:
            mid = (bid_val + ask_val) * 0.5
            if mid > 0:
                return (ask_val - bid_val) / mid * 10000.0
        high = getattr(bar, "high", None)
        low = getattr(bar, "low", None)
        try:
            high_val = float(high) if high is not None else float("nan")
            low_val = float(low) if low is not None else float("nan")
        except Exception:
            return None
        if math.isfinite(high_val) and math.isfinite(low_val):
            span = high_val - low_val
            if span > 0.0:
                mid_val = (high_val + low_val) * 0.5
                if not math.isfinite(mid_val) or mid_val == 0.0:
                    close = getattr(bar, "close", None)
                    try:
                        mid_val = float(close) if close is not None else float("nan")
                    except Exception:
                        mid_val = float("nan")
                if math.isfinite(mid_val) and mid_val != 0.0:
                    return span / abs(mid_val) * 10000.0
        return None

    def on_tick(self, tick: Tick) -> None:
        """Inject spread estimates from real-time book ticker events."""

        symbol = str(getattr(tick, "symbol", "")).upper()
        if not symbol:
            return
        price_val = self._coerce_price(getattr(tick, "price", None))
        bid = getattr(tick, "bid", None)
        ask = getattr(tick, "ask", None)
        if bid is None or ask is None:
            if price_val is not None:
                self._set_last_price(symbol, price_val)
            else:
                self._maybe_fetch_rest_price(symbol)
            return
        try:
            bid_val = float(bid)
            ask_val = float(ask)
        except Exception:
            return
        if not (
            math.isfinite(bid_val)
            and math.isfinite(ask_val)
            and bid_val > 0
            and ask_val > 0
        ):
            return
        mid_val: float | None = None
        mid_candidate = (bid_val + ask_val) * 0.5
        if math.isfinite(mid_candidate) and mid_candidate > 0:
            mid_val = mid_candidate
        if price_val is None and mid_val is not None:
            price_val = mid_val
        spread_val = getattr(tick, "spread_bps", None)
        try:
            spread_bps = float(spread_val) if spread_val is not None else None
        except Exception:
            spread_bps = None
        if spread_bps is None:
            if mid_val is None or mid_val <= 0:
                self._maybe_fetch_rest_price(symbol)
                return
            spread_bps = (ask_val - bid_val) / mid_val * 10000.0
        if spread_bps is None or not math.isfinite(spread_bps) or spread_bps <= 0:
            if price_val is not None:
                self._set_last_price(symbol, price_val)
            else:
                self._maybe_fetch_rest_price(symbol)
            return
        if price_val is not None:
            self._set_last_price(symbol, price_val)
        else:
            self._maybe_fetch_rest_price(symbol)
        ts_ms = int(getattr(tick, "ts", 0) or clock.now_ms())
        ttl_ms = self._spread_ttl_ms if self._spread_ttl_ms > 0 else self._spread_cache_max_ms
        expiry = ts_ms + ttl_ms if ttl_ms > 0 else None
        self._spread_injections[symbol] = {
            "spread": float(spread_bps),
            "ts": ts_ms,
            "expiry": expiry,
        }
        recorder = getattr(self._fp, "record_spread", None)
        if callable(recorder):
            try:
                recorder(
                    symbol,
                    spread_bps=spread_bps,
                    bid=bid_val,
                    ask=ask_val,
                    ts_ms=ts_ms,
                    ttl_ms=ttl_ms,
                )
            except Exception:
                pass

    def _log_signal_quality_drop(
        self,
        bar: Bar,
        snapshot: Any | None,
        bar_volume: float | None,
        detail: str | None,
        sigma_thr: float,
        vol_frac: float,
    ) -> None:
        reason_cfg = self._signal_quality_cfg.log_reason
        reason_label = Reason.OTHER.name
        should_log = False
        if isinstance(reason_cfg, bool):
            if reason_cfg:
                should_log = True
        else:
            reason_label_candidate = str(reason_cfg or "").strip()
            if reason_label_candidate:
                reason_label = reason_label_candidate
                should_log = True
        try:
            monitoring.inc_stage(Stage.POLICY)
        except Exception:
            pass
        try:
            monitoring.inc_reason(reason_label)
        except Exception:
            pass
        try:
            pipeline_stage_drop_count.labels(
                bar.symbol,
                Stage.POLICY.name,
                reason_label,
            ).inc()
        except Exception:
            pass
        payload: dict[str, Any] = {
            "stage": Stage.POLICY.name,
            "reason": reason_label,
            "symbol": bar.symbol,
            "sigma_threshold": sigma_thr,
            "vol_floor_frac": vol_frac,
            "bar_volume": bar_volume,
        }
        bar_close_ms = getattr(bar, "closed_at", None)
        if bar_close_ms is None:
            bar_close_ms = getattr(bar, "ts", None)
        try:
            if bar_close_ms is not None:
                bar_close_ms_int = int(bar_close_ms)
                payload["bar_close_ms"] = bar_close_ms_int
                payload["bar_close_at"] = (
                    datetime.fromtimestamp(bar_close_ms_int / 1000.0, tz=timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
        except Exception:
            pass
        if detail:
            payload["detail"] = detail
        if snapshot is not None:
            payload["current_sigma"] = getattr(snapshot, "current_sigma", None)
            payload["vol_median"] = getattr(snapshot, "current_vol_median", None)
            payload["window_ready"] = getattr(snapshot, "window_ready", None)
        if should_log:
            try:
                self._logger.info(_format_signal_quality_log(payload))
            except Exception:
                pass

    def _extract_features(
        self, bar: Bar, *, skip_metrics: bool = False
    ) -> dict[str, Any]:
        raw_feats = self._fp.update(bar, skip_metrics=skip_metrics)
        if isinstance(raw_feats, dict):
            return dict(raw_feats)
        try:
            return dict(raw_feats or {})
        except Exception:
            return {}

    def _apply_signal_quality_filter(
        self, bar: Bar, *, skip_metrics: bool = False
    ) -> tuple[bool, dict[str, Any]]:
        feats = self._extract_features(bar, skip_metrics=skip_metrics)

        sigma_thr = float(self._signal_quality_cfg.sigma_threshold or 0.0)
        vol_frac = float(self._signal_quality_cfg.vol_floor_frac or 0.0)
        raw_symbol = str(getattr(bar, "symbol", "") or "")
        symbol_key = raw_symbol.upper()
        snapshot = None
        signal_quality = getattr(self._fp, "signal_quality", None)
        getter = getattr(signal_quality, "get", None)
        if callable(getter):
            snapshot = getter(symbol_key)
            if snapshot is None and raw_symbol:
                snapshot = getter(raw_symbol)
        bar_volume = self._current_bar_volume(bar)

        blocked = False
        detail = None
        if snapshot is not None:
            if not getattr(snapshot, "window_ready", False):
                blocked = True
                detail = "WINDOW_NOT_READY"
            sigma = getattr(snapshot, "current_sigma", None)
            if (
                not blocked
                and sigma_thr > 0.0
                and sigma is not None
                and float(sigma) > sigma_thr
            ):
                blocked = True
                detail = "SIGMA_THRESHOLD"
            median = getattr(snapshot, "current_vol_median", None)
            if not blocked and vol_frac > 0.0:
                if median is None or float(median) <= 0.0:
                    blocked = True
                    detail = "VOL_MEDIAN"
                else:
                    threshold = float(median) * vol_frac
                    if bar_volume is None or bar_volume < threshold:
                        blocked = True
                        detail = "VOLUME_FLOOR"
        elif sigma_thr > 0.0 or vol_frac > 0.0:
            # Metrics missing; conservatively block to avoid trading without data.
            blocked = True
            detail = "METRICS_MISSING"

        if blocked:
            self._log_signal_quality_drop(bar, snapshot, bar_volume, detail, sigma_thr, vol_frac)
            return False, feats

        return True, feats

    def _evaluate_no_trade_windows(
        self,
        ts_ms: int,
        symbol: str,
        *,
        stage_cfg: PipelineStageConfig | None = None,
    ) -> tuple[PipelineResult, str | None]:
        try:
            monitoring.inc_stage(Stage.WINDOWS)
        except Exception:
            pass

        if stage_cfg is not None and not stage_cfg.enabled:
            return PipelineResult(action="pass", stage=Stage.WINDOWS), None

        if self._schedule_checker is None or not self._schedule_checker.enabled:
            return PipelineResult(action="pass", stage=Stage.WINDOWS), None

        blocked = False
        reason_label: str | None = None
        try:
            blocked, reason_label = self._schedule_checker.evaluate(ts_ms, symbol)
        except Exception:
            blocked = False
            reason_label = None

        if not blocked:
            return PipelineResult(action="pass", stage=Stage.WINDOWS), None

        try:
            monitoring.inc_reason(Reason.WINDOW)
        except Exception:
            pass

        return (
            PipelineResult(action="drop", stage=Stage.WINDOWS, reason=Reason.WINDOW),
            reason_label,
        )

    def _should_skip_idempotent(
        self, key: str | None, symbol: str, bar_ts_ms: int
    ) -> bool:
        if not key:
            return False
        if key in self._idempotency_cache:
            try:
                self._idempotency_cache.move_to_end(key)
            except Exception:
                pass
            try:
                self._logger.info(
                    "SKIP_DUPLICATE %s",
                    {
                        "stage": Stage.PUBLISH.name,
                        "symbol": symbol,
                        "bar_close_ms": bar_ts_ms,
                        "idempotency_key": key,
                    },
                )
            except Exception:
                pass
            try:
                monitoring.signal_idempotency_skipped_count.labels(symbol).inc()
            except Exception:
                pass
            return True
        return False

    def _remember_idempotency_key(self, key: str | None) -> None:
        if not key:
            return
        cache = self._idempotency_cache
        cache[key] = None
        try:
            cache.move_to_end(key)
        except Exception:
            pass
        if len(cache) > self._idempotency_cache_size:
            try:
                cache.popitem(last=False)
            except Exception:
                pass

    def _resolve_ttl_timeframe_ms(self, *, log_if_invalid: bool) -> int | None:
        timeframe_raw: Any = self._ws_dedup_timeframe_ms
        try:
            timeframe_ms = int(timeframe_raw)
        except (TypeError, ValueError):
            timeframe_ms = 0
        if timeframe_ms <= 0 and self._bar_timeframe_ms > 0:
            timeframe_ms = int(self._bar_timeframe_ms)
        if timeframe_ms <= 0:
            if log_if_invalid and self._last_invalid_ttl_timeframe != timeframe_raw:
                self._last_invalid_ttl_timeframe = timeframe_raw
                try:
                    self._logger.info(
                        "TTL_BYPASSED %s",
                        {
                            "stage": Stage.TTL.name,
                            "timeframe_ms": timeframe_raw,
                            "execution_mode": self._execution_mode,
                        },
                    )
                except Exception:
                    pass
            return None
        self._last_invalid_ttl_timeframe = None
        return timeframe_ms

    def _dispatch_signal_envelope(self, envelope: Any) -> None:
        order_obj: Any | None = None
        if self._execution_mode == "bar" and self._dispatch_stack:
            try:
                order_obj = self._dispatch_stack.pop()
            except IndexError:
                order_obj = None

        dispatcher = self._signal_dispatcher
        if dispatcher is not None:
            try:
                dispatcher(envelope)
            except Exception as exc:
                try:
                    self._logger.error(
                        "failed to dispatch signal envelope: %s", exc
                    )
                except Exception:
                    pass

        if order_obj is None:
            return

        symbol_key = str(getattr(order_obj, "symbol", "") or "").upper()
        if isinstance(envelope, MappingABC):
            symbol_candidate = envelope.get("symbol")
            if symbol_candidate:
                try:
                    symbol_key = str(symbol_candidate).upper()
                except Exception:
                    pass

        meta_map = self._ensure_order_meta(order_obj)
        bar_value = meta_map.get("bar")
        if bar_value is None:
            bar_value = self._last_bar_snapshot.get(symbol_key)
            if bar_value is not None:
                meta_map["bar"] = bar_value
                payload_map = self._coerce_mapping(meta_map.get("payload"))
                if payload_map is not None:
                    payload_map.setdefault("bar", bar_value)
                    meta_map["payload"] = payload_map

        execute = getattr(self._executor, "execute", None)
        if not callable(execute):
            return
        try:
            execute(order_obj)
        except Exception:
            try:
                self._logger.exception(
                    "failed to execute bar order from envelope", exc_info=True
                )
            except Exception:
                pass
        else:
            try:
                setattr(order_obj, "_bar_dispatched", True)
            except Exception:
                pass

    def _emit(
        self,
        o: Any,
        symbol: str,
        bar_close_ms: int,
        *,
        bar_open_ms: int | None = None,
    ) -> bool:
        ttl_stage_cfg = self._pipeline_cfg.get("ttl") if self._pipeline_cfg else None
        ttl_enabled = ttl_stage_cfg is None or ttl_stage_cfg.enabled

        if self._guards is not None:
            checked, reason = self._guards.apply(bar_close_ms, symbol, [o])
            if reason or not checked:
                try:
                    self._logger.info(
                        "DROP %s",
                        {"stage": Stage.PUBLISH.name, "reason": reason or ""},
                    )
                except Exception:
                    pass
                try:
                    pipeline_stage_drop_count.labels(
                        symbol, Stage.PUBLISH.name, Reason.RISK_POSITION.name
                    ).inc()
                except Exception:
                    pass
                self._log_drop_envelope(o, symbol, bar_close_ms, reason or "RISK")
                return False
            o = list(checked)[0]

        created_ts = self._extract_created_ts_ms(o)
        now_ms = clock.now_ms()
        expires_at_ms = max(created_ts, bar_close_ms)
        dedup_expires_at_ms: int | None = None
        if self._ws_dedup_timeframe_ms > 0:
            dedup_expires_at_ms = created_ts + self._ws_dedup_timeframe_ms
            expires_at_ms = max(expires_at_ms, dedup_expires_at_ms)
        ttl_expires_at_ms: int | None = None
        ttl_timeframe_ms = self._resolve_ttl_timeframe_ms(log_if_invalid=ttl_enabled)
        if ttl_enabled and ttl_timeframe_ms is not None:
            try:
                monitoring.inc_stage(Stage.TTL)
            except Exception:
                pass
            ok, expires_at_ms, _ = check_ttl(
                bar_close_ms=bar_close_ms,
                now_ms=now_ms,
                timeframe_ms=ttl_timeframe_ms,
            )
            ttl_expires_at_ms = expires_at_ms
            if not ok:
                try:
                    self._logger.info(
                        "TTL_EXPIRED_PUBLISH %s",
                        {
                            "symbol": symbol,
                            "created_ts_ms": created_ts,
                            "bar_close_ms": bar_close_ms,
                            "now_ms": now_ms,
                            "expires_at_ms": expires_at_ms,
                        },
                    )
                except Exception:
                    pass
                try:
                    monitoring.signal_absolute_count.labels(symbol).inc()
                except Exception:
                    pass
                try:
                    pipeline_stage_drop_count.labels(
                        symbol, Stage.TTL.name, Reason.OTHER.name
                    ).inc()
                except Exception:
                    pass
                return False
        if ttl_expires_at_ms is not None:
            expires_at_ms = min(expires_at_ms, ttl_expires_at_ms)
        try:
            age_ms = now_ms - created_ts
            if self._execution_mode != "bar":
                monitoring.age_at_publish_ms.labels(symbol).observe(age_ms)
            monitoring.signal_published_count.labels(symbol).inc()
        except Exception:
            pass

        score = float(getattr(o, "score", 0) or 0)
        fh = str(getattr(o, "features_hash", "") or "")
        side = getattr(o, "side", "")
        side = side.value if hasattr(side, "value") else str(side)
        side = str(side).upper()
        writer = getattr(signal_bus, "OUT_WRITER", None)
        if writer is not None:
            vol = getattr(o, "volume_frac", None)
            if vol is None:
                vol = getattr(o, "quantity", 0)
            try:
                vol_val = float(vol)
            except Exception:
                vol_val = 0.0
            row = {
                "ts_ms": bar_close_ms,
                "symbol": symbol,
                "side": side,
                "volume_frac": vol_val,
                "score": score,
                "features_hash": fh,
            }
            try:
                writer.write(row)
            except Exception:
                pass
        try:
            self._logger.info("order %s", o)
        except Exception:
            pass
        meta_map = self._ensure_order_meta(o)
        payload, payload_valid_until_ms, adv_quote = self._build_envelope_payload(
            o, symbol
        )
        if adv_quote is not None:
            meta_map["adv_quote"] = adv_quote
        equity_override = self._resolve_order_equity(o, payload, symbol)
        if equity_override is None:
            fallback_equity = self._portfolio_equity
            if fallback_equity is not None:
                try:
                    fallback_val = float(fallback_equity)
                except (TypeError, ValueError):
                    fallback_val = None
                else:
                    if math.isfinite(fallback_val) and fallback_val > 0.0:
                        equity_override = fallback_val
        if (
            equity_override is not None
            and math.isfinite(equity_override)
            and equity_override > 0.0
        ):
            try:
                meta_map["equity_usd"] = float(equity_override)
            except Exception:
                pass
        dedup_key: str | None = None
        if isinstance(meta_map, MappingABC):
            raw_key = meta_map.get("dedup_key")
            if raw_key is not None:
                dedup_key = str(raw_key)
        idempotency_key: str | None = None
        if isinstance(meta_map, MappingABC):
            for candidate in ("idempotency_key", "signal_id", "dedup_key"):
                raw_value = meta_map.get(candidate)
                if raw_value is None:
                    continue
                try:
                    idempotency_key = str(raw_value)
                except Exception:
                    continue
                if idempotency_key:
                    break
        idempotency_ts_ms = bar_open_ms if bar_open_ms is not None else bar_close_ms
        if not idempotency_key:
            try:
                idempotency_key = envelope_signal_id(symbol, idempotency_ts_ms)
            except Exception:
                idempotency_key = None
        if self._should_skip_idempotent(idempotency_key, symbol, idempotency_ts_ms):
            return False
        valid_until_ms_int: int | None = None
        if payload_valid_until_ms is not None:
            try:
                valid_until_ms_int = int(payload_valid_until_ms)
            except (TypeError, ValueError):
                valid_until_ms_int = None
        if valid_until_ms_int is not None:
            expires_at_ms = min(expires_at_ms, valid_until_ms_int)
        expires_at_ms = max(expires_at_ms, bar_close_ms)
        published = True
        stack_pushed = False
        if getattr(signal_bus, "ENABLED", False):
            if self._execution_mode == "bar":
                self._dispatch_stack.append(o)
                stack_pushed = True
            try:
                published = publish_signal_envelope(
                    symbol,
                    bar_close_ms,
                    payload,
                    self._dispatch_signal_envelope,
                    expires_at_ms=expires_at_ms,
                    dedup_key=dedup_key,
                    valid_until_ms=valid_until_ms_int,
                )
            except Exception:
                published = False
            finally:
                if stack_pushed and (
                    not published
                    or (
                        self._dispatch_stack
                        and self._dispatch_stack[-1] is o
                    )
                ):
                    try:
                        self._dispatch_stack.pop()
                    except IndexError:
                        pass
        if not published:
            return False
        self._remember_idempotency_key(idempotency_key)
        dispatched_inline = bool(getattr(o, "_bar_dispatched", False))
        if (
            self._execution_mode == "bar"
            and self._signal_dispatcher is None
            and not dispatched_inline
        ):
            execute = getattr(self._executor, "execute", None)
            if callable(execute):
                try:
                    execute(o)
                except Exception:
                    try:
                        self._logger.exception(
                            "failed to execute bar order inline", exc_info=True
                        )
                    except Exception:
                        pass
        if self._execution_mode != "bar":
            submit = getattr(self._executor, "submit", None)
            if callable(submit):
                try:
                    submit(o)
                except Exception:
                    pass
        return True

    def publish_decision(
        self,
        o: Any,
        symbol: str,
        bar_open_ms: int,
        *,
        bar_close_ms: int | None = None,
        stage_cfg: PipelineStageConfig | None = None,
    ) -> PipelineResult:
        """Final pipeline stage: throttle and emit order."""
        if bar_close_ms is None:
            timeframe_ms = self._ws_dedup_timeframe_ms
            if timeframe_ms <= 0 and self._bar_timeframe_ms > 0:
                timeframe_ms = self._bar_timeframe_ms
            if timeframe_ms > 0:
                bar_close_ms = bar_open_ms + timeframe_ms
            else:
                bar_close_ms = bar_open_ms
        if stage_cfg is not None and not stage_cfg.enabled:
            self._commit_exposure(o, bar_close_ms=bar_close_ms)
            return PipelineResult(action="pass", stage=Stage.PUBLISH, decision=o)
        if self._maybe_drop_due_to_cooldown(symbol, o, bar_open_ms):
            self._rollback_exposure(o)
            return PipelineResult(
                action="drop", stage=Stage.PUBLISH, reason=Reason.OTHER
            )
        throttle_stage_cfg = (
            self._pipeline_cfg.get("throttle") if self._pipeline_cfg else None
        )
        throttle_enabled = (
            self._throttle_cfg is not None
            and self._throttle_cfg.enabled
            and (throttle_stage_cfg is None or throttle_stage_cfg.enabled)
        )
        if throttle_enabled:
            try:
                monitoring.inc_stage(Stage.THROTTLE)
            except Exception:
                pass
            ok, reason = self._acquire_tokens(symbol)
            if not ok:
                if self._throttle_cfg.mode == "queue" and self._queue is not None:
                    exp = time.monotonic() + self._throttle_cfg.queue.ttl_ms / 1000.0
                    self._queue.append((exp, symbol, bar_close_ms, bar_open_ms, o))
                    self._update_queue_metrics()
                    try:
                        monitoring.throttle_enqueued_count.labels(
                            symbol, reason or ""
                        ).inc()
                    except Exception:
                        pass
                    return PipelineResult(action="queue", stage=Stage.PUBLISH)
                self._rollback_exposure(o)
                self._log_drop_envelope(o, symbol, bar_close_ms, reason or "")
                try:
                    monitoring.throttle_dropped_count.labels(symbol, reason or "").inc()
                except Exception:
                    pass
                try:
                    pipeline_stage_drop_count.labels(
                        symbol,
                        Stage.THROTTLE.name,
                        (reason or Reason.OTHER.name),
                    ).inc()
                except Exception:
                    pass
                return PipelineResult(
                    action="drop", stage=Stage.PUBLISH, reason=Reason.OTHER
                )
        if not self._emit(o, symbol, bar_close_ms, bar_open_ms=bar_open_ms):
            self._refund_tokens(symbol)
            self._rollback_exposure(o)
            return PipelineResult(
                action="drop", stage=Stage.PUBLISH, reason=Reason.OTHER
            )
        self._commit_exposure(o, bar_close_ms=bar_close_ms)
        return PipelineResult(action="pass", stage=Stage.PUBLISH, decision=o)

    def _drain_queue(self) -> list[Any]:
        emitted: list[Any] = []
        if self._queue is None:
            return emitted
        while True:
            with self._throttle_lock:
                if self._queue is None or not self._queue:
                    break
                exp, symbol, bar_close_ms, bar_open_ms, order = self._queue[0]
                now = time.monotonic()
                if exp <= now:
                    self._queue.popleft()
                    expired = True
                else:
                    expired = False
            if expired:
                try:
                    self._rollback_exposure(order)
                except Exception:
                    pass
                self._log_drop_envelope(order, symbol, bar_close_ms, "QUEUE_EXPIRED")
                try:
                    monitoring.throttle_queue_expired_count.labels(symbol).inc()
                except Exception:
                    pass
                try:
                    monitoring.throttle_dropped_count.labels(
                        symbol, "QUEUE_EXPIRED"
                    ).inc()
                except Exception:
                    pass
                continue
            ok, _ = self._acquire_tokens(symbol)
            if not ok:
                break
            with self._throttle_lock:
                if self._queue is None:
                    self._refund_tokens(symbol)
                    break
                try:
                    self._queue.popleft()
                except IndexError:
                    self._refund_tokens(symbol)
                    break
            if not self._emit(order, symbol, bar_close_ms, bar_open_ms=bar_open_ms):
                self._refund_tokens(symbol)
                try:
                    self._rollback_exposure(order)
                except Exception:
                    pass
            else:
                emitted.append(order)
                self._commit_exposure(order, bar_close_ms=bar_close_ms)
        self._update_queue_metrics()
        return emitted

    def process(self, bar: Bar):
        duplicates = 0
        emitted: list[Any] = []

        def _format_ts(ms: int | None) -> str | None:
            if ms is None:
                return None
            return (
                datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

        symbol = bar.symbol.upper()
        bar_close_ms = int(bar.ts)
        ts_ms = bar_close_ms
        bar_open_ms = bar_close_ms
        self._last_bar_snapshot[symbol] = bar
        price_attr = getattr(self, "_bar_price_field", "close") or "close"
        price_val = self._coerce_price(getattr(bar, price_attr, None))
        if price_val is None and price_attr != "close":
            price_val = self._coerce_price(getattr(bar, "close", None))
        if price_val is not None:
            self._set_last_price(symbol, price_val)
        else:
            self._maybe_fetch_rest_price(symbol)
        if self._execution_mode == "bar":
            self._replay_pending_weight_entries()
        prev_ts = self._last_bar_ts.get(symbol)
        gap_ms: int | None = None
        duplicate_ts = False
        if prev_ts is not None:
            delta = ts_ms - prev_ts
            if delta <= 0:
                duplicate_ts = True
            elif self._ws_dedup_timeframe_ms > 0 and delta > self._ws_dedup_timeframe_ms:
                gap_ms = delta
        if prev_ts is None or ts_ms >= prev_ts:
            self._last_bar_ts[symbol] = ts_ms

        if gap_ms is not None:
            metrics = getattr(self._fp, "metrics", None)
            reset_symbol = getattr(metrics, "reset_symbol", None)
            if callable(reset_symbol):
                try:
                    reset_symbol(symbol)
                except Exception:
                    pass
            try:
                self._fp.signal_quality.pop(symbol, None)
            except Exception:
                pass
            try:
                payload = {
                    "symbol": bar.symbol,
                    "previous_open_ms": prev_ts,
                    "previous_open_at": _format_ts(prev_ts),
                    "current_open_ms": ts_ms,
                    "current_open_at": _format_ts(ts_ms),
                    "gap_ms": gap_ms,
                    "interval_ms": self._ws_dedup_timeframe_ms,
                }
                self._logger.warning("BAR_GAP_DETECTED %s", payload)
            except Exception:
                pass

        if duplicate_ts and prev_ts is not None:
            try:
                payload = {
                    "symbol": bar.symbol,
                    "previous_open_ms": prev_ts,
                    "previous_open_at": _format_ts(prev_ts),
                    "current_open_ms": ts_ms,
                    "current_open_at": _format_ts(ts_ms),
                }
                self._logger.info("BAR_DUPLICATE_TIMESTAMP %s", payload)
            except Exception:
                pass

        skip_metrics = duplicate_ts

        def _finalize() -> list[Any]:
            new_bar = prev_ts is not None and ts_ms > prev_ts
            self._advance_symbol_cooldown(bar.symbol, ts_ms, new_bar=new_bar)
            emitted_count = len(emitted)
            runtime_monitoring = self._monitoring
            duplicates_count = duplicates
            total_events = emitted_count + duplicates_count
            if runtime_monitoring is not None:
                try:
                    runtime_monitoring.record_signals(
                        bar.symbol, emitted_count, duplicates_count
                    )
                except Exception:
                    pass
                snapshot = self._extract_monitoring_snapshot(self._executor)
                daily_snapshot = self._daily_turnover_snapshot()
                if daily_snapshot:
                    try:
                        runtime_monitoring.update_daily_turnover(daily_snapshot)
                    except Exception:
                        pass
                    if not isinstance(snapshot, MappingABC):
                        snapshot = {}
                    else:
                        snapshot = dict(snapshot)
                    snapshot["daily_turnover"] = daily_snapshot
                if self._execution_mode != "bar":
                    requested, filled = self._extract_fill_metrics(snapshot)
                    if requested is not None and filled is not None:
                        try:
                            runtime_monitoring.record_fill(requested, filled)
                        except Exception:
                            pass
                else:
                    bar_metrics = self._extract_bar_execution_metrics(
                        snapshot,
                        expected_bar_ts=ts_ms,
                        total_signals=total_events,
                    )
                    if bar_metrics:
                        metrics_bar_ts = bar_metrics.get("bar_ts")
                        try:
                            metrics_bar_ts_int = (
                                int(metrics_bar_ts)
                                if metrics_bar_ts is not None
                                else None
                            )
                        except (TypeError, ValueError):
                            metrics_bar_ts_int = None
                        if metrics_bar_ts_int == ts_ms and total_events > 0:
                            try:
                                runtime_monitoring.record_bar_execution(
                                    bar.symbol,
                                    decisions=int(bar_metrics.get("decisions", 0)),
                                    act_now=int(bar_metrics.get("act_now", 0)),
                                    turnover_usd=float(
                                        bar_metrics.get("turnover_usd", 0.0)
                                    ),
                                    cap_usd=bar_metrics.get("cap_usd"),
                                    impact_mode=bar_metrics.get("impact_mode"),
                                    modeled_cost_bps=bar_metrics.get(
                                        "modeled_cost_bps"
                                    ),
                                    realized_slippage_bps=bar_metrics.get(
                                        "realized_slippage_bps"
                                    ),
                                    cost_bias_bps=bar_metrics.get("cost_bias_bps"),
                                    bar_ts=bar_metrics.get("bar_ts"),
                                )
                            except Exception:
                                pass
                            reason = str(bar_metrics.get("reason") or "").strip()
                            if reason:
                                try:
                                    self._logger.info(
                                        "BAR_EXECUTOR_SKIP %s",
                                        {"symbol": bar.symbol, "reason": reason},
                                    )
                                except Exception:
                                    pass
                                try:
                                    monitoring.inc_reason(reason)
                                except Exception:
                                    pass
                                try:
                                    pipeline_stage_drop_count.labels(
                                        bar.symbol,
                                        Stage.PUBLISH.name,
                                        reason,
                                    ).inc()
                                except Exception:
                                    pass
                pnl_value = self._extract_daily_pnl(snapshot)
                if pnl_value is not None:
                    try:
                        runtime_monitoring.record_pnl(pnl_value)
                    except Exception:
                        pass
                self._zero_signal_streak = 0
            else:
                try:
                    monitoring.record_signals(bar.symbol, emitted_count, duplicates_count)
                except Exception:
                    pass
                if emitted_count == 0:
                    self._zero_signal_streak += 1
                    if (
                        self._zero_signal_alert > 0
                        and self._zero_signal_streak >= self._zero_signal_alert
                    ):
                        try:
                            monitoring.alert_zero_signals(bar.symbol)
                        except Exception:
                            pass
                else:
                    self._zero_signal_streak = 0
            total = total_events
            if total > 0:
                try:
                    monitoring.signal_error_rate.labels(bar.symbol).set(
                        float(duplicates_count) / float(total)
                    )
                except Exception:
                    pass
            self._publish_status(
                symbol=bar.symbol,
                last_bar_ts=ts_ms,
                emitted=emitted_count,
                duplicates=duplicates_count,
            )
            return emitted

        if self._pipeline_cfg is not None and not self._pipeline_cfg.enabled:
            return _finalize()
        if self._safe_mode_fn():
            return _finalize()

        if self._queue is not None:
            emitted.extend(self._drain_queue())

        ttl_stage_cfg = (
            self._pipeline_cfg.get("ttl") if self._pipeline_cfg else None
        )
        ttl_enabled = ttl_stage_cfg is None or ttl_stage_cfg.enabled
        ttl_timeframe_ms = self._resolve_ttl_timeframe_ms(log_if_invalid=ttl_enabled)

        open_timeframe_ms = self._bar_timeframe_ms
        if open_timeframe_ms <= 0:
            open_timeframe_ms = self._ws_dedup_timeframe_ms
        if open_timeframe_ms <= 0 and ttl_timeframe_ms is not None and ttl_timeframe_ms > 0:
            open_timeframe_ms = ttl_timeframe_ms
        if open_timeframe_ms > 0:
            bar_open_ms = bar_close_ms - open_timeframe_ms
        else:
            bar_open_ms = bar_close_ms

        dedup_stage_cfg = self._pipeline_cfg.get("dedup") if self._pipeline_cfg else None
        dedup_enabled = self._ws_dedup_enabled and (
            dedup_stage_cfg is None or dedup_stage_cfg.enabled
        )
        if self._ws_dedup_enabled:
            if dedup_enabled:
                try:
                    monitoring.inc_stage(Stage.DEDUP)
                except Exception:
                    pass
                if signal_bus.should_skip(bar.symbol, bar_close_ms):
                    if self._ws_dedup_log_skips:
                        try:
                            self._logger.info("SKIP_DUPLICATE_BAR")
                        except Exception:
                            pass
                    try:
                        monitoring.ws_dup_skipped_count.labels(bar.symbol).inc()
                    except Exception:
                        pass
                    self._update_queue_metrics()
                    try:
                        pipeline_stage_drop_count.labels(
                            bar.symbol, Stage.DEDUP.name, Reason.OTHER.name
                        ).inc()
                    except Exception:
                        pass
                    duplicates = 1
                    return _finalize()
        guard_res = closed_bar_guard(
            bar=bar,
            now_ms=clock.now_ms(),
            enforce=self._enforce_closed_bars,
            lag_ms=self._close_lag_ms,
            stage_cfg=(
                self._pipeline_cfg.get("closed_bar") if self._pipeline_cfg else None
            ),
        )
        if guard_res.action == "drop":
            try:
                self._logger.info(
                    "DROP %s",
                    {
                        "stage": guard_res.stage.name,
                        "reason": getattr(guard_res.reason, "name", ""),
                    },
                )
            except Exception:
                pass
            try:
                skipped_incomplete_bars.labels(bar.symbol).inc()
            except Exception:
                pass
            try:
                pipeline_stage_drop_count.labels(
                    bar.symbol,
                    guard_res.stage.name,
                    guard_res.reason.name if guard_res.reason else "",
                ).inc()
            except Exception:
                pass
            self._update_queue_metrics()
            return _finalize()

        dyn_stage_cfg = self._pipeline_cfg.get("dynamic_guard") if self._pipeline_cfg else None
        dyn_stage_enabled = dyn_stage_cfg is None or dyn_stage_cfg.enabled
        if self._dynamic_guard is not None:
            fp_snapshot = None
            metrics_getter = getattr(self._fp, "get_market_metrics", None)
            if callable(metrics_getter):
                try:
                    fp_snapshot = metrics_getter(symbol)
                except Exception:
                    fp_snapshot = None
            spread_bps = self._extract_spread_bps(bar, snapshot=fp_snapshot)
            self._dynamic_guard.update(symbol, bar, spread=spread_bps)
            blocked, reason, snapshot = self._dynamic_guard.should_block(symbol)
            warmup_ready = True
            if fp_snapshot is not None:
                warmup_ready = bool(getattr(fp_snapshot, "window_ready", True))
            guard_ready = bool(snapshot.get("ready", True)) if snapshot else True
            if not guard_ready:
                blocked = False
                reason = None
            if not warmup_ready:
                blocked = False
                reason = None
            if blocked and dyn_stage_enabled:
                metrics: Dict[str, object] = {}
                if snapshot:
                    metrics = {
                        "sigma": snapshot.get("sigma"),
                        "sigma_k": snapshot.get("sigma_k"),
                        "spread_bps": snapshot.get("spread_bps"),
                        "spread_percentile": snapshot.get("spread_percentile"),
                        "cooldown": snapshot.get("cooldown"),
                        "trigger_reasons": snapshot.get("trigger_reasons"),
                        "ready": snapshot.get("ready"),
                    }
                if fp_snapshot is not None:
                    metrics.setdefault("warmup_ready", warmup_ready)
                    fp_spread = getattr(fp_snapshot, "spread_bps", None)
                    if fp_spread is not None:
                        metrics.setdefault("spread_bps", fp_spread)
                log_reason = reason
                triggers = snapshot.get("trigger_reasons") if snapshot else None
                if not log_reason and isinstance(triggers, SequenceABC):
                    triggers = list(triggers)
                    if "vol_extreme" in triggers:
                        log_reason = "vol_extreme"
                    elif "spread_wide" in triggers:
                        log_reason = "spread_wide"
                if not log_reason:
                    log_reason = "vol_extreme"
                try:
                    detail: Dict[str, object] = {
                        "stage": Stage.WINDOWS.name,
                        "reason": log_reason,
                        "symbol": bar.symbol,
                    }
                    if reason and reason != log_reason:
                        detail["detail"] = reason
                    if metrics:
                        detail["metrics"] = metrics
                    bar_ts = getattr(bar, "ts", None)
                    try:
                        if bar_ts is not None:
                            detail["bar_open_ms"] = int(bar_ts)
                    except Exception:
                        pass
                    self._logger.info("DROP %s", detail)
                except Exception:
                    pass
                try:
                    monitoring.inc_stage(Stage.WINDOWS)
                except Exception:
                    pass
                try:
                    reason_label = {
                        "vol_extreme": Reason.EXTREME_VOL,
                        "spread_wide": Reason.EXTREME_SPREAD,
                    }.get(log_reason, Reason.OTHER)
                    monitoring.inc_reason(reason_label)
                except Exception:
                    pass
                try:
                    pipeline_stage_drop_count.labels(
                        bar.symbol,
                        Stage.WINDOWS.name,
                        log_reason,
                    ).inc()
                except Exception:
                    pass
                return _finalize()

        policy_stage_cfg = (
            self._pipeline_cfg.get("policy") if self._pipeline_cfg else None
        )
        policy_stage_enabled = policy_stage_cfg is None or policy_stage_cfg.enabled
        precomputed_feats: dict[str, Any] | None = None
        if policy_stage_enabled and self._signal_quality_cfg.enabled:
            allowed, feats = self._apply_signal_quality_filter(
                bar, skip_metrics=skip_metrics
            )
            precomputed_feats = feats
            if not allowed:
                return _finalize()

        if policy_stage_enabled and precomputed_feats is None:
            precomputed_feats = self._extract_features(
                bar, skip_metrics=skip_metrics
            )
        elif not policy_stage_enabled and precomputed_feats is None:
            precomputed_feats = {}

        windows_stage_cfg = (
            self._pipeline_cfg.get("windows") if self._pipeline_cfg else None
        )
        win_res, win_reason = self._evaluate_no_trade_windows(
            int(bar.ts),
            bar.symbol,
            stage_cfg=windows_stage_cfg,
        )
        if win_res.action == "drop":
            try:
                log_reason = "maintenance" if win_reason else getattr(
                    win_res.reason, "name", ""
                )
                payload = {
                    "stage": win_res.stage.name,
                    "reason": log_reason,
                }
                if win_reason:
                    payload["detail"] = win_reason
                self._logger.info("DROP %s", payload)
            except Exception:
                pass
            try:
                reason_label = "maintenance" if win_reason else (
                    win_res.reason.name if win_res.reason else ""
                )
                pipeline_stage_drop_count.labels(
                    bar.symbol,
                    win_res.stage.name,
                    reason_label,
                ).inc()
            except Exception:
                pass
            return _finalize()

        pol_res = policy_decide(
            self._fp,
            self._policy,
            bar,
            stage_cfg=policy_stage_cfg,
            signal_quality_cfg=self._signal_quality_cfg,
            precomputed_features=precomputed_feats,
        )
        if self._state_enabled:
            consumer = getattr(self._policy, "consume_dirty_signal_state", None)
            if callable(consumer):
                try:
                    exported = consumer()
                except Exception:
                    exported = None
                if exported:
                    try:
                        state_storage.update_state(signal_states=exported)
                    except Exception:
                        pass
        if pol_res.action == "drop":
            try:
                self._logger.info(
                    "DROP %s",
                    {
                        "stage": pol_res.stage.name,
                        "reason": getattr(pol_res.reason, "name", ""),
                    },
                )
            except Exception:
                pass
            try:
                pipeline_stage_drop_count.labels(
                    bar.symbol,
                    pol_res.stage.name,
                    pol_res.reason.name if pol_res.reason else "",
                ).inc()
            except Exception:
                pass
            return _finalize()
        orders = list(pol_res.decision or [])

        try:
            transitions = list(self._policy.consume_signal_transitions())
        except Exception:
            transitions = []
        if transitions and self._entry_limiter.enabled:
            orders = self._apply_entry_limiter(
                orders,
                transitions,
                default_symbol=str(getattr(bar, "symbol", "")),
                ts_ms=int(getattr(bar, "ts", 0)),
            )

        risk_res = apply_risk(
            int(bar.ts),
            bar.symbol,
            self._guards,
            orders,
            stage_cfg=self._pipeline_cfg.get("risk") if self._pipeline_cfg else None,
        )
        if risk_res.action == "drop":
            try:
                self._logger.info(
                    "DROP %s",
                    {
                        "stage": risk_res.stage.name,
                        "reason": getattr(risk_res.reason, "name", ""),
                    },
                )
            except Exception:
                pass
            try:
                pipeline_stage_drop_count.labels(
                    bar.symbol,
                    risk_res.stage.name,
                    risk_res.reason.name if risk_res.reason else "",
                ).inc()
            except Exception:
                pass
            return _finalize()
        orders = list(risk_res.decision or [])

        if self._portfolio_guard is not None and orders:
            guard_orders, guard_reason = self._portfolio_guard.apply(
                int(bar.ts), bar.symbol, orders
            )
            guard_orders = list(guard_orders or [])
            if guard_reason:
                try:
                    self._logger.info(
                        "DROP %s",
                        {
                            "stage": Stage.RISK.name,
                            "reason": guard_reason,
                        },
                    )
                except Exception:
                    pass
                try:
                    pipeline_stage_drop_count.labels(
                        bar.symbol,
                        Stage.RISK.name,
                        guard_reason,
                    ).inc()
                except Exception:
                    pass
                return _finalize()
            orders = guard_orders

        if not orders:
            return _finalize()

        bar_snapshot = bar if self._execution_mode == "bar" else None
        bar_payload_snapshot: Any | None = None
        if bar_snapshot is not None:
            to_dict = getattr(bar_snapshot, "to_dict", None)
            if callable(to_dict):
                try:
                    bar_payload_snapshot = to_dict()
                except Exception:
                    bar_payload_snapshot = None
        if self._execution_mode == "bar":
            orders = self._apply_daily_turnover_limits(orders, bar.symbol, int(bar.ts))
            if not orders:
                return _finalize()
            for order_candidate in orders:
                meta_map = self._ensure_order_meta(order_candidate)
                meta_map["bar"] = bar_snapshot if bar_snapshot is not None else bar
                payload_map = self._coerce_mapping(meta_map.get("payload"))
                if payload_map is not None:
                    payload_map["bar"] = (
                        bar_payload_snapshot
                        if bar_payload_snapshot is not None
                        else bar_snapshot
                    )
                    meta_map["payload"] = payload_map

        created_ts_ms = clock.now_ms()
        self._stage_exposure_adjustments(orders, created_ts_ms)
        checked_orders = []

        for o in orders:
            if ttl_enabled and ttl_timeframe_ms is not None:
                try:
                    monitoring.inc_stage(Stage.TTL)
                except Exception:
                    pass
                ok, expires_at_ms, _ = check_ttl(
                    bar_close_ms=bar_close_ms,
                    now_ms=created_ts_ms,
                    timeframe_ms=ttl_timeframe_ms,
                )
                if not ok:
                    try:
                        self._logger.info(
                            "TTL_EXPIRED_BOUNDARY %s",
                            {
                                "symbol": bar.symbol,
                                "bar_close_ms": bar_close_ms,
                                "now_ms": created_ts_ms,
                                "expires_at_ms": expires_at_ms,
                            },
                        )
                    except Exception:
                        pass
                    try:
                        monitoring.ttl_expired_boundary_count.labels(bar.symbol).inc()
                        monitoring.signal_boundary_count.labels(bar.symbol).inc()
                    except Exception:
                        pass
                    try:
                        pipeline_stage_drop_count.labels(
                            bar.symbol, Stage.TTL.name, Reason.OTHER.name
                        ).inc()
                    except Exception:
                        pass
                    self._rollback_exposure(o)
                    continue

            self._store_created_ts(o, created_ts_ms)
            checked_orders.append(o)

        if checked_orders:
            checked_orders, _ = self._normalize_weight_targets(checked_orders)

        for o in checked_orders:
            res = self.publish_decision(
                o,
                bar.symbol,
                bar_open_ms,
                bar_close_ms=bar_close_ms,
                stage_cfg=(
                    self._pipeline_cfg.get("publish") if self._pipeline_cfg else None
                ),
            )
            if res.action == "pass":
                emitted.append(o)
            elif res.action == "drop":
                try:
                    self._logger.info(
                        "DROP %s",
                        {
                            "stage": res.stage.name,
                            "reason": getattr(res.reason, "name", ""),
                        },
                    )
                except Exception:
                    pass
                try:
                    pipeline_stage_drop_count.labels(
                        bar.symbol,
                        res.stage.name,
                        res.reason.name if res.reason else "",
                    ).inc()
                except Exception:
                    pass

        self._update_queue_metrics()

        if self._ws_dedup_enabled:
            try:
                signal_bus.update(bar.symbol, bar_close_ms)
            except Exception:
                pass
        return _finalize()

    # Backwards compatibility: legacy callers may still use ``on_bar``.
    on_bar = process


async def worker_loop(bus: "EventBus", worker: _Worker) -> None:
    """Consume bars from ``bus`` and pass them to ``worker``.

    The function exits when :meth:`EventBus.close` is called and all
    remaining events are processed.  On cancellation, it drains any queued
    events before re-raising :class:`asyncio.CancelledError`.
    """
    import asyncio

    try:
        while True:
            event = await bus.get()
            if event is None:
                break
            tick = getattr(event, "tick", None)
            if tick is not None:
                worker.on_tick(tick)
            bar = getattr(event, "bar", None)
            if bar is not None:
                worker.process(bar)
    except asyncio.CancelledError:
        # Drain remaining events best-effort before shutting down
        try:
            while True:
                ev = bus._queue.get_nowait()  # type: ignore[attr-defined]
                if ev is None:
                    break
                tick = getattr(ev, "tick", None)
                if tick is not None:
                    worker.on_tick(tick)
                bar = getattr(ev, "bar", None)
                if bar is not None:
                    worker.process(bar)
        except Exception:
            pass
        raise


def _attach_monitoring_target(
    target: Any, agg: MonitoringAggregator | None
) -> None:
    """Best-effort attachment of ``agg`` to ``target`` if supported."""

    if agg is None or target is None:
        return
    setter = getattr(target, "set_monitoring_aggregator", None)
    if callable(setter):
        try:
            setter(agg)
            return
        except Exception:
            pass
    setter = getattr(target, "set_monitoring", None)
    if callable(setter):
        try:
            setter(agg)
            return
        except Exception:
            pass
    try:
        setattr(target, "monitoring_agg", agg)
        return
    except Exception:
        pass
    if hasattr(target, "_monitoring"):
        try:
            setattr(target, "_monitoring", agg)
        except Exception:
            pass


class ServiceSignalRunner:
    def __init__(
        self,
        adapter: SimAdapter,
        feature_pipe: FeaturePipe,
        policy: SignalPolicy,
        risk_guards: Optional[RiskGuards] = None,
        cfg: Optional[SignalRunnerConfig] = None,
        clock_sync_cfg: ClockSyncConfig | None = None,
        throttle_cfg: ThrottleConfig | None = None,
        no_trade_cfg: NoTradeConfig | None = None,
        signal_quality_cfg: SignalQualityConfig | None = None,
        portfolio_limits: Mapping[str, Any] | None = None,
        pipeline_cfg: PipelineConfig | None = None,
        shutdown_cfg: Dict[str, Any] | None = None,
        monitoring_cfg: MonitoringConfig | None = None,
        monitoring_agg: MonitoringAggregator | None = None,
        run_config: CommonRunConfig | None = None,
        *,
        enforce_closed_bars: bool = True,
        close_lag_ms: int = 0,
        ws_dedup_enabled: bool = False,
        ws_dedup_log_skips: bool = False,
        ws_dedup_timeframe_ms: int = 0,
        signal_writer_options: Mapping[str, Any] | None = None,
        signal_bus_config: Mapping[str, Any] | None = None,
        risk_component_params: Mapping[str, Any] | None = None,
    ) -> None:
        self.adapter = adapter
        self.feature_pipe = feature_pipe
        self.policy = policy
        self.risk_guards = risk_guards
        self.cfg = cfg or SignalRunnerConfig()
        self.logger = logging.getLogger(__name__)
        self.clock_sync_cfg = clock_sync_cfg
        self.throttle_cfg = throttle_cfg
        self.no_trade_cfg = no_trade_cfg
        self.signal_quality_cfg = signal_quality_cfg or SignalQualityConfig()
        limits_payload = portfolio_limits
        if limits_payload is None:
            limits_payload = getattr(self.cfg, "portfolio_limits", None)
        self._portfolio_limits_cfg = dict(limits_payload or {})
        self.pipeline_cfg = pipeline_cfg
        self.shutdown_cfg = shutdown_cfg or {}
        self.monitoring_cfg = monitoring_cfg or MonitoringConfig()
        self._run_config = run_config
        exec_cfg = getattr(run_config, "execution", None) if run_config is not None else None
        mode = "bar"
        if exec_cfg is not None:
            mode = str(getattr(exec_cfg, "mode", mode) or mode)
        mode = mode.strip().lower()
        if mode not in {"bar", "order"}:
            mode = "bar"
        self._execution_mode = mode
        self.alerts: AlertManager | None = None
        self.monitoring_agg: MonitoringAggregator | None = None
        self._clock_safe_mode = False
        self._clock_stop = threading.Event()
        self._clock_thread: Optional[threading.Thread] = None
        self.enforce_closed_bars = enforce_closed_bars
        try:
            self.close_lag_ms = max(0, int(close_lag_ms))
        except (TypeError, ValueError):
            self.close_lag_ms = 0
        self.ws_dedup_enabled = ws_dedup_enabled
        self.ws_dedup_log_skips = ws_dedup_log_skips
        self.ws_dedup_timeframe_ms = ws_dedup_timeframe_ms
        self._metadata_persisted = False
        self._git_hash: str | None = None
        logs_dir = Path(self.cfg.logs_dir or "logs")
        self._runner_status_path = logs_dir / "runner_status.json"
        self._reload_flag_path = logs_dir / "reload_request.json"
        self._safe_stop_flag_path = logs_dir / "safe_stop.request"
        self._runner_status_lock = threading.Lock()
        self._runner_status_workers: Dict[str, Dict[str, Any]] = {}
        self._worker_ref: _Worker | None = None
        self._last_safe_mode_state: Dict[str, bool] = {}
        self._signal_writer_options = dict(signal_writer_options or {})
        self._signal_writer_stats: Dict[str, Any] = {}
        self._signal_writer_ref: SignalCSVWriter | None = None
        self._signal_writer_reopen_flag_path = logs_dir / "signal_writer_reopen.flag"
        self._signal_bus_config: Dict[str, Any] = dict(signal_bus_config or {})
        self._risk_component_params: Dict[str, Any] = dict(risk_component_params or {})
        self._reload_history: deque[Dict[str, Any]] = deque(maxlen=20)
        self._last_reload_event: Dict[str, Any] | None = None
        self._dirty_restart = False
        self._dirty_restart_detected_at: int | None = None
        self._dirty_marker_path: Path | None = None
        self._shutdown_ref: ShutdownManager | None = None
        self._base_pipeline_cfg = pipeline_cfg or PipelineConfig()
        self._ops_pipeline_cfg = PipelineConfig()
        self._runtime_pipeline_cfg = PipelineConfig()
        self._slippage_regime_listener: Callable[[Any], None] | None = None
        self._symbol_quote_assets: Dict[str, str] = {}
        self._symbol_quote_missing: list[str] = []
        self._resolved_quote_asset: str | None = None
        self._symbol_specs: Dict[str, Dict[str, Any]] = {}
        self._init_failure_reason: str | None = None
        self._init_failure_details: Dict[str, Any] = {}

        monitoring.clear_runtime_aggregator()
        if self.monitoring_cfg.enabled:
            agg: MonitoringAggregator | None = monitoring_agg
            if agg is not None:
                self.alerts = getattr(agg, "alerts", None)
            else:
                alert_cfg = getattr(self.monitoring_cfg, "alerts", None)
                try:
                    self.alerts = AlertManager(alert_cfg)
                except Exception:
                    self.alerts = None
                else:
                    try:
                        agg = MonitoringAggregator(self.monitoring_cfg, self.alerts)
                    except Exception:
                        agg = None
                        self.alerts = None
            if agg is not None and agg.enabled:
                self.monitoring_agg = agg
                monitoring.set_runtime_aggregator(self.monitoring_agg)
            else:
                self.monitoring_agg = None
        else:
            self.monitoring_agg = None

        if self.monitoring_agg is not None:
            _attach_monitoring_target(self.adapter, self.monitoring_agg)
            for name in ("source", "market_data"):
                _attach_monitoring_target(
                    getattr(self.adapter, name, None), self.monitoring_agg
                )

        if self.signal_quality_cfg.enabled:
            self.logger.info(
                "signal quality guard: enabled sigma_window=%s sigma_threshold=%s "
                "vol_median_window=%s vol_floor_frac=%s log_reason=%s",
                self.signal_quality_cfg.sigma_window,
                self.signal_quality_cfg.sigma_threshold,
                self.signal_quality_cfg.vol_median_window,
                self.signal_quality_cfg.vol_floor_frac,
                self.signal_quality_cfg.log_reason,
            )

        if self.throttle_cfg is not None:
            self.logger.info(
                "throttle limits: enabled=%s global_rps=%s global_burst=%s symbol_rps=%s symbol_burst=%s mode=%s",
                self.throttle_cfg.enabled,
                self.throttle_cfg.global_.rps,
                self.throttle_cfg.global_.burst,
                self.throttle_cfg.symbol.rps,
                self.throttle_cfg.symbol.burst,
                self.throttle_cfg.mode,
            )

        if self.no_trade_cfg is not None:
            try:
                maintenance_cfg = getattr(self.no_trade_cfg, "maintenance", None)
                if maintenance_cfg is not None:
                    try:
                        maintenance_payload = maintenance_cfg.dict()
                    except Exception:
                        maintenance_payload = {
                            "funding_buffer_min": getattr(
                                maintenance_cfg, "funding_buffer_min", None
                            ),
                            "daily_utc": list(
                                getattr(maintenance_cfg, "daily_utc", []) or []
                            ),
                            "custom_ms": list(
                                getattr(maintenance_cfg, "custom_ms", []) or []
                            ),
                        }
                else:
                    maintenance_payload = {
                        "funding_buffer_min": getattr(
                            self.no_trade_cfg, "funding_buffer_min", None
                        ),
                        "daily_utc": list(
                            getattr(self.no_trade_cfg, "daily_utc", []) or []
                        ),
                        "custom_ms": list(
                            getattr(self.no_trade_cfg, "custom_ms", []) or []
                        ),
                    }
                self.logger.info(
                    "no-trade maintenance config: %s", maintenance_payload
                )
            except Exception:
                pass
            try:
                maintenance_path = Path(DEFAULT_NO_TRADE_STATE_PATH)
                exists = maintenance_path.exists()
                age_sec: float | None = None
                if exists:
                    try:
                        age_sec = max(
                            0.0, time.time() - maintenance_path.stat().st_mtime
                        )
                    except Exception:
                        age_sec = None
                age_repr = None
                if age_sec is not None:
                    age_repr = f"{age_sec:.0f}"
                self.logger.info(
                    "no-trade maintenance file: path=%s exists=%s age_sec=%s",
                    str(maintenance_path),
                    exists,
                    age_repr,
                )
            except Exception:
                pass

        dyn_cfg = getattr(self.no_trade_cfg, "dynamic_guard", None) if self.no_trade_cfg else None
        if dyn_cfg is not None:
            enabled = bool(getattr(dyn_cfg, "enable", False))
            if enabled:
                self.logger.info(
                    "dynamic guard: enabled sigma_window=%s atr_window=%s vol_abs=%s vol_pctile=%s spread_abs_bps=%s spread_pctile=%s hysteresis=%s cooldown_bars=%s log_reason=%s",
                    getattr(dyn_cfg, "sigma_window", None),
                    getattr(dyn_cfg, "atr_window", None),
                    getattr(dyn_cfg, "vol_abs", None),
                    getattr(dyn_cfg, "vol_pctile", None),
                    getattr(dyn_cfg, "spread_abs_bps", None),
                    getattr(dyn_cfg, "spread_pctile", None),
                    getattr(dyn_cfg, "hysteresis", None),
                    getattr(dyn_cfg, "cooldown_bars", None),
                    getattr(dyn_cfg, "log_reason", None),
                )
            else:
                self.logger.info("dynamic guard: disabled")
        else:
            self.logger.info("dynamic guard: not configured")

        if self._execution_mode != "bar":
            run_id = self.cfg.run_id or "sim"
            logs_dir = self.cfg.logs_dir or "logs"
            sim = getattr(self.adapter, "sim", None) or getattr(self.adapter, "_sim", None)
            if sim is not None:
                logging_config = {
                    "trades_path": os.path.join(logs_dir, f"log_trades_{run_id}.csv"),
                    "reports_path": os.path.join(logs_dir, f"report_equity_{run_id}.csv"),
                }
                try:
                    from sim_logging import LogWriter, LogConfig  # type: ignore

                    sim._logger = LogWriter(
                        LogConfig.from_dict(logging_config), run_id=run_id
                    )
                except Exception:
                    pass

        self._setup_market_regime_bridge()
        self._validate_symbol_quotes()

    def _iter_market_regime_sources(self) -> Iterator[Any]:
        adapter = getattr(self, "adapter", None)
        if adapter is None:
            return
        seen: set[int] = set()

        def _candidates(obj: Any) -> Iterator[Any]:
            if obj is None:
                return
            yield obj
            for name in ("sim", "_sim", "executor", "_executor"):
                nested = getattr(obj, name, None)
                if nested is not None:
                    yield nested

        for candidate in _candidates(adapter):
            if candidate is None:
                continue
            ident = id(candidate)
            if ident in seen:
                continue
            seen.add(ident)
            yield candidate

    @staticmethod
    def _extract_market_regime(owner: Any) -> Any:
        attr_names = (
            "current_market_regime",
            "market_regime",
            "_current_market_regime",
            "_last_market_regime",
        )
        for name in attr_names:
            if not hasattr(owner, name):
                continue
            try:
                value = getattr(owner, name)
            except Exception:
                continue
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            if value is not None:
                return value
        getter = getattr(owner, "get_market_regime", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                return None
        return None

    def _resolve_market_regime_source(
        self,
    ) -> tuple[Callable[[Callable[[Any], None]], None], Any] | None:
        for owner in self._iter_market_regime_sources():
            register = getattr(owner, "register_market_regime_listener", None)
            if not callable(register):
                continue
            initial = self._extract_market_regime(owner)
            return register, initial
        return None

    def _resolve_slippage_target(self) -> tuple[Any, Callable[[Any], None]] | None:
        adapter = getattr(self, "adapter", None)
        if adapter is None:
            return None
        candidates: list[Any] = [adapter]
        possible_attrs = [
            "_slippage_impl",
            "slippage_impl",
            "slippage",
            "executor",
            "_executor",
            "sim",
            "_sim",
        ]
        for attr in possible_attrs:
            obj = getattr(adapter, attr, None)
            if obj is None:
                continue
            candidates.append(obj)
            inner = getattr(obj, "_slippage_impl", None)
            if inner is not None:
                candidates.append(inner)
        seen: set[int] = set()
        for obj in candidates:
            if obj is None:
                continue
            ident = id(obj)
            if ident in seen:
                continue
            seen.add(ident)
            setter = getattr(obj, "set_market_regime", None)
            if callable(setter):
                return obj, setter
        return None

    def _setup_market_regime_bridge(self) -> None:
        run_cfg = self._run_config
        updates_enabled = True
        calibration_enabled: bool | None = None
        if run_cfg is not None:
            updates_enabled = bool(getattr(run_cfg, "slippage_regime_updates", True))
            calibration_enabled = bool(
                getattr(run_cfg, "slippage_calibration_enabled", False)
            )
        slippage_target = self._resolve_slippage_target()
        if not updates_enabled or slippage_target is None:
            return
        slippage_obj, setter = slippage_target
        if calibration_enabled is None:
            calibration_enabled = bool(
                getattr(slippage_obj, "_calibration_enabled", False)
            )
        elif not calibration_enabled:
            calibration_enabled = bool(
                getattr(slippage_obj, "_calibration_enabled", False)
            )
        if not calibration_enabled:
            return
        source = self._resolve_market_regime_source()
        if source is None:
            initial = getattr(slippage_obj, "_current_market_regime", None)
            if initial is None:
                initial = self._extract_market_regime(slippage_obj)
            if initial is not None:
                try:
                    setter(initial)
                except Exception:
                    self.logger.debug(
                        "failed to seed slippage market regime", exc_info=True
                    )
            return
        register, initial = source
        if initial is not None:
            try:
                setter(initial)
            except Exception:
                self.logger.debug(
                    "failed to seed slippage market regime", exc_info=True
                )

        def _listener(regime: Any) -> None:
            try:
                setter(regime)
            except Exception:
                self.logger.debug(
                    "failed to forward market regime to slippage", exc_info=True
                )

        try:
            register(_listener)
        except Exception:
            self.logger.debug(
                "failed to register market regime listener", exc_info=True
            )
        else:
            self._slippage_regime_listener = _listener

    @staticmethod
    def _coerce_signal_dispatcher(owner: Any) -> Callable[[Any], None] | None:
        if owner is None:
            return None
        explicit = getattr(owner, "signal_dispatcher", None)
        if callable(explicit):
            return explicit
        method_names = (
            "dispatch_signal_envelope",
            "on_bar_signal",
            "on_signal_envelope",
            "send_signal_envelope",
            "handle_signal_envelope",
        )
        for name in method_names:
            candidate = getattr(owner, name, None)
            if callable(candidate):
                return candidate
        return None

    def _resolve_signal_dispatcher(self, executor: Any) -> Callable[[Any], None] | None:
        seen: set[int] = set()
        stack: list[Any] = []

        adapter = getattr(self, "adapter", None)
        if adapter is not None:
            stack.append(adapter)
        if executor is not None:
            stack.append(executor)

        nested_attrs = (
            "adapter",
            "_adapter",
            "executor",
            "_executor",
            "transport",
            "_transport",
            "sim",
            "_sim",
        )

        while stack:
            obj = stack.pop()
            if obj is None:
                continue
            ident = id(obj)
            if ident in seen:
                continue
            seen.add(ident)

            dispatcher = self._coerce_signal_dispatcher(obj)
            if dispatcher is not None:
                return dispatcher

            for attr in nested_attrs:
                try:
                    nested = getattr(obj, attr)
                except Exception:
                    nested = None
                if nested is None:
                    continue
                stack.append(nested)

        return None

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, MappingABC):
            return {
                str(key): ServiceSignalRunner._to_jsonable(val)
                for key, val in value.items()
                if val is not None
            }
        if isinstance(value, (set, tuple, list)):
            return [
                ServiceSignalRunner._to_jsonable(item)
                for item in value
                if item is not None
            ]
        try:
            return str(value)
        except Exception:
            return repr(value)

    def _resolve_worker_cooldown_settings(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        keys = (
            "cooldown_bars",
            "cooldown_small_delta",
            "small_delta_threshold",
            "small_delta",
            "cooldown_large_trade",
            "large_trade_threshold",
            "large_trade",
            "turnover_cap",
        )
        for key in keys:
            if key in self._risk_component_params:
                params[key] = self._risk_component_params[key]
        if (
            "cooldown_small_delta" not in params
            and "small_delta_threshold" not in params
            and "small_delta" not in params
        ):
            exec_cfg = getattr(self._run_config, "execution", None)
            if exec_cfg is not None:
                candidate = getattr(exec_cfg, "min_rebalance_step", None)
                if candidate is not None:
                    params.setdefault("cooldown_small_delta", candidate)
        return params

    def _record_init_failure(
        self, reason: str, details: Mapping[str, Any] | None = None
    ) -> None:
        self._init_failure_reason = reason
        if isinstance(details, MappingABC):
            sanitized = {
                str(key): self._to_jsonable(val)
                for key, val in details.items()
                if val is not None
            }
        elif details is not None:
            sanitized = {"detail": self._to_jsonable(details)}
        else:
            sanitized = {}
        self._init_failure_details = sanitized
        self._resolved_quote_asset = None
        try:
            self.logger.error(
                "runner initialisation aborted: reason=%s details=%s",
                reason,
                sanitized or {},
            )
        except Exception:
            pass
        self._write_runner_status()

    def _collect_configured_symbols(self) -> set[str]:
        run_cfg = self._run_config
        symbols: set[str] = set()
        if run_cfg is None:
            return symbols

        def _add(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, str):
                text = value.strip().upper()
                if text:
                    symbols.add(text)
                return
            if isinstance(value, MappingABC):
                if "symbol" in value:
                    _add(value.get("symbol"))
                if "symbols" in value:
                    _add(value.get("symbols"))
                return
            if isinstance(value, SequenceABC) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for item in value:
                    _add(item)

        _add(getattr(run_cfg, "symbol", None))
        _add(getattr(run_cfg, "symbols", None))

        data_cfg = getattr(run_cfg, "data", None)
        if data_cfg is not None:
            _add(getattr(data_cfg, "symbol", None))
            _add(getattr(data_cfg, "symbols", None))

        exec_cfg = getattr(run_cfg, "execution", None)
        if exec_cfg is not None:
            _add(getattr(exec_cfg, "symbol", None))
            _add(getattr(exec_cfg, "symbols", None))

        portfolio_cfg = getattr(run_cfg, "portfolio", None)
        if portfolio_cfg is not None:
            _add(getattr(portfolio_cfg, "symbol", None))
            _add(getattr(portfolio_cfg, "symbols", None))

        components = getattr(run_cfg, "components", None)
        component_names = (
            "market_data",
            "executor",
            "feature_pipe",
            "policy",
            "risk_guards",
            "backtest_engine",
        )
        for name in component_names:
            comp = getattr(components, name, None)
            params = getattr(comp, "params", None) if comp is not None else None
            if isinstance(params, MappingABC):
                _add(params.get("symbol"))
                _add(params.get("symbols"))

        return symbols

    def _load_symbol_metadata(self) -> tuple[dict[str, Dict[str, Any]], dict[str, str]]:
        run_cfg = self._run_config
        specs: dict[str, Dict[str, Any]] = {}
        quotes: dict[str, str] = {}
        if run_cfg is None:
            return specs, quotes

        candidate_payloads: list[Mapping[str, Any]] = []
        candidate_paths: list[str] = []

        def _push_payload(payload: Any) -> None:
            if isinstance(payload, MappingABC) and payload:
                candidate_payloads.append(payload)

        def _push_path(path_value: Any) -> None:
            if not path_value:
                return
            text = str(path_value).strip()
            if text:
                candidate_paths.append(text)

        _push_payload(getattr(run_cfg, "symbol_specs", None))
        _push_path(getattr(run_cfg, "symbol_specs_path", None))

        exec_cfg = getattr(run_cfg, "execution", None)
        if exec_cfg is not None:
            _push_payload(getattr(exec_cfg, "symbol_specs", None))
            _push_path(getattr(exec_cfg, "symbol_specs_path", None))

        portfolio_cfg = getattr(run_cfg, "portfolio", None)
        if portfolio_cfg is not None:
            _push_payload(getattr(portfolio_cfg, "symbol_specs", None))
            _push_path(getattr(portfolio_cfg, "symbol_specs_path", None))

        components = getattr(run_cfg, "components", None)
        component_names = (
            "market_data",
            "executor",
            "feature_pipe",
            "policy",
            "risk_guards",
            "backtest_engine",
        )
        for name in component_names:
            comp = getattr(components, name, None)
            params = getattr(comp, "params", None) if comp is not None else None
            if not isinstance(params, MappingABC):
                continue
            _push_payload(params.get("symbol_specs"))
            for key in ("symbol_specs_path", "exchange_specs_path"):
                if key in params:
                    _push_path(params.get(key))

        default_paths = [
            "configs/symbol_specs.yaml",
            "configs/symbol_specs.yml",
            "configs/symbol_specs.json",
            "data/symbol_specs.yaml",
            "data/symbol_specs.yml",
            "data/symbol_specs.json",
        ]

        ordered_paths: list[Path] = []
        seen_paths: set[str] = set()
        for raw in candidate_paths:
            path_obj = Path(raw).expanduser()
            key = str(path_obj)
            if key in seen_paths or not path_obj.exists():
                continue
            ordered_paths.append(path_obj)
            seen_paths.add(key)
        for raw in default_paths:
            path_obj = Path(raw)
            key = str(path_obj)
            if key in seen_paths or not path_obj.exists():
                continue
            ordered_paths.append(path_obj)
            seen_paths.add(key)

        for path_obj in ordered_paths:
            try:
                with path_obj.open("r", encoding="utf-8") as fh:
                    if path_obj.suffix.lower() in {".yaml", ".yml"}:
                        payload = yaml.safe_load(fh) or {}
                    else:
                        payload = json.load(fh)
            except FileNotFoundError:
                continue
            except Exception:
                self.logger.exception(
                    "failed to load symbol specs: path=%s", path_obj
                )
                continue
            if isinstance(payload, MappingABC):
                candidate_payloads.append(payload)
            else:
                self.logger.warning(
                    "symbol specs path %s did not contain a mapping; ignoring",
                    path_obj,
                )

        reserved_keys = {
            "META",
            "METADATA",
            "INFO",
            "NOTES",
            "SYMBOLS",
            "SPECS",
            "DATA",
            "SYMBOL",
            "QUOTE_ASSET",
            "QUOTEASSET",
            "QUOTE_CURRENCY",
            "QUOTECURRENCY",
            "QUOTE",
            "QUOTE_SYMBOL",
            "QUOTESYMBOL",
            "MIN_NOTIONAL",
            "MINNOTIONAL",
            "STEP_SIZE",
            "STEPSIZE",
            "TICK_SIZE",
            "TICKSIZE",
        }

        def _merge_spec(symbol: str, payload: Any) -> None:
            symbol_key = symbol.strip().upper()
            if not symbol_key:
                return
            entry = specs.setdefault(symbol_key, {})
            if isinstance(payload, MappingABC):
                for key, value in payload.items():
                    entry[str(key)] = value
            elif isinstance(payload, str):
                text = payload.strip()
                if text:
                    entry.setdefault("quote_asset", text)

        def _merge_payload(payload: Any, depth: int = 0) -> None:
            if depth > 6 or payload is None:
                return
            if isinstance(payload, MappingABC):
                symbol_candidate = None
                for key in ("symbol", "Symbol", "SYMBOL"):
                    if key in payload:
                        raw_symbol = payload.get(key)
                        if isinstance(raw_symbol, str) and raw_symbol.strip():
                            symbol_candidate = raw_symbol.strip().upper()
                            break
                if symbol_candidate:
                    _merge_spec(symbol_candidate, payload)
                for raw_key, raw_value in payload.items():
                    key_text = str(raw_key).strip()
                    if not key_text:
                        _merge_payload(raw_value, depth + 1)
                        continue
                    upper_key = key_text.upper()
                    if upper_key in reserved_keys:
                        _merge_payload(raw_value, depth + 1)
                        continue
                    if isinstance(raw_value, MappingABC):
                        _merge_spec(upper_key, raw_value)
                        _merge_payload(raw_value, depth + 1)
                        continue
                    if isinstance(raw_value, str):
                        text = raw_value.strip().upper()
                        if text:
                            entry = specs.setdefault(upper_key, {})
                            entry.setdefault("quote_asset", text)
                            quotes[upper_key] = text
                        continue
                    if isinstance(raw_value, SequenceABC) and not isinstance(
                        raw_value, (str, bytes, bytearray)
                    ):
                        for item in raw_value:
                            _merge_payload(item, depth + 1)
                        continue
            elif isinstance(payload, SequenceABC) and not isinstance(
                payload, (str, bytes, bytearray)
            ):
                for item in payload:
                    _merge_payload(item, depth + 1)

        def _extract_quote(value: Any, depth: int = 0) -> str | None:
            if depth > 6 or value is None:
                return None
            if isinstance(value, str):
                text = value.strip().upper()
                return text or None
            if isinstance(value, MappingABC):
                keys = (
                    "quote_asset",
                    "quoteAsset",
                    "quote_currency",
                    "quoteCurrency",
                    "quote",
                    "quote_symbol",
                    "quoteSymbol",
                )
                for key in keys:
                    if key in value:
                        found = _extract_quote(value.get(key), depth + 1)
                        if found:
                            return found
                for nested_key in (
                    "metadata",
                    "info",
                    "symbol",
                    "spec",
                    "specs",
                    "data",
                ):
                    if nested_key in value:
                        found = _extract_quote(value.get(nested_key), depth + 1)
                        if found:
                            return found
                for nested_value in value.values():
                    if isinstance(nested_value, MappingABC):
                        found = _extract_quote(nested_value, depth + 1)
                        if found:
                            return found
            return None

        for payload in candidate_payloads:
            _merge_payload(payload)

        if not quotes:
            for symbol, payload in specs.items():
                quote = _extract_quote(payload)
                if quote:
                    quotes[symbol] = quote

        return specs, quotes

    def _validate_symbol_quotes(self) -> None:
        run_cfg = self._run_config
        if run_cfg is None:
            return

        symbols = sorted(self._collect_configured_symbols())
        if not symbols:
            return

        specs_map, quotes_map = self._load_symbol_metadata()
        if specs_map:
            self._symbol_specs = {str(sym): dict(payload) for sym, payload in specs_map.items()}
        else:
            self._symbol_specs = {}
        if not quotes_map:
            try:
                self.logger.warning(
                    "symbol quote metadata unavailable; skipping quote guard"
                )
            except Exception:
                pass
            self._symbol_quote_assets = {}
            self._symbol_quote_missing = []
            return

        resolved: dict[str, str] = {}
        missing: list[str] = []
        for sym in symbols:
            quote = quotes_map.get(sym)
            if not quote:
                missing.append(sym)
                continue
            resolved[sym] = quote

        self._symbol_quote_assets = resolved
        self._symbol_quote_missing = missing

        if missing:
            try:
                self.logger.warning(
                    "quote metadata missing for symbols: %s",
                    ", ".join(missing),
                )
            except Exception:
                pass

        unique_quotes = sorted({quote for quote in resolved.values()})
        if len(unique_quotes) > 1:
            details: Dict[str, Any] = {
                "quotes": resolved,
                "unique_quotes": unique_quotes,
            }
            if missing:
                details["missing"] = missing
            self._record_init_failure("mixed_quote", details)
            raise MixedQuoteError(
                "configured symbols reference multiple quote assets: "
                + ", ".join(unique_quotes)
            )

        if unique_quotes:
            self._resolved_quote_asset = unique_quotes[0]
            try:
                self.logger.info(
                    "quote guard resolved quote asset=%s symbols=%d",
                    self._resolved_quote_asset,
                    len(resolved),
                )
            except Exception:
                pass
        else:
            self._resolved_quote_asset = None

        self._init_failure_reason = None
        self._init_failure_details = {}
        self._write_runner_status()

    def _current_safe_mode_state(self) -> Dict[str, bool]:
        clock_safe = bool(self._clock_safe_mode)
        try:
            kill_switch_active = bool(monitoring.kill_switch_triggered())
        except Exception:
            kill_switch_active = False
        try:
            ops_tripped = bool(ops_kill_switch.tripped())
        except Exception:
            ops_tripped = False
        return {
            "active": clock_safe or kill_switch_active or ops_tripped,
            "clock": clock_safe,
            "kill_switch": kill_switch_active,
            "ops": ops_tripped,
        }

    def _compose_runner_status(self) -> Dict[str, Any]:
        workers = sorted(
            (dict(payload) for payload in self._runner_status_workers.values()),
            key=lambda item: str(item.get("id", "")),
        )
        queue_size = 0
        queue_max = 0
        for payload in workers:
            queue_info = payload.get("queue") or {}
            try:
                queue_size += int(queue_info.get("size", 0) or 0)
            except (TypeError, ValueError):
                pass
            try:
                q_max = int(queue_info.get("max", 0) or 0)
            except (TypeError, ValueError):
                q_max = 0
            queue_max = max(queue_max, q_max)
        status = {
            "updated_at_ms": clock.now_ms(),
            "safe_mode": self._current_safe_mode_state(),
            "queue": {"size": queue_size, "max": queue_max},
            "workers": workers,
        }
        if self._signal_writer_stats:
            status["signal_writer"] = dict(self._signal_writer_stats)
        status["dirty_restart"] = {
            "active": bool(self._dirty_restart),
            "detected_at_ms": self._dirty_restart_detected_at,
        }
        if self._last_reload_event is not None:
            status["last_reload"] = dict(self._last_reload_event)
        if self._reload_history:
            status["reloads"] = list(self._reload_history)
        init_payload: Dict[str, Any] = {"state": "ok"}
        if self._resolved_quote_asset:
            init_payload["quote_asset"] = self._resolved_quote_asset
        if self._symbol_quote_assets:
            init_payload["quotes"] = dict(self._symbol_quote_assets)
        if self._symbol_quote_missing:
            init_payload["missing"] = list(self._symbol_quote_missing)
        if self._init_failure_reason:
            init_payload["state"] = "failed"
            init_payload["reason"] = self._init_failure_reason
            if self._init_failure_details:
                init_payload["details"] = dict(self._init_failure_details)
        status["init"] = init_payload
        return status

    def _write_runner_status_unlocked(self) -> None:
        status = self._compose_runner_status()
        safe_state = status.get("safe_mode")
        if isinstance(safe_state, MappingABC):
            self._last_safe_mode_state = {
                str(key): bool(value) for key, value in safe_state.items()
            }
        try:
            atomic_write_with_retry(
                self._runner_status_path,
                json.dumps(status, ensure_ascii=False),
                retries=3,
                backoff=0.1,
            )
        except Exception:
            self.logger.exception("failed to persist runner status")

    def _write_runner_status(self) -> None:
        with self._runner_status_lock:
            self._write_runner_status_unlocked()

    def _update_runner_status(self, worker_id: str, payload: Dict[str, Any]) -> None:
        with self._runner_status_lock:
            self._runner_status_workers[worker_id] = payload
            self._write_runner_status_unlocked()

    def _refresh_signal_writer_stats(self) -> bool:
        writer = getattr(signal_bus, "OUT_WRITER", None)
        if writer is None:
            if self._signal_writer_stats:
                self._signal_writer_stats = {}
                return True
            return False
        try:
            stats = writer.stats()
        except Exception:
            return False
        if stats != self._signal_writer_stats:
            self._signal_writer_stats = stats
            return True
        return False

    def _handle_signal_writer_reopen_flag(self) -> bool:
        flag_path = self._signal_writer_reopen_flag_path
        if not flag_path.exists():
            return False
        try:
            flag_path.unlink()
        except Exception:
            pass
        writer = getattr(signal_bus, "OUT_WRITER", None)
        if writer is None:
            return False
        try:
            writer.reopen()
        except Exception:
            self.logger.exception("failed to reopen signal writer")
            return False
        try:
            self._signal_writer_stats = writer.stats()
        except Exception:
            self._signal_writer_stats = {}
        return True

    def _apply_pipeline_configs(self) -> None:
        combined = (self._base_pipeline_cfg or PipelineConfig()).merge(
            self._ops_pipeline_cfg or PipelineConfig()
        )
        combined = combined.merge(self._runtime_pipeline_cfg or PipelineConfig())
        self.pipeline_cfg = combined
        if self._worker_ref is not None:
            try:
                self._worker_ref.update_pipeline_config(combined)
            except Exception:
                self.logger.exception("failed to update pipeline config")

    def _apply_runtime_overrides(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, MappingABC):
            return
        throttle_data = data.get("throttle")
        if throttle_data is not None:
            try:
                new_throttle = ThrottleConfig.parse_obj(throttle_data)
            except Exception as exc:
                self.logger.warning("invalid throttle override: %s", exc)
            else:
                self.throttle_cfg = new_throttle
                if self._worker_ref is not None:
                    try:
                        self._worker_ref.update_throttle_config(new_throttle)
                    except Exception:
                        self.logger.exception("failed to apply throttle override")
        ws_data = data.get("ws") or {}
        if isinstance(ws_data, MappingABC):
            enabled = ws_data.get("enabled")
            log_skips = ws_data.get("log_skips")
            timeframe = ws_data.get("timeframe_ms")
            if timeframe is None:
                timeframe = ws_data.get("timeframe")
            if enabled is not None:
                self.ws_dedup_enabled = bool(enabled)
            if log_skips is not None:
                self.ws_dedup_log_skips = bool(log_skips)
            if timeframe is not None:
                try:
                    self.ws_dedup_timeframe_ms = int(timeframe)
                except (TypeError, ValueError):
                    pass
            if self._worker_ref is not None:
                try:
                    self._worker_ref.update_ws_settings(
                        enabled=self.ws_dedup_enabled,
                        log_skips=self.ws_dedup_log_skips,
                        timeframe_ms=self.ws_dedup_timeframe_ms,
                    )
                except Exception:
                    self.logger.exception("failed to update ws settings")

    def _apply_ops_config(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, MappingABC):
            return
        self._apply_runtime_overrides(data)
        runtime_section = data.get("runtime")
        if isinstance(runtime_section, MappingABC):
            self._apply_runtime_overrides(runtime_section)
        pipeline_data = data.get("pipeline")
        if pipeline_data is not None:
            try:
                self._ops_pipeline_cfg = _parse_pipeline_config(pipeline_data)
            except Exception as exc:
                self.logger.warning("invalid ops pipeline override: %s", exc)
            else:
                self._apply_pipeline_configs()
        kill_cfg = data.get("kill_switch") or {}
        if isinstance(kill_cfg, MappingABC):
            cfg_obj = self.cfg.kill_switch_ops
            try:
                cfg_obj.enabled = bool(kill_cfg.get("enabled", cfg_obj.enabled))
            except Exception:
                pass
            for key, attr in (
                ("error_limit", "error_limit"),
                ("duplicate_limit", "duplicate_limit"),
                ("stale_intervals_limit", "stale_intervals_limit"),
            ):
                if key in kill_cfg:
                    try:
                        setattr(cfg_obj, attr, int(kill_cfg.get(key)))
                    except Exception:
                        pass
            if "reset_cooldown_sec" in kill_cfg:
                try:
                    cfg_obj.reset_cooldown_sec = int(kill_cfg.get("reset_cooldown_sec"))
                except Exception:
                    pass
            if "flag_path" in kill_cfg:
                cfg_obj.flag_path = kill_cfg.get("flag_path")
            if "alert_command" in kill_cfg:
                cfg_obj.alert_command = kill_cfg.get("alert_command")
            ops_cfg = {
                "rest_limit": cfg_obj.error_limit,
                "ws_limit": cfg_obj.error_limit,
                "duplicate_limit": cfg_obj.duplicate_limit,
                "stale_limit": cfg_obj.stale_intervals_limit,
                "reset_cooldown_sec": cfg_obj.reset_cooldown_sec,
            }
            if cfg_obj.flag_path:
                ops_cfg["flag_path"] = cfg_obj.flag_path
            if cfg_obj.alert_command:
                ops_cfg["alert_command"] = (
                    shlex.split(cfg_obj.alert_command)
                    if isinstance(cfg_obj.alert_command, str)
                    else cfg_obj.alert_command
                )
            try:
                ops_kill_switch.init(ops_cfg)
            except Exception:
                self.logger.exception("failed to reconfigure ops kill switch")

    def _reconfigure_signal_writer(self) -> None:
        existing = self._signal_writer_ref
        if existing is not None:
            try:
                existing.close()
            except Exception:
                pass
        self._signal_writer_ref = None
        self._signal_writer_stats = {}
        out_csv = getattr(signal_bus, "OUT_CSV", None)
        if not out_csv:
            return
        opts = self._signal_writer_options
        try:
            writer = SignalCSVWriter(
                out_csv,
                fsync_mode=str(opts.get("fsync_mode", "batch")),
                rotate_daily=bool(opts.get("rotate_daily", True)),
                flush_interval_s=opts.get("flush_interval_s"),
            )
        except Exception:
            self.logger.exception("failed to initialise signal writer")
            return
        signal_bus.OUT_WRITER = writer
        self._signal_writer_ref = writer
        try:
            self._signal_writer_stats = writer.stats()
        except Exception:
            self._signal_writer_stats = {}
        shutdown = self._shutdown_ref
        if shutdown is not None:
            shutdown.on_flush(lambda w=writer: w.flush_fsync(force=True))
            shutdown.on_finalize(writer.close)

    def _apply_signal_bus_config(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, MappingABC):
            return
        enabled = bool(data.get("enabled", self._signal_bus_config.get("enabled", False)))
        ttl = int(data.get("ttl_seconds", self._signal_bus_config.get("ttl_seconds", 0)) or 0)
        persist_path = data.get("dedup_persist", self._signal_bus_config.get("persist_path"))
        out_csv = data.get("out_csv", self._signal_bus_config.get("out_csv"))
        flush_interval = data.get(
            "flush_interval_s", self._signal_bus_config.get("flush_interval_s", 60.0)
        )
        try:
            flush_interval = float(flush_interval)
        except (TypeError, ValueError):
            flush_interval = 60.0
        bus_cfg = {
            "enabled": enabled,
            "ttl_seconds": ttl,
            "persist_path": persist_path,
            "out_csv": out_csv,
            "flush_interval_s": flush_interval,
        }
        self._signal_bus_config.update(bus_cfg)
        try:
            signal_bus.init(**bus_cfg)
        except Exception:
            self.logger.exception("failed to reconfigure signal bus")
        self.ws_dedup_enabled = enabled
        fsync_mode = str(data.get("fsync_mode", self._signal_writer_options.get("fsync_mode", "batch")) or "batch")
        rotate_daily = bool(
            data.get("rotate_daily", self._signal_writer_options.get("rotate_daily", True))
        )
        flush_opt = data.get(
            "flush_interval_s", self._signal_writer_options.get("flush_interval_s")
        )
        try:
            flush_opt_val: float | None
            if flush_opt is None:
                flush_opt_val = None
            else:
                flush_opt_val = float(flush_opt)
        except (TypeError, ValueError):
            flush_opt_val = None
        self._signal_writer_options.update(
            {
                "fsync_mode": fsync_mode,
                "rotate_daily": rotate_daily,
                "flush_interval_s": flush_opt_val,
            }
        )
        self._reconfigure_signal_writer()

    def _record_reload_event(self, event: Dict[str, Any]) -> None:
        with self._runner_status_lock:
            self._reload_history.append(event)
            self._last_reload_event = event
            self._write_runner_status_unlocked()

    def _reload_config_bundle(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        timestamp = clock.now_ms()
        details: list[Dict[str, Any]] = []
        errors: list[str] = []
        success = True

        def _append(name: str, path: Path | None, ok: bool, error: str | None = None) -> None:
            nonlocal success
            entry: Dict[str, Any] = {"name": name, "success": bool(ok)}
            if path is not None:
                entry["path"] = str(path)
            if error:
                entry["error"] = error
                errors.append(error)
            if not ok:
                success = False
            details.append(entry)

        # Validate and capture base config (config_live)
        live_path_raw = (
            payload.get("config_live")
            or payload.get("config")
            or self.cfg.snapshot_config_path
            or "configs/config_live.yaml"
        )
        live_path = Path(str(live_path_raw))
        try:
            cfg_obj = load_config(str(live_path))
        except FileNotFoundError:
            _append("config_live", live_path, False, "missing")
        except Exception as exc:
            _append("config_live", live_path, False, str(exc))
        else:
            _append("config_live", live_path, True)
            try:
                with live_path.open("r", encoding="utf-8") as fh:
                    live_raw = yaml.safe_load(fh) or {}
            except Exception:
                live_raw = {}
            if isinstance(live_raw, MappingABC):
                try:
                    self._base_pipeline_cfg = _parse_pipeline_config(
                        live_raw.get("pipeline", {})
                    )
                except Exception as exc:
                    self.logger.warning("invalid base pipeline config: %s", exc)
                else:
                    self._apply_pipeline_configs()
            self.cfg.run_id = getattr(cfg_obj, "run_id", self.cfg.run_id)

        # Runtime overrides
        runtime_targets: list[Path] = []
        paths_field = payload.get("paths")
        if isinstance(paths_field, str):
            runtime_targets.append(Path(paths_field))
        elif isinstance(paths_field, SequenceABC):
            for item in paths_field:
                if not item:
                    continue
                runtime_targets.append(Path(str(item)))
        runtime_key = payload.get("runtime")
        if runtime_key:
            runtime_targets.append(Path(str(runtime_key)))
        if not runtime_targets:
            runtime_targets.append(Path("configs/runtime.yaml"))
        seen_paths: set[str] = set()
        unique_runtime: list[Path] = []
        for item in runtime_targets:
            key = str(item)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            unique_runtime.append(item)
        for rt_path in unique_runtime:
            ok, err = self._reload_runtime_config(rt_path)
            _append("runtime", rt_path, ok, err)

        # Ops configuration (YAML or JSON)
        ops_override = payload.get("ops") or payload.get("ops_path")
        candidate_paths: list[Path] = []
        if ops_override:
            candidate_paths.append(Path(str(ops_override)))
        else:
            candidate_paths.extend([Path("configs/ops.yaml"), Path("configs/ops.json")])
        ops_data: Mapping[str, Any] | None = None
        ops_path_used: Path | None = None
        last_error: str | None = None
        for candidate in candidate_paths:
            if not candidate.exists():
                continue
            try:
                with candidate.open("r", encoding="utf-8") as fh:
                    if candidate.suffix.lower() == ".json":
                        ops_data = json.load(fh) or {}
                    else:
                        ops_data = yaml.safe_load(fh) or {}
                ops_path_used = candidate
            except Exception as exc:
                last_error = str(exc)
                ops_data = None
                ops_path_used = candidate
                break
            else:
                break
        if ops_data is None:
            if ops_path_used is not None:
                _append("ops", ops_path_used, False, last_error or "failed to load")
            else:
                fallback = candidate_paths[0] if candidate_paths else None
                _append("ops", fallback, False, "missing")
        else:
            _append("ops", ops_path_used, True)
            self._apply_ops_config(ops_data)

        # Signals configuration
        signals_path_raw = payload.get("signals") or "configs/signals.yaml"
        signals_path = Path(str(signals_path_raw))
        try:
            with signals_path.open("r", encoding="utf-8") as fh:
                signals_data = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            _append("signals", signals_path, False, "missing")
        except Exception as exc:
            _append("signals", signals_path, False, str(exc))
        else:
            if not isinstance(signals_data, MappingABC):
                _append("signals", signals_path, False, "invalid format")
            else:
                self._apply_signal_bus_config(signals_data)
                _append("signals", signals_path, True)

        event: Dict[str, Any] = {
            "timestamp_ms": timestamp,
            "success": success,
            "details": details,
        }
        if errors:
            event["errors"] = errors
        return event

def clear_dirty_restart(
    marker_path: str | Path,
    state_cfg: StateConfig,
    *,
    runner_status_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Remove dirty restart marker and wipe persistent state."""

    result: Dict[str, Any] = {
        "marker_removed": False,
        "state_cleared": False,
        "status_updated": False,
        "errors": [],
    }

    marker = Path(marker_path)
    try:
        marker.unlink()
    except FileNotFoundError:
        pass
    except Exception as exc:
        result["errors"].append(str(exc))
    else:
        result["marker_removed"] = True

    if getattr(state_cfg, "enabled", False) and getattr(state_cfg, "path", None):
        try:
            state_storage.clear_state(
                state_cfg.path,
                backend=getattr(state_cfg, "backend", "json"),
                lock_path=getattr(state_cfg, "lock_path", None),
                backup_keep=getattr(state_cfg, "backup_keep", 0),
            )
        except Exception as exc:
            result["errors"].append(str(exc))
        else:
            result["state_cleared"] = True

    if runner_status_path:
        status_path = Path(runner_status_path)
        try:
            current_status: Dict[str, Any] = {}
            if status_path.exists():
                with status_path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh) or {}
                if isinstance(payload, MappingABC):
                    current_status = dict(payload)
            dirty_payload = current_status.get("dirty_restart")
            update = {
                "active": False,
                "cleared_at_ms": clock.now_ms(),
            }
            if isinstance(dirty_payload, MappingABC):
                dirty_payload = dict(dirty_payload)
                dirty_payload.update(update)
                current_status["dirty_restart"] = dirty_payload
            else:
                current_status["dirty_restart"] = update
            atomic_write_with_retry(
                status_path,
                json.dumps(current_status, ensure_ascii=False),
                retries=3,
                backoff=0.1,
            )
        except Exception as exc:
            result["errors"].append(str(exc))
        else:
            result["status_updated"] = True
    return result


    def _reload_runtime_config(self, path: Path) -> tuple[bool, str | None]:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            self.logger.warning("runtime config missing: %s", path)
            return False, "missing"
        except Exception as exc:
            self.logger.exception("failed to read runtime config from %s", path)
            return False, str(exc)
        if not isinstance(data, MappingABC):
            self.logger.warning("runtime config %s is not a mapping", path)
            return False, "invalid format"
        self._apply_runtime_overrides(data)
        pipeline_data = data.get("pipeline")
        if pipeline_data is not None:
            try:
                self._runtime_pipeline_cfg = _parse_pipeline_config(pipeline_data)
            except Exception as exc:
                self.logger.warning("invalid runtime pipeline override: %s", exc)
            else:
                self._apply_pipeline_configs()
        self.logger.info("runtime overrides reloaded from %s", path)
        return True, None

    def _handle_reload_request(self) -> bool:
        flag = self._reload_flag_path
        if not flag.exists():
            return False
        payload: Mapping[str, Any]
        try:
            with flag.open("r", encoding="utf-8") as fh:
                raw_payload = json.load(fh) or {}
            payload = raw_payload if isinstance(raw_payload, MappingABC) else {}
        except Exception as exc:
            self.logger.exception("failed to parse reload request")
            event = {
                "timestamp_ms": clock.now_ms(),
                "success": False,
                "details": [],
                "errors": [str(exc)],
            }
            self._record_reload_event(event)
            payload = {}
        finally:
            try:
                flag.unlink()
            except Exception:
                pass
        event = self._reload_config_bundle(payload)
        self._record_reload_event(event)
        return True

    def _handle_safe_stop_flag(
        self, shutdown: ShutdownManager, stop_event: threading.Event
    ) -> None:
        flag = self._safe_stop_flag_path
        if not flag.exists() or stop_event.is_set():
            return
        try:
            self.logger.info("safe stop requested via %s", flag)
        except Exception:
            pass
        stop_event.set()
        shutdown.request_shutdown()
        try:
            flag.unlink()
        except Exception:
            pass
        self._write_runner_status()

    def _process_control_flags(
        self, shutdown: ShutdownManager, stop_event: threading.Event
    ) -> None:
        changed = False
        if self._handle_signal_writer_reopen_flag():
            changed = True
        if self._refresh_signal_writer_stats():
            changed = True
        if self._handle_reload_request():
            changed = True
        self._handle_safe_stop_flag(shutdown, stop_event)
        marker_path = self._dirty_marker_path
        if (
            self._dirty_restart
            and marker_path is not None
            and not marker_path.exists()
        ):
            self._dirty_restart = False
            self._dirty_restart_detected_at = None
            changed = True
        current_state = self._current_safe_mode_state()
        if current_state != self._last_safe_mode_state:
            changed = True
        if changed:
            self._write_runner_status()

    def run(self) -> Iterator[Dict[str, Any]]:
        # снапшот конфига, если задан
        if self.cfg.snapshot_config_path and self.cfg.artifacts_dir:
            snapshot_config(self.cfg.snapshot_config_path, self.cfg.artifacts_dir)

        logs_dir = self.cfg.logs_dir or "logs"
        marker_path = Path(
            self.cfg.marker_path or os.path.join(logs_dir, "shutdown.marker")
        )
        try:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        dirty_restart = marker_path.exists()
        self._dirty_marker_path = marker_path
        self._dirty_restart = dirty_restart
        self._dirty_restart_detected_at = clock.now_ms() if dirty_restart else None
        try:
            marker_path.touch()
        except Exception:
            pass
        entry_limits_state: Dict[str, Dict[str, Any]] | None = None
        bar_initial_weights: Dict[str, float] | None = None
        loaded_state: Any | None = None
        if self.cfg.state.enabled:
            state_path = self.cfg.state.path
            try:
                self.logger.info(
                    "loading persistent state path=%s backend=%s",
                    state_path,
                    self.cfg.state.backend,
                )
            except Exception:
                pass
            try:
                loaded_state = state_storage.load_state(
                    state_path,
                    backend=self.cfg.state.backend,
                    lock_path=self.cfg.state.lock_path,
                    backup_keep=self.cfg.state.backup_keep,
                )
            except Exception:
                self.logger.exception(
                    "failed to load persistent state from %s; continuing with empty state",
                    state_path,
                )
            else:
                state_version = getattr(loaded_state, "version", None)
                try:
                    self.logger.info(
                        "persistent state restored: version=%s path=%s",
                        state_version,
                        state_path,
                    )
                except Exception:
                    pass

                positions_summary: Dict[str, float] = {}
                try:
                    raw_positions = getattr(loaded_state, "positions", {}) or {}
                    if isinstance(raw_positions, MappingABC):
                        for sym, payload in raw_positions.items():
                            symbol = str(sym).upper()
                            if not symbol:
                                continue
                            qty_val: float | None = None
                            if hasattr(payload, "qty"):
                                try:
                                    qty_val = float(getattr(payload, "qty"))
                                except Exception:
                                    qty_val = None
                            elif isinstance(payload, MappingABC):
                                candidate = (
                                    payload.get("qty")
                                    or payload.get("quantity")
                                    or payload.get("position_qty")
                                )
                                try:
                                    qty_val = float(candidate)
                                except (TypeError, ValueError):
                                    qty_val = None
                            else:
                                try:
                                    qty_val = float(payload)
                                except (TypeError, ValueError):
                                    qty_val = None
                            if qty_val is None or not math.isfinite(qty_val):
                                continue
                            positions_summary[symbol] = qty_val
                except Exception:
                    positions_summary = {}

                orders_summary: Dict[str, Dict[str, Any]] = {}
                try:
                    raw_orders = getattr(loaded_state, "open_orders", []) or []
                    if isinstance(raw_orders, MappingABC):
                        iterable_orders = raw_orders.values()
                    else:
                        iterable_orders = raw_orders
                    for idx, payload in enumerate(iterable_orders):
                        order_data: Dict[str, Any] = {}
                        if hasattr(payload, "to_dict"):
                            try:
                                order_data = dict(payload.to_dict())  # type: ignore[misc]
                            except Exception:
                                order_data = {}
                        elif isinstance(payload, MappingABC):
                            order_data = dict(payload)
                        key = (
                            order_data.get("orderId")
                            or order_data.get("clientOrderId")
                            or order_data.get("order_id")
                            or order_data.get("client_order_id")
                            or str(idx)
                        )
                        symbol = str(order_data.get("symbol", ""))
                        qty = (
                            order_data.get("qty")
                            or order_data.get("quantity")
                            or order_data.get("origQty")
                        )
                        orders_summary[str(key)] = {
                            "symbol": symbol,
                            "side": order_data.get("side"),
                            "qty": qty,
                            "status": order_data.get("status"),
                        }
                except Exception:
                    orders_summary = {}

                candidate_weights = _extract_bar_initial_weights_from_state(
                    loaded_state
                )
                if candidate_weights:
                    bar_initial_weights = candidate_weights

                entry_limits_state = None
                try:
                    entry_limits_raw = getattr(loaded_state, "entry_limits", {}) or {}
                    if isinstance(entry_limits_raw, MappingABC):
                        entry_limits_state = {
                            str(symbol): dict(payload)
                            for symbol, payload in entry_limits_raw.items()
                            if isinstance(payload, MappingABC)
                        }
                except Exception:
                    entry_limits_state = None

                seen_signals_map: Dict[str, int] = {}
                try:
                    raw_seen = getattr(loaded_state, "seen_signals", None) or []
                    if isinstance(raw_seen, MappingABC):
                        iterable = raw_seen.items()
                    else:
                        iterable = raw_seen
                    for item in iterable:
                        sid: str | None = None
                        expires: Any = None
                        if isinstance(item, MappingABC):
                            sid_candidate = (
                                item.get("sid")
                                or item.get("id")
                                or item.get("signal_id")
                                or item.get("key")
                            )
                            expires = (
                                item.get("expires_at_ms")
                                or item.get("expires_at")
                                or item.get("expiry")
                                or item.get("value")
                                or item.get("ts_ms")
                            )
                            sid = str(sid_candidate) if sid_candidate is not None else None
                        elif isinstance(item, (tuple, list)) and len(item) >= 2:
                            sid = str(item[0])
                            expires = item[1]
                        else:
                            continue
                        if not sid:
                            continue
                        try:
                            expires_int = int(expires)
                        except (TypeError, ValueError):
                            continue
                        seen_signals_map[sid] = expires_int
                except Exception:
                    seen_signals_map = {}

                signal_state_payload: Dict[str, Any] = {}
                try:
                    raw_states = getattr(loaded_state, "signal_states", {}) or {}
                    if isinstance(raw_states, MappingABC):
                        signal_state_payload = {str(k): v for k, v in raw_states.items()}
                except Exception:
                    signal_state_payload = {}

                config_snapshot = {}
                try:
                    raw_snapshot = getattr(loaded_state, "config_snapshot", {}) or {}
                    if isinstance(raw_snapshot, MappingABC):
                        config_snapshot = dict(raw_snapshot)
                except Exception:
                    config_snapshot = {}

                git_hash = None
                try:
                    git_hash = getattr(loaded_state, "git_hash", None)
                except Exception:
                    git_hash = None

                last_processed = None
                try:
                    last_processed = getattr(loaded_state, "last_processed_bar_ms", None)
                except Exception:
                    last_processed = None

                try:
                    self.logger.info(
                        "state summary: positions=%d open_orders=%d entry_limits=%d seen_signals=%d last_processed_bar_ms=%s",
                        len(positions_summary),
                        len(orders_summary),
                        len(entry_limits_state or {}),
                        len(seen_signals_map),
                        last_processed,
                    )
                except Exception:
                    pass

                if positions_summary:
                    try:
                        preview_limit = 5
                        preview_items = list(positions_summary.items())
                        preview = dict(preview_items[:preview_limit])
                        if len(preview_items) > preview_limit:
                            preview["..."] = f"{len(preview_items) - preview_limit} more"
                        self.logger.info("positions restored: %s", preview)
                    except Exception:
                        pass

                if orders_summary:
                    try:
                        preview_limit = 5
                        preview_items = list(orders_summary.items())
                        preview_orders: Dict[str, Dict[str, Any]] = {}
                        for oid, data in preview_items[:preview_limit]:
                            preview_orders[oid] = {
                                "symbol": data.get("symbol"),
                                "side": data.get("side"),
                                "qty": data.get("qty"),
                                "status": data.get("status"),
                            }
                        if len(preview_items) > preview_limit:
                            preview_orders["..."] = {"count": len(preview_items) - preview_limit}
                        self.logger.info("open orders restored: %s", preview_orders)
                    except Exception:
                        pass

                if config_snapshot:
                    try:
                        self.logger.info(
                            "config snapshot restored (%d keys)", len(config_snapshot)
                        )
                    except Exception:
                        pass

                if git_hash:
                    try:
                        self.logger.info("state git hash: %s", git_hash)
                    except Exception:
                        pass

                loader = getattr(self.policy, "load_signal_state", None)
                if callable(loader):
                    try:
                        loader(signal_state_payload)
                        try:
                            self.logger.info(
                                "policy signal state restored (%d entries)",
                                len(signal_state_payload),
                            )
                        except Exception:
                            pass
                    except Exception:
                        self.logger.exception("failed to restore policy signal state")

                if self.ws_dedup_enabled and seen_signals_map:
                    try:
                        signal_bus.STATE.clear()
                        for sid, exp in seen_signals_map.items():
                            signal_bus.STATE[str(sid)] = int(exp)
                        try:
                            self.logger.info(
                                "restored %d websocket dedup entries",
                                len(signal_bus.STATE),
                            )
                        except Exception:
                            pass
                    except Exception:
                        self.logger.exception(
                            "failed to restore websocket dedup state from persistence"
                        )

                client = getattr(self.adapter, "client", None)
                if client is not None:
                    try:
                        summary = reconcile_state(loaded_state, client)
                        self.logger.info("state reconciliation: %s", summary)
                    except Exception as exc:
                        self.logger.warning(
                            "state reconciliation failed: %s",
                            exc,
                        )
                else:
                    try:
                        self.logger.info(
                            "state reconciliation skipped: private client not configured"
                        )
                    except Exception:
                        pass
        if dirty_restart and self.monitoring_cfg.enabled:
            try:
                monitoring.reset_kill_switch_counters()
            except Exception:
                pass

        shutdown = ShutdownManager(self.shutdown_cfg)
        self._shutdown_ref = shutdown
        stop_event = threading.Event()
        shutdown.on_stop(stop_event.set)

        self.feature_pipe.warmup()
        if self.monitoring_cfg.enabled:
            monitoring.configure_kill_switch(self.monitoring_cfg.thresholds)
        else:
            monitoring.configure_kill_switch(None)

        ops_cfg = {
            "rest_limit": self.cfg.kill_switch_ops.error_limit,
            "ws_limit": self.cfg.kill_switch_ops.error_limit,
            "duplicate_limit": self.cfg.kill_switch_ops.duplicate_limit,
            "stale_limit": self.cfg.kill_switch_ops.stale_intervals_limit,
            "reset_cooldown_sec": self.cfg.kill_switch_ops.reset_cooldown_sec,
        }
        if self.cfg.kill_switch_ops.flag_path:
            ops_cfg["flag_path"] = self.cfg.kill_switch_ops.flag_path
        if self.cfg.kill_switch_ops.alert_command:
            ops_cfg["alert_command"] = shlex.split(
                self.cfg.kill_switch_ops.alert_command
            )
        ops_kill_switch.init(ops_cfg)

        ops_flush_stop = threading.Event()
        rest_failures = 0

        def _ops_flush_loop() -> None:
            nonlocal rest_failures
            while not ops_flush_stop.wait(1.0):
                limit = int(self.cfg.kill_switch_ops.error_limit)
                try:
                    ops_kill_switch.tick()
                    if rest_failures:
                        try:
                            ops_kill_switch.manual_reset()
                        except Exception:
                            pass
                        rest_failures = 0
                except Exception:
                    rest_failures += 1
                    if limit > 0 and rest_failures >= limit:
                        try:
                            ops_kill_switch.record_error("rest")
                        except Exception:
                            pass
                        rest_failures = 0

        ops_flush_thread = threading.Thread(target=_ops_flush_loop, daemon=True)
        ops_flush_thread.start()
        shutdown.on_flush(ops_kill_switch.tick)
        shutdown.on_finalize(ops_flush_stop.set)
        shutdown.on_finalize(lambda: ops_flush_thread.join(timeout=1.0))

        monitoring_stop = threading.Event()
        monitoring_thread: threading.Thread | None = None
        if self.monitoring_agg is not None and self.monitoring_agg.enabled:

            def _monitoring_loop() -> None:
                try:
                    interval = float(self.monitoring_cfg.tick_sec)
                except Exception:
                    interval = 1.0
                if interval <= 0:
                    interval = 1.0
                while not monitoring_stop.is_set():
                    try:
                        self.monitoring_agg.tick(int(time.time() * 1000))
                    except Exception:
                        pass
                    if monitoring_stop.wait(interval):
                        break

            monitoring_thread = threading.Thread(target=_monitoring_loop, daemon=True)
            monitoring_thread.start()
            shutdown.on_stop(monitoring_stop.set)
            shutdown.on_stop(monitoring.clear_runtime_aggregator)
            shutdown.on_flush(self.monitoring_agg.flush)
            shutdown.on_finalize(monitoring.clear_runtime_aggregator)
            shutdown.on_finalize(lambda: monitoring_thread.join(timeout=1.0))

        rest_candidates = [
            getattr(self.adapter, "rest_helper", None),
            getattr(self.adapter, "client", None),
            getattr(self.adapter, "rest_client", None),
            getattr(self.adapter, "rest", None),
        ]

        executor_for_worker = self.adapter
        portfolio_equity = None
        portfolio_weight_cap = None
        if self._execution_mode == "bar":
            exec_cfg = getattr(self._run_config, "execution", None)
            bar_price = "close"
            min_step = 0.0
            cost_cfg = getattr(self._run_config, "costs", None)
            safety_margin = 0.0
            max_participation = None
            default_equity = None
            weight_cap_raw: Any = None
            if exec_cfg is not None:
                bar_price = getattr(exec_cfg, "bar_price", bar_price) or bar_price
                try:
                    min_step = float(getattr(exec_cfg, "min_rebalance_step", min_step) or min_step)
                except (TypeError, ValueError):
                    min_step = 0.0
                cost_cfg = getattr(exec_cfg, "costs", cost_cfg)
                try:
                    safety_margin = float(getattr(exec_cfg, "safety_margin_bps", 0.0) or 0.0)
                except (TypeError, ValueError):
                    safety_margin = 0.0
                participation_candidate = getattr(exec_cfg, "max_participation", None)
                if participation_candidate is None and isinstance(exec_cfg, MappingABC):
                    participation_candidate = exec_cfg.get("max_participation")
                if participation_candidate is not None:
                    try:
                        max_participation = float(participation_candidate)
                    except (TypeError, ValueError):
                        max_participation = None
                portfolio_cfg = getattr(exec_cfg, "portfolio", None)
                if portfolio_cfg is None:
                    portfolio_cfg = getattr(self._run_config, "portfolio", None)
                if portfolio_cfg is not None:
                    equity_candidate = getattr(portfolio_cfg, "equity_usd", None)
                    if equity_candidate is None and isinstance(portfolio_cfg, MappingABC):
                        equity_candidate = portfolio_cfg.get("equity_usd")
                    if equity_candidate is not None:
                        default_equity = equity_candidate
                    weight_candidate = getattr(portfolio_cfg, "max_total_weight", None)
                    if weight_candidate is None and isinstance(portfolio_cfg, MappingABC):
                        weight_candidate = portfolio_cfg.get("max_total_weight")
                    if weight_candidate is not None:
                        weight_cap_raw = weight_candidate
            if default_equity is None:
                portfolio_cfg = getattr(self._run_config, "portfolio", None)
                if portfolio_cfg is not None:
                    equity_candidate = getattr(portfolio_cfg, "equity_usd", None)
                    if equity_candidate is None and isinstance(portfolio_cfg, MappingABC):
                        equity_candidate = portfolio_cfg.get("equity_usd")
                    if equity_candidate is not None:
                        default_equity = equity_candidate
                    if weight_cap_raw is None:
                        weight_candidate = getattr(portfolio_cfg, "max_total_weight", None)
                        if weight_candidate is None and isinstance(portfolio_cfg, MappingABC):
                            weight_candidate = portfolio_cfg.get("max_total_weight")
                        if weight_candidate is not None:
                            weight_cap_raw = weight_candidate
            try:
                portfolio_equity = float(default_equity) if default_equity is not None else None
            except (TypeError, ValueError):
                portfolio_equity = None
            try:
                portfolio_weight_cap = (
                    float(weight_cap_raw) if weight_cap_raw is not None else None
                )
            except (TypeError, ValueError):
                portfolio_weight_cap = None
            if portfolio_weight_cap is not None:
                if not math.isfinite(portfolio_weight_cap) or portfolio_weight_cap <= 0.0:
                    portfolio_weight_cap = None
            if max_participation is not None:
                if not math.isfinite(max_participation) or max_participation <= 0.0:
                    max_participation = None
            executor_for_worker = BarExecutor(
                run_id=self.cfg.run_id or "bar",
                bar_price=bar_price,
                min_rebalance_step=min_step,
                cost_config=cost_cfg,
                safety_margin_bps=safety_margin,
                max_participation=max_participation,
                default_equity_usd=portfolio_equity or 0.0,
                initial_weights=bar_initial_weights,
                symbol_specs=self._symbol_specs,
            )

        bar_timeframe_candidates = (
            getattr(getattr(self.cfg, "timing", None), "timeframe_ms", None),
            getattr(getattr(self._run_config, "timing", None), "timeframe_ms", None),
            getattr(getattr(self._run_config, "execution", None), "timeframe_ms", None),
            self.ws_dedup_timeframe_ms,
        )
        bar_timeframe_ms = 0
        for candidate in bar_timeframe_candidates:
            candidate_ms = _Worker._coerce_timeframe_ms(candidate)
            if candidate_ms > 0:
                bar_timeframe_ms = candidate_ms
                break

        worker = _Worker(
            self.feature_pipe,
            self.policy,
            self.logger,
            executor_for_worker,
            self.risk_guards,
            lambda: self._clock_safe_mode
            or monitoring.kill_switch_triggered()
            or ops_kill_switch.tripped(),
            enforce_closed_bars=self.enforce_closed_bars,
            close_lag_ms=self.close_lag_ms,
            ws_dedup_enabled=self.ws_dedup_enabled,
            ws_dedup_log_skips=self.ws_dedup_log_skips,
            ws_dedup_timeframe_ms=self.ws_dedup_timeframe_ms,
            bar_timeframe_ms=bar_timeframe_ms,
            throttle_cfg=self.throttle_cfg,
            no_trade_cfg=self.no_trade_cfg,
            pipeline_cfg=self.pipeline_cfg,
            signal_quality_cfg=self.signal_quality_cfg,
            zero_signal_alert=getattr(
                self.monitoring_cfg.thresholds, "zero_signals", 0
            ),
            state_enabled=self.cfg.state.enabled,
            rest_candidates=rest_candidates,
            monitoring=self.monitoring_agg,
            monitoring_agg=self.monitoring_agg,
            worker_id="worker-0",
            status_callback=self._update_runner_status,
            execution_mode=self._execution_mode,
            portfolio_equity=portfolio_equity,
            max_total_weight=portfolio_weight_cap,
            cooldown_settings=self._resolve_worker_cooldown_settings(),
            signal_dispatcher=self._resolve_signal_dispatcher(executor_for_worker),
        )
        self._worker_ref = worker
        self._write_runner_status()
        if self._portfolio_limits_cfg:
            try:
                guard_cfg = PortfolioLimitConfig.from_mapping(
                    self._portfolio_limits_cfg
                )
            except Exception:
                guard_cfg = PortfolioLimitConfig()
            if guard_cfg.enabled:
                portfolio_guard = PortfolioLimitGuard(
                    config=guard_cfg,
                    get_positions=worker._portfolio_positions_snapshot,
                    get_total_notional=worker._portfolio_total_notional,
                    get_price=worker._portfolio_last_price,
                    get_equity=lambda: None,
                    leg_getter=worker._signal_leg,
                )
                worker.set_portfolio_guard(portfolio_guard)
                try:
                    self.logger.info(
                        "portfolio limit guard enabled: max_total_notional=%s max_total_exposure_pct=%s buffer=%s",
                        guard_cfg.max_total_notional,
                        guard_cfg.max_total_exposure_pct,
                        guard_cfg.exposure_buffer_frac,
                    )
                except Exception:
                    pass
        if entry_limits_state is not None:
            try:
                worker._entry_limiter.restore_state(entry_limits_state)
            except Exception:
                pass

        def _resolve_history_source(source: Any) -> Any:
            if source is None:
                return None
            attr_candidates = (
                "history",
                "history_bars",
                "recent_bars",
                "warmup_bars",
                "initial_bars",
                "cached_bars",
            )
            for name in attr_candidates:
                if not hasattr(source, name):
                    continue
                value = getattr(source, name)
                if callable(value):
                    try:
                        value = value()
                    except Exception:
                        continue
                if value in (None, False):
                    continue
                if isinstance(value, (str, bytes)):
                    continue
                if isinstance(value, MappingABC) and value:
                    return value
                try:
                    length = len(value)  # type: ignore[arg-type]
                except Exception:
                    length = None
                if length and length > 0:
                    return value
                if length == 0:
                    continue
                if length is None:
                    try:
                        materialized = list(value)  # type: ignore[arg-type]
                    except TypeError:
                        continue
                    if materialized:
                        return materialized
            getter = getattr(source, "get_history", None)
            if callable(getter):
                try:
                    value = getter()
                except Exception:
                    value = None
                if value:
                    return value
            return None

        seen_candidates: set[int] = set()
        for candidate in (
            _resolve_history_source(self.adapter),
            _resolve_history_source(getattr(self.adapter, "source", None)),
            _resolve_history_source(self.feature_pipe),
        ):
            if not candidate:
                continue
            ident = id(candidate)
            if ident in seen_candidates:
                continue
            seen_candidates.add(ident)
            worker.prewarm_dynamic_guard(candidate)

        out_csv = getattr(signal_bus, "OUT_CSV", None)
        if out_csv:
            try:
                opts = self._signal_writer_options
                signal_bus.OUT_WRITER = SignalCSVWriter(
                    out_csv,
                    fsync_mode=str(opts.get("fsync_mode", "batch")),
                    rotate_daily=bool(opts.get("rotate_daily", True)),
                    flush_interval_s=opts.get("flush_interval_s"),
                )
                self._signal_writer_ref = signal_bus.OUT_WRITER
                try:
                    self._signal_writer_stats = self._signal_writer_ref.stats()
                except Exception:
                    self._signal_writer_stats = {}
                writer = self._signal_writer_ref
                shutdown.on_flush(lambda w=writer: w.flush_fsync(force=True))
                shutdown.on_finalize(writer.close)
            except Exception:
                signal_bus.OUT_WRITER = None
                self._signal_writer_ref = None
                self._signal_writer_stats = {}
        else:
            self._signal_writer_ref = None
            self._signal_writer_stats = {}

        self._write_runner_status()

        json_path = self.cfg.snapshot_metrics_json or os.path.join(
            logs_dir, "snapshot_metrics.json"
        )
        csv_path = self.cfg.snapshot_metrics_csv or os.path.join(
            logs_dir, "snapshot_metrics.csv"
        )
        snapshot_stop = threading.Event()
        snapshot_thread: threading.Thread | None = None
        if self.monitoring_cfg.enabled and self.cfg.snapshot_metrics_sec > 0:

            def _snapshot_loop() -> None:
                while not snapshot_stop.wait(self.cfg.snapshot_metrics_sec):
                    try:
                        monitoring.snapshot_metrics(json_path, csv_path)
                    except Exception:
                        pass

            try:
                monitoring.snapshot_metrics(json_path, csv_path)
            except Exception:
                pass
            snapshot_thread = threading.Thread(target=_snapshot_loop, daemon=True)
            snapshot_thread.start()
            shutdown.on_stop(snapshot_stop.set)

        state_stop = threading.Event()
        state_thread: threading.Thread | None = None

        def _persist_state(reason: str) -> None:
            if not self.cfg.state.enabled:
                return
            path = self.cfg.state.path
            backend = self.cfg.state.backend
            try:
                state_storage.save_state(
                    path,
                    backend=backend,
                    lock_path=self.cfg.state.lock_path,
                    backup_keep=self.cfg.state.backup_keep,
                )
            except Exception:
                self.logger.warning(
                    "failed to save persistent state (reason=%s) path=%s backend=%s",
                    reason,
                    path,
                    backend,
                    exc_info=True,
                )
            else:
                self.logger.info(
                    "persistent state saved (reason=%s) path=%s backend=%s",
                    reason,
                    path,
                    backend,
                )

        if self.cfg.state.enabled and self.cfg.state.snapshot_interval_s > 0:

            def _state_loop() -> None:
                while not state_stop.wait(self.cfg.state.snapshot_interval_s):
                    try:
                        _persist_state("interval")
                    except Exception:
                        self.logger.exception(
                            "unexpected error during periodic state persistence"
                        )

            state_thread = threading.Thread(target=_state_loop, daemon=True)
            state_thread.start()
        if self.cfg.state.enabled:
            shutdown.on_stop(state_stop.set)

        # Optional asynchronous event bus processing
        bus = getattr(self.adapter, "bus", None)
        loop_thread: threading.Thread | None = None
        if bus is not None:
            import asyncio

            n_workers = max(1, int(getattr(self.adapter, "n_workers", 1)))
            loop = asyncio.new_event_loop()

            async def _run_workers() -> None:
                tasks = [
                    asyncio.create_task(worker_loop(bus, worker))
                    for _ in range(n_workers)
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)

            def _loop_runner() -> None:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_run_workers())

            loop_thread = threading.Thread(target=_loop_runner, daemon=True)
            loop_thread.start()
            shutdown.on_stop(bus.close)
            shutdown.on_finalize(
                lambda: loop_thread.join(timeout=1.0) if loop_thread else None
            )

        ws_client = getattr(self.adapter, "ws", None) or getattr(
            self.adapter, "source", None
        )
        _attach_monitoring_target(ws_client, self.monitoring_agg)
        if ws_client is not None and hasattr(ws_client, "stop"):

            async def _stop_ws_client() -> None:
                await ws_client.stop()

            shutdown.on_stop(_stop_ws_client)

        shutdown.on_stop(worker._drain_queue)

        sim = getattr(self.adapter, "sim", None) or getattr(self.adapter, "_sim", None)
        signal_log = getattr(sim, "_logger", None) if sim is not None else None
        if signal_log is not None:
            if hasattr(signal_log, "flush_fsync"):
                shutdown.on_flush(signal_log.flush_fsync)
            elif hasattr(signal_log, "flush"):
                shutdown.on_flush(signal_log.flush)
        if hasattr(signal_bus, "flush"):
            shutdown.on_flush(signal_bus.flush)

        def _flush_snapshot() -> None:
            try:
                summary, json_str, _ = monitoring.snapshot_metrics(json_path, csv_path)
                self.logger.info("SNAPSHOT %s", json_str)
            except Exception:
                pass

        shutdown.on_flush(_flush_snapshot)
        if self.cfg.state.enabled and self.cfg.state.flush_on_event:

            def _flush_persistent_state() -> None:
                _persist_state("flush")

            shutdown.on_flush(_flush_persistent_state)

        def _remove_marker() -> None:
            try:
                marker_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

        def _final_summary() -> None:
            try:
                summary, json_str, _ = monitoring.snapshot_metrics(json_path, csv_path)
                self.logger.info("SUMMARY %s", json_str)
            except Exception:
                pass

        shutdown.on_finalize(_remove_marker)
        if snapshot_thread is not None:
            shutdown.on_finalize(lambda: snapshot_thread.join(timeout=1.0))
        if self.cfg.state.enabled:

            def _finalize_persistent_state() -> None:
                _persist_state("finalize")

            shutdown.on_finalize(_finalize_persistent_state)
        if state_thread is not None:
            shutdown.on_finalize(lambda: state_thread.join(timeout=1.0))
        shutdown.on_finalize(lambda: self._clock_stop.set())
        shutdown.on_finalize(
            lambda: (
                self._clock_thread.join(
                    timeout=(
                        self.clock_sync_cfg.refresh_sec if self.clock_sync_cfg else 1.0
                    )
                )
                if self._clock_thread is not None
                else None
            )
        )
        shutdown.on_finalize(
            lambda: signal_bus.shutdown() if signal_bus.ENABLED else None
        )
        shutdown.on_finalize(_final_summary)

        shutdown.register(signal.SIGINT, signal.SIGTERM)

        client = getattr(self.adapter, "client", None)
        if client is not None and self.clock_sync_cfg is not None:
            drift = clock.sync_clock(client, self.clock_sync_cfg, monitoring)
            try:
                monitoring.report_clock_sync(drift, 0.0, True, clock.system_utc_ms())
            except Exception:
                pass
            self.logger.info(
                "clock drift %.2fms (warn=%dms kill=%dms)",
                drift,
                int(self.clock_sync_cfg.warn_threshold_ms),
                int(self.clock_sync_cfg.kill_threshold_ms),
            )

            def _sync_loop() -> None:
                backoff = self.clock_sync_cfg.refresh_sec
                while not self._clock_stop.wait(backoff):
                    before = clock.last_sync_at
                    drift_local = clock.sync_clock(
                        client, self.clock_sync_cfg, monitoring
                    )
                    success = clock.last_sync_at != before
                    try:
                        monitoring.report_clock_sync(
                            drift_local, 0.0, success, clock.system_utc_ms()
                        )
                    except Exception:
                        pass
                    if not success:
                        try:
                            monitoring.clock_sync_fail.inc()
                        except Exception:
                            pass
                        backoff = min(
                            backoff * 2.0, self.clock_sync_cfg.refresh_sec * 10.0
                        )
                        continue
                    backoff = self.clock_sync_cfg.refresh_sec
                    if drift_local > self.clock_sync_cfg.kill_threshold_ms:
                        self._clock_safe_mode = True
                        try:
                            monitoring.clock_sync_fail.inc()
                        except Exception:
                            pass
                        self.logger.error(
                            "clock drift %.2fms exceeds kill threshold %.2fms; entering safe mode",
                            drift_local,
                            self.clock_sync_cfg.kill_threshold_ms,
                        )
                    else:
                        self._clock_safe_mode = False
                        if drift_local > self.clock_sync_cfg.warn_threshold_ms:
                            self.logger.warning(
                                "clock drift %.2fms exceeds warn threshold %.2fms",
                                drift_local,
                                self.clock_sync_cfg.warn_threshold_ms,
                            )
                            try:
                                monitoring.clock_sync_fail.inc()
                            except Exception:
                                pass

            self._clock_thread = threading.Thread(target=_sync_loop, daemon=True)
            self._clock_thread.start()

        positions_cache: Dict[str, Dict[str, Any]] = {}
        open_orders_cache: Dict[str, Dict[str, Any]] = {}
        last_processed_cache: Dict[str, int] = {}
        seen_signals_cache: Dict[str, int] = {}

        def _safe_float(value: Any) -> float | None:
            try:
                if value is None:
                    return None
                num = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(num):
                return None
            return num

        def _safe_int(value: Any) -> int | None:
            try:
                if value is None:
                    return None
                return int(float(value))
            except (TypeError, ValueError):
                return None

        def _order_key(data: Mapping[str, Any]) -> str | None:
            for key in (
                "orderId",
                "order_id",
                "id",
                "clientOrderId",
                "client_order_id",
                "client_id",
            ):
                if key not in data:
                    continue
                value = data.get(key)
                if value is None:
                    continue
                text = str(value)
                if text:
                    return text
            return None

        def _normalize_position_entry(value: Any) -> Dict[str, Any] | None:
            if isinstance(value, state_storage.PositionState):
                data = value.to_dict()
            elif isinstance(value, MappingABC):
                data = dict(value)
            else:
                data = {"qty": value}
            qty_val = _safe_float(
                data.get("qty")
                or data.get("quantity")
                or data.get("position_qty")
                or data.get("size")
            )
            if qty_val is None:
                return None
            avg_val = _safe_float(
                data.get("avg_price")
                or data.get("avgPrice")
                or data.get("price")
            )
            ts_val = _safe_int(
                data.get("last_update_ms")
                or data.get("timestamp")
                or data.get("ts_ms")
                or data.get("time")
            )
            return {
                "qty": qty_val,
                "avg_price": avg_val if avg_val is not None else 0.0,
                "last_update_ms": ts_val,
            }

        def _normalize_order_payload(
            order: Any,
            *,
            default_symbol: str = "",
            default_ts: int | None = None,
        ) -> tuple[str, Dict[str, Any]]:
            if isinstance(order, state_storage.OrderState):
                raw: Mapping[str, Any] = order.to_dict()
            elif isinstance(order, MappingABC):
                raw = order
            else:
                raw = getattr(order, "__dict__", {}) or {}
            data = dict(raw)
            key = _order_key(data) or ""
            symbol = str(data.get("symbol") or default_symbol or "")
            client_id = data.get("clientOrderId") or data.get("client_order_id") or data.get("client_id")
            order_id = data.get("orderId") or data.get("order_id") or data.get("id")
            side = data.get("side")
            if side in (None, "") and data.get("is_buy") is not None:
                side = "BUY" if bool(data.get("is_buy")) else "SELL"
            qty = (
                data.get("qty")
                or data.get("quantity")
                or data.get("origQty")
                or data.get("size")
            )
            price = data.get("price") or data.get("avg_price") or data.get("limit_price")
            status = data.get("status") or data.get("state")
            ts_candidate = (
                data.get("ts_ms")
                or data.get("timestamp")
                or data.get("time")
                or data.get("updateTime")
                or default_ts
            )
            qty_val = _safe_float(qty)
            price_val = _safe_float(price)
            ts_val = _safe_int(ts_candidate)
            normalized = {
                "symbol": symbol,
                "clientOrderId": str(client_id) if client_id not in (None, "") else None,
                "orderId": str(order_id) if order_id not in (None, "") else None,
                "side": str(side).upper() if side not in (None, "") else None,
                "qty": qty_val if qty_val is not None else 0.0,
                "price": price_val,
                "status": str(status).upper() if status not in (None, "") else None,
                "ts_ms": ts_val,
            }
            normalized_key = (
                normalized["orderId"]
                or normalized["clientOrderId"]
                or (key if key else None)
            )
            if not normalized_key:
                normalized_key = (
                    f"auto:{symbol}:"
                    f"{normalized['side'] or ''}:{normalized['qty']}:{normalized['price']}:{normalized['ts_ms']}"
                )
            return normalized_key, normalized

        def _serialize_positions() -> Dict[str, Dict[str, Any]]:
            return {
                sym: {
                    "qty": payload["qty"],
                    "avg_price": payload.get("avg_price", 0.0),
                    "last_update_ms": payload.get("last_update_ms"),
                }
                for sym, payload in sorted(positions_cache.items())
            }

        def _serialize_open_orders() -> list[Dict[str, Any]]:
            items = sorted(
                open_orders_cache.items(),
                key=lambda kv: ((_safe_int(kv[1].get("ts_ms")) or 0), kv[0]),
            )
            return [dict(payload) for _, payload in items]

        def _extract_avg_price_from_report(
            rep: Mapping[str, Any], existing: float | None
        ) -> float | None:
            for key in (
                "avg_price",
                "avgPrice",
                "avg_entry_price",
                "avgEntryPrice",
                "entry_price",
                "entryPrice",
            ):
                if key in rep:
                    candidate = _safe_float(rep.get(key))
                    if candidate is not None:
                        return candidate
            for key in (
                "mark_price",
                "markPrice",
                "price",
                "mtm_price",
                "close",
                "bar_close",
                "bar_close_price",
                "ref_price",
            ):
                if key in rep:
                    candidate = _safe_float(rep.get(key))
                    if candidate is not None:
                        return candidate
            return existing

        def _capture_seen_signals() -> Dict[str, int]:
            state = getattr(signal_bus, "STATE", {})
            captured: Dict[str, int] = {}
            if isinstance(state, MappingABC):
                iterator = state.items()
            else:
                getter = getattr(state, "items", None)
                iterator = getter() if callable(getter) else []
            for key, value in iterator:
                sid = str(key)
                expires = _safe_int(value)
                if expires is None:
                    continue
                captured[sid] = expires
            return captured

        if self.cfg.state.enabled:
            try:
                current_state = state_storage.get_state()
            except Exception:
                current_state = None
            if current_state is not None:
                raw_positions = getattr(current_state, "positions", {}) or {}
                if isinstance(raw_positions, MappingABC):
                    for sym, payload in raw_positions.items():
                        normalized = _normalize_position_entry(payload)
                        if not normalized:
                            continue
                        positions_cache[str(sym)] = normalized
                raw_orders = getattr(current_state, "open_orders", []) or []
                if isinstance(raw_orders, MappingABC):
                    iterable_orders = raw_orders.values()
                else:
                    iterable_orders = raw_orders
                for item in iterable_orders:
                    if item is None:
                        continue
                    key, normalized = _normalize_order_payload(item)
                    open_orders_cache[str(key)] = normalized
                raw_last_processed = (
                    getattr(current_state, "last_processed_bar_ms", {}) or {}
                )
                if isinstance(raw_last_processed, MappingABC):
                    for sym, ts in raw_last_processed.items():
                        ts_val = _safe_int(ts)
                        if ts_val is None:
                            continue
                        last_processed_cache[str(sym)] = ts_val
            seen_signals_cache = _capture_seen_signals()

        self._persist_run_metadata()

        ws_failures = 0
        limit = int(self.cfg.kill_switch_ops.error_limit)

        def _handle_report(rep: Dict[str, Any]) -> None:
            if not self.cfg.state.enabled:
                return
            updates: Dict[str, Any] = {}
            positions_changed = False
            orders_changed = False
            last_processed_changed = False
            seen_changed = False
            try:
                symbol = str(rep.get("symbol") or "").upper()
                pos_qty = rep.get("position_qty")
                ts_ms = _safe_int(rep.get("ts_ms"))

                if symbol and pos_qty is not None:
                    qty_val = _safe_float(pos_qty)
                    if qty_val is not None:
                        existing = positions_cache.get(symbol)
                        existing_avg = existing.get("avg_price") if existing else None
                        avg_price = _extract_avg_price_from_report(rep, existing_avg)
                        if avg_price is None:
                            avg_price = existing_avg
                        if avg_price is None:
                            avg_price = 0.0
                        last_update = (
                            _safe_int(rep.get("position_ts_ms"))
                            or _safe_int(rep.get("position_timestamp"))
                            or ts_ms
                            or (existing.get("last_update_ms") if existing else None)
                        )
                        if math.isclose(qty_val, 0.0, rel_tol=0.0, abs_tol=1e-12):
                            if symbol in positions_cache:
                                positions_cache.pop(symbol, None)
                                positions_changed = True
                        else:
                            payload = {
                                "qty": qty_val,
                                "avg_price": float(avg_price),
                                "last_update_ms": last_update,
                            }
                            if existing != payload:
                                positions_cache[symbol] = payload
                                positions_changed = True

                orders = rep.get("core_orders") or []
                if orders:
                    for order in orders:
                        key, normalized = _normalize_order_payload(
                            order, default_symbol=symbol, default_ts=ts_ms
                        )
                        prev = open_orders_cache.get(key)
                        if prev != normalized:
                            open_orders_cache[key] = normalized
                            orders_changed = True

                reports = rep.get("core_exec_reports") or []
                if reports:
                    for report in reports:
                        if isinstance(report, MappingABC):
                            data = report
                        else:
                            data = getattr(report, "__dict__", {}) or {}
                        key = _order_key(data)
                        if not key:
                            continue
                        status_val = data.get("status") or data.get("exec_status")
                        status = str(status_val).upper() if status_val else None
                        if status in {"FILLED", "CANCELLED"}:
                            if key in open_orders_cache:
                                open_orders_cache.pop(key, None)
                                orders_changed = True
                            continue
                        payload = open_orders_cache.get(key)
                        if not payload:
                            continue
                        changed = False
                        if status and payload.get("status") != status:
                            payload["status"] = status
                            changed = True
                        qty_update = (
                            data.get("qty")
                            or data.get("quantity")
                            or data.get("origQty")
                            or data.get("remaining_qty")
                        )
                        qty_val = _safe_float(qty_update)
                        if qty_val is not None and qty_val != payload.get("qty"):
                            payload["qty"] = qty_val
                            changed = True
                        price_update = (
                            data.get("price")
                            or data.get("avg_price")
                            or data.get("fill_price")
                        )
                        price_val = _safe_float(price_update)
                        if price_val is not None and price_val != payload.get("price"):
                            payload["price"] = price_val
                            changed = True
                        report_ts = (
                            _safe_int(data.get("ts_ms"))
                            or _safe_int(data.get("timestamp"))
                            or _safe_int(data.get("time"))
                            or ts_ms
                        )
                        if report_ts is not None and payload.get("ts_ms") != report_ts:
                            payload["ts_ms"] = report_ts
                            changed = True
                        symbol_update = data.get("symbol")
                        if symbol_update and not payload.get("symbol"):
                            payload["symbol"] = str(symbol_update)
                            changed = True
                        if changed:
                            orders_changed = True

                if ts_ms is not None:
                    if symbol:
                        prev_symbol_ts = last_processed_cache.get(symbol)
                        if prev_symbol_ts is None or ts_ms >= prev_symbol_ts:
                            last_processed_cache[symbol] = ts_ms
                            last_processed_changed = True
                    prev_global = last_processed_cache.get(
                        state_storage.LAST_PROCESSED_GLOBAL_KEY
                    )
                    if prev_global is None or ts_ms > prev_global:
                        last_processed_cache[
                            state_storage.LAST_PROCESSED_GLOBAL_KEY
                        ] = ts_ms
                        last_processed_changed = True

                seen_snapshot = _capture_seen_signals()
                if seen_snapshot != seen_signals_cache:
                    seen_signals_cache.clear()
                    seen_signals_cache.update(seen_snapshot)
                    seen_changed = True

                if positions_changed:
                    updates["positions"] = _serialize_positions()
                if orders_changed:
                    updates["open_orders"] = _serialize_open_orders()
                if last_processed_changed:
                    updates["last_processed_bar_ms"] = dict(last_processed_cache)
                if seen_changed:
                    updates["seen_signals"] = dict(seen_signals_cache)
                if updates:
                    updates["last_update_ms"] = ts_ms or clock.now_ms()
                    state_storage.update_state(**updates)
            except Exception:
                self.logger.exception("failed to handle report for state persistence")
                updates.clear()
            if updates and self.cfg.state.flush_on_event:
                _persist_state("event")

        while not stop_event.is_set():
            self._process_control_flags(shutdown, stop_event)
            if stop_event.is_set():
                break
            try:
                for rep in self.adapter.run_events(worker):
                    self._process_control_flags(shutdown, stop_event)
                    if stop_event.is_set():
                        break
                    if ws_failures:
                        try:
                            ops_kill_switch.manual_reset()
                        except Exception:
                            pass
                        ws_failures = 0
                    _handle_report(rep)
                    yield rep
                break
            except Exception:
                ws_failures += 1
                if limit > 0 and ws_failures >= limit:
                    try:
                        ops_kill_switch.record_error("ws")
                    except Exception:
                        pass
                    ws_failures = 0
                time.sleep(1.0)

        shutdown.unregister(signal.SIGINT, signal.SIGTERM)
        shutdown.request_shutdown()


def from_config(
    cfg: CommonRunConfig,
    *,
    snapshot_config_path: str | None = None,
) -> Iterator[Dict[str, Any]]:
    """Build dependencies from ``cfg`` and run :class:`ServiceSignalRunner`."""

    timing_defaults, timing_profiles = load_timing_profiles()
    exec_profile = getattr(
        cfg, "execution_profile", ExecutionProfile.MKT_OPEN_NEXT_H1
    )
    resolved_timing = resolve_execution_timing(
        exec_profile, timing_defaults, timing_profiles
    )

    def _parse_timeframe_to_ms(value: Any) -> int:
        try:
            raw_int = int(value)
        except (TypeError, ValueError):
            if value is None:
                return 0
            text = str(value).strip().lower()
            if not text:
                return 0
            unit = text[-1]
            multipliers = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}
            if unit not in multipliers:
                return 0
            try:
                magnitude = float(text[:-1] or 0)
            except ValueError:
                return 0
            value_ms = int(magnitude * multipliers[unit])
        else:
            value_ms = raw_int
        return int(value_ms) if value_ms > 0 else 0

    timeframe_candidates: list[Any] = [getattr(cfg.timing, "timeframe_ms", None)]
    timeframe_candidates.append(getattr(cfg.execution, "timeframe_ms", None))
    data_cfg = getattr(cfg, "data", None)
    if data_cfg is not None:
        timeframe_candidates.append(getattr(data_cfg, "timeframe_ms", None))
        timeframe_candidates.append(getattr(data_cfg, "timeframe", None))
    timeframe_candidates.append(getattr(timing_defaults, "timeframe_ms", None))

    timeframe_ms = 0
    for candidate in timeframe_candidates:
        timeframe_ms = _parse_timeframe_to_ms(candidate)
        if timeframe_ms > 0:
            break
    if timeframe_ms <= 0:
        timeframe_ms = 60_000

    cfg.timing.timeframe_ms = int(timeframe_ms)
    timing_fields_set = getattr(cfg.timing, "model_fields_set", None)
    if timing_fields_set is None:
        timing_fields_set = getattr(cfg.timing, "__fields_set__", set())
    if "close_lag_ms" not in timing_fields_set:
        cfg.timing.close_lag_ms = int(resolved_timing.decision_delay_ms)
    else:
        cfg.timing.close_lag_ms = max(0, int(cfg.timing.close_lag_ms))
    cfg.execution.timeframe_ms = int(timeframe_ms)
    cfg.execution.latency_constant_ms = int(resolved_timing.decision_delay_ms)
    cfg.execution.latency_steps = int(resolved_timing.latency_steps)
    cfg.execution.min_lookback_ms = int(resolved_timing.min_lookback_ms)
    cfg.execution.decision_mode = resolved_timing.decision_mode

    logging.getLogger(__name__).info("timing settings: %s", cfg.timing.dict())

    # Load signal bus configuration
    signals_cfg: Dict[str, Any] = {}
    sig_cfg_path = Path("configs/signals.yaml")
    if sig_cfg_path.exists():
        try:
            with sig_cfg_path.open("r", encoding="utf-8") as f:
                signals_cfg = yaml.safe_load(f) or {}
        except Exception:
            signals_cfg = {}

    bus_enabled = bool(signals_cfg.get("enabled", False))
    ttl = int(signals_cfg.get("ttl_seconds", 0))
    out_csv = signals_cfg.get("out_csv")
    dedup_persist = signals_cfg.get("dedup_persist") or cfg.ws_dedup.persist_path
    fsync_mode = str(signals_cfg.get("fsync_mode", "batch") or "batch")
    rotate_daily = bool(signals_cfg.get("rotate_daily", True))
    flush_interval_raw = signals_cfg.get("flush_interval_s", 5.0)
    try:
        flush_interval_s = float(flush_interval_raw)
    except (TypeError, ValueError):
        flush_interval_s = 5.0
    signal_writer_options = {
        "fsync_mode": fsync_mode,
        "rotate_daily": rotate_daily,
        "flush_interval_s": flush_interval_s,
    }
    signal_bus_config = {
        "enabled": bus_enabled,
        "ttl_seconds": ttl,
        "persist_path": dedup_persist,
        "out_csv": out_csv,
        "flush_interval_s": flush_interval_s,
    }

    signal_bus.init(
        enabled=bus_enabled,
        ttl_seconds=ttl,
        persist_path=dedup_persist,
        out_csv=out_csv,
        flush_interval_s=flush_interval_s,
    )

    cfg.ws_dedup.enabled = bus_enabled
    cfg.ws_dedup.persist_path = str(dedup_persist)

    # Load runtime parameters
    rt_cfg: Dict[str, Any] = {}
    rt_cfg_path = Path("configs/runtime.yaml")
    if rt_cfg_path.exists():
        try:
            with rt_cfg_path.open("r", encoding="utf-8") as f:
                rt_cfg = yaml.safe_load(f) or {}
        except Exception:
            rt_cfg = {}

    # Load signal quality configuration
    signal_quality_cfg = SignalQualityConfig()
    sq_cfg_path = Path("configs/signal_quality.yaml")
    sq_data: Dict[str, Any] = {}
    if sq_cfg_path.exists():
        try:
            with sq_cfg_path.open("r", encoding="utf-8") as f:
                sq_data = yaml.safe_load(f) or {}
        except Exception:
            sq_data = {}

    def _apply_signal_quality(data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        if "enabled" in data:
            signal_quality_cfg.enabled = bool(data.get("enabled", signal_quality_cfg.enabled))
        if "sigma_window" in data:
            try:
                value = data.get("sigma_window")
                if value is not None:
                    signal_quality_cfg.sigma_window = int(value)
            except (TypeError, ValueError):
                pass
        if "sigma_threshold" in data:
            try:
                value = data.get("sigma_threshold")
                if value is not None:
                    signal_quality_cfg.sigma_threshold = float(value)
            except (TypeError, ValueError):
                pass
        if "vol_median_window" in data:
            try:
                value = data.get("vol_median_window")
                if value is not None:
                    signal_quality_cfg.vol_median_window = int(value)
            except (TypeError, ValueError):
                pass
        if "vol_floor_frac" in data:
            try:
                value = data.get("vol_floor_frac")
                if value is not None:
                    signal_quality_cfg.vol_floor_frac = float(value)
            except (TypeError, ValueError):
                pass
        if "log_reason" in data:
            value = data.get("log_reason")
            if value is None:
                signal_quality_cfg.log_reason = ""
            elif isinstance(value, bool):
                signal_quality_cfg.log_reason = value
            else:
                signal_quality_cfg.log_reason = str(value)

    _apply_signal_quality(sq_data)
    _apply_signal_quality(rt_cfg.get("signal_quality", {}) or {})

    # Queue configuration for the asynchronous event bus
    queue_cfg = rt_cfg.get("queue", {})
    queue_capacity = int(queue_cfg.get("capacity", 0))
    drop_policy = str(queue_cfg.get("drop_policy", "newest"))
    bus = EventBus(queue_size=queue_capacity, drop_policy=drop_policy)

    # WS deduplication overrides
    ws_cfg = rt_cfg.get("ws", {})
    cfg.ws_dedup.enabled = bool(ws_cfg.get("enabled", cfg.ws_dedup.enabled))
    cfg.ws_dedup.persist_path = str(
        ws_cfg.get("persist_path", cfg.ws_dedup.persist_path)
    )
    cfg.ws_dedup.log_skips = bool(ws_cfg.get("log_skips", cfg.ws_dedup.log_skips))

    # Throttle configuration overrides
    throttle_cfg = rt_cfg.get("throttle", {})
    if throttle_cfg:
        cfg.throttle.enabled = bool(throttle_cfg.get("enabled", cfg.throttle.enabled))
        global_cfg = throttle_cfg.get("global", {})
        cfg.throttle.global_.rps = float(
            global_cfg.get("rps", cfg.throttle.global_.rps)
        )
        cfg.throttle.global_.burst = int(
            global_cfg.get("burst", cfg.throttle.global_.burst)
        )
        sym_cfg = throttle_cfg.get("symbol", {})
        cfg.throttle.symbol.rps = float(sym_cfg.get("rps", cfg.throttle.symbol.rps))
        cfg.throttle.symbol.burst = int(sym_cfg.get("burst", cfg.throttle.symbol.burst))
        cfg.throttle.mode = str(throttle_cfg.get("mode", cfg.throttle.mode))
        q_cfg = throttle_cfg.get("queue", {})
        cfg.throttle.queue.max_items = int(
            q_cfg.get("max_items", cfg.throttle.queue.max_items)
        )
        cfg.throttle.queue.ttl_ms = int(q_cfg.get("ttl_ms", cfg.throttle.queue.ttl_ms))
        cfg.throttle.time_source = str(
            throttle_cfg.get("time_source", cfg.throttle.time_source)
        )

    # Kill switch overrides
    kill_cfg = rt_cfg.get("ops", {}).get("kill_switch", {})
    if kill_cfg:
        cfg.kill_switch.feed_lag_ms = float(
            kill_cfg.get("feed_lag_ms", cfg.kill_switch.feed_lag_ms)
        )
        cfg.kill_switch.ws_failures = float(
            kill_cfg.get("ws_failures", cfg.kill_switch.ws_failures)
        )
        cfg.kill_switch.error_rate = float(
            kill_cfg.get("error_rate", cfg.kill_switch.error_rate)
        )
        cfg.kill_switch_ops.enabled = bool(
            kill_cfg.get("enabled", cfg.kill_switch_ops.enabled)
        )
        cfg.kill_switch_ops.error_limit = int(
            kill_cfg.get("error_limit", cfg.kill_switch_ops.error_limit)
        )
        cfg.kill_switch_ops.duplicate_limit = int(
            kill_cfg.get("duplicate_limit", cfg.kill_switch_ops.duplicate_limit)
        )
        cfg.kill_switch_ops.stale_intervals_limit = int(
            kill_cfg.get(
                "stale_intervals_limit", cfg.kill_switch_ops.stale_intervals_limit
            )
        )
        cfg.kill_switch_ops.reset_cooldown_sec = int(
            kill_cfg.get("reset_cooldown_sec", cfg.kill_switch_ops.reset_cooldown_sec)
        )
        cfg.kill_switch_ops.flag_path = kill_cfg.get(
            "flag_path", cfg.kill_switch_ops.flag_path
        )
        cfg.kill_switch_ops.alert_command = kill_cfg.get(
            "alert_command", cfg.kill_switch_ops.alert_command
        )

    base_pipeline = PipelineConfig()
    base_shutdown: Dict[str, Any] = {}
    if snapshot_config_path:
        try:
            with open(snapshot_config_path, "r", encoding="utf-8") as f:
                base_data = yaml.safe_load(f) or {}
            base_pipeline = _parse_pipeline_config(base_data.get("pipeline", {}))
            base_shutdown = base_data.get("shutdown", {}) or {}
        except Exception:
            base_pipeline = PipelineConfig()
            base_shutdown = {}

    ops_pipeline = PipelineConfig()
    ops_shutdown: Dict[str, Any] = {}
    ops_retry: Dict[str, Any] = {}
    ops_data: Dict[str, Any] = {}
    for name, loader in (
        ("configs/ops.yaml", yaml.safe_load),
        ("configs/ops.json", json.load),
    ):
        ops_path = Path(name)
        if ops_path.exists():
            try:
                with ops_path.open("r", encoding="utf-8") as f:
                    ops_data = loader(f) or {}
            except Exception:
                ops_data = {}
            break
    if ops_data:
        ops_pipeline = _parse_pipeline_config(ops_data.get("pipeline", {}))
        ops_shutdown = ops_data.get("shutdown", {}) or {}
        ops_retry = ops_data.get("retry", {}) or {}
    elif rt_cfg.get("pipeline"):
        ops_pipeline = _parse_pipeline_config(rt_cfg.get("pipeline", {}))

    pipeline_cfg = base_pipeline.merge(ops_pipeline)

    shutdown_cfg: Dict[str, Any] = {}
    shutdown_cfg.update(base_shutdown)
    shutdown_cfg.update(ops_shutdown)
    shutdown_cfg.update(rt_cfg.get("shutdown", {}))

    if ops_retry:
        cfg.retry.max_attempts = int(
            ops_retry.get("max_attempts", cfg.retry.max_attempts)
        )
        cfg.retry.backoff_base_s = float(
            ops_retry.get("backoff_base_s", cfg.retry.backoff_base_s)
        )
        cfg.retry.max_backoff_s = float(
            ops_retry.get("max_backoff_s", cfg.retry.max_backoff_s)
        )

    # Ensure components receive the bus if they accept it
    try:
        cfg.components.market_data.params.setdefault("bus", bus)
    except Exception:
        pass
    try:
        cfg.components.executor.params.setdefault("bus", bus)
    except Exception:
        pass

    monitoring_cfg = MonitoringConfig()
    mon_path = Path("configs/monitoring.yaml")
    if mon_path.exists():
        try:
            with mon_path.open("r", encoding="utf-8") as f:
                mon_data = yaml.safe_load(f) or {}
        except Exception:
            mon_data = {}
        mon_section = mon_data.get("monitoring", {}) or {}
        monitoring_cfg.enabled = bool(
            mon_section.get("enabled", monitoring_cfg.enabled)
        )
        monitoring_cfg.snapshot_metrics_sec = int(
            mon_section.get("snapshot_metrics_sec", monitoring_cfg.snapshot_metrics_sec)
        )
        tick_sec = mon_section.get("tick_sec")
        if tick_sec is not None:
            try:
                monitoring_cfg.tick_sec = float(tick_sec)
            except (TypeError, ValueError):
                pass
        thr = mon_data.get("thresholds", {}) or {}
        for key in (
            "feed_lag_ms",
            "ws_failures",
            "error_rate",
            "fill_ratio_min",
            "pnl_min",
        ):
            if key in thr:
                try:
                    setattr(
                        monitoring_cfg.thresholds,
                        key,
                        float(thr.get(key)),
                    )
                except (TypeError, ValueError):
                    pass
        if "zero_signals" in thr:
            try:
                monitoring_cfg.thresholds.zero_signals = int(thr.get("zero_signals"))
            except (TypeError, ValueError):
                pass
        al = mon_data.get("alerts", {}) or {}
        monitoring_cfg.alerts.enabled = bool(
            al.get("enabled", monitoring_cfg.alerts.enabled)
        )
        monitoring_cfg.alerts.command = al.get(
            "command", monitoring_cfg.alerts.command
        )
        if "channel" in al:
            channel = al.get("channel")
            if channel is not None:
                monitoring_cfg.alerts.channel = str(channel)
        if "cooldown_sec" in al:
            try:
                monitoring_cfg.alerts.cooldown_sec = float(al.get("cooldown_sec"))
            except (TypeError, ValueError):
                pass

    risk_limits: Dict[str, Any] = dict(cfg.risk.exposure_limits)
    try:
        risk_component_params: Dict[str, Any] = dict(cfg.risk.component_params())
    except Exception:
        risk_component_params = {}
    risk_override_path = Path("configs/risk.yaml")
    if risk_override_path.exists():
        try:
            with risk_override_path.open("r", encoding="utf-8") as fh:
                override_payload = yaml.safe_load(fh) or {}
        except Exception:
            override_payload = {}
        if isinstance(override_payload, MappingABC):
            for key in (
                "max_total_notional",
                "max_total_exposure_pct",
                "exposure_buffer_frac",
            ):
                if key in override_payload:
                    risk_limits[key] = override_payload.get(key)
            for key, value in override_payload.items():
                if key in {
                    "max_total_notional",
                    "max_total_exposure_pct",
                    "exposure_buffer_frac",
                }:
                    continue
                risk_component_params[key] = value

    svc_cfg = SignalRunnerConfig(
        snapshot_config_path=snapshot_config_path,
        artifacts_dir=cfg.artifacts_dir,
        logs_dir=cfg.logs_dir,
        marker_path=os.path.join(cfg.logs_dir, "shutdown.marker"),
        run_id=cfg.run_id,
        snapshot_metrics_json=os.path.join(cfg.logs_dir, "snapshot_metrics.json"),
        snapshot_metrics_csv=os.path.join(cfg.logs_dir, "snapshot_metrics.csv"),
        snapshot_metrics_sec=monitoring_cfg.snapshot_metrics_sec,
        kill_switch_ops=cfg.kill_switch_ops,
        state=cfg.state,
    )
    sec = rt_cfg.get("ops", {}).get("snapshot_metrics_sec")
    if isinstance(sec, (int, float)) and sec > 0:
        svc_cfg.snapshot_metrics_sec = int(sec)
        monitoring_cfg.snapshot_metrics_sec = int(sec)

    container = di_registry.build_graph(cfg.components, cfg)
    adapter: SimAdapter = container["executor"]
    if not hasattr(adapter, "bus"):
        try:
            adapter.bus = bus
        except Exception:
            pass
    fp: FeaturePipe = container["feature_pipe"]
    policy: SignalPolicy = container["policy"]
    guards: RiskGuards | None = container.get("risk_guards")
    service = ServiceSignalRunner(
        adapter,
        fp,
        policy,
        guards,
        svc_cfg,
        cfg.clock_sync,
        cfg.throttle,
        NoTradeConfig(**(cfg.no_trade or {})),
        signal_quality_cfg=signal_quality_cfg,
        portfolio_limits=risk_limits,
        pipeline_cfg=pipeline_cfg,
        shutdown_cfg=shutdown_cfg,
        monitoring_cfg=monitoring_cfg,
        enforce_closed_bars=cfg.timing.enforce_closed_bars,
        close_lag_ms=cfg.timing.close_lag_ms,
        ws_dedup_enabled=cfg.ws_dedup.enabled,
        ws_dedup_log_skips=cfg.ws_dedup.log_skips,
        ws_dedup_timeframe_ms=cfg.timing.timeframe_ms,
        signal_writer_options=signal_writer_options,
        signal_bus_config=signal_bus_config,
        risk_component_params=risk_component_params,
        run_config=cfg,
    )
    return service.run()


__all__ = [
    "CooldownSettings",
    "SignalQualityConfig",
    "SignalRunnerConfig",
    "RunnerConfig",
    "ServiceSignalRunner",
    "MixedQuoteError",
    "clear_dirty_restart",
    "from_config",
]
