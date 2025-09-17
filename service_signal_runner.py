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

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Iterator, Protocol, Callable, Mapping, Tuple
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
import os
import logging
import math
import threading
import signal
from pathlib import Path
from collections import deque, defaultdict
import time
from datetime import datetime, timezone
import shlex

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
from services.signal_bus import log_drop
from services.event_bus import EventBus
from services.shutdown import ShutdownManager
from services.signal_csv_writer import SignalCSVWriter
from adapters.binance_spot_private import reconcile_state

from sandbox.sim_adapter import SimAdapter  # исп. как TradeExecutor-подобный мост
from core_models import Bar, Tick
from core_contracts import FeaturePipe, SignalPolicy
from services.utils_config import (
    snapshot_config,
)  # снапшот конфига (Фаза 3)  # noqa: F401
from core_config import (
    CommonRunConfig,
    ClockSyncConfig,
    ThrottleConfig,
    OpsKillSwitchConfig,
    MonitoringConfig,
    StateConfig,
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


# обратная совместимость
RunnerConfig = SignalRunnerConfig


class _Worker:
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
        ws_dedup_enabled: bool = False,
        ws_dedup_log_skips: bool = False,
        ws_dedup_timeframe_ms: int = 0,
        throttle_cfg: ThrottleConfig | None = None,
        no_trade_cfg: NoTradeConfig | None = None,
        pipeline_cfg: PipelineConfig | None = None,
        signal_quality_cfg: SignalQualityConfig | None = None,
        zero_signal_alert: int = 0,
        state_enabled: bool = False,
        rest_candidates: Sequence[Any] | None = None,
    ) -> None:
        self._fp = fp
        self._policy = policy
        self._logger = logger
        self._executor = executor
        self._guards = guards
        self._safe_mode_fn = safe_mode_fn or (lambda: False)
        self._enforce_closed_bars = enforce_closed_bars
        self._ws_dedup_enabled = ws_dedup_enabled
        self._ws_dedup_log_skips = ws_dedup_log_skips
        self._ws_dedup_timeframe_ms = ws_dedup_timeframe_ms
        self._throttle_cfg = throttle_cfg
        self._no_trade_cfg = no_trade_cfg
        self._pipeline_cfg = pipeline_cfg
        self._signal_quality_cfg = signal_quality_cfg or SignalQualityConfig()
        self._zero_signal_alert = int(zero_signal_alert)
        self._zero_signal_streak = 0
        self._state_enabled = bool(state_enabled)
        self._global_bucket = None
        self._symbol_bucket_factory = None
        self._symbol_buckets = None
        self._queue = None
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
        if throttle_cfg is not None and throttle_cfg.enabled:
            self._global_bucket = TokenBucket(
                rps=throttle_cfg.global_.rps, burst=throttle_cfg.global_.burst
            )
            self._symbol_bucket_factory = lambda: TokenBucket(
                rps=throttle_cfg.symbol.rps, burst=throttle_cfg.symbol.burst
            )
            self._symbol_buckets = defaultdict(self._symbol_bucket_factory)
            self._queue = deque(maxlen=throttle_cfg.queue.max_items)
        self._last_bar_ts: Dict[str, int] = {}
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
        if self._state_enabled:
            self._seed_last_prices_from_state()

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

    @staticmethod
    def _coerce_price(value: Any) -> float | None:
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(price) or price <= 0.0:
            return None
        return price

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
        if self._global_bucket is None:
            return True, None
        now = time.monotonic()
        if not self._global_bucket.consume(now=now):
            return False, "THROTTLED_GLOBAL"
        sb = self._symbol_buckets[symbol]
        if not sb.consume(now=now):
            self._global_bucket.tokens = min(
                self._global_bucket.tokens + 1.0, self._global_bucket.burst
            )
            return False, "THROTTLED_SYMBOL"
        return True, None

    def _refund_tokens(self, symbol: str) -> None:
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

    def _emit(self, o: Any, symbol: str, bar_close_ms: int) -> bool:
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
                try:
                    log_drop(symbol, bar_close_ms, o, reason or "RISK")
                except Exception:
                    pass
                return False
            o = list(checked)[0]

        created_ts = int(getattr(o, "created_ts_ms", 0) or 0)
        now_ms = clock.now_ms()
        ok, expires_at_ms, _ = check_ttl(
            bar_close_ms=created_ts,
            now_ms=now_ms,
            timeframe_ms=self._ws_dedup_timeframe_ms,
        )
        if not ok:
            try:
                self._logger.info(
                    "TTL_EXPIRED_PUBLISH %s",
                    {
                        "symbol": symbol,
                        "created_ts_ms": created_ts,
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
            return False
        try:
            age_ms = now_ms - created_ts
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
        if getattr(signal_bus, "ENABLED", False):
            try:
                signal_bus.publish_signal(
                    ts_ms=bar_close_ms,
                    symbol=symbol,
                    side=side,
                    score=score,
                    features_hash=fh,
                    bar_close_ms=bar_close_ms,
                )
            except Exception:
                pass
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
        bar_close_ms: int,
        *,
        stage_cfg: PipelineStageConfig | None = None,
    ) -> PipelineResult:
        """Final pipeline stage: throttle and emit order."""
        if stage_cfg is not None and not stage_cfg.enabled:
            return PipelineResult(action="pass", stage=Stage.PUBLISH, decision=o)
        if self._throttle_cfg and self._throttle_cfg.enabled:
            ok, reason = self._acquire_tokens(symbol)
            if not ok:
                if self._throttle_cfg.mode == "queue" and self._queue is not None:
                    exp = time.monotonic() + self._throttle_cfg.queue.ttl_ms / 1000.0
                    self._queue.append((exp, symbol, bar_close_ms, o))
                    try:
                        monitoring.throttle_enqueued_count.labels(
                            symbol, reason or ""
                        ).inc()
                    except Exception:
                        pass
                    return PipelineResult(action="queue", stage=Stage.PUBLISH)
                try:
                    log_drop(symbol, bar_close_ms, o, reason or "")
                    monitoring.throttle_dropped_count.labels(symbol, reason or "").inc()
                except Exception:
                    pass
                return PipelineResult(
                    action="drop", stage=Stage.PUBLISH, reason=Reason.OTHER
                )
        if not self._emit(o, symbol, bar_close_ms):
            self._refund_tokens(symbol)
            return PipelineResult(
                action="drop", stage=Stage.PUBLISH, reason=Reason.OTHER
            )
        return PipelineResult(action="pass", stage=Stage.PUBLISH, decision=o)

    def _drain_queue(self) -> list[Any]:
        emitted: list[Any] = []
        if self._queue is None:
            return emitted
        now = time.monotonic()
        while self._queue:
            exp, symbol, bar_close_ms, order = self._queue[0]
            if exp <= now:
                self._queue.popleft()
                try:
                    log_drop(symbol, bar_close_ms, order, "QUEUE_EXPIRED")
                    monitoring.throttle_queue_expired_count.labels(symbol).inc()
                    monitoring.throttle_dropped_count.labels(
                        symbol, "QUEUE_EXPIRED"
                    ).inc()
                except Exception:
                    pass
                continue
            ok, _ = self._acquire_tokens(symbol)
            if not ok:
                break
            self._queue.popleft()
            if not self._emit(order, symbol, bar_close_ms):
                self._refund_tokens(symbol)
            else:
                emitted.append(order)
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
        ts_ms = int(bar.ts)
        close_val = self._coerce_price(getattr(bar, "close", None))
        if close_val is not None:
            self._set_last_price(symbol, close_val)
        else:
            self._maybe_fetch_rest_price(symbol)
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
            emitted_count = len(emitted)
            try:
                monitoring.record_signals(bar.symbol, emitted_count, duplicates)
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
            return emitted

        if self._pipeline_cfg is not None and not self._pipeline_cfg.enabled:
            return _finalize()
        if self._safe_mode_fn():
            return _finalize()

        if self._queue is not None:
            emitted.extend(self._drain_queue())

        close_ms: int | None = None
        if self._ws_dedup_enabled:
            close_ms = int(bar.ts) + self._ws_dedup_timeframe_ms
            if signal_bus.should_skip(bar.symbol, close_ms):
                if self._ws_dedup_log_skips:
                    try:
                        self._logger.info("SKIP_DUPLICATE_BAR")
                    except Exception:
                        pass
                try:
                    monitoring.ws_dup_skipped_count.labels(bar.symbol).inc()
                except Exception:
                    pass
                try:
                    monitoring.queue_len.set(len(self._queue) if self._queue else 0)
                except Exception:
                    pass
                duplicates = 1
                return _finalize()
        guard_res = closed_bar_guard(
            bar=bar,
            now_ms=clock.now_ms(),
            enforce=self._enforce_closed_bars,
            lag_ms=0,
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
            try:
                monitoring.queue_len.set(len(self._queue) if self._queue else 0)
            except Exception:
                pass
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

        created_ts_ms = clock.now_ms()
        checked_orders = []
        for o in orders:
            ok, expires_at_ms, _ = check_ttl(
                bar_close_ms=int(bar.ts),
                now_ms=created_ts_ms,
                timeframe_ms=self._ws_dedup_timeframe_ms,
            )
            if not ok:
                try:
                    self._logger.info(
                        "TTL_EXPIRED_BOUNDARY %s",
                        {
                            "symbol": bar.symbol,
                            "bar_close_ms": int(bar.ts),
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
                continue

            setattr(o, "created_ts_ms", created_ts_ms)
            checked_orders.append(o)

        for o in checked_orders:
            res = self.publish_decision(
                o,
                bar.symbol,
                int(bar.ts),
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

        try:
            monitoring.queue_len.set(len(self._queue) if self._queue else 0)
        except Exception:
            pass

        if self._ws_dedup_enabled and close_ms is not None:
            try:
                signal_bus.update(bar.symbol, close_ms)
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
        pipeline_cfg: PipelineConfig | None = None,
        shutdown_cfg: Dict[str, Any] | None = None,
        monitoring_cfg: MonitoringConfig | None = None,
        *,
        enforce_closed_bars: bool = True,
        ws_dedup_enabled: bool = False,
        ws_dedup_log_skips: bool = False,
        ws_dedup_timeframe_ms: int = 0,
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
        self.pipeline_cfg = pipeline_cfg
        self.shutdown_cfg = shutdown_cfg or {}
        self.monitoring_cfg = monitoring_cfg or MonitoringConfig()
        self.alerts: AlertManager | None = None
        self.monitoring_agg: MonitoringAggregator | None = None
        self._clock_safe_mode = False
        self._clock_stop = threading.Event()
        self._clock_thread: Optional[threading.Thread] = None
        self.enforce_closed_bars = enforce_closed_bars
        self.ws_dedup_enabled = ws_dedup_enabled
        self.ws_dedup_log_skips = ws_dedup_log_skips
        self.ws_dedup_timeframe_ms = ws_dedup_timeframe_ms

        if self.monitoring_cfg.enabled:
            alert_cfg = getattr(self.monitoring_cfg, "alerts", None)
            channel = getattr(alert_cfg, "channel", "noop")
            cooldown = float(getattr(alert_cfg, "cooldown_sec", 0.0))
            try:
                self.alerts = AlertManager(channel, cooldown)
                self.monitoring_agg = MonitoringAggregator(
                    self.monitoring_cfg, self.alerts
                )
            except Exception:
                self.alerts = None
                self.monitoring_agg = None

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

        run_id = self.cfg.run_id or "sim"
        logs_dir = self.cfg.logs_dir or "logs"
        sim = getattr(self.adapter, "sim", None) or getattr(self.adapter, "_sim", None)
        if sim is not None:
            logging_config = {
                "trades_path": os.path.join(logs_dir, f"log_trades_{run_id}.csv"),
                "reports_path": os.path.join(logs_dir, f"report_equity_{run_id}.csv"),
            }
            try:
                from logging import LogWriter, LogConfig  # type: ignore

                sim._logger = LogWriter(
                    LogConfig.from_dict(logging_config), run_id=run_id
                )
            except Exception:
                pass

    def run(self) -> Iterator[Dict[str, Any]]:
        # снапшот конфига, если задан
        if self.cfg.snapshot_config_path and self.cfg.artifacts_dir:
            snapshot_config(self.cfg.snapshot_config_path, self.cfg.artifacts_dir)

        logs_dir = self.cfg.logs_dir or "logs"
        marker_path = Path(
            self.cfg.marker_path or os.path.join(logs_dir, "shutdown.marker")
        )
        dirty_restart = True
        entry_limits_state: Dict[str, Dict[str, Any]] | None = None
        try:
            if marker_path.exists():
                dirty_restart = False
            marker_path.unlink()
        except Exception:
            dirty_restart = True
        if self.cfg.state.enabled:
            try:
                loaded_state = state_storage.load_state(
                    self.cfg.state.path,
                    backend=self.cfg.state.backend,
                    lock_path=self.cfg.state.lock_path,
                    backup_keep=self.cfg.state.backup_keep,
                )
                try:
                    entry_limits_raw = getattr(loaded_state, "entry_limits", {}) or {}
                    if isinstance(entry_limits_raw, MappingABC):
                        entry_limits_state = {
                            str(symbol): dict(payload)
                            for symbol, payload in entry_limits_raw.items()
                            if isinstance(payload, MappingABC)
                        }
                    else:
                        entry_limits_state = None
                except Exception:
                    entry_limits_state = None
                loader = getattr(self.policy, "load_signal_state", None)
                if callable(loader):
                    try:
                        loader(getattr(loaded_state, "signal_states", {}) or {})
                    except Exception:
                        pass
                if self.ws_dedup_enabled:
                    try:
                        seen = dict(getattr(loaded_state, "seen_signals", {}) or {})
                        for sid, exp in seen.items():
                            signal_bus.STATE[str(sid)] = int(exp)
                    except Exception:
                        pass
                client = getattr(self.adapter, "client", None)
                if client is not None:
                    try:
                        summary = reconcile_state(loaded_state, client)
                        self.logger.info("state reconciliation: %s", summary)
                    except Exception as e:
                        self.logger.warning("state reconciliation skipped: %s", e)
            except Exception:
                pass
        if dirty_restart and self.monitoring_cfg.enabled:
            try:
                monitoring.reset_kill_switch_counters()
            except Exception:
                pass

        shutdown = ShutdownManager(self.shutdown_cfg)
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
            limit = int(self.cfg.kill_switch_ops.error_limit)
            while not ops_flush_stop.wait(1.0):
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
        if self.monitoring_agg is not None:

            def _monitoring_loop() -> None:
                interval = float(getattr(self.monitoring_cfg, "tick_sec", 1.0))
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
            shutdown.on_flush(self.monitoring_agg.flush)
            shutdown.on_finalize(lambda: monitoring_thread.join(timeout=1.0))

        rest_candidates = [
            getattr(self.adapter, "rest_helper", None),
            getattr(self.adapter, "client", None),
            getattr(self.adapter, "rest_client", None),
            getattr(self.adapter, "rest", None),
        ]

        worker = _Worker(
            self.feature_pipe,
            self.policy,
            self.logger,
            self.adapter,
            self.risk_guards,
            lambda: self._clock_safe_mode
            or monitoring.kill_switch_triggered()
            or ops_kill_switch.tripped(),
            enforce_closed_bars=self.enforce_closed_bars,
            ws_dedup_enabled=self.ws_dedup_enabled,
            ws_dedup_log_skips=self.ws_dedup_log_skips,
            ws_dedup_timeframe_ms=self.ws_dedup_timeframe_ms,
            throttle_cfg=self.throttle_cfg,
            no_trade_cfg=self.no_trade_cfg,
            pipeline_cfg=self.pipeline_cfg,
            signal_quality_cfg=self.signal_quality_cfg,
            zero_signal_alert=getattr(
                self.monitoring_cfg.thresholds, "zero_signals", 0
            ),
            state_enabled=self.cfg.state.enabled,
            rest_candidates=rest_candidates,
        )
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
                signal_bus.OUT_WRITER = SignalCSVWriter(out_csv)
                shutdown.on_flush(signal_bus.OUT_WRITER.flush_fsync)
                shutdown.on_finalize(signal_bus.OUT_WRITER.close)
            except Exception:
                signal_bus.OUT_WRITER = None

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
        if self.cfg.state.enabled and self.cfg.state.snapshot_interval_s > 0:

            def _state_loop() -> None:
                while not state_stop.wait(self.cfg.state.snapshot_interval_s):
                    try:
                        state_storage.save_state(
                            self.cfg.state.path,
                            backend=self.cfg.state.backend,
                            lock_path=self.cfg.state.lock_path,
                            backup_keep=self.cfg.state.backup_keep,
                        )
                    except Exception:
                        pass

            state_thread = threading.Thread(target=_state_loop, daemon=True)
            state_thread.start()
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
            shutdown.on_flush(
                lambda: state_storage.save_state(
                    self.cfg.state.path,
                    backend=self.cfg.state.backend,
                    lock_path=self.cfg.state.lock_path,
                    backup_keep=self.cfg.state.backup_keep,
                )
            )

        def _write_marker() -> None:
            try:
                marker_path.touch()
            except Exception:
                pass

        def _final_summary() -> None:
            try:
                summary, json_str, _ = monitoring.snapshot_metrics(json_path, csv_path)
                self.logger.info("SUMMARY %s", json_str)
            except Exception:
                pass

        shutdown.on_finalize(_write_marker)
        if snapshot_thread is not None:
            shutdown.on_finalize(lambda: snapshot_thread.join(timeout=1.0))
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

        ws_failures = 0
        limit = int(self.cfg.kill_switch_ops.error_limit)

        def _handle_report(rep: Dict[str, Any]) -> None:
            if not self.cfg.state.enabled:
                return
            updated = False
            try:
                symbol = rep.get("symbol")
                pos_qty = rep.get("position_qty")
                if symbol is not None and pos_qty is not None:
                    st = state_storage.get_state()
                    positions = dict(getattr(st, "positions", {}))
                    if positions.get(symbol) != pos_qty:
                        positions[symbol] = pos_qty
                        state_storage.update_state(positions=positions)
                        updated = True

                orders = rep.get("core_orders") or []
                if orders:
                    st = state_storage.get_state()
                    open_orders = dict(getattr(st, "open_orders", {}))
                    for o in orders:
                        oid = str(
                            o.get("order_id")
                            or o.get("client_order_id")
                            or len(open_orders)
                        )
                        open_orders[oid] = o
                    state_storage.update_state(open_orders=open_orders)
                    updated = True

                reports = rep.get("core_exec_reports") or []
                if reports:
                    st = state_storage.get_state()
                    open_orders = dict(getattr(st, "open_orders", {}))
                    for er in reports:
                        oid = str(er.get("order_id") or er.get("client_order_id"))
                        status = str(
                            er.get("status") or er.get("exec_status") or ""
                        ).upper()
                        if status in {"FILLED", "CANCELLED"}:
                            open_orders.pop(oid, None)
                        else:
                            if oid in open_orders:
                                try:
                                    open_orders[oid].update(er)
                                except Exception:
                                    open_orders[oid] = er
                    state_storage.update_state(open_orders=open_orders)
                    updated = True

                ts_ms = rep.get("ts_ms")
                if ts_ms is not None:
                    seen = getattr(signal_bus, "STATE", {})
                    try:
                        seen_items = list(getattr(seen, "items", lambda: [])())
                    except Exception:
                        seen_items = []
                    state_storage.update_state(
                        last_processed_bar_ms=int(ts_ms),
                        seen_signals=seen_items,
                    )
                    updated = True
            except Exception:
                pass
            if updated and self.cfg.state.flush_on_event:
                try:
                    state_storage.save_state(
                        self.cfg.state.path,
                        backend=self.cfg.state.backend,
                        lock_path=self.cfg.state.lock_path,
                        backup_keep=self.cfg.state.backup_keep,
                    )
                except Exception:
                    pass

        while not stop_event.is_set():
            try:
                for rep in self.adapter.run_events(worker):
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

    signal_bus.init(
        enabled=bus_enabled,
        ttl_seconds=ttl,
        persist_path=dedup_persist,
        out_csv=out_csv,
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

    # Pipeline configuration
    def _parse_pipeline(data: Dict[str, Any]) -> PipelineConfig:
        enabled = bool(data.get("enabled", True))
        stages_cfg = data.get("stages", {}) or {}
        stages: Dict[str, PipelineStageConfig] = {}
        for name, st in stages_cfg.items():
            if isinstance(st, dict):
                st_enabled = bool(st.get("enabled", True))
                params = {k: v for k, v in st.items() if k != "enabled"}
            else:
                st_enabled = bool(st)
                params = {}
            stages[name] = PipelineStageConfig(enabled=st_enabled, params=params)
        return PipelineConfig(enabled=enabled, stages=stages)

    base_pipeline = PipelineConfig()
    base_shutdown: Dict[str, Any] = {}
    if snapshot_config_path:
        try:
            with open(snapshot_config_path, "r", encoding="utf-8") as f:
                base_data = yaml.safe_load(f) or {}
            base_pipeline = _parse_pipeline(base_data.get("pipeline", {}))
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
        ops_pipeline = _parse_pipeline(ops_data.get("pipeline", {}))
        ops_shutdown = ops_data.get("shutdown", {}) or {}
        ops_retry = ops_data.get("retry", {}) or {}
    elif rt_cfg.get("pipeline"):
        ops_pipeline = _parse_pipeline(rt_cfg.get("pipeline", {}))

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
        thr = mon_data.get("thresholds", {}) or {}
        monitoring_cfg.thresholds.feed_lag_ms = float(
            thr.get("feed_lag_ms", monitoring_cfg.thresholds.feed_lag_ms)
        )
        monitoring_cfg.thresholds.ws_failures = float(
            thr.get("ws_failures", monitoring_cfg.thresholds.ws_failures)
        )
        monitoring_cfg.thresholds.error_rate = float(
            thr.get("error_rate", monitoring_cfg.thresholds.error_rate)
        )
        al = mon_data.get("alerts", {}) or {}
        monitoring_cfg.alerts.enabled = bool(
            al.get("enabled", monitoring_cfg.alerts.enabled)
        )
        monitoring_cfg.alerts.command = al.get("command", monitoring_cfg.alerts.command)

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
        pipeline_cfg=pipeline_cfg,
        shutdown_cfg=shutdown_cfg,
        monitoring_cfg=monitoring_cfg,
        enforce_closed_bars=cfg.timing.enforce_closed_bars,
        ws_dedup_enabled=cfg.ws_dedup.enabled,
        ws_dedup_log_skips=cfg.ws_dedup.log_skips,
        ws_dedup_timeframe_ms=cfg.timing.timeframe_ms,
    )
    return service.run()


__all__ = [
    "SignalQualityConfig",
    "SignalRunnerConfig",
    "RunnerConfig",
    "ServiceSignalRunner",
    "from_config",
]
