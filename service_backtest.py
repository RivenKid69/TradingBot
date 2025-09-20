# -*- coding: utf-8 -*-
"""
services/service_backtest.py
Оркестратор офлайн-бэктеста. Минимальная склейка компонентов.

Пример использования через конфиг:

```python
from core_config import CommonRunConfig
from service_backtest import from_config

cfg = CommonRunConfig(...)
df = ...  # pandas.DataFrame с ценами
reports = from_config(cfg, df)
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Mapping
import logging
import os
import math
import time
from datetime import datetime, timezone
import pandas as pd

from execution_sim import ExecutionSimulator  # type: ignore
from adv_store import ADVStore
from sandbox.backtest_adapter import BacktestAdapter
from sandbox.sim_adapter import SimAdapter
from core_contracts import SignalPolicy
from services.utils_config import snapshot_config  # сохранение снапшота конфига
from services.utils_sandbox import read_df
from core_config import CommonRunConfig
import di_registry


try:  # pragma: no cover - optional dependency in sandbox setups
    from impl_slippage import SlippageImpl  # type: ignore
except Exception:  # pragma: no cover - fallback when implementation missing
    SlippageImpl = None  # type: ignore


logger = logging.getLogger(__name__)


def _coerce_timeframe_ms(value: Any) -> Optional[int]:
    """Best-effort conversion of ``value`` to timeframe in milliseconds."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            ms = int(value)
            return ms if ms > 0 else None
        text = str(value).strip()
        if not text:
            return None
        if text.isdigit():
            ms = int(text)
            return ms if ms > 0 else None
        suffix = text[-1].lower()
        mult = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}
        if suffix not in mult:
            return None
        amount_text = text[:-1].strip()
        if not amount_text:
            return None
        amount = float(amount_text)
        ms = int(amount * mult[suffix])
        return ms if ms > 0 else None
    except (TypeError, ValueError):
        return None


def _extract_dynamic_slippage_cfg(
    run_cfg: CommonRunConfig | None,
) -> Optional[Dict[str, Any]]:
    if run_cfg is None:
        return None
    slip_cfg = getattr(run_cfg, "slippage", None)
    dyn_block: Any = None
    if isinstance(slip_cfg, dict):
        dyn_block = slip_cfg.get("dynamic") or slip_cfg.get("dynamic_spread")
    else:
        dyn_block = getattr(slip_cfg, "dynamic", None) or getattr(
            slip_cfg, "dynamic_spread", None
        )
    if dyn_block is None:
        return None
    payload: Optional[Dict[str, Any]] = None
    if isinstance(dyn_block, dict):
        payload = dict(dyn_block)
    if hasattr(dyn_block, "dict"):
        try:
            data = dyn_block.dict()  # type: ignore[attr-defined]
        except Exception:
            data = None
        if isinstance(data, dict):
            payload = dict(data)
    if hasattr(dyn_block, "model_dump"):
        try:
            data = dyn_block.model_dump()  # type: ignore[attr-defined]
        except Exception:
            data = None
        if isinstance(data, dict):
            payload = dict(data)
    if payload is None:
        try:
            payload = dict(dyn_block)
        except Exception:
            payload = None
    result: Dict[str, Any] = {}
    for key in (
        "enabled",
        "base_bps",
        "alpha_vol",
        "beta_illiquidity",
        "vol_mode",
        "liq_col",
        "liq_ref",
        "min_bps",
        "max_bps",
        "vol_metric",
        "vol_window",
    ):
        if hasattr(dyn_block, key):
            result[key] = getattr(dyn_block, key)
    if payload is None:
        payload = result if result else None
    if not payload:
        return None

    mapped = dict(payload)

    def _propagate(primary: str, *aliases: str) -> None:
        val = None
        for key in (primary, *aliases):
            if key in mapped and mapped[key] is not None:
                val = mapped[key]
                break
        if val is None:
            return
        mapped[primary] = val
        for alias in aliases:
            mapped.setdefault(alias, val)

    _propagate("alpha_bps", "alpha", "base_bps")
    _propagate("beta_coef", "beta", "alpha_vol")
    _propagate("min_spread_bps", "min_bps")
    _propagate("max_spread_bps", "max_bps")
    _propagate("smoothing_alpha", "smoothing")
    _propagate("vol_metric", "volatility_metric")
    _propagate("vol_window", "volatility_window")

    return mapped or None


def _slippage_to_dict(cfg: Any) -> Optional[Dict[str, Any]]:
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return dict(cfg)
    for attr in ("model_dump", "dict"):
        if hasattr(cfg, attr):
            try:
                method = getattr(cfg, attr)
                if attr == "model_dump":
                    payload = method(exclude_unset=False)  # type: ignore[call-arg]
                else:
                    payload = method(exclude_unset=False)  # type: ignore[call-arg]
            except TypeError:
                try:
                    payload = method()
                except Exception:
                    payload = None
            except Exception:
                payload = None
            if isinstance(payload, dict):
                return dict(payload)
    try:
        return dict(cfg)
    except Exception:
        return None


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


def _as_mapping(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        try:
            return {str(k): v for k, v in cfg.items()}
        except Exception:
            return dict(cfg)
    for attr in ("model_dump", "dict"):
        if hasattr(cfg, attr):
            try:
                method = getattr(cfg, attr)
                payload = method(exclude_unset=False)  # type: ignore[call-arg]
            except TypeError:
                try:
                    payload = method()
                except Exception:
                    payload = None
            except Exception:
                payload = None
            if isinstance(payload, Mapping):
                try:
                    return {str(k): v for k, v in payload.items()}
                except Exception:
                    return dict(payload)
    if hasattr(cfg, "__dict__"):
        try:
            return {
                str(k): v
                for k, v in vars(cfg).items()
                if not str(k).startswith("_")
            }
        except Exception:
            return {}
    return {}


_BAR_CAPACITY_KEY_ALIASES: Dict[str, set[str]] = {
    "enabled": {"enabled"},
    "adv_base_path": {"adv_base_path", "adv_path", "path", "dataset_path"},
    "capacity_frac_of_ADV_base": {
        "capacity_frac_of_adv_base",
        "capacity_fraction_of_adv_base",
        "capacity_frac_of_adv",
        "capacity_fraction_of_adv",
        "capacity_frac_of_adv_base_pct",
    },
    "floor_base": {"floor_base", "floor", "adv_floor", "adv_floor_base"},
    "timeframe_ms": {
        "timeframe_ms",
        "timeframe",
        "bar_timeframe_ms",
        "bar_timeframe",
    },
}


def _normalise_bar_capacity_block(block: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    stack: List[Mapping[str, Any]] = [block]
    while stack:
        current = stack.pop()
        for raw_key, value in current.items():
            try:
                key_lower = str(raw_key).strip().lower()
            except Exception:
                continue
            if key_lower == "extra" and isinstance(value, Mapping):
                stack.append(value)
                continue
            for target, aliases in _BAR_CAPACITY_KEY_ALIASES.items():
                if key_lower in aliases and target not in result:
                    result[target] = value
                    break
    return result


def _extract_bar_capacity_base_cfg(
    run_cfg: CommonRunConfig | None,
) -> Optional[Dict[str, Any]]:
    if run_cfg is None:
        return None
    exec_cfg = getattr(run_cfg, "execution", None)
    if exec_cfg is None:
        return None
    exec_payload = _as_mapping(exec_cfg)
    block: Any = None
    if "bar_capacity_base" in exec_payload:
        block = exec_payload.get("bar_capacity_base")
    elif hasattr(exec_cfg, "bar_capacity_base"):
        block = getattr(exec_cfg, "bar_capacity_base")
    if block is None and isinstance(exec_payload.get("extra"), Mapping):
        block = exec_payload["extra"].get("bar_capacity_base")
    block_map = _as_mapping(block)
    if not block_map:
        return None
    normalised = _normalise_bar_capacity_block(block_map)
    return normalised or None


def _finalise_bar_capacity_payload(
    raw_cfg: Optional[Dict[str, Any]],
    *,
    adv_store: Optional[ADVStore],
    default_timeframe_ms: Optional[int],
) -> tuple[Optional[Dict[str, Any]], List[tuple[str, Any]], List[str]]:
    if not raw_cfg:
        return None, [], []

    payload: Dict[str, Any] = {}
    fallbacks: List[tuple[str, Any]] = []
    missing: List[str] = []

    if "enabled" in raw_cfg:
        payload["enabled"] = bool(raw_cfg.get("enabled"))
    if "capacity_frac_of_ADV_base" in raw_cfg:
        payload["capacity_frac_of_ADV_base"] = raw_cfg.get(
            "capacity_frac_of_ADV_base"
        )
    if "floor_base" in raw_cfg:
        payload["floor_base"] = raw_cfg.get("floor_base")

    raw_path = raw_cfg.get("adv_base_path")
    path_val: Optional[str] = None
    if raw_path is not None:
        try:
            candidate = str(raw_path).strip()
        except Exception:
            candidate = ""
        if candidate:
            path_val = candidate
    if path_val:
        payload["adv_base_path"] = path_val
    else:
        fallback_path = getattr(adv_store, "path", None) if adv_store is not None else None
        if fallback_path:
            payload["adv_base_path"] = fallback_path
            fallbacks.append(("adv_base_path", fallback_path))
        else:
            missing.append("adv_base_path")

    timeframe_candidate = raw_cfg.get("timeframe_ms")
    timeframe_val = _coerce_timeframe_ms(timeframe_candidate)
    used_fallback_timeframe = False
    if timeframe_val is None and default_timeframe_ms is not None:
        timeframe_val = _coerce_timeframe_ms(default_timeframe_ms)
        if timeframe_val is not None:
            used_fallback_timeframe = True
    if timeframe_val is not None:
        payload["timeframe_ms"] = timeframe_val
        if used_fallback_timeframe:
            fallbacks.append(("timeframe_ms", timeframe_val))
    else:
        missing.append("timeframe_ms")

    payload = {k: v for k, v in payload.items() if v is not None}
    if not payload:
        return None, fallbacks, missing
    return payload, fallbacks, missing


def _apply_bar_capacity_base_config(
    sim: ExecutionSimulator,
    raw_cfg: Optional[Dict[str, Any]],
    *,
    adv_store: Optional[ADVStore],
    default_timeframe_ms: Optional[int],
    context: str,
) -> None:
    set_cfg = getattr(sim, "set_bar_capacity_base_config", None)
    if not callable(set_cfg):
        if raw_cfg:
            logger.warning(
                "%s: ExecutionSimulator lacks set_bar_capacity_base_config(); bar capacity base disabled",
                context,
            )
        return

    payload, fallbacks, missing = _finalise_bar_capacity_payload(
        raw_cfg, adv_store=adv_store, default_timeframe_ms=default_timeframe_ms
    )
    if not payload:
        return

    try:
        set_cfg(**payload)
    except Exception:
        logger.exception("%s: failed to configure bar capacity base", context)
        return

    if fallbacks:
        for field, value in fallbacks:
            if field == "adv_base_path":
                logger.warning(
                    "%s: bar_capacity_base.%s not configured; falling back to ADV dataset %s",
                    context,
                    field,
                    value,
                )
            elif field == "timeframe_ms":
                logger.warning(
                    "%s: bar_capacity_base.%s not configured; falling back to timeframe %s",
                    context,
                    field,
                    value,
                )
            else:
                logger.warning(
                    "%s: bar_capacity_base.%s missing; using fallback %s",
                    context,
                    field,
                    value,
                )

    enabled_flag = raw_cfg.get("enabled") if raw_cfg else None
    should_warn_missing = bool(enabled_flag) if enabled_flag is not None else bool(raw_cfg)
    if should_warn_missing:
        for field in missing:
            logger.warning(
                "%s: bar_capacity_base.%s not configured and no fallback available",
                context,
                field,
            )


def _yield_bar_capacity_meta(report: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    collected: List[Mapping[str, Any]] = []
    if not isinstance(report, Mapping):
        return collected

    core_reports = report.get("core_exec_reports")
    if isinstance(core_reports, list) and core_reports:
        for entry in core_reports:
            if not isinstance(entry, Mapping):
                continue
            meta = entry.get("meta")
            if not isinstance(meta, Mapping):
                continue
            bc_meta = meta.get("bar_capacity_base")
            if isinstance(bc_meta, Mapping):
                collected.append(bc_meta)
        if collected:
            return collected

    trades = report.get("trades")
    if isinstance(trades, list):
        for entry in trades:
            if not isinstance(entry, Mapping):
                continue
            meta = entry.get("meta")
            if isinstance(meta, Mapping):
                bc_meta = meta.get("bar_capacity_base")
                if isinstance(bc_meta, Mapping):
                    collected.append(bc_meta)
                    continue
            reason = entry.get("capacity_reason")
            if isinstance(reason, str) and reason.upper() == "BAR_CAPACITY_BASE":
                collected.append(entry)
    return collected


def _configure_adv_runtime(
    sim: ExecutionSimulator,
    run_cfg: CommonRunConfig | None,
    *,
    context: str,
) -> tuple[Optional[ADVStore], Optional[Dict[str, Any]]]:
    bar_capacity_cfg = _extract_bar_capacity_base_cfg(run_cfg)
    if run_cfg is None:
        return None, bar_capacity_cfg
    adv_cfg = getattr(run_cfg, "adv", None)
    if adv_cfg is None or not getattr(adv_cfg, "enabled", False):
        return None, bar_capacity_cfg
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
        return existing_store, bar_capacity_cfg
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
        return None, bar_capacity_cfg
    _log_adv_runtime_warnings(store, getattr(sim, "symbol", None), adv_cfg, context)
    return store, bar_capacity_cfg


@dataclass
class BacktestConfig:
    symbol: str
    timeframe: str
    exchange_specs_path: Optional[str] = None
    dynamic_spread_config: Optional[Dict[str, Any]] = None
    guards_config: Optional[Dict[str, Any]] = None
    signal_cooldown_s: int = 0
    no_trade_config: Optional[Dict[str, Any]] = None
    snapshot_config_path: Optional[str] = None
    artifacts_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    run_id: Optional[str] = None
    timing_config: Optional[Dict[str, Any]] = None


class ServiceBacktest:
    """
    Сервис работает через BacktestAdapter, который использует SimAdapter.step.
    """

    class _EmptySource:
        """Заглушка источника данных для SimAdapter."""

        def stream_bars(
            self, symbols, interval_ms
        ):  # pragma: no cover - простая заглушка
            return iter(())

        def stream_ticks(self, symbols):  # pragma: no cover - простая заглушка
            return iter(())

    def __init__(
        self,
        policy: SignalPolicy,
        sim: ExecutionSimulator,
        cfg: BacktestConfig,
        *,
        run_config: CommonRunConfig | None = None,
    ) -> None:
        self.policy = policy
        self.sim = sim
        self.cfg = cfg
        self._adv_store, _bar_capacity_cfg = _configure_adv_runtime(
            sim, run_config, context="service_backtest"
        )
        self._run_config = (
            run_config
            or getattr(sim, "run_config", None)
            or getattr(sim, "_run_config", None)
        )

        self._fees_expected_payload: Optional[Dict[str, Any]] = None
        self._fees_metadata: Optional[Dict[str, Any]] = None
        self._fees_enabled: bool = True
        (
            self._fees_expected_payload,
            self._fees_metadata,
            self._fees_enabled,
        ) = self._resolve_fee_expected_info()
        self._log_fee_metadata_warnings()

        if SlippageImpl is not None:
            slip_attached = callable(getattr(self.sim, "_slippage_get_trade_cost", None))
            if not slip_attached:
                rc_slip_cfg = getattr(self._run_config, "slippage", None)
                slip_payload = _slippage_to_dict(rc_slip_cfg)
                if slip_payload:
                    try:
                        SlippageImpl.from_dict(
                            slip_payload, run_config=self._run_config
                        ).attach_to(self.sim)
                    except Exception:
                        logger.exception("Failed to attach slippage config to simulator")
        elif getattr(self._run_config, "slippage", None):
            logger.debug("SlippageImpl is unavailable; using simulator defaults")

        timeframe_ms: Optional[int] = None
        exec_cfg = getattr(self._run_config, "execution", None)
        if exec_cfg is not None:
            timeframe_ms = _coerce_timeframe_ms(getattr(exec_cfg, "timeframe_ms", None))
        if timeframe_ms is None:
            data_cfg = getattr(self._run_config, "data", None)
            tf_value = getattr(data_cfg, "timeframe", None) if data_cfg is not None else None
            if tf_value is None:
                tf_value = getattr(self.cfg, "timeframe", None)
            timeframe_ms = _coerce_timeframe_ms(tf_value)
        if timeframe_ms is not None:
            try:
                setattr(self.sim, "_execution_timeframe_ms", int(timeframe_ms))
            except Exception:
                pass

        _apply_bar_capacity_base_config(
            self.sim,
            _bar_capacity_cfg,
            adv_store=self._adv_store,
            default_timeframe_ms=timeframe_ms,
            context="service_backtest",
        )

        run_id = self.cfg.run_id or "sim"
        logs_dir = self.cfg.logs_dir or "logs"
        logging_config = {
            "trades_path": os.path.join(logs_dir, f"log_trades_{run_id}.csv"),
            "reports_path": os.path.join(logs_dir, f"report_equity_{run_id}.csv"),
        }
        try:  # переподключаем логгер симулятора с нужными путями
            from logging import LogWriter, LogConfig  # type: ignore

            self.sim._logger = LogWriter(
                LogConfig.from_dict(logging_config), run_id=run_id
            )
        except Exception:
            pass

        self.sim_bridge = SimAdapter(
            sim,
            symbol=self.cfg.symbol,
            timeframe=self.cfg.timeframe,
            source=self._EmptySource(),
            run_config=self._run_config,
        )

        dyn_spread_cfg = self.cfg.dynamic_spread_config
        if dyn_spread_cfg is None:
            dyn_spread_cfg = _extract_dynamic_slippage_cfg(self._run_config)
        if dyn_spread_cfg is not None:
            dyn_spread_cfg = dict(dyn_spread_cfg)

        self._bt = BacktestAdapter(
            policy=self.policy,
            sim_bridge=self.sim_bridge,
            dynamic_spread_config=dyn_spread_cfg,
            exchange_specs_path=self.cfg.exchange_specs_path,
            guards_config=self.cfg.guards_config,
            signal_cooldown_s=self.cfg.signal_cooldown_s,
            no_trade_config=self.cfg.no_trade_config,
            timing_config=self.cfg.timing_config,
        )

    def run(
        self,
        df: pd.DataFrame,
        *,
        ts_col: str = "ts_ms",
        symbol_col: str = "symbol",
        price_col: str = "ref_price",
    ) -> List[Dict[str, Any]]:
        if self.cfg.snapshot_config_path and self.cfg.artifacts_dir:
            snapshot_config(self.cfg.snapshot_config_path, self.cfg.artifacts_dir)
        reports = self._bt.run(
            df, ts_col=ts_col, symbol_col=symbol_col, price_col=price_col
        )

        expected_payload: Dict[str, Any] = (
            dict(self._fees_expected_payload)
            if isinstance(self._fees_expected_payload, Mapping)
            else {}
        )
        fees_enabled = bool(self._fees_enabled)
        share_expected_flag = False
        fee_expected_flag = False

        total_trades = 0
        total_fill_sum = 0.0
        total_fill_count = 0
        share_values: List[float] = []
        fee_values: List[float] = []
        spread_values: List[float] = []
        component_sums: Dict[str, float] = {}
        component_counts: Dict[str, int] = {}
        for rep in reports:
            if not isinstance(rep, dict):
                continue
            capacity_meta = _yield_bar_capacity_meta(rep)
            per_count = 0
            per_fill_sum = 0.0
            per_fill_count = 0
            for meta in capacity_meta:
                if not isinstance(meta, Mapping):
                    continue
                per_count += 1
                fill_raw = meta.get("fill_ratio")
                try:
                    fill_val = float(fill_raw) if fill_raw is not None else None
                except (TypeError, ValueError):
                    fill_val = None
                if fill_val is not None and math.isfinite(fill_val):
                    per_fill_sum += fill_val
                    per_fill_count += 1
            rep["bar_capacity_base_trade_count"] = per_count
            rep["bar_capacity_base_fill_ratio_avg"] = (
                per_fill_sum / per_fill_count if per_fill_count else None
            )
            total_trades += per_count
            total_fill_sum += per_fill_sum
            total_fill_count += per_fill_count

            maker_share_raw = rep.get("maker_share")
            fee_bps_raw = rep.get("expected_fee_bps")
            spread_bps_raw = rep.get("expected_spread_bps")
            try:
                maker_share_val = float(maker_share_raw)
            except (TypeError, ValueError):
                maker_share_val = None
            else:
                if math.isfinite(maker_share_val):
                    share_values.append(maker_share_val)
            try:
                fee_val = float(fee_bps_raw)
            except (TypeError, ValueError):
                fee_val = None
            else:
                if math.isfinite(fee_val):
                    fee_values.append(fee_val)
            try:
                spread_val = float(spread_bps_raw)
            except (TypeError, ValueError):
                spread_val = None
            else:
                if math.isfinite(spread_val):
                    spread_values.append(spread_val)

            components = rep.get("expected_cost_components")
            if isinstance(components, Mapping):
                for key, value in components.items():
                    try:
                        num = float(value)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(num):
                        continue
                    component_sums[key] = component_sums.get(key, 0.0) + num
                    component_counts[key] = component_counts.get(key, 0) + 1

        if fees_enabled and expected_payload:
            default_share = self._safe_float(expected_payload.get("maker_share"))
            if default_share is not None and not share_values:
                share_values.append(default_share)
                share_expected_flag = True
            default_fee = self._safe_float(
                expected_payload.get("expected_fee_bps")
            )
            if default_fee is None:
                default_fee = self._safe_float(
                    expected_payload.get("taker_fee_bps")
                )
            if default_fee is not None and not fee_values:
                fee_values.append(default_fee)
                fee_expected_flag = True
            if not component_sums:
                for key in (
                    "maker_fee_bps",
                    "taker_fee_bps",
                    "expected_fee_bps",
                ):
                    fallback_val = self._safe_float(expected_payload.get(key))
                    if fallback_val is None:
                        continue
                    component_sums[key] = float(fallback_val)
                    component_counts[key] = 1

        if total_trades:
            if total_fill_count:
                overall_avg = total_fill_sum / total_fill_count
                logger.info(
                    "%s: bar_capacity_base trades=%d fill_ratio_avg=%.6f (n=%d)",
                    "service_backtest",
                    total_trades,
                    overall_avg,
                    total_fill_count,
                )
            else:
                logger.info(
                    "%s: bar_capacity_base trades=%d (fill_ratio unavailable)",
                    "service_backtest",
                    total_trades,
                )
        share_count = len(share_values)
        fee_count = len(fee_values)
        def _avg(series: List[float]) -> Optional[float]:
            if not series:
                return None
            return sum(series) / len(series)

        maker_share_avg = _avg(share_values)
        fee_bps_avg = _avg(fee_values)
        spread_bps_avg = _avg(spread_values)
        component_avg: Dict[str, float] = {}
        for key, total in component_sums.items():
            count = component_counts.get(key, 0)
            if count > 0:
                component_avg[key] = total / count

        if (
            maker_share_avg is not None
            or fee_bps_avg is not None
            or spread_bps_avg is not None
            or component_avg
        ):
            def _fmt(value: Optional[float], expected: bool = False) -> str:
                if value is None:
                    return "None"
                suffix = " (expected)" if expected else ""
                return f"{value:.4f}{suffix}"

            comp_avg_repr = (
                "{"
                + ", ".join(
                    f"{k}={component_avg[k]:.4f}" for k in sorted(component_avg)
                )
                + "}"
                if component_avg
                else "{}"
            )
            comp_sum_repr = (
                "{"
                + ", ".join(f"{k}={component_sums[k]:.4f}" for k in sorted(component_sums))
                + "}"
                if component_sums
                else "{}"
            )
            logger.info(
                "%s: maker_share=%s fee_bps=%s spread_bps=%s cost_components_avg=%s cost_components_sum=%s",
                "service_backtest",
                _fmt(maker_share_avg, share_expected_flag and share_count == 1),
                _fmt(fee_bps_avg, fee_expected_flag and fee_count == 1),
                _fmt(spread_bps_avg),
                comp_avg_repr,
                comp_sum_repr,
            )

        summary: Optional[Mapping[str, int]] = None
        getter = getattr(self.sim, "get_filter_rejection_summary", None)
        if callable(getter):
            try:
                summary_candidate = getter()
            except Exception:
                summary_candidate = None
            if summary_candidate:
                summary = dict(summary_candidate)
        if summary:
            logger.info(
                "%s: filter_rejections=%s",
                "service_backtest",
                summary,
            )
            cleared = False
            clearer = getattr(self.sim, "clear_filter_rejection_summary", None)
            if callable(clearer):
                try:
                    clearer()
                except Exception:
                    pass
                else:
                    cleared = True
            if not cleared:
                counts_attr = getattr(self.sim, "_filter_rejection_counts", None)
                if hasattr(counts_attr, "clear"):
                    try:
                        counts_attr.clear()
                    except Exception:
                        pass

        try:
            if getattr(self.sim, "_logger", None):
                self.sim._logger.flush()
        except Exception:
            pass
        return reports

    def _resolve_fee_expected_info(
        self,
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
        expected_payload: Optional[Dict[str, Any]] = None
        metadata_payload: Optional[Dict[str, Any]] = None

        getter = getattr(self.sim, "_fees_get_expected_info", None)
        info_payload: Any = None
        if callable(getter):
            try:
                info_payload = getter()
            except Exception:
                logger.debug(
                    "service_backtest: failed to fetch fees expected info", exc_info=True
                )
        if isinstance(info_payload, Mapping):
            expected_candidate = info_payload.get("expected")
            if isinstance(expected_candidate, Mapping):
                expected_payload = {k: v for k, v in expected_candidate.items()}
            metadata_candidate = info_payload.get("metadata")
            if isinstance(metadata_candidate, Mapping):
                metadata_payload = {k: v for k, v in metadata_candidate.items()}

        if expected_payload is None:
            candidate = getattr(self.sim, "fees_expected_payload", None)
            if isinstance(candidate, Mapping):
                expected_payload = {k: v for k, v in candidate.items()}

        if metadata_payload is None:
            candidate = getattr(self.sim, "fees_metadata", None)
            if isinstance(candidate, Mapping):
                metadata_payload = {k: v for k, v in candidate.items()}

        enabled = True
        for source in (metadata_payload, getattr(self.sim, "fees_config_payload", None)):
            if not isinstance(source, Mapping):
                continue
            flag = source.get("enabled")
            if flag is None:
                continue
            try:
                enabled = bool(flag)
            except Exception:
                continue
            else:
                break

        if not enabled:
            expected_payload = None

        return expected_payload, metadata_payload, enabled

    def _log_fee_metadata_warnings(self) -> None:
        if not self._fees_enabled:
            return
        metadata = self._fees_metadata
        if not isinstance(metadata, Mapping):
            return
        table_meta = metadata.get("table")
        if not isinstance(table_meta, Mapping):
            return
        path = table_meta.get("path")
        refresh_days = self._safe_float(table_meta.get("refresh_days"))
        age_days = self._metadata_age_days(table_meta)
        stale_flag = bool(table_meta.get("stale"))
        error = table_meta.get("error")

        warnings: List[str] = []
        if isinstance(error, str) and error:
            warnings.append(f"error={error}")
        if refresh_days is not None and age_days is not None:
            if age_days > refresh_days:
                built_repr = table_meta.get("built_at")
                detail = f"age={age_days:.1f}d refresh_days={refresh_days}"
                if built_repr is not None:
                    detail += f" built_at={built_repr}"
                warnings.append(detail)
        elif stale_flag:
            detail = "metadata flagged as stale"
            built_repr = table_meta.get("built_at")
            if built_repr is not None:
                detail += f" built_at={built_repr}"
            if age_days is not None:
                detail += f" age={age_days:.1f}d"
            warnings.append(detail)

        if not warnings:
            return

        logger.warning(
            "%s: fees table %s appears stale (%s)",
            "service_backtest",
            path or "<unknown>",
            "; ".join(warnings),
        )

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(num):
            return None
        return float(num)

    @staticmethod
    def _parse_metadata_timestamp(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(num):
                return None
            return num
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                num = float(text)
            except (TypeError, ValueError):
                normalised = text
                if normalised.endswith("Z"):
                    normalised = normalised[:-1] + "+00:00"
                try:
                    dt = datetime.fromisoformat(normalised)
                except Exception:
                    return None
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            else:
                if not math.isfinite(num):
                    return None
                return num
        return None

    def _metadata_age_days(self, meta: Mapping[str, Any]) -> Optional[float]:
        age = self._safe_float(meta.get("age_days"))
        if age is not None:
            return max(age, 0.0)
        built_ts = self._parse_metadata_timestamp(meta.get("built_at"))
        if built_ts is None:
            return None
        return max((time.time() - built_ts) / 86400.0, 0.0)


def from_config(
    cfg: CommonRunConfig,
    *,
    snapshot_config_path: str | None = None,
) -> List[Dict[str, Any]]:
    """Run :class:`ServiceBacktest` using dependencies from ``cfg``."""

    params = cfg.components.backtest_engine.params or {}
    bt_kwargs = {k: v for k, v in params.items() if k in BacktestConfig.__annotations__}

    symbol = bt_kwargs.get("symbol") or (
        cfg.data.symbols[0]
        if getattr(getattr(cfg, "data", None), "symbols", [])
        else None
    )
    timeframe = bt_kwargs.get("timeframe") or getattr(
        getattr(cfg, "data", None), "timeframe", None
    )
    if not symbol or not timeframe:
        raise ValueError("Config must provide symbols and data.timeframe")

    svc_cfg = BacktestConfig(
        symbol=symbol,
        timeframe=timeframe,
        exchange_specs_path=bt_kwargs.get("exchange_specs_path"),
        dynamic_spread_config=bt_kwargs.get("dynamic_spread_config"),
        guards_config=bt_kwargs.get("guards_config"),
        signal_cooldown_s=bt_kwargs.get("signal_cooldown_s", 0),
        no_trade_config=bt_kwargs.get("no_trade_config")
        or getattr(cfg, "no_trade", None),
        snapshot_config_path=snapshot_config_path,
        artifacts_dir=cfg.artifacts_dir,
        logs_dir=bt_kwargs.get("logs_dir") or cfg.logs_dir,
        run_id=bt_kwargs.get("run_id") or cfg.run_id,
        timing_config=bt_kwargs.get("timing_config") or cfg.timing.dict(),
    )

    logging.getLogger(__name__).info("timing settings: %s", svc_cfg.timing_config)

    data_path = getattr(cfg.data, "prices_path", None)
    if data_path is None:
        md_params = cfg.components.market_data.params or {}
        paths = md_params.get("paths") or []
        data_path = paths[0] if paths else None
    if not data_path:
        raise ValueError("Data path must be specified in config")

    df = read_df(data_path)

    ts_col = params.get("ts_col", "ts_ms")
    sym_col = params.get("symbol_col", "symbol")
    price_col = params.get("price_col", "ref_price")

    exec_spec = cfg.components.executor
    if exec_spec and isinstance(exec_spec.params, dict):
        target = exec_spec.target or ""
        try:
            lat_cfg_dict = cfg.latency.dict(exclude_unset=False)
        except Exception:
            lat_cfg_dict = {}
        if lat_cfg_dict and "ExecutionSimulator" in target and not exec_spec.params.get("latency_config"):
            exec_spec.params["latency_config"] = dict(lat_cfg_dict)

    container = di_registry.build_graph(cfg.components, cfg)
    policy: SignalPolicy = container["policy"]
    executor_obj = container["executor"]
    sim: ExecutionSimulator
    if isinstance(executor_obj, ExecutionSimulator):
        sim = executor_obj
    else:
        candidate = getattr(executor_obj, "_sim", None)
        if isinstance(candidate, ExecutionSimulator):
            sim = candidate
        else:
            candidate = getattr(executor_obj, "sim", None)
            if isinstance(candidate, ExecutionSimulator):
                sim = candidate
            else:
                raise TypeError(
                    "Executor component must provide an ExecutionSimulator instance"
                )
    service = ServiceBacktest(policy, sim, svc_cfg, run_config=cfg)
    reports = service.run(df, ts_col=ts_col, symbol_col=sym_col, price_col=price_col)

    out_path = params.get("out_reports", "logs/sandbox_reports.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.lower().endswith(".parquet"):
        pd.DataFrame(reports).to_parquet(out_path, index=False)
    else:
        pd.DataFrame(reports).to_csv(out_path, index=False)
    print(f"Wrote {len(reports)} rows to {out_path}")

    return reports


__all__ = ["BacktestConfig", "ServiceBacktest", "from_config"]
