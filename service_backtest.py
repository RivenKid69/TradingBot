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
from typing import Any, Dict, List, Optional
import logging
import os
import pandas as pd

from execution_sim import ExecutionSimulator  # type: ignore
from sandbox.backtest_adapter import BacktestAdapter
from sandbox.sim_adapter import SimAdapter
from core_contracts import SignalPolicy
from services.utils_config import snapshot_config  # сохранение снапшота конфига
from services.utils_sandbox import read_df
from core_config import CommonRunConfig
import di_registry


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
        self._run_config = (
            run_config
            or getattr(sim, "run_config", None)
            or getattr(sim, "_run_config", None)
        )

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
        sim_spread_getter = getattr(self.sim, "get_spread_bps", None)
        if not callable(sim_spread_getter):
            sim_spread_getter = getattr(self.sim, "_slippage_get_spread", None)
        if callable(sim_spread_getter):
            dyn_spread_cfg = {}

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
        try:
            if getattr(self.sim, "_logger", None):
                self.sim._logger.flush()
        except Exception:
            pass
        return reports


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
