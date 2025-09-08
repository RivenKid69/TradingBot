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
import pandas as pd

from execution_sim import ExecutionSimulator  # type: ignore
from sandbox.backtest_adapter import BacktestAdapter
from sandbox.sim_adapter import SimAdapter
from core_strategy import Strategy
from services.utils_config import snapshot_config  # сохранение снапшота конфига
from core_config import CommonRunConfig
import di_registry


@dataclass
class BacktestConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "1m"
    exchange_specs_path: Optional[str] = None
    dynamic_spread_config: Optional[Dict[str, Any]] = None
    guards_config: Optional[Dict[str, Any]] = None
    signal_cooldown_s: int = 0
    no_trade_config: Optional[Dict[str, Any]] = None
    snapshot_config_path: Optional[str] = None
    artifacts_dir: Optional[str] = None


class ServiceBacktest:
    """
    Сервис работает через BacktestAdapter, который использует SimAdapter.step.
    """

    class _EmptySource:
        """Заглушка источника данных для SimAdapter."""

        def stream_bars(self, symbols, interval_ms):  # pragma: no cover - простая заглушка
            return iter(())

        def stream_ticks(self, symbols):  # pragma: no cover - простая заглушка
            return iter(())

    def __init__(self, strategy: Strategy, sim: ExecutionSimulator, cfg: Optional[BacktestConfig] = None) -> None:
        self.strategy = strategy
        self.sim = sim
        self.cfg = cfg or BacktestConfig()

        self.sim_bridge = SimAdapter(
            sim,
            symbol=self.cfg.symbol,
            timeframe=self.cfg.timeframe,
            source=self._EmptySource(),
        )

        self._bt = BacktestAdapter(
            strategy=self.strategy,
            sim_bridge=self.sim_bridge,
            dynamic_spread_config=self.cfg.dynamic_spread_config,
            exchange_specs_path=self.cfg.exchange_specs_path,
            guards_config=self.cfg.guards_config,
            signal_cooldown_s=self.cfg.signal_cooldown_s,
            no_trade_config=self.cfg.no_trade_config,
        )

    def run(self, df: pd.DataFrame, *, ts_col: str = "ts_ms", symbol_col: str = "symbol", price_col: str = "ref_price") -> List[Dict[str, Any]]:
        if self.cfg.snapshot_config_path and self.cfg.artifacts_dir:
            snapshot_config(self.cfg.snapshot_config_path, self.cfg.artifacts_dir)
        return self._bt.run(df, ts_col=ts_col, symbol_col=symbol_col, price_col=price_col)


def from_config(
    cfg: CommonRunConfig,
    df: pd.DataFrame,
    *,
    ts_col: str = "ts_ms",
    symbol_col: str = "symbol",
    price_col: str = "ref_price",
    svc_cfg: BacktestConfig | None = None,
) -> List[Dict[str, Any]]:
    """Run :class:`ServiceBacktest` using dependencies from ``cfg``.

    Parameters
    ----------
    cfg: CommonRunConfig
        Configuration describing components to build.
    df: pandas.DataFrame
        Input price data for backtesting.
    ts_col, symbol_col, price_col:
        Column names of timestamp, symbol and price respectively.
    svc_cfg: BacktestConfig, optional
        Additional service configuration.
    """
    container = di_registry.build_graph(cfg.components, cfg)
    strat: Strategy = container["strategy"]
    sim: ExecutionSimulator = container["executor"]  # type: ignore[assignment]
    service = ServiceBacktest(strat, sim, svc_cfg)
    return service.run(df, ts_col=ts_col, symbol_col=symbol_col, price_col=price_col)


__all__ = ["BacktestConfig", "ServiceBacktest", "from_config"]
