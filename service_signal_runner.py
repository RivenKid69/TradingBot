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

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Iterator, Protocol
import os

from sandbox.sim_adapter import SimAdapter, DecisionsProvider  # исп. как TradeExecutor-подобный мост
from core_models import Bar
from core_contracts import FeaturePipe
from services.utils_config import snapshot_config  # снапшот конфига (Фаза 3)  # noqa: F401
from core_config import CommonRunConfig
import di_registry
from core_strategy import Strategy


class RiskGuards(Protocol):
    def apply(self, ts_ms: int, symbol: str, decisions: Sequence[Any]) -> Sequence[Any]: ...


@dataclass
class SignalRunnerConfig:
    snapshot_config_path: Optional[str] = None
    artifacts_dir: Optional[str] = None
    logs_dir: Optional[str] = None
    run_id: Optional[str] = None


# обратная совместимость
RunnerConfig = SignalRunnerConfig


class _Provider(DecisionsProvider):
    def __init__(self, fp: FeaturePipe, strat: Strategy, guards: Optional[RiskGuards] = None):
        self._fp = fp
        self._strat = strat
        self._guards = guards

    def on_bar(self, bar: Bar):
        feats = self._fp.on_bar(bar)
        self._strat.on_features({**feats, "ref_price": float(bar.close)})
        dec = list(self._strat.decide({"ts_ms": int(bar.ts), "symbol": bar.symbol, "ref_price": float(bar.close), "features": feats}) or [])
        if self._guards:
            dec = list(self._guards.apply(int(bar.ts), bar.symbol, dec) or [])
        return dec


class ServiceSignalRunner:
    def __init__(
        self,
        adapter: SimAdapter,
        feature_pipe: FeaturePipe,
        strategy: Strategy,
        risk_guards: Optional[RiskGuards] = None,
        cfg: Optional[SignalRunnerConfig] = None,
    ) -> None:
        self.adapter = adapter
        self.feature_pipe = feature_pipe
        self.strategy = strategy
        self.risk_guards = risk_guards
        self.cfg = cfg or SignalRunnerConfig()

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

                sim._logger = LogWriter(LogConfig.from_dict(logging_config), run_id=run_id)
            except Exception:
                pass

    def run(self) -> Iterator[Dict[str, Any]]:
        # снапшот конфига, если задан
        if self.cfg.snapshot_config_path and self.cfg.artifacts_dir:
            snapshot_config(self.cfg.snapshot_config_path, self.cfg.artifacts_dir)

        self.feature_pipe.warmup()
        provider = _Provider(self.feature_pipe, self.strategy, self.risk_guards)
        try:
            for rep in self.adapter.run_events(provider):
                yield rep
        finally:
            sim = getattr(self.adapter, "sim", None) or getattr(self.adapter, "_sim", None)
            try:
                if getattr(sim, "_logger", None):
                    sim._logger.flush()
            except Exception:
                pass


def from_config(cfg: CommonRunConfig, svc_cfg: SignalRunnerConfig | None = None) -> Iterator[Dict[str, Any]]:
    """Build dependencies from ``cfg`` and run :class:`ServiceSignalRunner`.

    Parameters
    ----------
    cfg:
        Runtime configuration describing component graph.
    svc_cfg:
        Optional additional service configuration.

    Returns
    -------
    Iterator[Dict[str, Any]]
        Stream of execution reports produced by the service.
    """
    svc_cfg = svc_cfg or SignalRunnerConfig()
    if svc_cfg.logs_dir is None:
        svc_cfg.logs_dir = cfg.logs_dir
    if svc_cfg.run_id is None:
        svc_cfg.run_id = cfg.run_id

    container = di_registry.build_graph(cfg.components, cfg)
    adapter: SimAdapter = container["executor"]
    fp: FeaturePipe = container["feature_pipe"]
    strat: Strategy = container["strategy"]
    guards: RiskGuards | None = container.get("risk_guards")
    service = ServiceSignalRunner(adapter, fp, strat, guards, svc_cfg)
    return service.run()


__all__ = ["SignalRunnerConfig", "RunnerConfig", "ServiceSignalRunner", "from_config"]
