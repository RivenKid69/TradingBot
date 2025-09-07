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

from sandbox.sim_adapter import SimAdapter, DecisionsProvider  # исп. как TradeExecutor-подобный мост
from core_models import Bar
from core_contracts import FeaturePipe
from services.utils_config import snapshot_config  # снапшот конфига (Фаза 3)  # noqa: F401
from core_config import CommonRunConfig
import di_registry


class Strategy(Protocol):
    def on_features(self, feats: Dict[str, Any]) -> None: ...
    def decide(self, ctx: Dict[str, Any]) -> Sequence[Any]: ...


class RiskGuards(Protocol):
    def apply(self, ts_ms: int, symbol: str, decisions: Sequence[Any]) -> Sequence[Any]: ...


@dataclass
class RunnerConfig:
    snapshot_config_path: Optional[str] = None
    artifacts_dir: Optional[str] = None


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
    def __init__(self, adapter: SimAdapter, feature_pipe: FeaturePipe, strategy: Strategy, risk_guards: Optional[RiskGuards] = None, cfg: Optional[RunnerConfig] = None) -> None:
        self.adapter = adapter
        self.feature_pipe = feature_pipe
        self.strategy = strategy
        self.risk_guards = risk_guards
        self.cfg = cfg or RunnerConfig()

    def run(self) -> Iterator[Dict[str, Any]]:
        # снапшот конфига, если задан
        if self.cfg.snapshot_config_path and self.cfg.artifacts_dir:
            snapshot_config(self.cfg.snapshot_config_path, self.cfg.artifacts_dir)

        self.feature_pipe.warmup()
        provider = _Provider(self.feature_pipe, self.strategy, self.risk_guards)
        for rep in self.adapter.run_events(provider):
            yield rep


def from_config(cfg: CommonRunConfig, svc_cfg: RunnerConfig | None = None) -> Iterator[Dict[str, Any]]:
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
    container = di_registry.build_graph(cfg.components, cfg)
    adapter: SimAdapter = container["executor"]
    fp: FeaturePipe = container["feature_pipe"]
    strat: Strategy = container["strategy"]
    guards: RiskGuards | None = container.get("risk_guards")
    service = ServiceSignalRunner(adapter, fp, strat, guards, svc_cfg)
    return service.run()


__all__ = ["RunnerConfig", "ServiceSignalRunner", "from_config"]
