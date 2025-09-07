# -*- coding: utf-8 -*-
"""
services/service_signal_runner.py
Онлайн-оркестратор: MarketDataSource -> FeaturePipe -> Strategy -> RiskGuards -> TradeExecutor.
Не содержит бизнес-логики, только склейка компонентов.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Iterator, Protocol

from sandbox.sim_adapter import SimAdapter, DecisionsProvider  # исп. как TradeExecutor-подобный мост
from market_data_port import Bar
from services.utils_config import snapshot_config  # снапшот конфига (Фаза 3)  # noqa: F401


class FeaturePipe(Protocol):
    def warmup(self) -> None: ...
    def on_bar(self, bar: Bar) -> Dict[str, Any]: ...


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
