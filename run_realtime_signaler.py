"""Run realtime signaler using ServiceSignalRunner.

This script wires together a market data source, feature pipeline,
strategy and optional guards, then passes them into
:class:`ServiceSignalRunner`. It demonstrates configuring the service
purely via dependencies without relying on the legacy ``SignalRunner``
class.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any, Dict, List

import yaml

from impl_binance_public import BinancePublicBarSource
from execution_sim import ExecutionSimulator
from sandbox.sim_adapter import SimAdapter
from service_signal_runner import ServiceSignalRunner
from feature_pipe import FeatureConfig, FeaturePipe


class _HistoryGuard:
    """Blocks decisions until enough history bars accumulated."""

    def __init__(self, min_history_bars: int = 0) -> None:
        self._min = int(min_history_bars)
        self._cnt: Dict[str, int] = {}

    def apply(self, ts_ms: int, symbol: str, decisions):
        c = self._cnt.get(symbol, 0) + 1
        self._cnt[symbol] = c
        if c < self._min:
            return []
        return decisions


def main() -> None:
    p = argparse.ArgumentParser(description="Run realtime signaler (public Binance WS, no keys).")
    p.add_argument("--config", default="configs/config_live.yaml", help="Путь к YAML конфигу")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    symbols: List[str] = [str(s).upper() for s in cfg.get("symbols", [])]
    if not symbols:
        raise ValueError("At least one symbol must be provided")
    interval: str = str(cfg.get("interval", "1m"))

    strat_cfg = cfg.get("strategy", {}) or {}
    strat_mod = str(strat_cfg.get("module", "strategies.momentum"))
    strat_cls = str(strat_cfg.get("class", "MomentumStrategy"))
    strat_params = dict(strat_cfg.get("params", {}))
    strategy = getattr(importlib.import_module(strat_mod), strat_cls)(**strat_params)

    feats_cfg = cfg.get("features", {}) or {}
    fp_cfg = {
        "lookbacks_prices": list(feats_cfg.get("lookbacks_prices", [5, 15, 60])),
        "rsi_period": int(feats_cfg.get("rsi_period", 14)),
    }
    feature_pipe = FeaturePipe(FeatureConfig(**fp_cfg))

    guards_cfg = cfg.get("guards", {}) or {}
    guards = _HistoryGuard(int(guards_cfg.get("min_history_bars", 0)))

    source = BinancePublicBarSource(interval)
    sim = ExecutionSimulator(symbol=symbols[0])
    adapter = SimAdapter(sim, symbol=symbols[0], timeframe=interval, source=source)

    runner = ServiceSignalRunner(adapter, feature_pipe, strategy, guards)

    for report in runner.run():
        print(report)


if __name__ == "__main__":
    main()
