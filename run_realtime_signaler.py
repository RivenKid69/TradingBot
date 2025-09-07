# scripts/run_realtime_signaler.py
from __future__ import annotations

import argparse
import asyncio
import importlib
import os
from typing import Any, Dict, List

import yaml

from binance_public import BinancePublicClient
from signal_runner import SignalRunner


def main():
    p = argparse.ArgumentParser(description="Run realtime signaler (public Binance WS, no keys).")
    p.add_argument("--config", default="configs/realtime.yaml", help="Путь к YAML конфигу")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    symbols: List[str] = [str(s).upper() for s in cfg.get("symbols", [])]
    interval: str = str(cfg.get("interval", "1m"))
    out_csv: str = str(cfg.get("out_csv", "logs/signals.csv"))
    min_gap: int = int(cfg.get("min_signal_gap_s", 300))
    backfill_on_gap: bool = bool(cfg.get("backfill_on_gap", True))
    market: str = str(cfg.get("market", "futures")).lower()

    strat = cfg.get("strategy", {})
    strat_mod = str(strat.get("module", "strategies.momentum"))
    strat_cls = str(strat.get("class", "MomentumStrategy"))
    strat_params = dict(strat.get("params", {}))

    feats = cfg.get("features", {})
    features_cfg = {
        "lookbacks_prices": list(feats.get("lookbacks_prices", [5, 15, 60])),
        "rsi_period": int(feats.get("rsi_period", 14)),
    }

    rest = BinancePublicClient()

    runner = SignalRunner(
        symbols=symbols,
        interval=interval,
        strategy_module=strat_mod,
        strategy_class=strat_cls,
        strategy_params=strat_params,
        features_cfg=features_cfg,
        out_csv=out_csv,
        min_signal_gap_s=min_gap,
        backfill_on_gap=backfill_on_gap,
        rest_client=rest,
    )

    # Запуск бесконечного цикла
    asyncio.run(runner.start())


if __name__ == "__main__":
    main()
