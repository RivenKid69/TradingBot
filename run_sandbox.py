# scripts/run_sandbox.py
from __future__ import annotations

import argparse
import os

import pandas as pd
import yaml

from execution_sim import ExecutionSimulator  # type: ignore
from service_backtest import BacktestConfig, ServiceBacktest
from services.utils_sandbox import build_strategy, read_df


def main() -> None:
    p = argparse.ArgumentParser(description="Strategy sandbox runner")
    p.add_argument("--config", default="configs/sandbox.yaml", help="Путь к YAML-конфигу песочницы")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # симулятор
    with open(cfg["sim_config_path"], "r", encoding="utf-8") as f:
        sim_cfg = yaml.safe_load(f)

    sim = ExecutionSimulator(
        symbol=cfg.get("symbol", "BTCUSDT"),
        latency_steps=int(cfg.get("latency_steps", 0)),
        filters_path=sim_cfg["filters"]["path"],
        enforce_ppbs=sim_cfg["filters"]["enforce_percent_price_by_side"],
        strict_filters=sim_cfg["filters"]["strict"],
        fees_config=sim_cfg.get("fees", {}),
        funding_config=sim_cfg.get("funding", {}),
        slippage_config=sim_cfg.get("slippage", {}),
        execution_config=sim_cfg.get("execution", {}),
        latency_config=sim_cfg.get("latency", {}),
        pnl_config=sim_cfg.get("pnl", {}),
        risk_config=sim_cfg.get("risk", {}),
        logging_config=sim_cfg.get("logging", {}),
    )

    # стратегия
    strat = build_strategy(
        cfg["strategy"]["module"],
        cfg["strategy"]["class"],
        cfg["strategy"].get("params", {}),
    )

    # бэктест
    data_cfg = cfg["data"]
    df = read_df(data_cfg["path"])
    ts_col = data_cfg.get("ts_col", "ts_ms")
    sym_col = data_cfg.get("symbol_col", "symbol")
    price_col = data_cfg.get("price_col", "ref_price")

    bt_cfg = BacktestConfig(
        symbol=cfg.get("symbol", "BTCUSDT"),
        timeframe=data_cfg.get("timeframe", "1m"),
        dynamic_spread_config=cfg.get("dynamic_spread", {}),
        exchange_specs_path=cfg.get("exchange_specs_path", "data/exchange_specs.json"),
        guards_config=cfg.get("sim_guards", {}),
        signal_cooldown_s=int(cfg.get("min_signal_gap_s", 0)),
        no_trade_config=cfg.get("no_trade", {}),
    )

    svc = ServiceBacktest(strategy=strat, sim=sim, cfg=bt_cfg)
    reports = svc.run(df, ts_col=ts_col, symbol_col=sym_col, price_col=price_col)

    # сохранить
    out_path = cfg.get("out_reports", "logs/sandbox_reports.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.lower().endswith(".parquet"):
        pd.DataFrame(reports).to_parquet(out_path, index=False)
    else:
        pd.DataFrame(reports).to_csv(out_path, index=False)
    print(f"Wrote {len(reports)} rows to {out_path}")


if __name__ == "__main__":
    main()
