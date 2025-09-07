# scripts/run_sandbox.py
from __future__ import annotations

import argparse
import os

import pandas as pd

from core_config import load_config
from service_backtest import BacktestConfig, from_config
from services.utils_sandbox import read_df


def main() -> None:
    p = argparse.ArgumentParser(description="Strategy sandbox runner")
    p.add_argument(
        "--config",
        default="configs/config_sim.yaml",
        help="Путь к YAML-конфигу запуска",
    )
    args = p.parse_args()

    cfg = load_config(args.config)

    data_path = getattr(cfg.data, "prices_path", None)
    if data_path is None:
        md_params = cfg.components.market_data.params or {}
        paths = md_params.get("paths") or []
        data_path = paths[0] if paths else None
    if not data_path:
        raise ValueError("Data path must be specified in config")

    df = read_df(data_path)

    params = cfg.components.backtest_engine.params or {}
    bt_kwargs = {k: v for k, v in params.items() if k in BacktestConfig.__annotations__}
    svc_cfg = BacktestConfig(**bt_kwargs, snapshot_config_path=args.config, artifacts_dir=cfg.artifacts_dir)

    ts_col = params.get("ts_col", "ts_ms")
    sym_col = params.get("symbol_col", "symbol")
    price_col = params.get("price_col", "ref_price")

    reports = from_config(cfg, df, ts_col=ts_col, symbol_col=sym_col, price_col=price_col, svc_cfg=svc_cfg)

    out_path = params.get("out_reports", "logs/sandbox_reports.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.lower().endswith(".parquet"):
        pd.DataFrame(reports).to_parquet(out_path, index=False)
    else:
        pd.DataFrame(reports).to_csv(out_path, index=False)
    print(f"Wrote {len(reports)} rows to {out_path}")


if __name__ == "__main__":
    main()
