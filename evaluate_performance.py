# scripts/evaluate_performance.py
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import compute_equity_metrics, compute_trade_metrics


def _read_any(path: str) -> pd.DataFrame:
    # поддержка glob
    if any(ch in path for ch in ["*", "?", "["]):
        parts: List[str] = glob.glob(path)
        dfs = []
        for p in parts:
            dfs.append(_read_any(p))
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _try_plot_equity(reports: pd.DataFrame, out_path: str) -> None:
    if reports.empty:
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    r = reports.sort_values("ts_ms")
    x = r["ts_ms"].values.astype("int64")
    y = r["equity"].values.astype(float)
    plt.figure(figsize=(12, 5))
    plt.plot(x, y)
    plt.xlabel("ts_ms")
    plt.ylabel("equity")
    plt.title("Equity curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _compute_turnover_from_trades(trades: pd.DataFrame, capital_base: float) -> float:
    if trades is None or trades.empty or capital_base <= 0:
        return float("nan")
    notional = trades["notional"].astype(float) if "notional" in trades.columns else None
    if notional is None:
        return float("nan")
    gross = float(notional.abs().sum())
    return float(gross / float(capital_base))


def main():
    p = argparse.ArgumentParser(description="Evaluate strategy performance from execution logs.")
    p.add_argument("--trades", required=True, help="Путь к трейдам (CSV/Parquet или glob)")
    p.add_argument("--reports", required=True, help="Путь к отчётам (CSV/Parquet или glob)")
    p.add_argument("--out-json", default="logs/metrics.json", help="Куда сохранить метрики в JSON")
    p.add_argument("--out-md", default="logs/metrics.md", help="Куда сохранить краткий Markdown-отчёт")
    p.add_argument("--equity-png", default="logs/equity.png", help="PNG с кривой капитала")
    p.add_argument("--capital-base", type=float, default=10000.0, help="Базовый капитал для нормировки доходностей")
    p.add_argument("--rf-annual", type=float, default=0.0, help="Годовая безрисковая ставка (например, 0.03)")
    args = p.parse_args()

    trades = _read_any(args.trades)
    reports = _read_any(args.reports)

    # Унификация колонок для нового формата трейдов (log_trades_*.csv): quantity -> qty
    if set(['ts','run_id','symbol','side','order_type','price','quantity']).issubset(set(trades.columns)):
        trades = trades.rename(columns={'quantity':'qty'})
    # Приводим side к верхнему регистру для консистентности
    if 'side' in trades.columns:
        trades['side'] = trades['side'].astype(str).str.upper()

    eqm = compute_equity_metrics(reports, capital_base=float(args.capital_base), rf_annual=float(args.rf_annual))
    trm = compute_trade_metrics(trades)

    # Дооценим turnover по трейдам и добавим в eqm.to_dict() как post-fix
    turnover = _compute_turnover_from_trades(trades, float(args.capital_base))
    eqd = eqm.to_dict()
    eqd["turnover"] = float(turnover)

    # Сохранить JSON
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"equity": eqd, "trades": trm.to_dict()}, f, ensure_ascii=False, indent=2)

    # Сохранить Markdown
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# Performance Metrics\n\n")
        f.write("## Equity\n")
        for k, v in eqd.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Trades\n")
        for k, v in trm.to_dict().items():
            f.write(f"- **{k}**: {v}\n")

    # График equity
    try:
        _try_plot_equity(reports, args.equity_png)
    except Exception:
        pass

    print(f"Wrote metrics JSON -> {args.out_json}")
    print(f"Wrote metrics MD   -> {args.out_md}")
    print(f"Wrote equity PNG   -> {args.equity_png}")


if __name__ == "__main__":
    main()
