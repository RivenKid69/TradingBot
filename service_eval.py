# -*- coding: utf-8 -*-
"""Service for evaluating strategy performance.

Example
-------
```python
from core_config import CommonRunConfig
from service_eval import from_config, EvalConfig

cfg = CommonRunConfig(...)
eval_cfg = EvalConfig(trades_path="trades.csv", reports_path="reports.csv")
metrics = from_config(cfg, eval_cfg)
```
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict

import pandas as pd

from services.metrics import calculate_metrics, read_any, plot_equity_curve
from core_config import CommonRunConfig
import di_registry


@dataclass
class EvalConfig:
    """Configuration for :class:`ServiceEval`."""

    trades_path: str
    reports_path: str
    out_json: str = "logs/metrics.json"
    out_md: str = "logs/metrics.md"
    equity_png: str = "logs/equity.png"
    capital_base: float = 10_000.0
    rf_annual: float = 0.0


class ServiceEval:
    """High-level service that reads logs, computes metrics and stores artefacts."""

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg

    def run(self) -> Dict[str, Dict[str, float]]:
        trades = read_any(self.cfg.trades_path)
        reports = read_any(self.cfg.reports_path)

        if set([
            "ts",
            "run_id",
            "symbol",
            "side",
            "order_type",
            "price",
            "quantity",
        ]).issubset(set(trades.columns)):
            trades = trades.rename(columns={"quantity": "qty"})
        if "side" in trades.columns:
            trades["side"] = trades["side"].astype(str).str.upper()

        metrics = calculate_metrics(
            trades,
            reports,
            capital_base=float(self.cfg.capital_base),
            rf_annual=float(self.cfg.rf_annual),
        )

        os.makedirs(os.path.dirname(self.cfg.out_json) or ".", exist_ok=True)
        with open(self.cfg.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        os.makedirs(os.path.dirname(self.cfg.out_md) or ".", exist_ok=True)
        with open(self.cfg.out_md, "w", encoding="utf-8") as f:
            f.write("# Performance Metrics\n\n")
            f.write("## Equity\n")
            for k, v in metrics["equity"].items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n## Trades\n")
            for k, v in metrics["trades"].items():
                f.write(f"- **{k}**: {v}\n")

        try:
            plot_equity_curve(reports, self.cfg.equity_png)
        except Exception:
            pass

        print(f"Wrote metrics JSON -> {self.cfg.out_json}")
        print(f"Wrote metrics MD   -> {self.cfg.out_md}")
        print(f"Wrote equity PNG   -> {self.cfg.equity_png}")

        return metrics


def from_config(cfg: CommonRunConfig, eval_cfg: EvalConfig) -> Dict[str, Dict[str, float]]:
    """Run :class:`ServiceEval` using dependencies described in ``cfg``."""
    di_registry.build_graph(cfg.components, cfg)
    service = ServiceEval(eval_cfg)
    return service.run()


__all__ = ["EvalConfig", "ServiceEval", "from_config"]

