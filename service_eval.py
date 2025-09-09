# -*- coding: utf-8 -*-
"""Service for evaluating strategy performance.

Includes equity metrics such as Sharpe, Sortino and Conditional Value at Risk (CVaR).

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
from typing import Any, Dict, Optional

import pandas as pd
from services.utils_config import snapshot_config

from services.metrics import calculate_metrics, read_any, plot_equity_curve
from core_config import CommonRunConfig
import di_registry


@dataclass
class EvalConfig:
    """Configuration for :class:`ServiceEval`."""

    trades_path: str | Dict[str, str]
    reports_path: str | Dict[str, str]
    out_json: str = "logs/metrics.json"
    out_md: str = "logs/metrics.md"
    equity_png: str = "logs/equity.png"
    capital_base: float = 10_000.0
    rf_annual: float = 0.0
    snapshot_config_path: Optional[str] = None
    artifacts_dir: Optional[str] = None


class ServiceEval:
    """High-level service that reads logs, computes metrics and stores artefacts."""

    def __init__(self, cfg: EvalConfig, container: Optional[Dict[str, Any]] = None):
        self.cfg = cfg
        self.container = container or {}

    def run(self) -> Dict[str, Dict[str, float]]:
        if self.cfg.snapshot_config_path and self.cfg.artifacts_dir:
            snapshot_config(self.cfg.snapshot_config_path, self.cfg.artifacts_dir)

        def _read(path: str | Dict[str, str]) -> Any:
            if isinstance(path, dict):
                return {k: read_any(p) for k, p in path.items()}
            return read_any(path)

        trades = _read(self.cfg.trades_path)
        reports = _read(self.cfg.reports_path)

        def _normalize(tr: pd.DataFrame) -> pd.DataFrame:
            if set([
                "ts",
                "run_id",
                "symbol",
                "side",
                "order_type",
                "price",
                "quantity",
            ]).issubset(set(tr.columns)):
                tr = tr.rename(columns={"quantity": "qty"})
            if "side" in tr.columns:
                tr["side"] = tr["side"].astype(str).str.upper()
            return tr

        if isinstance(trades, dict):
            trades = {k: _normalize(v) for k, v in trades.items()}
        else:
            trades = _normalize(trades)

        metrics = calculate_metrics(
            trades,
            reports,
            capital_base=float(self.cfg.capital_base),
            rf_annual=float(self.cfg.rf_annual),
        )

        def _suffix(path: str, name: str) -> str:
            root, ext = os.path.splitext(path)
            return f"{root}_{name}{ext}"

        if isinstance(metrics, dict) and "equity" not in metrics:
            for name, data in metrics.items():
                out_json = _suffix(self.cfg.out_json, name)
                out_md = _suffix(self.cfg.out_md, name)
                eq_png = _suffix(self.cfg.equity_png, name)

                os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
                with open(out_md, "w", encoding="utf-8") as f:
                    f.write("# Performance Metrics\n\n")
                    f.write("## Equity\n")
                    for k, v in data["equity"].items():
                        f.write(f"- **{k}**: {v}\n")
                    f.write("\n## Trades\n")
                    for k, v in data["trades"].items():
                        f.write(f"- **{k}**: {v}\n")

                try:
                    rep = reports[name] if isinstance(reports, dict) else reports
                    plot_equity_curve(rep, eq_png)
                except Exception:
                    pass

                print(f"Wrote metrics JSON -> {out_json}")
                print(f"Wrote metrics MD   -> {out_md}")
                print(f"Wrote equity PNG   -> {eq_png}")
            return metrics

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


def from_config(
    cfg: CommonRunConfig,
    *,
    snapshot_config_path: str | None = None,
    profile: str | None = None,
    all_profiles: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Run :class:`ServiceEval` using dependencies described in ``cfg``."""

    trades_path = cfg.input.trades_path
    reports_path = getattr(cfg.input, "equity_path", "")
    if isinstance(trades_path, dict):
        if profile:
            trades_path = {profile: trades_path[profile]}
            reports_path = (
                {profile: reports_path[profile]}
                if isinstance(reports_path, dict)
                else {}
            )
        elif all_profiles:
            if not isinstance(reports_path, dict):
                reports_path = {k: "" for k in trades_path}
        else:
            first = next(iter(trades_path))
            trades_path = trades_path[first]
            reports_path = reports_path[first] if isinstance(reports_path, dict) else ""

    eval_cfg = EvalConfig(
        trades_path=trades_path,
        reports_path=reports_path,
        out_json=f"{cfg.logs_dir}/metrics.json",
        out_md=f"{cfg.logs_dir}/metrics.md",
        equity_png=f"{cfg.logs_dir}/equity.png",
        capital_base=10_000.0,
        rf_annual=0.0,
        snapshot_config_path=snapshot_config_path,
        artifacts_dir=cfg.artifacts_dir,
    )

    container = di_registry.build_graph(cfg.components, cfg)
    service = ServiceEval(eval_cfg, container)
    return service.run()


__all__ = ["EvalConfig", "ServiceEval", "from_config"]

