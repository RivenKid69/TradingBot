# -*- coding: utf-8 -*-
"""Service for evaluating strategy performance.

Includes equity metrics such as Sharpe, Sortino and Conditional Value at Risk (CVaR).

Example
-------
```python
from core_config import CommonRunConfig, ExecutionProfile
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
import math

import pandas as pd
from services.utils_config import snapshot_config

from services.metrics import calculate_metrics, read_any, plot_equity_curve
from core_config import CommonRunConfig, ExecutionProfile
import di_registry


@dataclass
class EvalConfig:
    """Configuration for :class:`ServiceEval`."""

    trades_path: str | Dict[str, str]
    reports_path: str | Dict[str, str]
    profile: Optional[str] = None
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

        def _ensure_equity_frame(data: Any) -> Any:
            if isinstance(data, dict):
                return {k: _ensure_equity_frame(v) for k, v in data.items()}
            if not isinstance(data, pd.DataFrame) or data.empty:
                return data
            if "equity" in data.columns:
                return data
            df = data.copy()
            if "equity_after_costs" in df.columns:
                df["equity"] = df["equity_after_costs"].astype(float)
            elif "bar_pnl" in df.columns:
                base = float(self.cfg.capital_base)
                cumulative = df["bar_pnl"].astype(float).cumsum()
                if math.isfinite(base) and base != 0.0:
                    df["equity"] = base + cumulative
                else:
                    df["equity"] = cumulative
            return df

        def _synth_trades_frame(source: Any) -> pd.DataFrame:
            if not isinstance(source, pd.DataFrame) or source.empty:
                return pd.DataFrame()
            if "pnl" in source.columns:
                return source
            if "bar_pnl" not in source.columns:
                return pd.DataFrame()
            payload: Dict[str, Any] = {}
            if "ts_ms" in source.columns:
                payload["ts_ms"] = source["ts_ms"]
            if "symbol" in source.columns:
                payload["symbol"] = source["symbol"]
            payload["pnl"] = source["bar_pnl"].astype(float)
            return pd.DataFrame(payload)

        reports = _ensure_equity_frame(reports)

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

        if isinstance(trades, dict):
            enriched: Dict[str, pd.DataFrame] = {}
            for name, tdf in trades.items():
                if isinstance(tdf, pd.DataFrame) and not tdf.empty and "pnl" in tdf.columns:
                    enriched[name] = tdf
                    continue
                report_source = reports.get(name) if isinstance(reports, dict) else reports
                synthetic = _synth_trades_frame(report_source)
                enriched[name] = synthetic if not synthetic.empty else tdf
            trades = enriched
        else:
            needs_synth = not isinstance(trades, pd.DataFrame) or trades.empty or "pnl" not in trades.columns
            if needs_synth:
                if isinstance(reports, dict):
                    first_report = next(iter(reports.values()), pd.DataFrame())
                else:
                    first_report = reports
                trades = _synth_trades_frame(first_report)

        def _filter_profile(data: Any, prof: str) -> Any:
            def _trim(df: Any) -> Any:
                if not isinstance(df, pd.DataFrame):
                    return df
                if "execution_profile" in df.columns:
                    mask = df["execution_profile"].astype(str) == prof
                    df = df.loc[mask].drop(columns=["execution_profile"])
                return df

            if isinstance(data, dict):
                if prof in data:
                    return _trim(data[prof])
                if data:
                    first = next(iter(data.values()))
                    return _trim(first)
                return pd.DataFrame()
            return _trim(data)

        if self.cfg.profile:
            trades = _filter_profile(trades, self.cfg.profile)
            reports = _filter_profile(reports, self.cfg.profile)

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
    """Run :class:`ServiceEval` using dependencies described in ``cfg``.

    If ``all_profiles`` is set, metrics are computed separately for each
    :class:`core_config.ExecutionProfile` and returned as a dictionary
    ``{"PROFILE": metrics_dict}``.
    """

    def _apply_profile(path: str, prof: str) -> str:
        if not path:
            return ""
        if "<profile>" in path:
            return path.replace("<profile>", prof)
        if "{profile}" in path:
            try:
                return path.format(profile=prof)
            except Exception:
                pass
        root, ext = os.path.splitext(path)
        candidate = f"{root}_{prof}{ext}"
        return candidate if os.path.exists(candidate) else path

    container = di_registry.build_graph(cfg.components, cfg)

    if all_profiles:
        results: Dict[str, Dict[str, float]] = {}
        base_trades = cfg.input.trades_path
        base_reports = getattr(cfg.input, "equity_path", "")
        for prof in ExecutionProfile:
            tp = _apply_profile(base_trades, prof.value) if isinstance(base_trades, str) else base_trades.get(prof.value, "")
            rp: str = ""
            if base_reports:
                if isinstance(base_reports, str):
                    rp = _apply_profile(base_reports, prof.value)
                else:
                    rp = base_reports.get(prof.value, "")
            eval_cfg = EvalConfig(
                trades_path=tp,
                reports_path=rp,
                profile=prof.value,
                out_json=_apply_profile(f"{cfg.logs_dir}/metrics.json", prof.value),
                out_md=_apply_profile(f"{cfg.logs_dir}/metrics.md", prof.value),
                equity_png=_apply_profile(f"{cfg.logs_dir}/equity.png", prof.value),
                capital_base=10_000.0,
                rf_annual=0.0,
                snapshot_config_path=snapshot_config_path,
                artifacts_dir=cfg.artifacts_dir,
            )
            service = ServiceEval(eval_cfg, container)
            results[prof.value] = service.run()
        return results

    trades_path = cfg.input.trades_path
    reports_path = getattr(cfg.input, "equity_path", "")

    effective_profile: Optional[str] = None
    if profile:
        effective_profile = profile.value if isinstance(profile, ExecutionProfile) else str(profile)
    elif not all_profiles:
        cfg_prof = getattr(cfg, "execution_profile", None)
        if isinstance(cfg_prof, ExecutionProfile):
            effective_profile = cfg_prof.value
        elif isinstance(cfg_prof, str) and cfg_prof:
            effective_profile = cfg_prof

    if effective_profile:
        trades_path = (
            trades_path[effective_profile]
            if isinstance(trades_path, dict)
            else _apply_profile(str(trades_path), effective_profile)
        )
        reports_path = (
            reports_path.get(effective_profile, "")
            if isinstance(reports_path, dict)
            else _apply_profile(str(reports_path), effective_profile)
        )
    elif isinstance(trades_path, dict):
        first = next(iter(trades_path))
        trades_path = trades_path[first]
        reports_path = reports_path[first] if isinstance(reports_path, dict) else ""

    eval_cfg = EvalConfig(
        trades_path=trades_path,
        reports_path=reports_path,
        profile=effective_profile,
        out_json=f"{cfg.logs_dir}/metrics.json",
        out_md=f"{cfg.logs_dir}/metrics.md",
        equity_png=f"{cfg.logs_dir}/equity.png",
        capital_base=10_000.0,
        rf_annual=0.0,
        snapshot_config_path=snapshot_config_path,
        artifacts_dir=cfg.artifacts_dir,
    )

    service = ServiceEval(eval_cfg, container)
    return service.run()


__all__ = ["EvalConfig", "ServiceEval", "from_config"]

