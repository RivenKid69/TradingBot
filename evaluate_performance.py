"""CLI wrapper around :mod:`service_eval` using YAML config."""

from __future__ import annotations

import argparse

from core_config import load_config
from service_eval import EvalConfig, EvalServiceConfig, from_config


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate strategy performance via ServiceEval",
    )
    p.add_argument(
        "--config",
        default="configs/config_eval.yaml",
        help="Путь к YAML-конфигу запуска",
    )
    args = p.parse_args()

    cfg = load_config(args.config)

    eval_cfg = EvalConfig(
        trades_path=cfg.input.trades_path,
        reports_path=getattr(cfg.input, "equity_path", ""),
        out_json=f"{cfg.logs_dir}/metrics.json",
        out_md=f"{cfg.logs_dir}/metrics.md",
        equity_png=f"{cfg.logs_dir}/equity.png",
        capital_base=10_000.0,
        rf_annual=0.0,
    )
    svc_cfg = EvalServiceConfig(snapshot_config_path=args.config, artifacts_dir=cfg.artifacts_dir)
    from_config(cfg, eval_cfg, svc_cfg=svc_cfg)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

