"""CLI wrapper around :mod:`service_eval`."""

from __future__ import annotations

import argparse

from service_eval import EvalConfig, ServiceEval


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate strategy performance via ServiceEval",
    )
    p.add_argument("--trades", required=True, help="Path to trades (CSV/Parquet or glob)")
    p.add_argument("--reports", required=True, help="Path to reports (CSV/Parquet or glob)")
    p.add_argument("--out-json", default="logs/metrics.json", help="Where to save metrics JSON")
    p.add_argument("--out-md", default="logs/metrics.md", help="Where to save Markdown report")
    p.add_argument("--equity-png", default="logs/equity.png", help="PNG with equity curve")
    p.add_argument("--capital-base", type=float, default=10_000.0, help="Base capital for returns")
    p.add_argument("--rf-annual", type=float, default=0.0, help="Annual risk-free rate")
    args = p.parse_args()

    cfg = EvalConfig(
        trades_path=args.trades,
        reports_path=args.reports,
        out_json=args.out_json,
        out_md=args.out_md,
        equity_png=args.equity_png,
        capital_base=float(args.capital_base),
        rf_annual=float(args.rf_annual),
    )
    ServiceEval(cfg).run()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

