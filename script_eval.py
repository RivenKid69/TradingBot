"""Evaluate strategy performance via :mod:`service_eval`."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from core_config import load_config
from service_eval import from_config


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate strategy performance via ServiceEval",
    )
    p.add_argument(
        "--config",
        default="configs/config_eval.yaml",
        help="Путь к YAML-конфигу запуска",
    )
    p.add_argument("--profile", help="Оценить конкретный профиль", default=None)
    p.add_argument(
        "--all-profiles",
        action="store_true",
        help="Оценить все профили из конфигурации",
    )
    p.add_argument(
        "--rc-historical-trades",
        help="Путь к историческому логу сделок для проверки реалистичности",
    )
    p.add_argument(
        "--rc-benchmark",
        help="Путь к эталонной кривой капитала",
    )
    p.add_argument(
        "--rc-equity",
        help="Путь к логу капитальной кривой симуляции; если не задан, строится из сделок",
    )
    p.add_argument(
        "--rc-thresholds",
        default="benchmarks/sim_kpi_thresholds.json",
        help="JSON с допустимыми диапазонами KPI",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    metrics = from_config(
        cfg,
        snapshot_config_path=args.config,
        profile=args.profile,
        all_profiles=args.all_profiles or getattr(cfg, "all_profiles", False),
    )
    print(metrics)

    if args.rc_historical_trades and args.rc_benchmark:
        trades_path = Path(cfg.input.trades_path)
        equity_path = Path(args.rc_equity) if args.rc_equity else getattr(cfg.input, "equity_path", None)
        cmd = [
            sys.executable,
            "scripts/sim_reality_check.py",
            "--trades",
            trades_path.as_posix(),
            "--historical-trades",
            args.rc_historical_trades,
            "--benchmark",
            args.rc_benchmark,
            "--kpi-thresholds",
            args.rc_thresholds,
        ]
        if equity_path:
            cmd.extend(["--equity", Path(equity_path).as_posix()])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        rc_json = trades_path.parent / "sim_reality_check.json"
        flags = {}
        try:
            with open(rc_json, "r", encoding="utf-8") as fh:
                flags = json.load(fh).get("flags", {})
        except Exception:
            pass
        if any(v == "нереалистично" for v in flags.values()):
            raise SystemExit("Reality check flagged 'нереалистично' KPIs")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

