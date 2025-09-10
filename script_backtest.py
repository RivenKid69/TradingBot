"""Run backtest via :mod:`service_backtest`."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from core_config import load_config
from service_backtest import from_config


def main() -> None:
    p = argparse.ArgumentParser(description="Strategy backtest runner")
    p.add_argument(
        "--config",
        default="configs/config_sim.yaml",
        help="Путь к YAML-конфигу запуска",
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
        "--rc-thresholds",
        default="benchmarks/sim_kpi_thresholds.json",
        help="JSON с допустимыми диапазонами KPI",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    reports = from_config(cfg, snapshot_config_path=args.config)
    print(f"Produced {len(reports)} reports")

    if args.rc_historical_trades and args.rc_benchmark:
        run_id = cfg.run_id or "sim"
        logs_dir = Path(cfg.logs_dir)
        trades_path = logs_dir / f"log_trades_{run_id}.csv"
        equity_path = logs_dir / f"report_equity_{run_id}.csv"
        cmd = [
            sys.executable,
            "scripts/sim_reality_check.py",
            "--trades",
            trades_path.as_posix(),
            "--historical-trades",
            args.rc_historical_trades,
            "--benchmark",
            args.rc_benchmark,
            "--equity",
            equity_path.as_posix(),
            "--kpi-thresholds",
            args.rc_thresholds,
        ]
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


if __name__ == "__main__":
    main()
