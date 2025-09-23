"""Evaluate strategy performance via :mod:`service_eval`."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from core_config import load_config
from service_eval import from_config
from scripts.offline_utils import resolve_split_bundle


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
    p.add_argument(
        "--offline-config",
        default="configs/offline.yaml",
        help="Path to offline dataset configuration",
    )
    p.add_argument(
        "--dataset-split",
        default="test",
        help="Dataset split identifier (use 'none' to disable offline bundle integration)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    split_key = (args.dataset_split or "").strip()
    seasonality_path: str | None = None
    fees_path: str | None = None
    adv_path: str | None = None
    seasonality_hash: str | None = None
    if split_key and split_key.lower() not in {"none", "null"}:
        try:
            offline_bundle = resolve_split_bundle(args.offline_config, split_key)
        except FileNotFoundError as exc:
            raise SystemExit(f"Offline config not found: {args.offline_config}") from exc
        except KeyError as exc:
            raise SystemExit(
                f"Dataset split '{split_key}' not found in offline config {args.offline_config}"
            ) from exc
        except ValueError as exc:
            raise SystemExit(f"Failed to resolve offline split '{split_key}': {exc}") from exc
        if offline_bundle.version:
            print(
                f"Resolved offline dataset split '{offline_bundle.name}' version {offline_bundle.version}"
            )
        else:
            print(f"Resolved offline dataset split '{offline_bundle.name}'")
        seasonality_art = offline_bundle.artifacts.get("seasonality")
        if seasonality_art:
            seasonality_path = seasonality_art.path.as_posix()
            raw_hash = seasonality_art.info.artifact.get("verification_hash")
            if raw_hash:
                seasonality_hash = str(raw_hash)
        fees_art = offline_bundle.artifacts.get("fees")
        if fees_art:
            fees_path = fees_art.path.as_posix()
        adv_art = offline_bundle.artifacts.get("adv")
        if adv_art:
            adv_path = adv_art.path.as_posix()

    cfg_dict = cfg.dict()
    if seasonality_path:
        cfg_dict["liquidity_seasonality_path"] = seasonality_path
        cfg_dict["latency_seasonality_path"] = seasonality_path
        latency_block = cfg_dict.setdefault("latency", {})
        if not latency_block.get("latency_seasonality_path"):
            latency_block["latency_seasonality_path"] = seasonality_path
        if seasonality_hash:
            cfg_dict["liquidity_seasonality_hash"] = seasonality_hash
    if fees_path:
        fees_block = cfg_dict.setdefault("fees", {})
        if not fees_block.get("path"):
            fees_block["path"] = fees_path
    if adv_path:
        adv_block = cfg_dict.setdefault("adv", {})
        if not adv_block.get("path"):
            adv_block["path"] = adv_path
        execution_block = cfg_dict.setdefault("execution", {})
        if isinstance(execution_block, dict):
            bar_block = execution_block.setdefault("bar_capacity_base", {})
            if not bar_block.get("adv_base_path"):
                bar_block["adv_base_path"] = adv_path
    cfg = cfg.__class__.parse_obj(cfg_dict)

    if seasonality_path and not Path(seasonality_path).exists():
        raise FileNotFoundError(
            f"Liquidity seasonality file not found: {seasonality_path}. Run offline builders first."
        )

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

