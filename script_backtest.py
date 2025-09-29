"""Run backtest via :mod:`service_backtest`."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from core_config import load_config
from service_backtest import from_config
from scripts.offline_utils import resolve_split_bundle


logger = logging.getLogger(__name__)


def _apply_runtime_overrides(
    cfg_dict: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """Apply CLI-provided runtime overrides to a config mapping."""

    def _require_non_negative(value: float, label: str) -> float:
        if value < 0:
            raise SystemExit(f"{label} must be non-negative")
        return float(value)

    exec_block = dict(cfg_dict.get("execution") or {})
    exec_changed = False

    if args.execution_mode:
        exec_block["mode"] = str(args.execution_mode).strip().lower()
        exec_changed = True

    if args.execution_bar_price is not None:
        bar_price = str(args.execution_bar_price or "").strip()
        if bar_price:
            exec_block["bar_price"] = bar_price
        else:
            exec_block.pop("bar_price", None)
        exec_changed = True

    if args.execution_min_step is not None:
        exec_block["min_rebalance_step"] = _require_non_negative(
            args.execution_min_step, "execution-min-step"
        )
        exec_changed = True

    if args.execution_safety_margin_bps is not None:
        exec_block["safety_margin_bps"] = _require_non_negative(
            args.execution_safety_margin_bps, "execution-safety-margin-bps"
        )
        exec_changed = True

    if args.execution_max_participation is not None:
        exec_block["max_participation"] = _require_non_negative(
            args.execution_max_participation, "execution-max-participation"
        )
        exec_changed = True

    if args.portfolio_equity_usd is not None:
        equity = _require_non_negative(args.portfolio_equity_usd, "portfolio-equity-usd")
        portfolio_block = dict(cfg_dict.get("portfolio") or {})
        portfolio_block["equity_usd"] = equity
        cfg_dict["portfolio"] = portfolio_block
        exec_portfolio = dict(exec_block.get("portfolio") or {})
        exec_portfolio["equity_usd"] = equity
        exec_block["portfolio"] = exec_portfolio
        exec_changed = True

    if any(
        value is not None
        for value in (
            args.costs_taker_fee_bps,
            args.costs_half_spread_bps,
            args.costs_impact_sqrt,
            args.costs_impact_linear,
            args.costs_turnover_cap_symbol_bps,
            args.costs_turnover_cap_symbol_usd,
            args.costs_turnover_cap_portfolio_bps,
            args.costs_turnover_cap_portfolio_usd,
            args.costs_turnover_cap_symbol_daily_bps,
            args.costs_turnover_cap_symbol_daily_usd,
            args.costs_turnover_cap_portfolio_daily_bps,
            args.costs_turnover_cap_portfolio_daily_usd,
        )
    ):
        costs_block = dict(cfg_dict.get("costs") or {})
        exec_costs = dict(exec_block.get("costs") or {})
        impact_block = dict(costs_block.get("impact") or {})
        exec_impact = dict(exec_costs.get("impact") or {})
        turnover_caps_block = dict(costs_block.get("turnover_caps") or {})
        exec_turnover_caps = dict(exec_costs.get("turnover_caps") or {})
        symbol_caps_block = dict(turnover_caps_block.get("per_symbol") or {})
        exec_symbol_caps_block = dict(exec_turnover_caps.get("per_symbol") or {})
        portfolio_caps_block = dict(turnover_caps_block.get("portfolio") or {})
        exec_portfolio_caps_block = dict(exec_turnover_caps.get("portfolio") or {})

        if args.costs_taker_fee_bps is not None:
            fee = _require_non_negative(args.costs_taker_fee_bps, "costs-taker-fee-bps")
            costs_block["taker_fee_bps"] = fee
            exec_costs["taker_fee_bps"] = fee

        if args.costs_half_spread_bps is not None:
            half = _require_non_negative(args.costs_half_spread_bps, "costs-half-spread-bps")
            costs_block["half_spread_bps"] = half
            exec_costs["half_spread_bps"] = half

        if args.costs_impact_sqrt is not None:
            sqrt_coeff = _require_non_negative(args.costs_impact_sqrt, "costs-impact-sqrt")
            impact_block["sqrt_coeff"] = sqrt_coeff
            exec_impact["sqrt_coeff"] = sqrt_coeff

        if args.costs_impact_linear is not None:
            linear_coeff = _require_non_negative(
                args.costs_impact_linear, "costs-impact-linear"
            )
            impact_block["linear_coeff"] = linear_coeff
            exec_impact["linear_coeff"] = linear_coeff

        if args.costs_turnover_cap_symbol_bps is not None:
            symbol_caps_block["bps"] = _require_non_negative(
                args.costs_turnover_cap_symbol_bps, "costs-turnover-cap-symbol-bps"
            )
            exec_symbol_caps_block["bps"] = symbol_caps_block["bps"]

        if args.costs_turnover_cap_symbol_usd is not None:
            symbol_caps_block["usd"] = _require_non_negative(
                args.costs_turnover_cap_symbol_usd, "costs-turnover-cap-symbol-usd"
            )
            exec_symbol_caps_block["usd"] = symbol_caps_block["usd"]

        if args.costs_turnover_cap_symbol_daily_bps is not None:
            symbol_caps_block["daily_bps"] = _require_non_negative(
                args.costs_turnover_cap_symbol_daily_bps,
                "costs-turnover-cap-symbol-daily-bps",
            )
            exec_symbol_caps_block["daily_bps"] = symbol_caps_block["daily_bps"]

        if args.costs_turnover_cap_symbol_daily_usd is not None:
            symbol_caps_block["daily_usd"] = _require_non_negative(
                args.costs_turnover_cap_symbol_daily_usd,
                "costs-turnover-cap-symbol-daily-usd",
            )
            exec_symbol_caps_block["daily_usd"] = symbol_caps_block["daily_usd"]

        if args.costs_turnover_cap_portfolio_bps is not None:
            portfolio_caps_block["bps"] = _require_non_negative(
                args.costs_turnover_cap_portfolio_bps,
                "costs-turnover-cap-portfolio-bps",
            )
            exec_portfolio_caps_block["bps"] = portfolio_caps_block["bps"]

        if args.costs_turnover_cap_portfolio_usd is not None:
            portfolio_caps_block["usd"] = _require_non_negative(
                args.costs_turnover_cap_portfolio_usd,
                "costs-turnover-cap-portfolio-usd",
            )
            exec_portfolio_caps_block["usd"] = portfolio_caps_block["usd"]

        if args.costs_turnover_cap_portfolio_daily_bps is not None:
            portfolio_caps_block["daily_bps"] = _require_non_negative(
                args.costs_turnover_cap_portfolio_daily_bps,
                "costs-turnover-cap-portfolio-daily-bps",
            )
            exec_portfolio_caps_block["daily_bps"] = portfolio_caps_block["daily_bps"]

        if args.costs_turnover_cap_portfolio_daily_usd is not None:
            portfolio_caps_block["daily_usd"] = _require_non_negative(
                args.costs_turnover_cap_portfolio_daily_usd,
                "costs-turnover-cap-portfolio-daily-usd",
            )
            exec_portfolio_caps_block["daily_usd"] = portfolio_caps_block["daily_usd"]

        if impact_block:
            costs_block["impact"] = impact_block
        else:
            costs_block.pop("impact", None)

        if exec_impact:
            exec_costs["impact"] = exec_impact
        else:
            exec_costs.pop("impact", None)

        if symbol_caps_block:
            turnover_caps_block["per_symbol"] = symbol_caps_block
            exec_turnover_caps["per_symbol"] = exec_symbol_caps_block
        else:
            turnover_caps_block.pop("per_symbol", None)
            exec_turnover_caps.pop("per_symbol", None)

        if portfolio_caps_block:
            turnover_caps_block["portfolio"] = portfolio_caps_block
            exec_turnover_caps["portfolio"] = exec_portfolio_caps_block
        else:
            turnover_caps_block.pop("portfolio", None)
            exec_turnover_caps.pop("portfolio", None)

        if turnover_caps_block:
            costs_block["turnover_caps"] = turnover_caps_block
        else:
            costs_block.pop("turnover_caps", None)

        if exec_turnover_caps:
            exec_costs["turnover_caps"] = exec_turnover_caps
        else:
            exec_costs.pop("turnover_caps", None)

        cfg_dict["costs"] = costs_block
        if exec_costs:
            exec_block["costs"] = exec_costs
        else:
            exec_block.pop("costs", None)
        exec_changed = True

    if exec_changed:
        cfg_dict["execution"] = exec_block

    return cfg_dict


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
    p.add_argument(
        "--offline-config",
        default="configs/offline.yaml",
        help="Path to offline dataset configuration",
    )
    p.add_argument(
        "--dataset-split",
        default="val",
        help="Dataset split identifier (use 'none' to disable offline bundle integration)",
    )
    runtime_group = p.add_argument_group("Runtime overrides")
    runtime_group.add_argument(
        "--execution-mode",
        choices=["order", "bar"],
        help="Override execution.mode for the run (order/bar mode switch)",
    )
    runtime_group.add_argument(
        "--execution-bar-price",
        help="Override execution.bar_price (empty string to clear)",
    )
    runtime_group.add_argument(
        "--execution-min-step",
        type=float,
        help="Override execution.min_rebalance_step (fraction, >=0)",
    )
    runtime_group.add_argument(
        "--execution-safety-margin-bps",
        type=float,
        help="Override execution.safety_margin_bps applied by the bar executor",
    )
    runtime_group.add_argument(
        "--execution-max-participation",
        type=float,
        help="Override execution.max_participation (fraction of ADV, >=0)",
    )
    runtime_group.add_argument(
        "--portfolio-equity-usd",
        type=float,
        help="Override portfolio.equity_usd assumption (>=0)",
    )
    runtime_group.add_argument(
        "--costs-taker-fee-bps",
        type=float,
        help="Override costs.taker_fee_bps (>=0)",
    )
    runtime_group.add_argument(
        "--costs-half-spread-bps",
        type=float,
        help="Override costs.half_spread_bps (>=0)",
    )
    runtime_group.add_argument(
        "--costs-impact-sqrt",
        type=float,
        help="Override costs.impact.sqrt_coeff (>=0)",
    )
    runtime_group.add_argument(
        "--costs-impact-linear",
        type=float,
        help="Override costs.impact.linear_coeff (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-symbol-bps",
        type=float,
        help="Override costs.turnover_caps.per_symbol.bps (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-symbol-usd",
        type=float,
        help="Override costs.turnover_caps.per_symbol.usd (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-symbol-daily-bps",
        type=float,
        help="Override costs.turnover_caps.per_symbol.daily_bps (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-symbol-daily-usd",
        type=float,
        help="Override costs.turnover_caps.per_symbol.daily_usd (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-portfolio-bps",
        type=float,
        help="Override costs.turnover_caps.portfolio.bps (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-portfolio-usd",
        type=float,
        help="Override costs.turnover_caps.portfolio.usd (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-portfolio-daily-bps",
        type=float,
        help="Override costs.turnover_caps.portfolio.daily_bps (>=0)",
    )
    runtime_group.add_argument(
        "--costs-turnover-cap-portfolio-daily-usd",
        type=float,
        help="Override costs.turnover_caps.portfolio.daily_usd (>=0)",
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
    cfg_dict = _apply_runtime_overrides(cfg_dict, args)
    cfg = cfg.__class__.parse_obj(cfg_dict)

    if seasonality_path and not Path(seasonality_path).exists():
        raise FileNotFoundError(
            f"Liquidity seasonality file not found: {seasonality_path}. Run offline builders first."
        )

    # Trades are filled against the current bar and their PnL shows up when the
    # adapter advances to the subsequent bar, matching :class:`BarBacktestSimBridge`.
    logger.debug(
        "Running backtest: trades fill on the current bar and PnL applies on the next bar",
    )
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
