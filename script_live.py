"""Run realtime signaler using :mod:`service_signal_runner`."""

from __future__ import annotations

import argparse
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml
from pydantic import BaseModel

from services.universe import get_symbols
from core_config import StateConfig, load_config
from service_signal_runner import from_config
from runtime_trade_defaults import (
    DEFAULT_RUNTIME_TRADE_PATH,
    load_runtime_trade_defaults,
    merge_runtime_trade_defaults,
)

try:
    from box import Box  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Box = None  # type: ignore


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


def _merge_state_config(state_obj: Any, payload: Mapping[str, Any]) -> Any:
    if not payload:
        return state_obj
    if isinstance(state_obj, BaseModel):
        return state_obj.copy(update=payload)
    if state_obj is None:
        try:
            return StateConfig.parse_obj(payload)
        except Exception:
            return payload
    if Box is not None and isinstance(state_obj, Box):
        state_obj.update(payload)
        return state_obj
    if isinstance(state_obj, dict):
        state_obj.update(payload)
        return state_obj
    for key, value in payload.items():
        try:
            setattr(state_obj, key, value)
        except Exception:
            continue
    return state_obj


def _reset_state_files(state_obj: Any) -> None:
    path_value = getattr(state_obj, "path", None)
    if path_value:
        p = Path(path_value)
        with suppress(Exception):
            p.unlink()
        for backup in p.parent.glob(f"{p.name}.bak*"):
            with suppress(Exception):
                backup.unlink()
        plain_backup = p.with_name(p.name + ".bak")
        with suppress(Exception):
            if plain_backup.exists():
                plain_backup.unlink()
        derived_lock = p.with_suffix(p.suffix + ".lock")
        with suppress(Exception):
            if derived_lock.exists():
                derived_lock.unlink()
    lock_value = getattr(state_obj, "lock_path", None)
    if lock_value:
        lock_path = Path(lock_value)
        with suppress(Exception):
            if lock_path.exists():
                lock_path.unlink()


def _ensure_state_dir(state_obj: Any) -> None:
    target_dir = getattr(state_obj, "dir", None)
    if not target_dir:
        path_value = getattr(state_obj, "path", None)
        if path_value:
            target_dir = Path(path_value).parent
    if not target_dir:
        return
    Path(target_dir).mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run realtime signaler (public Binance WS, no keys).",
    )
    p.add_argument(
        "--config",
        default="configs/config_live.yaml",
        help="Путь к YAML-конфигу запуска",
    )
    p.add_argument(
        "--state-config",
        default="configs/state.yaml",
        help="Путь к YAML-конфигу состояния",
    )
    p.add_argument(
        "--reset-state",
        action="store_true",
        help="Удалить файлы состояния перед запуском",
    )
    p.add_argument(
        "--symbols",
        default="",
        help="Список символов через запятую; пусто = загрузить из universe",
    )
    runtime_group = p.add_argument_group("Runtime overrides")
    runtime_group.add_argument(
        "--execution-mode",
        choices=["order", "bar"],
        help="Override execution.mode (order/bar)",
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
        help="Override execution.safety_margin_bps used by the bar executor",
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
    p.add_argument(
        "--runtime-trade-config",
        default=DEFAULT_RUNTIME_TRADE_PATH,
        help="Путь к runtime_trade.yaml с дефолтами исполнения",
    )
    args = p.parse_args()
    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else get_symbols()
    )

    try:
        with open(args.state_config, "r", encoding="utf-8") as f:
            state_data_raw = yaml.safe_load(f) or {}
    except Exception:
        state_data_raw = {}
    state_data = state_data_raw if isinstance(state_data_raw, Mapping) else {}

    cfg = load_config(args.config)
    cfg.data.symbols = symbols
    try:
        cfg.components.executor.params["symbol"] = symbols[0]
    except Exception:
        pass
    if state_data:
        merged_state = _merge_state_config(cfg.state, state_data)
        if merged_state is not cfg.state:
            cfg.state = merged_state
    state_cfg = cfg.state

    if args.reset_state:
        _reset_state_files(state_cfg)

    if getattr(state_cfg, "enabled", False):
        _ensure_state_dir(state_cfg)

    cfg_dict = cfg.dict()
    runtime_trade_defaults = load_runtime_trade_defaults(args.runtime_trade_config)
    cfg_dict = merge_runtime_trade_defaults(cfg_dict, runtime_trade_defaults)
    cfg_dict = _apply_runtime_overrides(cfg_dict, args)
    cfg = cfg.__class__.parse_obj(cfg_dict)
    cfg.data.symbols = symbols
    try:
        cfg.components.executor.params["symbol"] = symbols[0]
    except Exception:
        pass

    for report in from_config(cfg, snapshot_config_path=args.config):
        print(report)


if __name__ == "__main__":
    main()
