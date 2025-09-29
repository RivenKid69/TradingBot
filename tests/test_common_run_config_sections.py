import sys
import pathlib
import textwrap
from types import SimpleNamespace

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_config import load_config_from_str, PortfolioConfig, SpotCostConfig
from runtime_trade_defaults import (
    load_runtime_trade_defaults,
    merge_runtime_trade_defaults,
)
from script_live import _apply_runtime_overrides


def _wrap_payload(payload: str) -> str:
    base = textwrap.dedent(
        """
        mode: sim
        symbols: ["BTCUSDT"]
        components:
          market_data:
            target: "module:Cls"
            params: {}
          executor:
            target: "module:Cls"
            params: {}
          feature_pipe:
            target: "module:Cls"
            params: {}
          policy:
            target: "module:Cls"
            params: {}
          risk_guards:
            target: "module:Cls"
            params: {}
        data:
          symbols: ["BTCUSDT"]
          timeframe: "1m"
        """
    ).strip()
    payload_block = textwrap.dedent(payload).strip()
    return "\n".join([base, payload_block, ""]) if payload_block else f"{base}\n"


def test_common_run_config_syncs_top_level_sections():
    yaml_cfg = _wrap_payload(
        """
        portfolio:
          equity_usd: 500000.0
        costs:
          taker_fee_bps: 8.0
          half_spread_bps: 1.0
          impact:
            sqrt_coeff: 10.0
            linear_coeff: 3.0
        """
    )
    cfg = load_config_from_str(yaml_cfg)

    assert isinstance(cfg.portfolio, PortfolioConfig)
    assert cfg.portfolio.equity_usd == 500000.0
    assert isinstance(cfg.execution.portfolio, PortfolioConfig)
    assert cfg.execution.portfolio.equity_usd == 500000.0

    assert isinstance(cfg.costs, SpotCostConfig)
    assert cfg.costs.taker_fee_bps == 8.0
    assert cfg.costs.half_spread_bps == 1.0
    assert cfg.costs.impact.sqrt_coeff == 10.0
    assert cfg.execution.costs.impact.linear_coeff == 3.0


def test_common_run_config_reads_embedded_sections():
    yaml_cfg = _wrap_payload(
        """
        execution:
          mode: bar
          portfolio:
            equity_usd: 250000.0
          costs:
            taker_fee_bps: 4.5
            half_spread_bps: 0.8
            impact:
              sqrt_coeff: 6.0
              linear_coeff: 1.5
        """
    )
    cfg = load_config_from_str(yaml_cfg)

    assert cfg.execution.mode == "bar"
    assert isinstance(cfg.portfolio, PortfolioConfig)
    assert cfg.portfolio.equity_usd == 250000.0
    assert isinstance(cfg.costs, SpotCostConfig)
    assert cfg.costs.half_spread_bps == 0.8
    assert cfg.execution.costs.taker_fee_bps == 4.5
    assert cfg.costs.impact.sqrt_coeff == 6.0
    assert cfg.execution.costs.impact.linear_coeff == 1.5


def test_execution_runtime_config_serializes_new_fields():
    yaml_cfg = _wrap_payload(
        """
        execution:
          mode: bar
          safety_margin_bps: 7.5
          max_participation: 0.05
          costs:
            turnover_caps:
              per_symbol:
                bps: 250
                daily_usd: 2000.0
              portfolio:
                usd: 50000.0
                daily_bps: 150
        """
    )
    cfg = load_config_from_str(yaml_cfg)

    assert pytest.approx(cfg.execution.safety_margin_bps) == 7.5
    assert cfg.execution.max_participation is not None
    assert pytest.approx(cfg.execution.max_participation, rel=1e-9) == 0.05
    caps = cfg.execution.costs.turnover_caps
    assert caps.per_symbol is not None
    assert caps.per_symbol.bps == 250
    assert caps.per_symbol.daily_usd == pytest.approx(2000.0)
    assert caps.portfolio is not None
    assert caps.portfolio.usd == 50_000.0
    assert caps.portfolio.daily_bps == pytest.approx(150.0)

    dumped = cfg.execution.dict()
    assert dumped["safety_margin_bps"] == 7.5
    assert pytest.approx(dumped["max_participation"], rel=1e-9) == 0.05
    turnover_caps = dumped["costs"]["turnover_caps"]
    assert turnover_caps["per_symbol"]["bps"] == 250
    assert turnover_caps["per_symbol"]["daily_usd"] == 2000.0
    assert turnover_caps["portfolio"]["usd"] == 50_000.0
    assert turnover_caps["portfolio"]["daily_bps"] == 150


@pytest.mark.parametrize("cli_equity", [None, 2_500_000.0])
def test_runtime_trade_defaults_precedence(tmp_path, cli_equity):
    runtime_trade_path = tmp_path / "runtime_trade.yaml"
    runtime_trade_path.write_text(
        textwrap.dedent(
            """
            portfolio:
              equity_usd: 1_000_000.0
            """
        ).strip()
        + "\n"
    )

    cfg = load_config_from_str(_wrap_payload(""))
    cfg_dict = cfg.dict()

    defaults = load_runtime_trade_defaults(str(runtime_trade_path))
    merge_runtime_trade_defaults(cfg_dict, defaults)

    args = SimpleNamespace(
        execution_mode=None,
        execution_bar_price=None,
        execution_min_step=None,
        execution_safety_margin_bps=None,
        execution_max_participation=None,
        portfolio_equity_usd=cli_equity,
        costs_taker_fee_bps=None,
        costs_half_spread_bps=None,
        costs_impact_sqrt=None,
        costs_impact_linear=None,
        costs_turnover_cap_symbol_bps=None,
        costs_turnover_cap_symbol_usd=None,
        costs_turnover_cap_portfolio_bps=None,
        costs_turnover_cap_portfolio_usd=None,
        costs_turnover_cap_symbol_daily_bps=None,
        costs_turnover_cap_symbol_daily_usd=None,
        costs_turnover_cap_portfolio_daily_bps=None,
        costs_turnover_cap_portfolio_daily_usd=None,
    )

    merged = _apply_runtime_overrides(cfg_dict, args)
    parsed = cfg.__class__.parse_obj(merged)

    expected_equity = cli_equity if cli_equity is not None else 1_000_000.0
    assert parsed.portfolio.equity_usd == expected_equity
    assert parsed.execution.portfolio.equity_usd == expected_equity
