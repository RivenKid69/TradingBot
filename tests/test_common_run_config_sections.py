import textwrap
import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core_config import load_config_from_str, PortfolioConfig, SpotCostConfig


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
