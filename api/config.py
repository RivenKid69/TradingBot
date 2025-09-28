"""Structured configuration objects for the Trading environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from core_constants import PRICE_SCALE


@dataclass(slots=True)
class MarketConfig:
    """Market-specific configuration."""

    initial_balance: float = 1_000_000.0
    initial_price: float = 100.0
    initial_atr: float = 1.0
    price_scale: int = PRICE_SCALE


@dataclass(slots=True)
class ExecutionConfig:
    """Execution engine parameters."""

    taker_fee: float = 0.0005
    maker_fee: float = 0.0002
    slip_k: float = 0.0
    half_spread_bps: float = 0.0
    impact_coefficient: float = 0.0
    impact_exponent: float = 1.0
    adv_quote: float = 0.0


@dataclass(slots=True)
class RewardConfig:
    """Reward shaping parameters."""

    profit_close_bonus: float = 0.0
    loss_close_penalty: float = 0.0
    trade_frequency_penalty: float = 0.0
    turnover_penalty_coef: float = 0.0
    use_potential_shaping: bool = False
    use_legacy_log_reward: bool = False
    gamma: float = 0.99
    potential_shaping_coef: float = 1.0
    risk_aversion_variance: float = 0.0
    risk_aversion_drawdown: float = 0.0


@dataclass(slots=True)
class RiskConfig:
    """Risk management parameters."""

    bankruptcy_threshold: float = 0.0
    max_drawdown: float = 1.0
    use_dynamic_risk: bool = False
    risk_off_level: float = 25.0
    risk_on_level: float = 75.0
    max_position_risk_off: float = 0.25
    max_position_risk_on: float = 0.75
    use_atr_stop: bool = False
    use_trailing_stop: bool = False
    tp_atr_mult: float = 0.0
    terminate_on_sl_tp: bool = False
    fear_greed_value: float = 50.0


@dataclass(slots=True)
class MicroConfig:
    """Microstructure generator parameters."""

    events_per_step: int = 32
    p_limit_order: float = 0.6
    p_market_order: float = 0.3
    p_cancel_order: float = 0.1


@dataclass(slots=True)
class EnvConfig:
    """Aggregate environment configuration."""

    execution_mode: str = "FAST"
    market: MarketConfig = field(default_factory=MarketConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    micro: MicroConfig = field(default_factory=MicroConfig)

    DEFAULT_MODE: ClassVar[str] = "FAST"

    @classmethod
    def default(cls) -> "EnvConfig":
        """Return a default configuration instance."""
        return cls()
