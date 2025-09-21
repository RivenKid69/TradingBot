from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import yaml


class PolicyConfig(BaseModel):
    module: str = "strategies.momentum"
    class_name: str = Field("MomentumStrategy", alias="class")
    params: Dict[str, Any] = Field(default_factory=dict)


class DataConfig(BaseModel):
    path: str
    ts_col: str = "ts_ms"
    symbol_col: str = "symbol"
    price_col: str = "ref_price"


class SandboxConfig(BaseModel):
    mode: str = "backtest"
    symbol: str = "BTCUSDT"
    latency_steps: int = 0
    sim_config_path: str
    exchange_specs_path: Optional[str] = None
    sim_guards: Dict[str, Any] = Field(default_factory=dict)
    min_signal_gap_s: int = 0
    no_trade: Dict[str, Any] = Field(default_factory=dict)
    policy: PolicyConfig = PolicyConfig()
    data: DataConfig
    dynamic_spread: Dict[str, Any] = Field(default_factory=dict)
    out_reports: Optional[str] = None
    bar_report_path: Optional[str] = None


def load_config(path: str) -> SandboxConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return SandboxConfig(**data)


def load_config_from_str(content: str) -> SandboxConfig:
    data = yaml.safe_load(content) or {}
    return SandboxConfig(**data)


__all__ = ["SandboxConfig", "load_config", "load_config_from_str"]
