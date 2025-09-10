# -*- coding: utf-8 -*-
"""
core_config.py
Pydantic-модели конфигураций: sim/live/train/eval + декларация компонентов для DI.
Поддерживается dotted path формата "module.submodule:ClassName".
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Mapping, Union, Literal
from enum import Enum

import yaml
from pydantic import BaseModel, Field, validator


class ComponentSpec(BaseModel):
    """
    Описание компонента для DI. target — dotted path "module:Class",
    params — аргументы конструктора.
    """
    target: str = Field(..., description='Например: "impl_offline_data:OfflineBarSource"')
    params: Dict[str, Any] = Field(default_factory=dict)


class Components(BaseModel):
    """
    Карта используемых компонентов запуском.
    """
    market_data: ComponentSpec
    executor: ComponentSpec
    feature_pipe: ComponentSpec
    policy: ComponentSpec
    risk_guards: ComponentSpec
    backtest_engine: Optional[ComponentSpec] = None


class CommonRunConfig(BaseModel):
    run_id: Optional[str] = Field(default=None, description="Идентификатор запуска; если None — генерируется.")
    seed: Optional[int] = Field(default=None)
    logs_dir: str = Field(default="logs")
    artifacts_dir: str = Field(default="artifacts")
    timezone: Optional[str] = None
    liquidity_seasonality_path: Optional[str] = Field(default=None)
    liquidity_seasonality_hash: Optional[str] = Field(default=None)
    components: Components


class ExecutionProfile(str, Enum):
    """Подход к исполнению заявок."""

    MKT_OPEN_NEXT_H1 = "MKT_OPEN_NEXT_H1"
    VWAP_CURRENT_H1 = "VWAP_CURRENT_H1"
    LIMIT_MID_BPS = "LIMIT_MID_BPS"


class ExecutionParams(BaseModel):
    slippage_bps: float = 0.0
    limit_offset_bps: float = 0.0
    ttl_steps: int = 0
    tif: str = "GTC"


class SimulationDataConfig(BaseModel):
    symbols: List[str]
    timeframe: str = Field(..., description="Например: '1m', '5m'")
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None
    prices_path: Optional[str] = Field(default=None, description="Путь к parquet/csv с историческими данными.")


class SimulationConfig(CommonRunConfig):
    mode: str = Field(default="sim")
    market: Literal["spot", "futures"] = Field(default="spot")
    symbols: List[str] = Field(default_factory=list)
    quantizer: Dict[str, Any] = Field(default_factory=dict)
    fees: Dict[str, Any] = Field(default_factory=dict)
    slippage: Dict[str, Any] = Field(default_factory=dict)
    latency: Dict[str, Any] = Field(default_factory=dict)
    risk: Dict[str, Any] = Field(default_factory=dict)
    no_trade: Dict[str, Any] = Field(default_factory=dict)
    data: SimulationDataConfig
    limits: Dict[str, Any] = Field(default_factory=dict)
    execution_profile: ExecutionProfile = Field(default=ExecutionProfile.MKT_OPEN_NEXT_H1)
    execution_params: ExecutionParams = Field(default_factory=ExecutionParams)

    @validator("symbols", always=True)
    def validate_symbols(cls, v, values):
        if values.get("market") and not v:
            raise ValueError("symbols must be provided for the selected market")
        return v


class LiveAPIConfig(BaseModel):
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True
    ws_endpoint: Optional[str] = None
    rest_endpoint: Optional[str] = None


class LiveDataConfig(BaseModel):
    symbols: List[str]
    timeframe: str
    reconnect: bool = True
    heartbeat_ms: int = 10_000


class LiveConfig(CommonRunConfig):
    mode: str = Field(default="live")
    api: LiveAPIConfig
    data: LiveDataConfig
    limits: Dict[str, Any] = Field(default_factory=dict)
    fees: Dict[str, Any] = Field(default_factory=dict)
    slippage: Dict[str, Any] = Field(default_factory=dict)


class TrainDataConfig(BaseModel):
    symbols: List[str]
    timeframe: str
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None
    features_params: Dict[str, Any] = Field(default_factory=dict)
    target_params: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    algo: str = Field(..., description="Например: 'ppo', 'xgboost', 'lgbm'")
    params: Dict[str, Any] = Field(default_factory=dict)


class TrainConfig(CommonRunConfig):
    mode: str = Field(default="train")
    market: Literal["spot", "futures"] = Field(default="spot")
    symbols: List[str] = Field(default_factory=list)
    quantizer: Dict[str, Any] = Field(default_factory=dict)
    fees: Dict[str, Any] = Field(default_factory=dict)
    slippage: Dict[str, Any] = Field(default_factory=dict)
    latency: Dict[str, Any] = Field(default_factory=dict)
    risk: Dict[str, Any] = Field(default_factory=dict)
    no_trade: Dict[str, Any] = Field(default_factory=dict)
    data: TrainDataConfig
    model: ModelConfig
    execution_profile: ExecutionProfile = Field(default=ExecutionProfile.MKT_OPEN_NEXT_H1)
    execution_params: ExecutionParams = Field(default_factory=ExecutionParams)

    @validator("symbols", always=True)
    def validate_symbols(cls, v, values):
        if values.get("market") and not v:
            raise ValueError("symbols must be provided for the selected market")
        return v


class EvalInputConfig(BaseModel):
    trades_path: str
    equity_path: Optional[str] = None


class EvalConfig(CommonRunConfig):
    mode: str = Field(default="eval")
    input: EvalInputConfig
    metrics: List[str] = Field(default_factory=lambda: ["sharpe", "sortino", "mdd", "pnl"])
    execution_profile: ExecutionProfile = Field(default=ExecutionProfile.MKT_OPEN_NEXT_H1)
    execution_params: ExecutionParams = Field(default_factory=ExecutionParams)
    all_profiles: bool = Field(default=False)


def load_config(path: str) -> CommonRunConfig:
    """Загрузить конфигурацию запуска из YAML-файла."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    mode = data.get("mode")
    mapping = {
        "sim": SimulationConfig,
        "live": LiveConfig,
        "train": TrainConfig,
        "eval": EvalConfig,
    }
    cfg_cls = mapping.get(mode)
    if cfg_cls is None:
        raise ValueError(f"Unknown mode: {mode}")
    # parse_obj ensures all newly added optional fields are preserved
    return cfg_cls.parse_obj(data)


def load_config_from_str(content: str) -> CommonRunConfig:
    """Parse configuration from YAML string."""
    data = yaml.safe_load(content) or {}
    mode = data.get("mode")
    mapping = {
        "sim": SimulationConfig,
        "live": LiveConfig,
        "train": TrainConfig,
        "eval": EvalConfig,
    }
    cfg_cls = mapping.get(mode)
    if cfg_cls is None:
        raise ValueError(f"Unknown mode: {mode}")
    # parse_obj ensures all newly added optional fields are preserved
    return cfg_cls.parse_obj(data)


__all__ = [
    "ComponentSpec",
    "Components",
    "CommonRunConfig",
    "SimulationDataConfig",
    "SimulationConfig",
    "LiveAPIConfig",
    "LiveDataConfig",
    "LiveConfig",
    "TrainDataConfig",
    "ModelConfig",
    "TrainConfig",
    "EvalInputConfig",
    "EvalConfig",
    "ExecutionProfile",
    "ExecutionParams",
    "load_config",
    "load_config_from_str",
]
