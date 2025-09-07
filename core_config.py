# -*- coding: utf-8 -*-
"""
core_config.py
Pydantic-модели конфигураций: sim/live/train/eval + декларация компонентов для DI.
Поддерживается dotted path формата "module.submodule:ClassName".
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Mapping, Union
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
    strategy: ComponentSpec
    risk_guards: ComponentSpec
    backtest_engine: Optional[ComponentSpec] = None


class CommonRunConfig(BaseModel):
    run_id: Optional[str] = Field(default=None, description="Идентификатор запуска; если None — генерируется.")
    seed: Optional[int] = Field(default=None)
    logs_dir: str = Field(default="logs")
    artifacts_dir: str = Field(default="artifacts")
    timezone: Optional[str] = None
    components: Components


class SimulationDataConfig(BaseModel):
    symbols: List[str]
    timeframe: str = Field(..., description="Например: '1m', '5m'")
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None
    prices_path: Optional[str] = Field(default=None, description="Путь к parquet/csv с историческими данными.")


class SimulationConfig(CommonRunConfig):
    mode: str = Field(default="sim")
    data: SimulationDataConfig
    fees: Dict[str, Any] = Field(default_factory=dict)
    latency: Dict[str, Any] = Field(default_factory=dict)
    slippage: Dict[str, Any] = Field(default_factory=dict)
    limits: Dict[str, Any] = Field(default_factory=dict)


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
    data: TrainDataConfig
    model: ModelConfig


class EvalInputConfig(BaseModel):
    trades_path: str
    equity_path: Optional[str] = None


class EvalConfig(CommonRunConfig):
    mode: str = Field(default="eval")
    input: EvalInputConfig
    metrics: List[str] = Field(default_factory=lambda: ["sharpe", "sortino", "mdd", "pnl"])
