# -*- coding: utf-8 -*-
"""
core_config.py
Pydantic-модели конфигураций: sim/live/train/eval + декларация компонентов для DI.
Поддерживается dotted path формата "module.submodule:ClassName".
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Mapping, Union, Literal
from enum import Enum
from dataclasses import dataclass, field

import yaml
from pydantic import BaseModel, Field, validator
import logging


class ComponentSpec(BaseModel):
    """
    Описание компонента для DI. target — dotted path "module:Class",
    params — аргументы конструктора.
    """

    target: str = Field(
        ..., description='Например: "impl_offline_data:OfflineBarSource"'
    )
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


class ClockSyncConfig(BaseModel):
    """Настройки синхронизации часов между процессами."""

    refresh_sec: float = Field(
        default=60.0, description="How often to refresh clock sync in seconds"
    )
    warn_threshold_ms: float = Field(
        default=500.0, description="Log warning if drift exceeds this many ms"
    )
    kill_threshold_ms: float = Field(
        default=2000.0, description="Enter safe mode if drift exceeds this many ms"
    )
    attempts: int = Field(default=5, description="Number of samples per sync attempt")
    ema_alpha: float = Field(
        default=0.1, description="EMA coefficient for skew updates"
    )
    max_step_ms: float = Field(
        default=1000.0, description="Maximum skew adjustment per sync in ms"
    )


class TimingConfig(BaseModel):
    """Настройки тайминга обработки баров и задержек закрытия."""

    enforce_closed_bars: bool = Field(default=True)
    timeframe_ms: int = Field(default=60_000)
    close_lag_ms: int = Field(default=2000)


class WSDedupConfig(BaseModel):
    """Конфигурация дедупликации данных вебсокета."""

    enabled: bool = Field(default=False)
    persist_path: str = Field(default="state/last_bar_seen.json")
    log_skips: bool = Field(default=False)


class TTLConfig(BaseModel):
    """Настройки TTL для сигналов и дедупликации."""

    enabled: bool = Field(default=False)
    ttl_seconds: int = Field(default=60)
    out_csv: Optional[str] = Field(default=None)
    dedup_persist: Optional[str] = Field(default=None)


class TokenBucketConfig(BaseModel):
    """Token bucket limiter settings."""

    rps: float = 0.0
    burst: int = 0


class ThrottleQueueConfig(BaseModel):
    """Settings for queued throttle mode."""

    max_items: int = 0
    ttl_ms: int = 0


class ThrottleConfig(BaseModel):
    """Global throttling configuration."""

    enabled: bool = False
    global_: TokenBucketConfig = Field(default_factory=TokenBucketConfig, alias="global")
    symbol: TokenBucketConfig = Field(default_factory=TokenBucketConfig)
    mode: str = "drop"
    queue: ThrottleQueueConfig = Field(default_factory=ThrottleQueueConfig)
    time_source: str = "monotonic"


class KillSwitchConfig(BaseModel):
    """Thresholds for entering safe mode based on runtime metrics."""

    feed_lag_ms: float = Field(
        default=0.0,
        description=
        "Enter safe mode if worst feed lag exceeds this many milliseconds; non-positive disables",
    )
    ws_failures: float = Field(
        default=0.0,
        description=
        "Enter safe mode if websocket failures for any symbol exceed this count; non-positive disables",
    )
    error_rate: float = Field(
        default=0.0,
        description=
        "Enter safe mode if signal error rate for any symbol exceeds this fraction; non-positive disables",
    )


class OpsKillSwitchConfig(BaseModel):
    """Operational kill switch settings."""

    enabled: bool = False
    error_limit: int = 0
    duplicate_limit: int = 0
    stale_intervals_limit: int = 0
    reset_cooldown_sec: int = 60
    flag_path: Optional[str] = None
    alert_command: Optional[str] = None


class RetryConfig(BaseModel):
    """Retry strategy settings."""

    max_attempts: int = Field(
        default=5, description="Maximum number of retry attempts; non-positive disables"
    )
    backoff_base_s: float = Field(
        default=2.0, description="Initial backoff in seconds for retry backoff"
    )
    max_backoff_s: float = Field(
        default=60.0, description="Maximum backoff in seconds for retry backoff"
    )




class StateConfig(BaseModel):
    """Settings for persisting runner state."""

    enabled: bool = Field(default=False)
    backend: str = Field(default="json")
    path: str = Field(default="state/state_store.json")
    snapshot_interval_s: int = Field(default=0)
    flush_on_event: bool = Field(default=True)
    backup_keep: int = Field(default=0)
    lock_path: str = Field(default="state/state.lock")

@dataclass
class MonitoringThresholdsConfig:
    """Monitoring thresholds for automatic safe-mode triggers."""

    feed_lag_ms: float = 0.0
    ws_failures: float = 0.0
    error_rate: float = 0.0
    fill_ratio_min: float = 0.0
    pnl_min: float = 0.0


@dataclass
class MonitoringAlertConfig:
    """External alert command configuration."""

    enabled: bool = False
    command: Optional[str] = None


@dataclass
class MonitoringConfig:
    """Top-level monitoring configuration."""

    enabled: bool = False
    snapshot_metrics_sec: int = 60
    thresholds: MonitoringThresholdsConfig = field(default_factory=MonitoringThresholdsConfig)
    alerts: MonitoringAlertConfig = field(default_factory=MonitoringAlertConfig)

class CommonRunConfig(BaseModel):
    run_id: Optional[str] = Field(
        default=None, description="Идентификатор запуска; если None — генерируется."
    )
    seed: Optional[int] = Field(default=None)
    logs_dir: str = Field(default="logs")
    artifacts_dir: str = Field(default="artifacts")
    timezone: Optional[str] = None
    liquidity_seasonality_path: Optional[str] = Field(default=None)
    liquidity_seasonality_hash: Optional[str] = Field(default=None)
    seasonality_log_level: str = Field(
        default="INFO", description="Logging level for seasonality namespace"
    )
    max_signals_per_sec: Optional[float] = Field(
        default=None,
        description="Maximum outbound signals per second; non-positive disables limiting.",
    )
    backoff_base_s: float = Field(
        default=2.0, description="Initial backoff in seconds for rate limiter"
    )
    max_backoff_s: float = Field(
        default=60.0, description="Maximum backoff in seconds for rate limiter"
    )
    timing: TimingConfig = Field(default_factory=TimingConfig)
    clock_sync: ClockSyncConfig = Field(default_factory=ClockSyncConfig)
    ws_dedup: WSDedupConfig = Field(default_factory=WSDedupConfig)
    ttl: TTLConfig = Field(default_factory=TTLConfig)
    throttle: ThrottleConfig = Field(default_factory=ThrottleConfig)
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)
    kill_switch_ops: OpsKillSwitchConfig = Field(default_factory=OpsKillSwitchConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    state: StateConfig = Field(default_factory=StateConfig)
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
    prices_path: Optional[str] = Field(
        default=None, description="Путь к parquet/csv с историческими данными."
    )


class SimulationConfig(CommonRunConfig):
    mode: str = Field(default="sim")
    timing: TimingConfig = Field(default_factory=TimingConfig)
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
    execution_profile: ExecutionProfile = Field(
        default=ExecutionProfile.MKT_OPEN_NEXT_H1
    )
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
    execution_profile: ExecutionProfile = Field(
        default=ExecutionProfile.MKT_OPEN_NEXT_H1
    )
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
    metrics: List[str] = Field(
        default_factory=lambda: ["sharpe", "sortino", "mdd", "pnl"]
    )
    execution_profile: ExecutionProfile = Field(
        default=ExecutionProfile.MKT_OPEN_NEXT_H1
    )
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
    cfg = cfg_cls.parse_obj(data)
    _set_seasonality_log_level(cfg)
    return cfg


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
    cfg = cfg_cls.parse_obj(data)
    _set_seasonality_log_level(cfg)
    return cfg


def _set_seasonality_log_level(cfg: CommonRunConfig) -> None:
    """Configure log level for ``seasonality`` namespace."""
    level = getattr(cfg, "seasonality_log_level", "INFO")
    if isinstance(level, str):
        level_num = getattr(logging, level.upper(), logging.INFO)
    else:
        try:
            level_num = int(level)
        except Exception:
            level_num = logging.INFO
    logging.getLogger("seasonality").setLevel(level_num)


__all__ = [
    "ComponentSpec",
    "Components",
    "ClockSyncConfig",
    "TimingConfig",
    "WSDedupConfig",
    "TTLConfig",
    "ThrottleConfig",
    "KillSwitchConfig",
    "OpsKillSwitchConfig",
    "MonitoringThresholdsConfig",
    "MonitoringAlertConfig",
    "MonitoringConfig",
    "RetryConfig",
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
