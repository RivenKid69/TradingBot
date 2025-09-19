# -*- coding: utf-8 -*-
"""
core_config.py
Pydantic-модели конфигураций: sim/live/train/eval + декларация компонентов для DI.
Поддерживается dotted path формата "module.submodule:ClassName".
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Mapping, Union, Literal, Sequence
from enum import Enum
from dataclasses import dataclass, field

import yaml
from pydantic import BaseModel, Field, root_validator
import logging

from services.universe import get_symbols


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


class RiskConfigSection(BaseModel):
    """Top-level risk configuration shared across run modes."""

    max_total_notional: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Maximum aggregate notional exposure across all symbols; ``None`` disables the limit."
        ),
    )
    max_total_exposure_pct: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Maximum aggregate exposure expressed as a fraction of equity; ``None`` disables the limit."
        ),
    )
    exposure_buffer_frac: float = Field(
        default=0.0,
        ge=0.0,
        description="Fractional buffer applied when evaluating aggregate exposure limits.",
    )

    class Config:
        extra = "allow"

    def component_params(self) -> Dict[str, Any]:
        """Return component-specific parameters excluding aggregate exposure limits."""

        data = self.dict(exclude_unset=False)
        data.pop("max_total_notional", None)
        data.pop("max_total_exposure_pct", None)
        data.pop("exposure_buffer_frac", None)
        return data

    @property
    def exposure_limits(self) -> Dict[str, Optional[float]]:
        """Expose aggregate exposure limit knobs for downstream consumers."""

        return {
            "max_total_notional": self.max_total_notional,
            "max_total_exposure_pct": self.max_total_exposure_pct,
            "exposure_buffer_frac": self.exposure_buffer_frac,
        }


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
    global_: TokenBucketConfig = Field(
        default_factory=TokenBucketConfig, alias="global"
    )
    symbol: TokenBucketConfig = Field(default_factory=TokenBucketConfig)
    mode: str = "drop"
    queue: ThrottleQueueConfig = Field(default_factory=ThrottleQueueConfig)
    time_source: str = "monotonic"


class KillSwitchConfig(BaseModel):
    """Thresholds for entering safe mode based on runtime metrics."""

    feed_lag_ms: float = Field(
        default=0.0,
        description="Enter safe mode if worst feed lag exceeds this many milliseconds; non-positive disables",
    )
    ws_failures: float = Field(
        default=0.0,
        description="Enter safe mode if websocket failures for any symbol exceed this count; non-positive disables",
    )
    error_rate: float = Field(
        default=0.0,
        description="Enter safe mode if signal error rate for any symbol exceeds this fraction; non-positive disables",
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
    thresholds: MonitoringThresholdsConfig = field(
        default_factory=MonitoringThresholdsConfig
    )
    alerts: MonitoringAlertConfig = field(default_factory=MonitoringAlertConfig)


class LatencyConfig(BaseModel):
    """Latency configuration preserved on ``CommonRunConfig``."""

    use_seasonality: bool = Field(default=True)
    latency_seasonality_path: Optional[str] = Field(default=None)
    refresh_period_days: int = Field(default=30)
    seasonality_default: Optional[Union[float, Sequence[float]]] = Field(default=1.0)

    class Config:
        extra = "allow"

    def dict(self, *args, **kwargs):  # type: ignore[override]
        if "exclude_unset" not in kwargs:
            kwargs["exclude_unset"] = False
        return super().dict(*args, **kwargs)


class ExecutionBridgeConfig(BaseModel):
    """Configuration payload for execution bridge adapters."""

    intrabar_price_model: Optional[str] = Field(default=None)
    timeframe_ms: Optional[int] = Field(default=None)
    use_latency_from: Optional[str] = Field(default=None)
    latency_constant_ms: Optional[int] = Field(default=None)

    class Config:
        extra = "allow"


class ExecutionRuntimeConfig(BaseModel):
    """Runtime execution configuration shared across run modes."""

    intrabar_price_model: Optional[str] = Field(default=None)
    timeframe_ms: Optional[int] = Field(default=None)
    use_latency_from: Optional[str] = Field(default=None)
    latency_constant_ms: Optional[int] = Field(default=None)
    bridge: ExecutionBridgeConfig = Field(default_factory=ExecutionBridgeConfig)

    class Config:
        extra = "allow"

    def dict(self, *args, **kwargs):  # type: ignore[override]
        if "exclude_unset" not in kwargs:
            kwargs["exclude_unset"] = False
        return super().dict(*args, **kwargs)


class AdvRuntimeConfig(BaseModel):
    """Runtime configuration for ADV/turnover data access."""

    enabled: bool = Field(
        default=False,
        description="Enable ADV dataset integration for runtime components.",
    )
    path: Optional[str] = Field(
        default=None,
        description="Path to the ADV dataset (parquet/json).",
    )
    dataset: Optional[str] = Field(
        default=None,
        description="Optional dataset identifier when ``path`` points to a directory.",
    )
    window_days: int = Field(
        default=30,
        ge=1,
        description="Lookback window (days) used when aggregating ADV metrics.",
    )
    refresh_days: int = Field(
        default=7,
        ge=1,
        description="How often to refresh cached ADV data (days).",
    )
    auto_refresh: bool = Field(
        default=True,
        description="Automatically refresh ADV data when ``refresh_days`` elapsed.",
    )
    missing_symbol_policy: Literal["warn", "skip", "error"] = Field(
        default="warn",
        description="Behaviour when ADV quote is missing for a symbol.",
    )
    floor_quote: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Lower bound applied to resolved ADV quote values.",
    )
    default_quote: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Fallback ADV quote when symbol data is unavailable.",
    )
    capacity_fraction: float = Field(
        default=1.0,
        ge=0.0,
        description="Fraction of per-bar ADV capacity used for execution sizing.",
    )
    bars_per_day_override: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Override for bars-per-day when deriving per-bar ADV capacity.",
    )
    seasonality_path: Optional[str] = Field(
        default=None,
        description="Optional path to seasonality multipliers applied to ADV quotes.",
    )
    seasonality_profile: Optional[str] = Field(
        default=None,
        description="Profile key to extract from ``seasonality_path`` payload.",
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Container for legacy knobs preserved for compatibility.",
    )

    class Config:
        extra = "allow"

    def dict(self, *args, **kwargs):  # type: ignore[override]
        if "exclude_unset" not in kwargs:
            kwargs["exclude_unset"] = False
        payload = super().dict(*args, **kwargs)
        return payload

    @root_validator(pre=True)
    def _capture_unknown(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(values, dict):
            return values
        known = set(cls.__fields__.keys())
        extras = {k: values[k] for k in list(values.keys()) if k not in known}
        if extras:
            existing = values.get("extra")
            merged: Dict[str, Any] = {}
            if isinstance(existing, Mapping):
                merged.update(existing)
            merged.update(extras)
            values["extra"] = merged
            for key in extras:
                values.pop(key, None)
        return values


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
    latency_seasonality_path: Optional[str] = Field(default=None)
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
    risk: RiskConfigSection = Field(default_factory=RiskConfigSection)
    adv: AdvRuntimeConfig = Field(default_factory=AdvRuntimeConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    execution: ExecutionRuntimeConfig = Field(default_factory=ExecutionRuntimeConfig)
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
    symbols: List[str] = Field(default_factory=get_symbols)
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
    symbols: List[str] = Field(default_factory=get_symbols)
    quantizer: Dict[str, Any] = Field(default_factory=dict)
    fees: Dict[str, Any] = Field(default_factory=dict)
    slippage: Dict[str, Any] = Field(default_factory=dict)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    risk: RiskConfigSection = Field(default_factory=RiskConfigSection)
    no_trade: Dict[str, Any] = Field(default_factory=dict)
    data: SimulationDataConfig
    limits: Dict[str, Any] = Field(default_factory=dict)
    execution_profile: ExecutionProfile = Field(
        default=ExecutionProfile.MKT_OPEN_NEXT_H1
    )
    execution_params: ExecutionParams = Field(default_factory=ExecutionParams)
    execution: ExecutionRuntimeConfig = Field(default_factory=ExecutionRuntimeConfig)

    @root_validator(pre=True)
    def _sync_symbols(cls, values):
        syms = values.get("symbols")
        data = values.get("data") or {}
        if syms and isinstance(data, dict) and not data.get("symbols"):
            data["symbols"] = syms
            values["data"] = data
        return values


class LiveAPIConfig(BaseModel):
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True
    ws_endpoint: Optional[str] = None
    rest_endpoint: Optional[str] = None


class LiveDataConfig(BaseModel):
    symbols: List[str] = Field(default_factory=get_symbols)
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
    symbols: List[str] = Field(default_factory=get_symbols)
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
    symbols: List[str] = Field(default_factory=get_symbols)
    quantizer: Dict[str, Any] = Field(default_factory=dict)
    fees: Dict[str, Any] = Field(default_factory=dict)
    slippage: Dict[str, Any] = Field(default_factory=dict)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    risk: RiskConfigSection = Field(default_factory=RiskConfigSection)
    no_trade: Dict[str, Any] = Field(default_factory=dict)
    data: TrainDataConfig
    model: ModelConfig
    execution_profile: ExecutionProfile = Field(
        default=ExecutionProfile.MKT_OPEN_NEXT_H1
    )
    execution_params: ExecutionParams = Field(default_factory=ExecutionParams)

    @root_validator(pre=True)
    def _sync_symbols(cls, values):
        syms = values.get("symbols")
        data = values.get("data") or {}
        if syms and isinstance(data, dict) and not data.get("symbols"):
            data["symbols"] = syms
            values["data"] = data
        return values


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


def _inject_quantizer_config(cfg: CommonRunConfig, data: Dict[str, Any]) -> None:
    """Ensure quantizer configuration is preserved on ``cfg``."""

    q_raw = data.get("quantizer")
    if q_raw is None:
        return
    try:
        q_dict = dict(q_raw)  # type: ignore[arg-type]
    except Exception:
        q_dict = {}
    try:
        existing = getattr(cfg, "quantizer")
    except AttributeError:
        object.__setattr__(cfg, "quantizer", q_dict)
        return
    if existing is None or not isinstance(existing, dict):
        object.__setattr__(cfg, "quantizer", q_dict)
        return
    existing.clear()
    existing.update(q_dict)


def _inject_adv_config(cfg: CommonRunConfig, data: Dict[str, Any]) -> None:
    """Populate ``cfg.adv`` with structured configuration if present."""

    adv_raw = data.get("adv")
    if adv_raw is None:
        return
    if isinstance(adv_raw, AdvRuntimeConfig):
        adv_cfg = adv_raw
    else:
        try:
            adv_cfg = AdvRuntimeConfig.parse_obj(adv_raw)
        except Exception:
            adv_cfg = AdvRuntimeConfig()
    try:
        setattr(cfg, "adv", adv_cfg)
    except Exception:
        object.__setattr__(cfg, "adv", adv_cfg)


def _inject_risk_config(cfg: CommonRunConfig, data: Dict[str, Any]) -> None:
    """Populate ``cfg.risk`` with structured configuration if present."""

    r_raw = data.get("risk")
    if r_raw is None:
        return
    if isinstance(r_raw, RiskConfigSection):
        risk_cfg = r_raw
    else:
        try:
            risk_cfg = RiskConfigSection.parse_obj(r_raw)
        except Exception:
            risk_cfg = RiskConfigSection()
    try:
        setattr(cfg, "risk", risk_cfg)
    except Exception:
        object.__setattr__(cfg, "risk", risk_cfg)


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
    _inject_quantizer_config(cfg, data)
    _inject_adv_config(cfg, data)
    _inject_risk_config(cfg, data)
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
    _inject_quantizer_config(cfg, data)
    _inject_adv_config(cfg, data)
    _inject_risk_config(cfg, data)
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
    "RiskConfigSection",
    "KillSwitchConfig",
    "OpsKillSwitchConfig",
    "LatencyConfig",
    "ExecutionBridgeConfig",
    "ExecutionRuntimeConfig",
    "AdvRuntimeConfig",
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
