# -*- coding: utf-8 -*-
"""
core_config.py
Pydantic-модели конфигураций: sim/live/train/eval + декларация компонентов для DI.
Поддерживается dotted path формата "module.submodule:ClassName".
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Mapping, Union, Literal, Sequence, Tuple
from enum import Enum
from dataclasses import dataclass, field

import yaml
from pydantic import BaseModel, Field, root_validator, model_validator
import logging
import math

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


class TimingProfileSpec(BaseModel):
    """Профиль тайминга для конкретного :class:`ExecutionProfile`."""

    decision_mode: Optional[str] = Field(default=None)
    decision_delay_ms: Optional[int] = Field(default=None)
    latency_steps: Optional[int] = Field(default=None)
    min_lookback_ms: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def _sanitize(cls, values: "TimingProfileSpec") -> "TimingProfileSpec":
        delay = getattr(values, "decision_delay_ms", None)
        if delay is not None:
            object.__setattr__(values, "decision_delay_ms", max(0, int(delay)))
        latency = getattr(values, "latency_steps", None)
        if latency is not None:
            object.__setattr__(values, "latency_steps", max(0, int(latency)))
        lookback = getattr(values, "min_lookback_ms", None)
        if lookback is not None:
            object.__setattr__(values, "min_lookback_ms", max(0, int(lookback)))
        mode = getattr(values, "decision_mode", None)
        if mode is not None:
            object.__setattr__(values, "decision_mode", str(mode).strip())
        return values


@dataclass(frozen=True)
class ResolvedTiming:
    """Разрешённые параметры тайминга для среды исполнения."""

    decision_mode: str
    decision_delay_ms: int
    latency_steps: int
    min_lookback_ms: int


class WSDedupConfig(BaseModel):
    """Конфигурация дедупликации данных вебсокета."""

    enabled: bool = Field(default=False)
    persist_path: str = Field(default="state/last_bar_seen.json")
    log_skips: bool = Field(default=False)


class TTLConfig(BaseModel):
    """Настройки TTL для сигналов и дедупликации."""

    enabled: bool = Field(default=False)
    ttl_seconds: int = Field(default=60, ge=0, le=24 * 60 * 60)
    out_csv: Optional[str] = Field(default=None)
    dedup_persist: Optional[str] = Field(default=None)
    mode: Literal["off", "relative", "absolute"] = Field(
        default="relative",
        description="Алгоритм TTL: relative — относительно закрытия бара, absolute — жёсткая отсечка, off — выключено.",
    )
    guard_ms: int = Field(
        default=5_000,
        ge=0,
        le=15 * 60 * 1000,
        description="Дополнительный защитный буфер в миллисекундах перед публикацией решений.",
    )
    absolute_failsafe_ms: int = Field(
        default=15 * 60 * 1000,
        ge=0,
        le=60 * 60 * 1000,
        description="Абсолютный предел возраста решений в миллисекундах независимо от режима.",
    )
    state_path: str = Field(
        default="state/ttl_state.json",
        description="Путь к файлу состояния TTL-гварда.",
    )

    @model_validator(mode="after")
    def _clamp_values(cls, values: "TTLConfig") -> "TTLConfig":
        guard = int(getattr(values, "guard_ms", 0) or 0)
        failsafe = int(getattr(values, "absolute_failsafe_ms", 0) or 0)
        guard = max(0, min(guard, 15 * 60 * 1000))
        failsafe = max(0, min(failsafe, 60 * 60 * 1000))
        if failsafe and guard and guard > failsafe:
            guard = failsafe
        object.__setattr__(values, "guard_ms", guard)
        object.__setattr__(values, "absolute_failsafe_ms", failsafe)
        return values


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
    snapshot_interval_ms: Optional[int] = Field(default=None)
    flush_on_event: bool = Field(default=True)
    backup_keep: int = Field(default=0)
    lock_path: str = Field(default="state/state.lock")
    dir: Optional[str] = Field(default=None)
    last_processed_per_symbol: bool = Field(default=False)


@dataclass
class MonitoringThresholdsConfig:
    """Monitoring thresholds for automatic safe-mode triggers."""

    feed_lag_ms: float = 0.0
    ws_failures: float = 0.0
    error_rate: float = 0.0
    fill_ratio_min: float = 0.0
    pnl_min: float = 0.0
    zero_signals: int = 0
    cost_bias_bps: float = 0.0


@dataclass
class MonitoringAlertConfig:
    """External alert command configuration."""

    enabled: bool = False
    command: Optional[str] = None
    channel: str = "noop"
    cooldown_sec: float = 0.0
    telegram: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """Top-level monitoring configuration."""

    enabled: bool = False
    snapshot_metrics_sec: int = 60
    tick_sec: float = 1.0
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
    reference_prices_path: Optional[str] = Field(
        default=None,
        description="Optional path to an intrabar reference dataset consumed by bridge adapters.",
    )

    class Config:
        extra = "allow"


class ExecutionEntryMode(str, Enum):
    """Available strategies for deriving execution entry points."""

    DEFAULT = "default"
    STRICT = "strict"


class ClipToBarConfig(BaseModel):
    """Control clipping of intrabar prices to the observed bar range."""

    enabled: bool = Field(
        default=True,
        description="Clip simulated prices to the current bar's high/low range.",
    )
    strict_open_fill: bool = Field(
        default=False,
        description="When enabled, force opening bar fills to honour the clipped price strictly.",
    )


class SpotImpactConfig(BaseModel):
    """Coefficients for simple spot-market impact models."""

    sqrt_coeff: float = Field(
        default=0.0,
        description="Coefficient applied to the square-root participation term (bps).",
    )
    linear_coeff: float = Field(
        default=0.0,
        description="Coefficient applied to the linear participation term (bps).",
    )
    power_coefficient: float = Field(
        default=0.0,
        ge=0.0,
        description="General power-law coefficient applied to participation (bps).",
    )
    power_exponent: float = Field(
        default=1.0,
        ge=0.0,
        description="Exponent for the power-law impact component.",
    )

    class Config:
        extra = "allow"


@dataclass(frozen=True)
class ResolvedTurnoverLimit:
    """Resolved turnover cap expressed in USD terms."""

    bps: Optional[float] = None
    usd: Optional[float] = None
    daily_bps: Optional[float] = None
    daily_usd: Optional[float] = None

    def limit_for_equity(self, equity_usd: float) -> Optional[float]:
        """Compute the USD cap implied by the specification."""

        values: List[float] = []
        if self.usd is not None:
            usd_value = float(self.usd)
            if math.isfinite(usd_value) and usd_value > 0.0:
                values.append(usd_value)
        if self.bps is not None and equity_usd > 0.0:
            bps_value = float(self.bps)
            if math.isfinite(bps_value) and bps_value > 0.0:
                values.append(equity_usd * bps_value / 10_000.0)
        if not values:
            return None
        return min(values)

    def daily_limit_for_equity(self, equity_usd: float) -> Optional[float]:
        """Compute the daily USD cap implied by the specification."""

        values: List[float] = []
        if self.daily_usd is not None:
            usd_value = float(self.daily_usd)
            if math.isfinite(usd_value) and usd_value > 0.0:
                values.append(usd_value)
        if self.daily_bps is not None and equity_usd > 0.0:
            bps_value = float(self.daily_bps)
            if math.isfinite(bps_value) and bps_value > 0.0:
                values.append(equity_usd * bps_value / 10_000.0)
        if not values:
            return None
        return min(values)


@dataclass(frozen=True)
class ResolvedTurnoverCaps:
    """Resolved turnover caps for per-symbol and portfolio aggregates."""

    per_symbol: ResolvedTurnoverLimit = field(default_factory=ResolvedTurnoverLimit)
    portfolio: ResolvedTurnoverLimit = field(default_factory=ResolvedTurnoverLimit)


class SpotTurnoverLimit(BaseModel):
    """Per-entity turnover guard expressed in USD/bps terms."""

    bps: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Maximum turnover expressed as basis points of portfolio equity.",
    )
    usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Absolute turnover cap in USD.",
    )
    daily_bps: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Maximum daily turnover expressed as basis points of portfolio equity.",
    )
    daily_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Absolute daily turnover cap in USD.",
    )

    class Config:
        extra = "allow"

    def resolve(self) -> ResolvedTurnoverLimit:
        return ResolvedTurnoverLimit(
            bps=float(self.bps) if self.bps is not None else None,
            usd=float(self.usd) if self.usd is not None else None,
            daily_bps=float(self.daily_bps) if self.daily_bps is not None else None,
            daily_usd=float(self.daily_usd) if self.daily_usd is not None else None,
        )


class SpotTurnoverCaps(BaseModel):
    """Turnover guardrails applied to bar-execution decisions."""

    per_symbol: Optional[SpotTurnoverLimit] = Field(
        default=None,
        description="Cap applied per symbol (per bar).",
    )
    portfolio: Optional[SpotTurnoverLimit] = Field(
        default=None,
        description="Aggregate cap applied across all symbols (per bar).",
    )

    class Config:
        extra = "allow"

    def resolve(self) -> ResolvedTurnoverCaps:
        return ResolvedTurnoverCaps(
            per_symbol=self.per_symbol.resolve() if self.per_symbol else ResolvedTurnoverLimit(),
            portfolio=self.portfolio.resolve() if self.portfolio else ResolvedTurnoverLimit(),
        )


class SpotCostConfig(BaseModel):
    """Container describing spot execution cost assumptions."""

    taker_fee_bps: float = Field(
        default=0.0,
        ge=0.0,
        description="Taker fee expressed in basis points.",
    )
    half_spread_bps: float = Field(
        default=0.0,
        ge=0.0,
        description="Half spread assumption used for slippage modelling (bps).",
    )
    impact: SpotImpactConfig = Field(
        default_factory=SpotImpactConfig,
        description="Coefficients for the simple impact model applied to participation rates.",
    )
    turnover_caps: SpotTurnoverCaps = Field(
        default_factory=SpotTurnoverCaps,
        description="Turnover guardrails evaluated by the bar executor.",
    )

    class Config:
        extra = "allow"


class PortfolioConfig(BaseModel):
    """High-level portfolio assumptions shared between runtime components."""

    equity_usd: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Capital base in USD; ``None`` leaves it unspecified (legacy behaviour).",
    )
    max_total_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Optional cap on the sum of absolute portfolio weights submitted in a single bar "
            "run. When the requested targets exceed the cap the runner will scale them "
            "pro-rata. ``None`` disables the guard."
        ),
    )

    class Config:
        extra = "allow"


class ExecutionRuntimeConfig(BaseModel):
    """Runtime execution configuration shared across run modes."""

    intrabar_price_model: Optional[str] = Field(default=None)
    timeframe_ms: Optional[int] = Field(default=None)
    use_latency_from: Optional[str] = Field(default=None)
    latency_constant_ms: Optional[int] = Field(default=None)
    reference_prices_path: Optional[str] = Field(
        default=None,
        description="Path to M1 reference bars used by the 'reference' intrabar price model.",
    )
    entry_mode: ExecutionEntryMode = Field(
        default=ExecutionEntryMode.DEFAULT,
        description="Режим выбора точки входа; ``default`` соответствует текущему поведению.",
    )
    mode: Literal["order", "bar"] = Field(
        default="order",
        description="Execution runtime mode. ``order`` preserves legacy per-order behaviour.",
    )
    bar_price: Optional[str] = Field(
        default=None,
        description="Reference price to use when ``mode`` is ``bar`` (e.g. 'close').",
    )
    min_rebalance_step: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum rebalance weight threshold expressed as a fraction of portfolio equity.",
    )
    safety_margin_bps: float = Field(
        default=0.0,
        ge=0.0,
        description="Additional safety margin applied to bar-mode economics (bps).",
    )
    max_participation: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Default per-slice participation cap expressed as a fraction of ADV.",
    )
    clip_to_bar: ClipToBarConfig = Field(default_factory=ClipToBarConfig)
    bridge: ExecutionBridgeConfig = Field(default_factory=ExecutionBridgeConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    costs: SpotCostConfig = Field(default_factory=SpotCostConfig)

    class Config:
        extra = "allow"

    def _export_payload(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        if "exclude_unset" not in kwargs:
            kwargs["exclude_unset"] = False
        exporter = getattr(super(), method)
        payload = exporter(*args, **kwargs)
        if isinstance(payload, dict):
            for key in ("clip_to_bar", "bridge", "portfolio", "costs"):
                nested = payload.get(key)
                if isinstance(nested, BaseModel):
                    payload[key] = nested.dict(exclude_unset=False)
        return payload

    def dict(self, *args, **kwargs):  # type: ignore[override]
        return self._export_payload("dict", *args, **kwargs)

    def model_dump(self, *args, **kwargs):  # type: ignore[override]
        if hasattr(super(), "model_dump"):
            return self._export_payload("model_dump", *args, **kwargs)
        return self._export_payload("dict", *args, **kwargs)


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
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    costs: SpotCostConfig = Field(default_factory=SpotCostConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    execution: ExecutionRuntimeConfig = Field(default_factory=ExecutionRuntimeConfig)
    slippage_calibration_enabled: bool = Field(
        default=False,
        description=(
            "Enable loading calibrated slippage profiles from runtime artifacts when available."
        ),
    )
    slippage_calibration_path: Optional[str] = Field(
        default=None,
        description="Optional override path to a calibrated slippage artifact (JSON/YAML).",
    )
    slippage_calibration_default_symbol: Optional[str] = Field(
        default=None,
        description="Default symbol key used when the calibration artifact omits explicit mapping.",
    )
    slippage_regime_updates: bool = Field(
        default=True,
        description="Forward market regime updates to the slippage component for calibrated overrides.",
    )
    components: Components
    symbol_specs_path: Optional[str] = Field(
        default=None,
        description="Optional path to symbol metadata containing quote assets.",
    )
    symbol_specs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Inline symbol metadata mapping keyed by symbol.",
    )

    @model_validator(mode="after")
    def _sync_runtime_sections(cls, values: "CommonRunConfig") -> "CommonRunConfig":
        exec_cfg = getattr(values, "execution", None)
        if not isinstance(exec_cfg, ExecutionRuntimeConfig):
            try:
                exec_cfg = ExecutionRuntimeConfig.parse_obj(exec_cfg or {})
            except Exception:
                exec_cfg = ExecutionRuntimeConfig()
            object.__setattr__(values, "execution", exec_cfg)

        fields_set = getattr(values, "__fields_set__", set())
        exec_fields_set = getattr(exec_cfg, "__fields_set__", set())

        portfolio_cfg = getattr(values, "portfolio", None)
        if isinstance(portfolio_cfg, Mapping):
            try:
                portfolio_cfg = PortfolioConfig.parse_obj(portfolio_cfg)
            except Exception:
                portfolio_cfg = PortfolioConfig()
        elif not isinstance(portfolio_cfg, PortfolioConfig):
            portfolio_cfg = PortfolioConfig()
        exec_portfolio_raw = getattr(exec_cfg, "portfolio", None)
        if isinstance(exec_portfolio_raw, Mapping):
            try:
                exec_portfolio = PortfolioConfig.parse_obj(exec_portfolio_raw)
            except Exception:
                exec_portfolio = PortfolioConfig()
        elif isinstance(exec_portfolio_raw, PortfolioConfig):
            exec_portfolio = exec_portfolio_raw
        else:
            exec_portfolio = None
        exec_provided = "portfolio" in exec_fields_set and exec_portfolio is not None
        top_provided = "portfolio" in fields_set
        if exec_provided and exec_portfolio is not None:
            portfolio_cfg = exec_portfolio
        elif top_provided:
            exec_portfolio = portfolio_cfg
        else:
            portfolio_cfg = exec_portfolio or portfolio_cfg
        if portfolio_cfg is None:
            portfolio_cfg = PortfolioConfig()
        object.__setattr__(exec_cfg, "portfolio", portfolio_cfg)
        object.__setattr__(values, "portfolio", portfolio_cfg)

        costs_cfg = getattr(values, "costs", None)
        if isinstance(costs_cfg, Mapping):
            try:
                costs_cfg = SpotCostConfig.parse_obj(costs_cfg)
            except Exception:
                costs_cfg = SpotCostConfig()
        elif not isinstance(costs_cfg, SpotCostConfig):
            costs_cfg = SpotCostConfig()
        exec_costs_raw = getattr(exec_cfg, "costs", None)
        if isinstance(exec_costs_raw, Mapping):
            try:
                exec_costs = SpotCostConfig.parse_obj(exec_costs_raw)
            except Exception:
                exec_costs = SpotCostConfig()
        elif isinstance(exec_costs_raw, SpotCostConfig):
            exec_costs = exec_costs_raw
        else:
            exec_costs = None
        exec_costs_provided = "costs" in exec_fields_set and exec_costs is not None
        top_costs_provided = "costs" in fields_set
        if exec_costs_provided and exec_costs is not None:
            costs_cfg = exec_costs
        elif top_costs_provided:
            exec_costs = costs_cfg
        else:
            costs_cfg = exec_costs or costs_cfg
        if costs_cfg is None:
            costs_cfg = SpotCostConfig()
        object.__setattr__(exec_cfg, "costs", costs_cfg)
        object.__setattr__(values, "costs", costs_cfg)
        return values


class ExecutionProfile(str, Enum):
    """Подход к исполнению заявок."""

    MKT_OPEN_NEXT_H1 = "MKT_OPEN_NEXT_H1"
    VWAP_CURRENT_H1 = "VWAP_CURRENT_H1"
    LIMIT_MID_BPS = "LIMIT_MID_BPS"


def _coerce_execution_profile(value: Any) -> ExecutionProfile:
    if isinstance(value, ExecutionProfile):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty execution profile")
    if text in ExecutionProfile.__members__:
        return ExecutionProfile[text]
    try:
        return ExecutionProfile(text)
    except ValueError as exc:
        raise ValueError(f"Unknown execution profile: {value}") from exc


def _initial_timing_profile_map() -> Dict[ExecutionProfile, TimingProfileSpec]:
    return {prof: TimingProfileSpec() for prof in ExecutionProfile}


def load_timing_profiles(
    path: str = "configs/timing.yaml",
) -> Tuple[TimingConfig, Dict[ExecutionProfile, TimingProfileSpec]]:
    """Загрузить конфигурацию тайминга и профили исполнения из YAML."""

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        defaults = TimingConfig()
        return defaults, _initial_timing_profile_map()

    if not isinstance(raw, Mapping):
        raw = {}

    if "defaults" in raw or "profiles" in raw:
        defaults_payload = raw.get("defaults") or {}
        profiles_payload = raw.get("profiles") or {}
    else:
        defaults_payload = dict(raw)
        profiles_payload = defaults_payload.pop("profiles", {}) if isinstance(defaults_payload, dict) else {}

    defaults_cfg = TimingConfig.parse_obj(defaults_payload)
    profile_map = _initial_timing_profile_map()

    for key, payload in (profiles_payload or {}).items():
        prof = _coerce_execution_profile(key)
        profile_map[prof] = TimingProfileSpec.parse_obj(payload or {})

    return defaults_cfg, profile_map


def resolve_execution_timing(
    profile: ExecutionProfile,
    timing_defaults: TimingConfig,
    profiles: Mapping[ExecutionProfile, TimingProfileSpec] | None,
) -> ResolvedTiming:
    """Сформировать параметры тайминга для среды по профилю исполнения."""

    spec = profiles.get(profile) if profiles else None
    if spec is None:
        spec = TimingProfileSpec()

    delay = spec.decision_delay_ms
    if delay is None:
        delay = int(getattr(timing_defaults, "close_lag_ms", 0) or 0)
    delay = max(0, int(delay))

    min_lookback = int(getattr(spec, "min_lookback_ms", 0) or 0)

    mode_text = spec.decision_mode or "CLOSE_TO_OPEN"
    mode_key = mode_text.strip().upper()
    valid_modes = {"CLOSE_TO_OPEN", "INTRA_HOUR_WITH_LATENCY"}
    if mode_key not in valid_modes:
        raise ValueError(
            f"Unsupported decision_mode '{mode_text}' for execution profile {profile.value}"
        )

    latency = spec.latency_steps
    if latency is None:
        if mode_key == "INTRA_HOUR_WITH_LATENCY":
            timeframe_ms = int(getattr(timing_defaults, "timeframe_ms", 0) or 0)
            if timeframe_ms > 0:
                latency = max(1, math.ceil(delay / timeframe_ms))
            else:
                latency = 1
        else:
            latency = 0
    else:
        latency = max(0, int(latency))
        if mode_key != "INTRA_HOUR_WITH_LATENCY":
            latency = max(0, latency)

    return ResolvedTiming(
        decision_mode=mode_key,
        decision_delay_ms=int(delay),
        latency_steps=int(latency),
        min_lookback_ms=int(max(0, min_lookback)),
    )


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
    execution_profile: ExecutionProfile = Field(
        default=ExecutionProfile.MKT_OPEN_NEXT_H1
    )
    execution_params: ExecutionParams = Field(default_factory=ExecutionParams)


class TrainDataConfig(BaseModel):
    symbols: List[str] = Field(default_factory=get_symbols)
    timeframe: str
    start_ts: Optional[int] = None
    end_ts: Optional[int] = None
    train_start_ts: Optional[int] = None
    train_end_ts: Optional[int] = None
    val_start_ts: Optional[int] = None
    val_end_ts: Optional[int] = None
    test_start_ts: Optional[int] = None
    test_end_ts: Optional[int] = None
    processed_dir: str = Field(default="data/processed")
    split_path: Optional[str] = None
    split_version: Optional[str] = None
    split_overrides: Optional[Dict[str, Any]] = None
    timestamp_column: str = Field(default="timestamp")
    role_column: str = Field(default="wf_role")
    features_params: Dict[str, Any] = Field(default_factory=dict)
    target_params: Dict[str, Any] = Field(default_factory=dict)

    @root_validator(pre=True)
    def _sync_train_window_aliases(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        start = values.get("start_ts")
        train_start = values.get("train_start_ts")
        if start is not None and train_start is not None and start != train_start:
            raise ValueError("start_ts and train_start_ts must match when both are provided")
        if start is None and train_start is not None:
            values["start_ts"] = train_start
        elif train_start is None and start is not None:
            values["train_start_ts"] = start

        end = values.get("end_ts")
        train_end = values.get("train_end_ts")
        if end is not None and train_end is not None and end != train_end:
            raise ValueError("end_ts and train_end_ts must match when both are provided")
        if end is None and train_end is not None:
            values["end_ts"] = train_end
        elif train_end is None and end is not None:
            values["train_end_ts"] = end

        return values

    @model_validator(mode="after")
    def _ensure_train_window_aliases(cls, values: "TrainDataConfig") -> "TrainDataConfig":
        start = getattr(values, "start_ts", None)
        train_start = getattr(values, "train_start_ts", None)
        if train_start is None:
            object.__setattr__(values, "train_start_ts", start)
        elif start is None:
            object.__setattr__(values, "start_ts", train_start)
        elif start != train_start:
            raise ValueError("start_ts and train_start_ts diverged during validation")

        end = getattr(values, "end_ts", None)
        train_end = getattr(values, "train_end_ts", None)
        if train_end is None:
            object.__setattr__(values, "train_end_ts", end)
        elif end is None:
            object.__setattr__(values, "end_ts", train_end)
        elif end != train_end:
            raise ValueError("end_ts and train_end_ts diverged during validation")

        return values


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
    "TimingProfileSpec",
    "ResolvedTiming",
    "WSDedupConfig",
    "TTLConfig",
    "ThrottleConfig",
    "RiskConfigSection",
    "KillSwitchConfig",
    "OpsKillSwitchConfig",
    "LatencyConfig",
    "ExecutionBridgeConfig",
    "ExecutionRuntimeConfig",
    "PortfolioConfig",
    "SpotCostConfig",
    "SpotImpactConfig",
    "SpotTurnoverLimit",
    "SpotTurnoverCaps",
    "ResolvedTurnoverLimit",
    "ResolvedTurnoverCaps",
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
    "load_timing_profiles",
    "resolve_execution_timing",
    "load_config",
    "load_config_from_str",
]
