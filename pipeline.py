from __future__ import annotations

"""Utilities for basic pipeline time-to-live checks."""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Tuple, Sequence, Protocol, Dict, Mapping

import numpy as np
from collections import deque

from core_models import Bar
from core_contracts import FeaturePipe, SignalPolicy, PolicyCtx
from utils_time import is_bar_closed
from no_trade import (
    _parse_daily_windows_min,
    _in_daily_window,
    _in_funding_buffer,
    _in_custom_window,
)
from no_trade_config import NoTradeConfig
from services.monitoring import inc_stage, inc_reason


class Stage(Enum):
    """Pipeline stages for decision making."""

    CLOSED_BAR = auto()
    OPEN_BAR = auto()
    WINDOWS = auto()
    ANOMALY = auto()
    EXTREME = auto()
    POLICY = auto()
    RISK = auto()
    TTL = auto()
    DEDUP = auto()
    THROTTLE = auto()
    PUBLISH = auto()


class Reason(Enum):
    """Reasons for halting or skipping pipeline stages."""

    INCOMPLETE_BAR = auto()
    MAINTENANCE = auto()
    WINDOW = auto()
    ANOMALY_RET = auto()
    ANOMALY_SPREAD = auto()
    EXTREME_VOL = auto()
    EXTREME_SPREAD = auto()
    RISK_POSITION = auto()
    OTHER = auto()


@dataclass
class PipelineResult:
    """Result produced by each pipeline stage."""

    action: str
    stage: Stage
    reason: Reason | None = None
    decision: Any | None = None


@dataclass
class PipelineStageConfig:
    """Configuration for a single pipeline stage."""

    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    enabled: bool = True
    stages: Dict[str, PipelineStageConfig] = field(default_factory=dict)

    def get(self, name: str) -> PipelineStageConfig | None:
        return self.stages.get(name)

    def merge(self, other: "PipelineConfig") -> "PipelineConfig":
        """Merge two configs, with ``other`` taking precedence."""

        enabled = other.enabled if other is not None else self.enabled
        stages: Dict[str, PipelineStageConfig] = {
            k: PipelineStageConfig(v.enabled, dict(v.params))
            for k, v in self.stages.items()
        }
        if other is not None:
            for name, cfg in other.stages.items():
                if name in stages:
                    stages[name].enabled = cfg.enabled
                    stages[name].params.update(cfg.params)
                else:
                    stages[name] = PipelineStageConfig(cfg.enabled, dict(cfg.params))
        return PipelineConfig(enabled=enabled, stages=stages)


def closed_bar_guard(
    bar: Bar, now_ms: int, enforce: bool, lag_ms: int, *, stage_cfg: PipelineStageConfig | None = None
) -> PipelineResult:
    """Ensure that incoming bars are fully closed before processing.

    Parameters
    ----------
    bar : Bar
        The bar under consideration.
    now_ms : int
        Current timestamp in milliseconds.
    enforce : bool
        Whether the guard is active.
    lag_ms : int
        Allowed closing lag in milliseconds. ``0`` implies websocket bars
        where ``bar.is_final`` should be used.

    Returns
    -------
    PipelineResult
        Result with action ``"pass"`` if the bar is closed, otherwise
        ``"drop"`` with reason :class:`Reason.INCOMPLETE_BAR`.
    """

    inc_stage(Stage.CLOSED_BAR)
    if stage_cfg is not None and not stage_cfg.enabled:
        return PipelineResult(action="pass", stage=Stage.CLOSED_BAR)

    if not enforce:
        return PipelineResult(action="pass", stage=Stage.CLOSED_BAR)

    if lag_ms <= 0:
        if not getattr(bar, "is_final", True):
            inc_reason(Reason.INCOMPLETE_BAR)
            return PipelineResult(
                action="drop", stage=Stage.CLOSED_BAR, reason=Reason.INCOMPLETE_BAR
            )
        return PipelineResult(action="pass", stage=Stage.CLOSED_BAR)

    if not is_bar_closed(int(bar.ts), now_ms, lag_ms):
        inc_reason(Reason.INCOMPLETE_BAR)
        return PipelineResult(
            action="drop", stage=Stage.CLOSED_BAR, reason=Reason.INCOMPLETE_BAR
        )

    return PipelineResult(action="pass", stage=Stage.CLOSED_BAR)


def open_bar_guard(
    bar: Bar,
    now_ms: int,
    enforce: bool,
    lag_ms: int,
    *,
    stage_cfg: PipelineStageConfig | None = None,
) -> PipelineResult:
    """Ensure that websocket bars are sufficiently aged before processing.

    Parameters
    ----------
    bar : Bar
        Incoming bar under consideration.
    now_ms : int
        Current timestamp in milliseconds.
    enforce : bool
        Whether the guard is active.
    lag_ms : int
        Minimum allowed lag between ``bar.ts`` and ``now_ms``.

    Returns
    -------
    PipelineResult
        ``"pass"`` when the bar is considered safe to process. Otherwise
        ``"drop"`` with :class:`Reason.INCOMPLETE_BAR`.
    """

    inc_stage(Stage.OPEN_BAR)
    if stage_cfg is not None and not stage_cfg.enabled:
        return PipelineResult(action="pass", stage=Stage.OPEN_BAR)

    if not enforce:
        return PipelineResult(action="pass", stage=Stage.OPEN_BAR)

    ts = getattr(bar, "ts", None)
    if ts is None:
        inc_reason(Reason.INCOMPLETE_BAR)
        return PipelineResult(
            action="drop", stage=Stage.OPEN_BAR, reason=Reason.INCOMPLETE_BAR
        )

    if lag_ms <= 0:
        if not getattr(bar, "is_final", True):
            inc_reason(Reason.INCOMPLETE_BAR)
            return PipelineResult(
                action="drop", stage=Stage.OPEN_BAR, reason=Reason.INCOMPLETE_BAR
            )
        return PipelineResult(action="pass", stage=Stage.OPEN_BAR)

    try:
        ts_ms = int(ts)
    except (TypeError, ValueError):
        inc_reason(Reason.INCOMPLETE_BAR)
        return PipelineResult(
            action="drop", stage=Stage.OPEN_BAR, reason=Reason.INCOMPLETE_BAR
        )

    if now_ms < ts_ms + lag_ms:
        inc_reason(Reason.INCOMPLETE_BAR)
        return PipelineResult(
            action="drop", stage=Stage.OPEN_BAR, reason=Reason.INCOMPLETE_BAR
        )

    return PipelineResult(action="pass", stage=Stage.OPEN_BAR)


_NO_TRADE_CACHE: dict[str, tuple[list[tuple[int, int]], int, list[dict[str, int]]]] = {}


def apply_no_trade_windows(
    ts_ms: int,
    symbol: str,
    nt_cfg: NoTradeConfig,
    *,
    stage_cfg: PipelineStageConfig | None = None,
) -> PipelineResult:
    """Check whether ``ts_ms`` falls into any no-trade window.

    Parameters
    ----------
    ts_ms:
        Timestamp in milliseconds since epoch.
    symbol:
        Trading symbol used for cache key.
    nt_cfg:
        No-trade configuration.

    Returns
    -------
    PipelineResult
        ``"drop"`` with reason :class:`Reason.WINDOW` if the timestamp is
        blocked, otherwise ``"pass"``.
    """

    inc_stage(Stage.WINDOWS)
    if stage_cfg is not None and not stage_cfg.enabled:
        return PipelineResult(action="pass", stage=Stage.WINDOWS)

    cached = _NO_TRADE_CACHE.get(symbol)
    if cached is None:
        daily_min = _parse_daily_windows_min(nt_cfg.daily_utc or [])
        buf_min = int(nt_cfg.funding_buffer_min or 0)
        custom = nt_cfg.custom_ms or []
        cached = (daily_min, buf_min, custom)
        _NO_TRADE_CACHE[symbol] = cached
    daily_min, buf_min, custom = cached

    ts_arr = np.asarray([ts_ms], dtype=np.int64)
    blocked = (
        _in_daily_window(ts_arr, daily_min)[0]
        or _in_funding_buffer(ts_arr, buf_min)[0]
        or _in_custom_window(ts_arr, custom)[0]
    )
    if blocked:
        inc_reason(Reason.WINDOW)
        return PipelineResult(action="drop", stage=Stage.WINDOWS, reason=Reason.WINDOW)
    return PipelineResult(action="pass", stage=Stage.WINDOWS)


class RiskGuards(Protocol):
    def apply(
        self, ts_ms: int, symbol: str, decisions: Sequence[Any]
    ) -> Tuple[Sequence[Any], str | None]: ...


def policy_decide(
    fp: FeaturePipe,
    policy: SignalPolicy,
    bar: Bar,
    *,
    stage_cfg: PipelineStageConfig | None = None,
    signal_quality_cfg: Any | None = None,
    precomputed_features: Mapping[str, Any] | None = None,
) -> PipelineResult:
    inc_stage(Stage.POLICY)
    if stage_cfg is not None and not stage_cfg.enabled:
        return PipelineResult(action="pass", stage=Stage.POLICY, decision=[])
    if precomputed_features is None:
        feats = fp.update(bar)
    else:
        feats = dict(precomputed_features)
    ctx = PolicyCtx(
        ts=int(bar.ts),
        symbol=bar.symbol,
        signal_quality_cfg=signal_quality_cfg,
    )
    decisions = list(policy.decide({**feats}, ctx) or [])
    return PipelineResult(action="pass", stage=Stage.POLICY, decision=decisions)


def apply_risk(
    ts_ms: int,
    symbol: str,
    guards: RiskGuards | None,
    decisions: Sequence[Any],
    *,
    stage_cfg: PipelineStageConfig | None = None,
) -> PipelineResult:
    inc_stage(Stage.RISK)
    if stage_cfg is not None and not stage_cfg.enabled:
        return PipelineResult(action="pass", stage=Stage.RISK, decision=list(decisions))
    if guards is None:
        return PipelineResult(action="pass", stage=Stage.RISK, decision=list(decisions))
    checked, reason = guards.apply(ts_ms, symbol, decisions)
    checked = list(checked or [])
    if reason:
        inc_reason(Reason.RISK_POSITION)
        return PipelineResult(
            action="drop", stage=Stage.RISK, reason=Reason.RISK_POSITION, decision=checked
        )
    return PipelineResult(action="pass", stage=Stage.RISK, decision=checked)


@dataclass
class AnomalyDetector:
    """Stateful anomaly detector for returns and spread.

    Maintains rolling statistics over ``window`` bars and drops bars when
    extreme moves are observed. After triggering, the detector enforces a
    cooldown for ``cooldown_bars`` bars.
    """

    window: int = 100
    cooldown_bars: int = 0
    sigma_mult: float = 5.0
    spread_pct: float = 99.0

    _rets: deque[float] = field(default_factory=deque, init=False)
    _spreads: deque[float] = field(default_factory=deque, init=False)
    _cooldown_left: int = 0
    _last_reason: Reason | None = None

    def __post_init__(self) -> None:
        self._rets = deque(maxlen=int(self.window))
        self._spreads = deque(maxlen=int(self.window))

    def update(self, ret: float, spread: float) -> PipelineResult:
        """Update detector with new return and spread values.

        Parameters
        ----------
        ret:
            Return of the latest bar.
        spread:
            Bid/ask spread of the latest bar in the same units as history.
        """

        inc_stage(Stage.ANOMALY)
        self._rets.append(float(ret))
        self._spreads.append(float(spread))

        if len(self._rets) < self._rets.maxlen:
            return PipelineResult(action="pass", stage=Stage.ANOMALY)

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            if self._last_reason is not None:
                inc_reason(self._last_reason)
            return PipelineResult(
                action="drop", stage=Stage.ANOMALY, reason=self._last_reason
            )

        rets_arr = np.asarray(self._rets, dtype=np.float64)
        cur_ret = rets_arr[-1]
        sigma = np.std(rets_arr[:-1]) if len(rets_arr) > 1 else 0.0
        if sigma > 0 and abs(cur_ret) > float(self.sigma_mult) * sigma:
            self._cooldown_left = int(self.cooldown_bars)
            self._last_reason = Reason.ANOMALY_RET
            inc_reason(Reason.ANOMALY_RET)
            return PipelineResult(
                action="drop", stage=Stage.ANOMALY, reason=Reason.ANOMALY_RET
            )

        sp_arr = np.asarray(self._spreads, dtype=np.float64)
        cur_spread = sp_arr[-1]
        thr = np.percentile(sp_arr[:-1], self.spread_pct) if len(sp_arr) > 1 else 0.0
        if cur_spread > thr:
            self._cooldown_left = int(self.cooldown_bars)
            self._last_reason = Reason.ANOMALY_SPREAD
            inc_reason(Reason.ANOMALY_SPREAD)
            return PipelineResult(
                action="drop", stage=Stage.ANOMALY, reason=Reason.ANOMALY_SPREAD
            )

        return PipelineResult(action="pass", stage=Stage.ANOMALY)


@dataclass
class _SymbolState:
    """Internal per-symbol state for :class:`MetricKillSwitch`."""

    active: bool = False
    last_metric: float = 0.0
    cooldown_left: int = 0


@dataclass
class MetricKillSwitch:
    """Guard trading based on a metric with hysteresis and cooldown.

    The switch tracks state separately for each symbol.  Trading is disabled
    when the observed ``metric`` exceeds ``upper`` and re-enabled once it
    falls below ``lower`` after ``cooldown_bars`` updates.

    Parameters
    ----------
    upper:
        Threshold that triggers the kill switch.
    lower:
        Threshold for leaving the kill state once cooldown has elapsed.
    cooldown_bars:
        Number of updates to wait after triggering before re-evaluating the
        exit condition.
    """

    upper: float
    lower: float
    cooldown_bars: int = 0

    _states: dict[str, _SymbolState] = field(default_factory=dict, init=False)

    def _get_state(self, symbol: str) -> _SymbolState:
        return self._states.setdefault(symbol, _SymbolState())

    def update(self, symbol: str, metric: float) -> PipelineResult:
        """Update state for ``symbol`` and return pipeline decision.

        Parameters
        ----------
        symbol:
            Trading symbol to update.
        metric:
            Observed metric value.
        """

        inc_stage(Stage.POLICY)
        st = self._get_state(symbol)
        st.last_metric = float(metric)

        if st.active:
            if st.cooldown_left > 0:
                st.cooldown_left -= 1
            if st.cooldown_left <= 0 and st.last_metric <= self.lower:
                st.active = False
                return PipelineResult(action="pass", stage=Stage.POLICY)
            inc_reason(Reason.MAINTENANCE)
            return PipelineResult(
                action="drop", stage=Stage.POLICY, reason=Reason.MAINTENANCE
            )

        if st.last_metric >= self.upper:
            st.active = True
            st.cooldown_left = int(self.cooldown_bars)
            inc_reason(Reason.MAINTENANCE)
            return PipelineResult(
                action="drop", stage=Stage.POLICY, reason=Reason.MAINTENANCE
            )

        return PipelineResult(action="pass", stage=Stage.POLICY)

    def is_active(self, symbol: str) -> bool:
        """Return whether trading is currently disabled for ``symbol``."""

        return self._get_state(symbol).active

    def last_metric_value(self, symbol: str) -> float:
        """Return last observed metric value for ``symbol``."""

        return self._get_state(symbol).last_metric


def compute_expires_at(bar_close_ms: int, timeframe_ms: int) -> int:
    """Compute expiration timestamp for a bar.

    Parameters
    ----------
    bar_close_ms : int
        Close timestamp of the bar in milliseconds since epoch.
    timeframe_ms : int
        Timeframe of the bar in milliseconds.

    Returns
    -------
    int
        The timestamp (ms since epoch) when the bar's TTL expires.

    Notes
    -----
    This helper normalises both inputs to integers and guarantees that the TTL
    spans exactly one full bar after ``bar_close_ms``.
    """

    try:
        close_ms = int(bar_close_ms)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        raise ValueError("bar_close_ms must be an integer timestamp") from None

    try:
        tf_ms = int(timeframe_ms)
    except (TypeError, ValueError):
        raise ValueError("timeframe_ms must be an integer duration") from None

    if tf_ms <= 0:
        raise ValueError("timeframe_ms must be a positive duration")

    return close_ms + tf_ms


def check_ttl(
    bar_close_ms: int, now_ms: int, timeframe_ms: int
) -> Tuple[bool, int, str]:
    """Validate that a bar has not exceeded its time-to-live.

    The TTL for a bar is one full timeframe after its close. This function
    checks the absolute age of the bar against that limit.

    Parameters
    ----------
    bar_close_ms : int
        Close timestamp of the bar.
    now_ms : int
        Current time in milliseconds since epoch.
    timeframe_ms : int
        Bar timeframe in milliseconds.

    Returns
    -------
    Tuple[bool, int, str]
        A tuple of ``(valid, expires_at_ms, reason)`` where ``valid`` indicates
        whether the bar is still within its TTL, ``expires_at_ms`` is the
        absolute expiration timestamp, and ``reason`` provides context when the
        bar is no longer valid.
    """
    expires_at_ms = compute_expires_at(bar_close_ms, timeframe_ms)
    age_ms = now_ms - bar_close_ms
    if now_ms <= expires_at_ms:
        return True, expires_at_ms, ""
    return False, expires_at_ms, f"age {age_ms}ms exceeds {timeframe_ms}ms"


__all__ = [
    "Stage",
    "Reason",
    "PipelineResult",
    "closed_bar_guard",
    "open_bar_guard",
    "apply_no_trade_windows",
    "policy_decide",
    "apply_risk",
    "AnomalyDetector",
    "MetricKillSwitch",
    "compute_expires_at",
    "check_ttl",
]
