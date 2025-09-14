from __future__ import annotations

"""Utilities for basic pipeline time-to-live checks."""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Tuple

import numpy as np
from collections import deque

from core_models import Bar
from utils_time import next_bar_open_ms, is_bar_closed
from no_trade import (
    _parse_daily_windows_min,
    _in_daily_window,
    _in_funding_buffer,
    _in_custom_window,
)
from no_trade_config import NoTradeConfig


class Stage(Enum):
    """Pipeline stages for decision making."""

    CLOSED_BAR = auto()
    WINDOWS = auto()
    ANOMALY = auto()
    EXTREME = auto()
    POLICY = auto()
    RISK = auto()
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


def closed_bar_guard(
    bar: Bar, now_ms: int, enforce: bool, lag_ms: int
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

    if not enforce:
        return PipelineResult(action="pass", stage=Stage.CLOSED_BAR)

    if lag_ms <= 0:
        if not getattr(bar, "is_final", True):
            return PipelineResult(
                action="drop", stage=Stage.CLOSED_BAR, reason=Reason.INCOMPLETE_BAR
            )
        return PipelineResult(action="pass", stage=Stage.CLOSED_BAR)

    if not is_bar_closed(int(bar.ts), now_ms, lag_ms):
        return PipelineResult(
            action="drop", stage=Stage.CLOSED_BAR, reason=Reason.INCOMPLETE_BAR
        )

    return PipelineResult(action="pass", stage=Stage.CLOSED_BAR)


_NO_TRADE_CACHE: dict[str, tuple[list[tuple[int, int]], int, list[dict[str, int]]]] = {}


def apply_no_trade_windows(
    ts_ms: int, symbol: str, cfg: NoTradeConfig
) -> PipelineResult:
    """Check whether ``ts_ms`` falls into any no-trade window.

    Parameters
    ----------
    ts_ms:
        Timestamp in milliseconds since epoch.
    symbol:
        Trading symbol used for cache key.
    cfg:
        No-trade configuration.

    Returns
    -------
    PipelineResult
        ``"drop"`` with reason :class:`Reason.WINDOW` if the timestamp is
        blocked, otherwise ``"pass"``.
    """

    cached = _NO_TRADE_CACHE.get(symbol)
    if cached is None:
        daily_min = _parse_daily_windows_min(cfg.daily_utc or [])
        buf_min = int(cfg.funding_buffer_min or 0)
        custom = cfg.custom_ms or []
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
        return PipelineResult(action="drop", stage=Stage.WINDOWS, reason=Reason.WINDOW)
    return PipelineResult(action="pass", stage=Stage.WINDOWS)


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

        self._rets.append(float(ret))
        self._spreads.append(float(spread))

        if len(self._rets) < self._rets.maxlen:
            return PipelineResult(action="pass", stage=Stage.ANOMALY)

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return PipelineResult(
                action="drop", stage=Stage.ANOMALY, reason=self._last_reason
            )

        rets_arr = np.asarray(self._rets, dtype=np.float64)
        cur_ret = rets_arr[-1]
        sigma = np.std(rets_arr[:-1]) if len(rets_arr) > 1 else 0.0
        if sigma > 0 and abs(cur_ret) > float(self.sigma_mult) * sigma:
            self._cooldown_left = int(self.cooldown_bars)
            self._last_reason = Reason.ANOMALY_RET
            return PipelineResult(
                action="drop", stage=Stage.ANOMALY, reason=Reason.ANOMALY_RET
            )

        sp_arr = np.asarray(self._spreads, dtype=np.float64)
        cur_spread = sp_arr[-1]
        thr = np.percentile(sp_arr[:-1], self.spread_pct) if len(sp_arr) > 1 else 0.0
        if cur_spread > thr:
            self._cooldown_left = int(self.cooldown_bars)
            self._last_reason = Reason.ANOMALY_SPREAD
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

        st = self._get_state(symbol)
        st.last_metric = float(metric)

        if st.active:
            if st.cooldown_left > 0:
                st.cooldown_left -= 1
            if st.cooldown_left <= 0 and st.last_metric <= self.lower:
                st.active = False
                return PipelineResult(action="pass", stage=Stage.POLICY)
            return PipelineResult(
                action="drop", stage=Stage.POLICY, reason=Reason.MAINTENANCE
            )

        if st.last_metric >= self.upper:
            st.active = True
            st.cooldown_left = int(self.cooldown_bars)
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
        The timestamp (ms since epoch) when the next bar opens.
    """
    return next_bar_open_ms(bar_close_ms, timeframe_ms)


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
    "apply_no_trade_windows",
    "AnomalyDetector",
    "MetricKillSwitch",
    "compute_expires_at",
    "check_ttl",
]
