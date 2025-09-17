"""Stateful dynamic no-trade guard used by the live signal runner.

The guard mirrors the structured configuration used by the historical helper
in :mod:`no_trade`.  It keeps rolling volatility/spread statistics and applies
``sigma_k`` / ``spread_percentile`` triggers with upper/lower hysteresis and a
cooldown to avoid rapid toggling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple
import math

from core_models import Bar
from no_trade_config import DynamicGuardConfig


def _to_float(value: object) -> float:
    """Coerce ``value`` to ``float`` returning ``nan`` on failure."""

    try:
        return float(value)
    except Exception:
        return float("nan")


def _deque_std(values: Deque[float], min_periods: int) -> float:
    """Sample standard deviation of ``values`` ignoring ``nan`` entries."""

    if min_periods <= 0:
        return float("nan")
    finite = [v for v in values if math.isfinite(v)]
    n = len(finite)
    if n < max(2, min_periods):
        return float("nan")
    mean = sum(finite) / n
    var = sum((v - mean) ** 2 for v in finite)
    if n <= 1:
        return float("nan")
    return math.sqrt(var / (n - 1))


def _deque_percentile(values: Deque[float], min_periods: int) -> float:
    """Percentile of the latest value relative to the window."""

    if not values:
        return float("nan")
    current = values[-1]
    if not math.isfinite(current):
        return float("nan")
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < max(1, min_periods):
        return float("nan")
    total = len(finite)
    if total == 0:
        return float("nan")
    less_or_equal = sum(1 for v in finite if v <= current)
    return float(less_or_equal) / float(total)


def _finite_float(value: object | None) -> float | None:
    """Return ``value`` as ``float`` when finite, otherwise ``None``."""

    try:
        result = float(value)
    except Exception:
        return None
    if math.isfinite(result):
        return result
    return None


def _deque_count_finite(values: Deque[float]) -> int:
    """Number of finite entries in ``values``."""

    return sum(1 for v in values if math.isfinite(v))


@dataclass
class _SymbolState:
    """Per-symbol rolling statistics and guard status."""

    returns: Deque[float] = field(default_factory=deque)
    spread: Deque[float] = field(default_factory=deque)
    last_close: float | None = None
    blocked: bool = False
    cooldown: int = 0
    reason: str | None = None
    last_trigger: Tuple[str, ...] = ()
    last_snapshot: Dict[str, object] = field(default_factory=dict)


class DynamicNoTradeGuard:
    """Online evaluator for dynamic no-trade rules."""

    def __init__(self, cfg: DynamicGuardConfig) -> None:
        self._cfg = cfg

        vol_cfg = getattr(cfg, "volatility", None)
        spread_cfg = getattr(cfg, "spread", None)

        sigma_window = int(getattr(vol_cfg, "window", None) or cfg.sigma_window or 120)
        if sigma_window <= 1:
            sigma_window = 120
        sigma_min = int(
            getattr(vol_cfg, "min_periods", None)
            or cfg.sigma_min_periods
            or min(sigma_window, max(2, sigma_window // 2))
        )
        if sigma_min <= 1:
            sigma_min = min(sigma_window, max(2, sigma_window // 2))

        spread_window = int(
            getattr(spread_cfg, "pctile_window", None)
            or cfg.spread_pctile_window
            or sigma_window
        )
        if spread_window <= 1:
            spread_window = max(2, sigma_window // 2)
        spread_min = int(
            getattr(spread_cfg, "pctile_min_periods", None)
            or cfg.spread_pctile_min_periods
            or min(spread_window, max(1, spread_window // 2))
        )
        if spread_min <= 0:
            spread_min = min(spread_window, max(1, spread_window // 2))

        sigma_upper = _finite_float(getattr(vol_cfg, "upper_multiplier", None))
        sigma_lower = _finite_float(getattr(vol_cfg, "lower_multiplier", None))
        if sigma_upper is None:
            sigma_upper = _finite_float(getattr(cfg, "sigma_k", None))
        spread_upper = _finite_float(getattr(spread_cfg, "upper_pctile", None))
        spread_lower = _finite_float(getattr(spread_cfg, "lower_pctile", None))
        if spread_upper is None:
            spread_upper = _finite_float(getattr(cfg, "spread_percentile", None))
        if spread_upper is None:
            spread_upper = _finite_float(getattr(cfg, "spread_pctile", None))

        spread_abs_upper = _finite_float(getattr(spread_cfg, "abs_bps", None))
        if spread_abs_upper is None:
            spread_abs_upper = _finite_float(getattr(cfg, "spread_abs_bps", None))
        spread_abs_lower = _finite_float(getattr(spread_cfg, "abs_release_bps", None))
        if spread_abs_lower is None:
            spread_abs_lower = _finite_float(getattr(cfg, "spread_abs_release_bps", None))

        hysteresis_ratio = _finite_float(getattr(cfg, "hysteresis", None))
        if hysteresis_ratio is not None and hysteresis_ratio < 0:
            hysteresis_ratio = 0.0

        if sigma_lower is None and sigma_upper is not None and hysteresis_ratio is not None:
            sigma_lower = max(0.0, sigma_upper * (1.0 - hysteresis_ratio))
        if spread_lower is None and spread_upper is not None and hysteresis_ratio is not None:
            spread_lower = max(0.0, spread_upper - hysteresis_ratio)

        if (
            spread_abs_lower is None
            and spread_abs_upper is not None
            and hysteresis_ratio is not None
        ):
            spread_abs_lower = max(0.0, spread_abs_upper * (1.0 - hysteresis_ratio))

        cooldown_candidates = [int(cfg.cooldown_bars or 0)]
        if vol_cfg is not None:
            cooldown_candidates.append(int(getattr(vol_cfg, "cooldown_bars", 0) or 0))
        if spread_cfg is not None:
            cooldown_candidates.append(int(getattr(spread_cfg, "cooldown_bars", 0) or 0))
        cooldown_bars = max(0, max(cooldown_candidates))

        self._sigma_window = sigma_window
        self._sigma_min = min(sigma_window, max(2, sigma_min))
        self._spread_window = spread_window
        self._spread_min = min(spread_window, max(1, spread_min))
        self._sigma_upper = sigma_upper
        self._sigma_lower = sigma_lower
        self._spread_upper = spread_upper
        self._spread_lower = spread_lower
        self._spread_abs_upper = spread_abs_upper
        self._spread_abs_lower = spread_abs_lower
        self._cooldown_bars = cooldown_bars

        self._needs_sigma = sigma_upper is not None
        self._needs_spread = spread_upper is not None or spread_abs_upper is not None
        self._states: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prewarm(self, symbol: str, bars: Sequence[object]) -> None:
        """Seed rolling windows with historical ``bars`` and optional spreads."""

        state = self._get_state(symbol)
        for entry in bars:
            bar_obj: Bar | None = None
            spread_val: float | None = None
            if isinstance(entry, Bar):
                bar_obj = entry
            elif isinstance(entry, tuple) and len(entry) == 2:
                first, second = entry
                if isinstance(first, Bar):
                    bar_obj = first
                    spread_val = _finite_float(second)
                elif isinstance(second, Bar):
                    bar_obj = second
                    spread_val = _finite_float(first)
            if bar_obj is None:
                continue
            self._update_from_bar(state, bar_obj, spread=spread_val, evaluate=False)

    def update(self, symbol: str, bar: Bar, spread: float | None) -> None:
        """Update guard state with the latest ``bar`` and optional ``spread``."""

        state = self._get_state(symbol)
        self._update_from_bar(state, bar, spread=spread, evaluate=True)

    def should_block(self, symbol: str) -> Tuple[bool, Optional[str], Mapping[str, object]]:
        """Return whether trading should be blocked for ``symbol``."""

        state = self._states.get(symbol)
        if state is None:
            return False, None, {}
        snapshot = dict(state.last_snapshot)
        return bool(state.blocked), state.reason, snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_state(self, symbol: str) -> _SymbolState:
        state = self._states.get(symbol)
        if state is None:
            state = _SymbolState(
                returns=deque(maxlen=self._sigma_window),
                spread=deque(maxlen=self._spread_window),
            )
            self._states[symbol] = state
        return state

    def _update_from_bar(
        self,
        state: _SymbolState,
        bar: Bar,
        *,
        spread: float | None,
        evaluate: bool,
    ) -> None:
        close = _to_float(bar.close)
        ret = float("nan")
        if math.isfinite(close) and state.last_close is not None:
            prev_close = float(state.last_close)
            if math.isfinite(prev_close) and prev_close != 0.0:
                ret = (close - prev_close) / prev_close
        elif state.last_close is None and math.isfinite(close):
            ret = float("nan")

        if math.isfinite(close):
            state.last_close = close

        if math.isfinite(ret):
            state.returns.append(ret)
        else:
            state.returns.append(float("nan"))

        spread_val = float("nan")
        if spread is not None:
            spread_val = _to_float(spread)
        if not math.isfinite(spread_val):
            alt = getattr(bar, "spread_bps", None)
            if alt is not None:
                spread_val = _to_float(alt)
        if not math.isfinite(spread_val):
            try:
                high_val = float(getattr(bar, "high", float("nan")))
                low_val = float(getattr(bar, "low", float("nan")))
            except Exception:
                high_val = low_val = float("nan")
            if math.isfinite(high_val) and math.isfinite(low_val):
                span = high_val - low_val
                if span > 0.0:
                    mid_val = (high_val + low_val) * 0.5
                    if not math.isfinite(mid_val) or mid_val == 0.0:
                        try:
                            mid_val = float(getattr(bar, "close", float("nan")))
                        except Exception:
                            mid_val = float("nan")
                    if math.isfinite(mid_val) and mid_val != 0.0:
                        spread_val = span / abs(mid_val) * 10000.0
        state.spread.append(spread_val if math.isfinite(spread_val) else float("nan"))

        sigma = _deque_std(state.returns, self._sigma_min)
        spread_pct = _deque_percentile(state.spread, self._spread_min)

        sigma_ready = _deque_count_finite(state.returns) >= self._sigma_min
        spread_ready = _deque_count_finite(state.spread) >= self._spread_min

        abs_ret = abs(ret) if math.isfinite(ret) else float("nan")
        sigma_ratio = float("nan")
        if sigma_ready and math.isfinite(sigma) and sigma > 0.0 and math.isfinite(abs_ret):
            sigma_ratio = abs_ret / sigma

        guard_ready = True
        if self._needs_sigma and not sigma_ready:
            guard_ready = False
        if self._needs_spread and not spread_ready:
            guard_ready = False

        trigger_reasons: List[str] = []
        if evaluate and guard_ready:
            if (
                self._sigma_upper is not None
                and math.isfinite(sigma_ratio)
                and sigma_ratio >= self._sigma_upper
            ):
                trigger_reasons.append("vol_extreme")
            if (
                self._spread_upper is not None
                and math.isfinite(spread_pct)
                and spread_pct >= self._spread_upper
            ):
                trigger_reasons.append("spread_wide")
            if (
                self._spread_abs_upper is not None
                and math.isfinite(spread_val)
                and spread_val >= self._spread_abs_upper
            ):
                trigger_reasons.append("spread_abs")

        if evaluate and guard_ready and trigger_reasons:
            state.blocked = True
            state.cooldown = max(state.cooldown, self._cooldown_bars)
            state.last_trigger = tuple(trigger_reasons)
            if "vol_extreme" in trigger_reasons:
                state.reason = "vol_extreme"
            elif "spread_abs" in trigger_reasons:
                state.reason = "spread_abs"
            else:
                state.reason = "spread_wide"
        elif evaluate and guard_ready and state.blocked:
            release_ready = True
            if "vol_extreme" in state.last_trigger and self._sigma_upper is not None:
                release_thr = (
                    self._sigma_lower
                    if self._sigma_lower is not None
                    else self._sigma_upper
                )
                if not (
                    math.isfinite(sigma_ratio)
                    and release_thr is not None
                    and sigma_ratio <= release_thr
                ):
                    release_ready = False
            if "spread_wide" in state.last_trigger and self._spread_upper is not None:
                release_thr = (
                    self._spread_lower
                    if self._spread_lower is not None
                    else self._spread_upper
                )
                if not (
                    math.isfinite(spread_pct)
                    and release_thr is not None
                    and spread_pct <= release_thr
                ):
                    release_ready = False
            if "spread_abs" in state.last_trigger and self._spread_abs_upper is not None:
                release_thr = (
                    self._spread_abs_lower
                    if self._spread_abs_lower is not None
                    else self._spread_abs_upper
                )
                if not (
                    math.isfinite(spread_val)
                    and release_thr is not None
                    and spread_val <= release_thr
                ):
                    release_ready = False

            if release_ready:
                if state.cooldown > 0:
                    state.cooldown -= 1
                    if state.reason:
                        state.reason = f"{state.reason}_cooldown"
                    else:
                        state.reason = "cooldown"
                else:
                    state.blocked = False
                    state.reason = None
                    state.last_trigger = ()
            else:
                state.cooldown = max(0, state.cooldown)
                if "vol_extreme" in state.last_trigger:
                    state.reason = "vol_extreme"
                elif "spread_wide" in state.last_trigger:
                    state.reason = "spread_wide"
                elif "spread_abs" in state.last_trigger:
                    state.reason = "spread_abs"
                if state.cooldown > 0:
                    if state.reason:
                        state.reason = f"{state.reason}_cooldown"
                    else:
                        state.reason = "cooldown"
        elif evaluate and not guard_ready:
            state.blocked = False
            state.cooldown = 0
            state.reason = None
            state.last_trigger = ()
        elif not evaluate:
            pass
        else:
            state.blocked = False
            state.cooldown = 0
            state.reason = None
            state.last_trigger = ()

        snapshot = {
            "sigma": sigma if math.isfinite(sigma) else None,
            "sigma_ready": sigma_ready,
            "abs_return": abs_ret if math.isfinite(abs_ret) else None,
            "sigma_k": sigma_ratio if math.isfinite(sigma_ratio) else None,
            "spread_bps": spread_val if math.isfinite(spread_val) else None,
            "spread": spread_val if math.isfinite(spread_val) else None,
            "spread_ready": spread_ready,
            "spread_percentile": spread_pct if math.isfinite(spread_pct) else None,
            "blocked": state.blocked,
            "cooldown": state.cooldown,
            "trigger_reasons": list(state.last_trigger),
            "reason": state.reason,
            "ready": guard_ready,
        }
        state.last_snapshot = snapshot

