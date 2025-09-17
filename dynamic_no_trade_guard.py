"""Stateful dynamic no-trade guard used by the live signal runner.

The guard mirrors the behaviour of the historical helper in :mod:`no_trade`
but works in an online, per-symbol fashion.  It keeps rolling statistics for
returns, ATR and spreads, applies configurable trigger thresholds and enforces
hysteresis/cooldown rules to avoid rapid toggling.
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


def _deque_mean(values: Deque[float], min_periods: int) -> float:
    """Simple moving average of ``values`` ignoring ``nan`` entries."""

    if min_periods <= 0:
        return float("nan")
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < min_periods or not finite:
        return float("nan")
    return sum(finite) / float(len(finite))


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


@dataclass
class _SymbolState:
    """Per-symbol rolling statistics and guard status."""

    returns: Deque[float] = field(default_factory=deque)
    true_range: Deque[float] = field(default_factory=deque)
    vol_metric: Deque[float] = field(default_factory=deque)
    spread: Deque[float] = field(default_factory=deque)
    last_close: float | None = None
    blocked: bool = False
    cooldown: int = 0
    reason: str | None = None
    last_trigger: Tuple[str, ...] = ()
    last_snapshot: Dict[str, float | int | str | List[str] | None] = field(default_factory=dict)


class DynamicNoTradeGuard:
    """Online evaluator for dynamic no-trade rules."""

    def __init__(self, cfg: DynamicGuardConfig) -> None:
        self._cfg = cfg
        self._sigma_window = max(1, int(cfg.sigma_window or 120))
        self._atr_window = max(1, int(cfg.atr_window or 14))
        self._sigma_min = min(self._sigma_window, max(2, self._sigma_window // 2))
        self._atr_min = min(self._atr_window, max(1, self._atr_window // 2))
        self._vol_pct_min = min(self._sigma_window, max(1, self._sigma_window // 5))
        self._spread_pct_min = min(self._atr_window, max(1, self._atr_window // 5))

        self._vol_abs = float(cfg.vol_abs) if cfg.vol_abs is not None else None
        self._vol_pctile = (
            float(cfg.vol_pctile) if cfg.vol_pctile is not None else None
        )
        self._spread_abs = (
            float(cfg.spread_abs_bps) if cfg.spread_abs_bps is not None else None
        )
        self._spread_pctile = (
            float(cfg.spread_pctile) if cfg.spread_pctile is not None else None
        )
        self._hysteresis = max(0.0, float(cfg.hysteresis or 0.0))
        self._cooldown_bars = max(0, int(cfg.cooldown_bars or 0))

        self._log_reason = bool(cfg.log_reason)
        self._has_thresholds = any(
            x is not None
            for x in (
                self._vol_abs,
                self._vol_pctile,
                self._spread_abs,
                self._spread_pctile,
            )
        )
        self._states: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prewarm(self, symbol: str, bars: Sequence[Bar]) -> None:
        """Seed rolling windows with historical ``bars``."""

        state = self._get_state(symbol)
        for bar in bars:
            self._update_from_bar(state, bar, spread=None, evaluate=False)

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
                true_range=deque(maxlen=self._atr_window),
                vol_metric=deque(maxlen=self._sigma_window),
                spread=deque(maxlen=self._atr_window),
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
        high = _to_float(bar.high)
        low = _to_float(bar.low)

        if math.isfinite(close) and state.last_close is not None and math.isfinite(state.last_close) and state.last_close != 0.0:
            ret = (close - state.last_close) / state.last_close
            if math.isfinite(ret):
                state.returns.append(ret)
        elif state.last_close is None and math.isfinite(close):
            # no return yet but keep window aligned by appending nan placeholder
            state.returns.append(float("nan"))

        if math.isfinite(high) and math.isfinite(low):
            tr = high - low
            if state.last_close is not None and math.isfinite(state.last_close):
                tr = max(tr, abs(high - state.last_close), abs(low - state.last_close))
            if math.isfinite(tr):
                state.true_range.append(tr)
            else:
                state.true_range.append(float("nan"))
        else:
            state.true_range.append(float("nan"))

        if math.isfinite(close):
            state.last_close = close

        sigma = _deque_std(state.returns, self._sigma_min)
        atr = _deque_mean(state.true_range, self._atr_min)
        atr_pct = float("nan")
        if math.isfinite(atr) and math.isfinite(close) and close != 0.0:
            atr_pct = atr / abs(close)

        vol_metric = sigma if math.isfinite(sigma) else atr_pct
        if math.isfinite(vol_metric):
            state.vol_metric.append(vol_metric)
        else:
            state.vol_metric.append(float("nan"))

        spread_val = float("nan")
        if spread is not None:
            spread_val = _to_float(spread)
        elif math.isfinite(atr_pct):
            spread_val = atr_pct * 10000.0
        state.spread.append(spread_val if math.isfinite(spread_val) else float("nan"))

        vol_pctile = _deque_percentile(state.vol_metric, self._vol_pct_min)
        spread_pctile = _deque_percentile(state.spread, self._spread_pct_min)

        snapshot = {
            "sigma": sigma,
            "atr": atr,
            "atr_pct": atr_pct,
            "vol_metric": vol_metric,
            "vol_pctile": vol_pctile,
            "spread": spread_val,
            "spread_pctile": spread_pctile,
            "blocked": state.blocked,
            "cooldown": state.cooldown,
            "trigger_reasons": list(state.last_trigger),
            "reason": state.reason,
        }

        if not evaluate or not self._has_thresholds:
            state.last_snapshot = snapshot
            return

        trigger_reasons: List[str] = []
        if self._vol_abs is not None and math.isfinite(vol_metric) and vol_metric >= self._vol_abs:
            trigger_reasons.append("vol_abs")
        if self._vol_pctile is not None and math.isfinite(vol_pctile) and vol_pctile >= self._vol_pctile:
            trigger_reasons.append("vol_pctile")
        if self._spread_abs is not None and math.isfinite(spread_val) and spread_val >= self._spread_abs:
            trigger_reasons.append("spread_abs")
        if self._spread_pctile is not None and math.isfinite(spread_pctile) and spread_pctile >= self._spread_pctile:
            trigger_reasons.append("spread_pctile")

        if trigger_reasons:
            state.blocked = True
            state.cooldown = max(state.cooldown, self._cooldown_bars)
            state.last_trigger = tuple(trigger_reasons)
            base_reason = ",".join(trigger_reasons)
            state.reason = f"trigger:{base_reason}"
        elif state.blocked:
            release_ready = True
            if self._vol_abs is not None and math.isfinite(vol_metric):
                release_thr = self._vol_abs * (1.0 - self._hysteresis)
                if vol_metric > release_thr:
                    release_ready = False
            if self._vol_pctile is not None and math.isfinite(vol_pctile):
                release_thr = max(0.0, self._vol_pctile - self._hysteresis)
                if vol_pctile > release_thr:
                    release_ready = False
            if self._spread_abs is not None and math.isfinite(spread_val):
                release_thr = self._spread_abs * (1.0 - self._hysteresis)
                if spread_val > release_thr:
                    release_ready = False
            if self._spread_pctile is not None and math.isfinite(spread_pctile):
                release_thr = max(0.0, self._spread_pctile - self._hysteresis)
                if spread_pctile > release_thr:
                    release_ready = False

            if release_ready:
                if state.cooldown > 0:
                    state.cooldown -= 1
                    trigger = ",".join(state.last_trigger) if state.last_trigger else ""
                    suffix = f"|trigger={trigger}" if trigger else ""
                    state.reason = f"hold:cooldown{suffix}" if self._log_reason else "hold:cooldown"
                else:
                    state.blocked = False
                    state.reason = None
                    state.last_trigger = ()
            else:
                trigger = ",".join(state.last_trigger) if state.last_trigger else ""
                suffix = f"|trigger={trigger}" if trigger and self._log_reason else ""
                state.reason = f"hold:hysteresis{suffix}" if self._log_reason else "hold:hysteresis"
        else:
            state.cooldown = 0
            state.reason = None
            state.last_trigger = ()

        snapshot.update(
            {
                "blocked": state.blocked,
                "cooldown": state.cooldown,
                "trigger_reasons": list(state.last_trigger),
                "reason": state.reason,
            }
        )
        state.last_snapshot = snapshot

