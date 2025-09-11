# sim/latency.py
from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence
import threading
import logging

from utils_time import SEASONALITY_MULT_MAX


logger = logging.getLogger(__name__)
seasonality_logger = logging.getLogger("seasonality").getChild(__name__)


def validate_multipliers(
    multipliers: Sequence[float], *, expected_len: int = 168, cap: float = SEASONALITY_MULT_MAX
) -> List[float]:
    """Return ``multipliers`` as a list after validation.

    Parameters
    ----------
    multipliers:
        Sequence of multiplier values to validate.
    expected_len:
        Required length of the sequence. Defaults to ``168``.
    cap:
        Maximum allowed value for each multiplier. Values must also be
        positive and finite.

    Returns
    -------
    List[float]
        Validated multiplier values.

    Raises
    ------
    ValueError
        If the sequence length differs from ``expected_len`` or any value is
        non-finite, non-positive or exceeds ``cap``.
    """

    arr = [float(x) for x in multipliers]
    if len(arr) != int(expected_len):
        raise ValueError(f"multipliers must have length {expected_len}")
    for i, v in enumerate(arr):
        if not math.isfinite(v):
            raise ValueError(f"multipliers[{i}] is not finite")
        if v <= 0:
            raise ValueError(f"multipliers[{i}] must be positive")
        if v > cap:
            raise ValueError(f"multipliers[{i}] {v} exceeds cap {cap}")
    return arr


@dataclass
class LatencyModel:
    """
    Простейшая стохастическая модель латентности с редкими «спайками» и таймаутами.

    Механика:
      total_ms = base_ms + U[0, jitter_ms]
      с вероятностью spike_p → total_ms *= spike_mult
      timeout = (total_ms > timeout_ms)
      если timeout и retries > 0 → повторить выбор ещё до retries раз, суммируя total_ms

    Возвращаемый словарь:
      {
        "total_ms": int,     # суммарная задержка по успешной попытке или по последней, если таймаут после всех ретраев
        "spike": bool,       # был ли спайк на успешной попытке (или последней попытке, если неуспех)
        "timeout": bool,     # True если после всех попыток остался таймаут (считай ордер не прошёл)
        "attempts": int      # сколько было попыток
      }
    """
    base_ms: int = 250
    jitter_ms: int = 50
    spike_p: float = 0.01
    spike_mult: float = 5.0
    timeout_ms: int = 2500
    retries: int = 1
    seed: int = 0
    # Accumulators for latency statistics
    lat_samples: List[int] = field(default_factory=list)
    timeouts: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(int(self.seed))

    def _one_draw(self) -> Dict[str, int | float | bool]:
        base = max(0, int(self.base_ms))
        jitter = max(0, int(self.jitter_ms))
        t = base + (self._rng.randint(0, jitter) if jitter > 0 else 0)
        is_spike = (self._rng.random() < float(self.spike_p))
        if is_spike:
            t = int(t * float(self.spike_mult))
        timeout = (t > int(self.timeout_ms))
        return {"total_ms": int(t), "spike": bool(is_spike), "timeout": bool(timeout)}

    def sample(self) -> Dict[str, int | float | bool]:
        """
        Выполнить серию попыток с ретраями. Суммируем задержки всех попыток.
        Если после всех попыток timeout=True — считаем, что запрос не удался.
        """
        attempts = 0
        agg_ms = 0
        spike_on_success = False
        last_timeout = False

        while True:
            d = self._one_draw()
            attempts += 1
            agg_ms += int(d["total_ms"])
            last_timeout = bool(d["timeout"])
            spike_on_success = spike_on_success or bool(d["spike"])
            if not last_timeout:
                break
            if attempts > int(self.retries) + 1:
                break

        result = {
            "total_ms": int(agg_ms),
            "spike": bool(spike_on_success),
            "timeout": bool(last_timeout),
            "attempts": int(attempts),
        }
        # Update statistics accumulators
        self.lat_samples.append(int(agg_ms))
        if last_timeout:
            self.timeouts += 1
        return result

    def stats(self) -> Dict[str, float]:
        """Return latency statistics."""
        n = len(self.lat_samples)
        if n == 0:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "timeout_rate": 0.0}
        sorted_samples = sorted(self.lat_samples)
        # Helper to compute percentile with linear interpolation
        def percentile(p: float) -> float:
            if n == 1:
                return float(sorted_samples[0])
            k = (n - 1) * p
            f = int(k)
            c = min(f + 1, n - 1)
            if f == c:
                return float(sorted_samples[f])
            return float(sorted_samples[f] + (sorted_samples[c] - sorted_samples[f]) * (k - f))

        p50 = percentile(0.5)
        p95 = percentile(0.95)
        timeout_rate = float(self.timeouts) / n
        return {"p50_ms": p50, "p95_ms": p95, "timeout_rate": timeout_rate}

    def reset_stats(self) -> None:
        """Reset collected latency statistics."""
        self.lat_samples.clear()
        self.timeouts = 0


class SeasonalLatencyModel:
    """Wrapper around :class:`LatencyModel` applying hourly seasonality multipliers.

    The :meth:`sample` method is thread-safe and can be called from multiple
    threads concurrently. An internal lock guards access to the underlying
    ``LatencyModel`` to prevent races when its parameters are temporarily
    adjusted for seasonality.
    """

    def __init__(self, model: LatencyModel, multipliers: Sequence[float]) -> None:
        self._model = model
        self._mult = validate_multipliers(multipliers, expected_len=168)
        self._lock = threading.Lock()

    def sample(self, ts_ms: int) -> Dict[str, int | float | bool]:
        hour = ((int(ts_ms) // 3_600_000) + 72) % len(self._mult)
        m = float(self._mult[hour])
        with self._lock:
            base, jitter, timeout = (
                self._model.base_ms,
                self._model.jitter_ms,
                self._model.timeout_ms,
            )
            seed = getattr(self._model, "seed", None)
            state_after = None
            try:
                scaled_base = int(round(base * m))
                if scaled_base > timeout:
                    seasonality_logger.warning(
                        "scaled base_ms %s exceeds timeout_ms %s; capping",
                        scaled_base,
                        timeout,
                    )
                    scaled_base = timeout
                self._model.base_ms = scaled_base
                self._model.jitter_ms = int(round(jitter * m))
                res = self._model.sample()
                if hasattr(self._model, "_rng"):
                    state_after = self._model._rng.getstate()
                return res
            finally:
                self._model.base_ms = base
                self._model.jitter_ms = jitter
                self._model.timeout_ms = timeout
                if seed is not None:
                    self._model.seed = seed
                if state_after is not None and hasattr(self._model, "_rng"):
                    self._model._rng.setstate(state_after)

    def update_multipliers(self, multipliers: Sequence[float]) -> None:
        """Atomically replace the internal multipliers array."""

        arr = validate_multipliers(multipliers, expected_len=len(self._mult))
        with self._lock:
            self._mult = arr

    def __getattr__(self, name: str):  # pragma: no cover - simple delegation
        return getattr(self._model, name)
