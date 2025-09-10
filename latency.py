# sim/latency.py
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


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
    """Wrapper around :class:`LatencyModel` applying hourly seasonality multipliers."""

    def __init__(self, model: LatencyModel, multipliers: Sequence[float]) -> None:
        if len(multipliers) != 168:
            raise ValueError("multipliers must have length 168")
        self._model = model
        self._mult: List[float] = [float(x) for x in multipliers]

    def sample(self, ts_ms: int) -> Dict[str, int | float | bool]:
        hour = ((int(ts_ms) // 3_600_000) + 72) % len(self._mult)
        m = float(self._mult[hour])
        base, jitter, timeout = (
            self._model.base_ms,
            self._model.jitter_ms,
            self._model.timeout_ms,
        )
        seed = getattr(self._model, "seed", None)
        state_after = None
        try:
            self._model.base_ms = int(round(base * m))
            self._model.jitter_ms = int(round(jitter * m))
            self._model.timeout_ms = int(round(timeout * m))
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

    def __getattr__(self, name: str):  # pragma: no cover - simple delegation
        return getattr(self._model, name)
