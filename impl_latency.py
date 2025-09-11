# -*- coding: utf-8 -*-
"""
impl_latency.py
Обёртка над latency.LatencyModel. Подключает задержки к симулятору.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Sequence
import json
import os
import importlib.util
import sysconfig
from pathlib import Path

import numpy as np

try:
    from utils.time import hour_of_week
except Exception:  # pragma: no cover - fallback
    hour_of_week = lambda ts: int((int(ts) // 3_600_000 + 72) % 168)  # type: ignore

_logging_spec = importlib.util.spec_from_file_location(
    "py_logging", Path(sysconfig.get_path("stdlib")) / "logging/__init__.py"
)
logging = importlib.util.module_from_spec(_logging_spec)
_logging_spec.loader.exec_module(logging)

try:
    from utils_time import load_hourly_seasonality
except Exception:  # pragma: no cover - fallback
    try:
        import pathlib, sys
        sys.path.append(str(pathlib.Path(__file__).resolve().parent))
        from utils_time import load_hourly_seasonality
    except Exception:  # pragma: no cover
        load_hourly_seasonality = lambda *a, **k: None  # type: ignore

logger = logging.getLogger(__name__)

try:
    from latency import LatencyModel
except Exception:  # pragma: no cover
    LatencyModel = None  # type: ignore


@dataclass
class LatencyCfg:
    base_ms: int = 250
    jitter_ms: int = 50
    spike_p: float = 0.01
    spike_mult: float = 5.0
    timeout_ms: int = 2500
    retries: int = 1
    seed: int = 0
    seasonality_path: str | None = None
    use_seasonality: bool = True
    seasonality_override: Sequence[float] | None = None
    seasonality_override_path: str | None = None
    seasonality_hash: str | None = None


class _LatencyWithSeasonality:
    """Wraps LatencyModel applying hourly multipliers and collecting stats."""

    def __init__(self, model: LatencyModel, multipliers: Sequence[float]):  # type: ignore[name-defined]
        self._model = model
        self._mult = (
            np.asarray(multipliers, dtype=float)
            if len(multipliers) == 168
            else np.ones(168, dtype=float)
        )
        self._mult_sum: List[float] = [0.0] * 168
        self._lat_sum: List[float] = [0.0] * 168
        self._count: List[int] = [0] * 168

    def sample(self, ts_ms: int | None = None):
        if ts_ms is None:
            return self._model.sample()
        hour = hour_of_week(int(ts_ms)) % len(self._mult)
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
        finally:
            self._model.base_ms, self._model.jitter_ms, self._model.timeout_ms = base, jitter, timeout
            if seed is not None:
                self._model.seed = seed
            if state_after is not None and hasattr(self._model, "_rng"):
                self._model._rng.setstate(state_after)
        self._mult_sum[hour] += m
        self._lat_sum[hour] += float(res.get("total_ms", 0))
        self._count[hour] += 1
        return res

    def stats(self):  # pragma: no cover - simple delegation
        return self._model.stats()

    def reset_stats(self) -> None:  # pragma: no cover - simple delegation
        self._model.reset_stats()
        self._mult_sum = [0.0] * 168
        self._lat_sum = [0.0] * 168
        self._count = [0] * 168

    def hourly_stats(self) -> Dict[str, List[float]]:
        avg_mult = [self._mult_sum[i] / self._count[i] if self._count[i] else 0.0 for i in range(168)]
        avg_lat = [self._lat_sum[i] / self._count[i] if self._count[i] else 0.0 for i in range(168)]
        return {"multiplier": avg_mult, "latency_ms": avg_lat, "count": list(self._count)}

class LatencyImpl:
    def __init__(self, cfg: LatencyCfg) -> None:
        self.cfg = cfg
        self._model = LatencyModel(
            base_ms=int(cfg.base_ms),
            jitter_ms=int(cfg.jitter_ms),
            spike_p=float(cfg.spike_p),
            spike_mult=float(cfg.spike_mult),
            timeout_ms=int(cfg.timeout_ms),
            retries=int(cfg.retries),
            seed=int(cfg.seed),
        ) if LatencyModel is not None else None
        self.latency: List[float] = [1.0] * 168
        path = cfg.seasonality_path or "configs/liquidity_latency_seasonality.json"
        self._has_seasonality = bool(cfg.use_seasonality)
        if self._has_seasonality:
            arr = load_hourly_seasonality(
                path, "latency", expected_hash=cfg.seasonality_hash
            )
            if arr is not None:
                self.latency = [float(x) for x in arr]
            else:
                logger.warning(
                    "Latency seasonality config %s not found; using default multipliers.",
                    path,
                )
                self._has_seasonality = False
            if not self._has_seasonality:
                logger.warning(
                    "Using default latency seasonality multipliers of 1.0; "
                    "run scripts/build_hourly_seasonality.py to generate them.",
                )
            override = cfg.seasonality_override
            o_path = cfg.seasonality_override_path
            if override is None and o_path:
                override = load_hourly_seasonality(o_path, "latency")
                if override is None:
                    logger.warning(
                        "Latency override %s not found or invalid; ignoring.", o_path
                    )
            if override is not None:
                arr = np.asarray(override, dtype=float)
                if len(arr) == 168:
                    self.latency = (
                        np.asarray(self.latency, dtype=float) * arr
                    ).tolist()
                else:
                    logger.warning(
                        "Latency override array must have length 168; ignoring."
                    )
        self.attached_sim = None
        self._wrapper: _LatencyWithSeasonality | None = None

    @property
    def model(self):
        return self._model

    def attach_to(self, sim) -> None:
        if self._model is not None:
            mult = self.latency if self._has_seasonality else [1.0] * 168
            self._wrapper = _LatencyWithSeasonality(self._model, mult)
            setattr(sim, "latency", self._wrapper)
        self.attached_sim = sim

    def get_stats(self):
        if self._wrapper is not None:
            return self._wrapper.stats()
        if self._model is None:
            return None
        return self._model.stats()

    def reset_stats(self) -> None:
        if self._wrapper is not None:
            self._wrapper.reset_stats()
        elif self._model is not None:
            self._model.reset_stats()

    def get_hourly_stats(self):
        if self._wrapper is None:
            return None
        return self._wrapper.hourly_stats()

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LatencyImpl":
        return LatencyImpl(LatencyCfg(
            base_ms=int(d.get("base_ms", 250)),
            jitter_ms=int(d.get("jitter_ms", 50)),
            spike_p=float(d.get("spike_p", 0.01)),
            spike_mult=float(d.get("spike_mult", 5.0)),
            timeout_ms=int(d.get("timeout_ms", 2500)),
            retries=int(d.get("retries", 1)),
            seed=int(d.get("seed", 0)),
            seasonality_path=d.get("seasonality_path"),
            use_seasonality=bool(d.get("use_seasonality", True)),
            seasonality_override=d.get("seasonality_override"),
            seasonality_override_path=d.get("seasonality_override_path"),
            seasonality_hash=d.get("seasonality_hash"),
        ))
