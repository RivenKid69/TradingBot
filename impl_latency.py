# -*- coding: utf-8 -*-
"""
impl_latency.py
Обёртка над latency.LatencyModel. Подключает задержки к симулятору.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

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
        self.attached_sim = None

    @property
    def model(self):
        return self._model

    def attach_to(self, sim) -> None:
        if self._model is not None:
            setattr(sim, "latency", self._model)
        self.attached_sim = sim

    def get_stats(self):
        if self._model is None:
            return None
        return self._model.stats()

    def reset_stats(self) -> None:
        if self._model is not None:
            self._model.reset_stats()

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
        ))
