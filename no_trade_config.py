from __future__ import annotations

"""Utility helpers for loading no-trade configuration.

This module provides :func:`get_no_trade_config` which reads the ``no_trade``
section from a YAML file and returns a :class:`NoTradeConfig` object.  All
consumers should use this function so that the configuration is loaded from a
single source of truth.
"""

from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class DynamicGuardConfig(BaseModel):
    """Configuration for a dynamic no-trade guard."""

    enable: bool = False
    sigma_window: Optional[int] = None
    atr_window: Optional[int] = None
    vol_abs: Optional[float] = None
    vol_pctile: Optional[float] = None
    spread_abs_bps: Optional[float] = None
    spread_pctile: Optional[float] = None
    hysteresis: Optional[float] = None
    cooldown_bars: int = 0
    log_reason: bool = False


class NoTradeConfig(BaseModel):
    """Pydantic model for the ``no_trade`` section."""

    funding_buffer_min: int = 0
    daily_utc: List[str] = Field(default_factory=list)
    custom_ms: List[Dict[str, int]] = Field(default_factory=list)
    dynamic_guard: Optional[DynamicGuardConfig] = None


def get_no_trade_config(path: str) -> NoTradeConfig:
    """Load :class:`NoTradeConfig` from ``path``.

    Parameters
    ----------
    path:
        Path to a YAML file containing a top-level ``no_trade`` section.
    """

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    config = NoTradeConfig(**(data.get("no_trade", {}) or {}))
    if config.dynamic_guard is None:
        config.dynamic_guard = DynamicGuardConfig()
    return config
