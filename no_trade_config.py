from __future__ import annotations

"""Utility helpers for loading no-trade configuration.

This module provides :func:`get_no_trade_config` which reads the ``no_trade``
section from a YAML file and returns a :class:`NoTradeConfig` object.  All
consumers should use this function so that the configuration is loaded from a
single source of truth.
"""

from typing import Dict, List

import yaml
from pydantic import BaseModel, Field


class NoTradeConfig(BaseModel):
    """Pydantic model for the ``no_trade`` section."""

    funding_buffer_min: int = 0
    daily_utc: List[str] = Field(default_factory=list)
    custom_ms: List[Dict[str, int]] = Field(default_factory=list)


def get_no_trade_config(path: str) -> NoTradeConfig:
    """Load :class:`NoTradeConfig` from ``path``.

    Parameters
    ----------
    path:
        Path to a YAML file containing a top-level ``no_trade`` section.
    """

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return NoTradeConfig(**(data.get("no_trade", {}) or {}))
