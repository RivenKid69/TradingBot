"""Service for slippage coefficient calibration.

The previous implementation lived in ``calibrate_slippage.py`` and handled
argument parsing on its own.  The logic is preserved here and exposed through
``run``/``from_config`` helpers so that it can be reused from other
applications.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yaml

from slippage import SlippageConfig


def fit_k_closed_form(df: pd.DataFrame) -> float:
    """Closed form fit for the impact coefficient ``k``.

    The model assumes

    ``observed_slip_bps - half_spread_bps ≈ k * vol_factor * sqrt(size/liquidity)``
    and solves it in least squares sense.
    """

    d = df.copy()
    d = d[(d["size"] > 0) & (d["liquidity"] > 0)]
    if d.empty:
        return 0.8
    x = d["vol_factor"] * (d["size"] / d["liquidity"]).pow(0.5)
    y = d["observed_slip_bps"] - d["half_spread_bps"]
    num = float((x * y).sum())
    den = float((x * x).sum())
    if den <= 0.0 or not pd.notna(den):
        return 0.8
    k = num / den
    if not pd.notna(k):
        k = 0.8
    return float(max(0.0, k))


@dataclass
class SlippageCalibrateConfig:
    """Configuration for slippage calibration."""

    trades: str
    out: str
    fmt: Optional[str] = None
    min_half_spread_bps: float = 0.0
    default_spread_bps: float = 2.0


def run(cfg: SlippageCalibrateConfig) -> float:
    """Load trades, fit coefficient and dump JSON report."""

    fmt = cfg.fmt
    path = cfg.trades
    if fmt is None:
        fmt = "parquet" if path.lower().endswith(".parquet") else "csv"
    if fmt == "csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    cfg_slip = SlippageConfig(
        k=0.8,
        min_half_spread_bps=float(cfg.min_half_spread_bps),
        default_spread_bps=float(cfg.default_spread_bps),
    )
    df = df.copy()
    if "half_spread_bps" not in df.columns:
        df["half_spread_bps"] = df["spread_bps"].fillna(cfg_slip.default_spread_bps) * 0.5
        df["half_spread_bps"] = df["half_spread_bps"].clip(lower=cfg_slip.min_half_spread_bps)

    k = fit_k_closed_form(df)

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w", encoding="utf-8") as f:
        json.dump({"k": float(k)}, f, ensure_ascii=False, indent=2, sort_keys=True)

    return float(k)


def from_config(path: str, *, out: str) -> float:
    """Load YAML configuration and execute calibration."""

    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}

    cfg = SlippageCalibrateConfig(
        trades=d["trades"],
        out=out,
        fmt=d.get("format"),
        min_half_spread_bps=float(d.get("min_half_spread_bps", 0.0)),
        default_spread_bps=float(d.get("default_spread_bps", 2.0)),
    )
    return run(cfg)


__all__ = [
    "SlippageCalibrateConfig",
    "fit_k_closed_form",
    "run",
    "from_config",
]

