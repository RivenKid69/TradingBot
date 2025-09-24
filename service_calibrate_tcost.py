"""Calibration of T-cost parameters.

The heavy lifting previously lived in ``calibrate_tcost.py`` which mixed the
core logic with command line parsing.  This module keeps only the numerical
parts and exposes a small service style API with ``run`` and
``from_config`` helpers.

Usage from code::

    from service_calibrate_tcost import from_config
    report = from_config("configs/legacy_sandbox.yaml", out="logs/tcost.json")

Both ``run`` and ``from_config`` return the JSON serialisable report with
fitted parameters and statistics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from legacy_sandbox_config import load_config as load_sandbox_config


# ---------------------------------------------------------------------------
# Helper functions copied from the original script
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.fspath(path)) or "."
    os.makedirs(directory, exist_ok=True)


def _safe_abs_log_ret(df: pd.DataFrame, sym_col: str, ts_col: str, price_col: str) -> pd.Series:
    df = df.sort_values([sym_col, ts_col]).copy()
    prev = df.groupby(sym_col)[price_col].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.log(df[price_col].astype(float).values / prev.astype(float).values)
    ret = np.where(np.isfinite(ret), np.abs(ret), np.nan)
    return pd.Series(ret, index=df.index)


def _compute_vol_bps(df: pd.DataFrame, vol_mode: str, price_col: str) -> pd.Series:
    vol_mode = str(vol_mode).lower().strip()
    if vol_mode == "hl" and ("high" in df.columns) and ("low" in df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            vol = (df["high"].astype(float).values - df["low"].astype(float).values) / df[price_col].astype(float).values
        vol = np.where(np.isfinite(vol), np.maximum(0.0, vol), np.nan)
        return pd.Series(vol * 10000.0, index=df.index)
    if "ret_1m" in df.columns:
        v = np.abs(df["ret_1m"].astype(float).values) * 10000.0
        v = np.where(np.isfinite(v), v, np.nan)
        return pd.Series(v, index=df.index)
    # fallback: abs log return
    v = _safe_abs_log_ret(df, "symbol", "ts_ms", price_col) * 10000.0
    return v


def _compute_illq_ratio(df: pd.DataFrame, liq_col: str, liq_ref: float) -> pd.Series:
    liq_col = str(liq_col)
    if liq_col in df.columns:
        liq = df[liq_col].astype(float).values
    elif "volume" in df.columns:
        liq = df["volume"].astype(float).values
    else:
        liq = np.ones(len(df), dtype=float)
    liq_ref = float(liq_ref) if float(liq_ref) > 0 else 1.0
    ratio = np.maximum(0.0, (liq_ref - liq) / liq_ref)
    ratio = np.where(np.isfinite(ratio), ratio, 0.0)
    return pd.Series(ratio, index=df.index)


def _target_spread_bps(df: pd.DataFrame, price_col: str, mode: str, k: float) -> pd.Series:
    mode = str(mode).lower().strip()
    if mode == "hl":
        if ("high" not in df.columns) or ("low" not in df.columns):
            raise ValueError("Для target=hl требуются колонки 'high' и 'low'.")
        with np.errstate(divide="ignore", invalid="ignore"):
            y = (df["high"].astype(float).values - df["low"].astype(float).values) / df[price_col].astype(float).values
        y = np.where(np.isfinite(y), np.maximum(0.0, y), np.nan) * (10000.0 * float(k))
        return pd.Series(y, index=df.index)
    if mode == "oc":
        if ("open" not in df.columns) or ("close" not in df.columns):
            raise ValueError("Для target=oc требуются колонки 'open' и 'close'.")
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.abs(df["close"].astype(float).values - df["open"].astype(float).values) / df[price_col].astype(float).values
        y = np.where(np.isfinite(y), y, np.nan) * (10000.0 * float(k))
        return pd.Series(y, index=df.index)
    raise ValueError("Неизвестный режим target. Используйте 'hl' или 'oc'.")


def _fit_linear_nonneg(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Простая NNLS-заглушка: сначала lstsq, затем обрезаем <0 до 0."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef = np.maximum(0.0, coef)
    return coef


def calibrate(
    df: pd.DataFrame,
    *,
    price_col: str,
    vol_mode: str,
    target_mode: str,
    target_k: float,
    liq_col: str,
    liq_ref: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return fitted parameters and simple statistics."""
    y = _target_spread_bps(df, price_col, target_mode, target_k)
    vol_bps = _compute_vol_bps(df, vol_mode, price_col)
    illq_ratio = _compute_illq_ratio(df, liq_col, liq_ref)

    data = pd.DataFrame({
        "y": y,
        "vol_bps": vol_bps,
        "illq_ratio": illq_ratio,
    }).dropna()

    data = data[(data["y"] > 0) & np.isfinite(data["y"]) & np.isfinite(data["vol_bps"]) & np.isfinite(data["illq_ratio"])]
    if data.empty:
        raise ValueError("Недостаточно данных для калибровки (после очистки пусто).")

    X = np.column_stack([
        np.ones(len(data), dtype=float),
        data["vol_bps"].values.astype(float),
        data["illq_ratio"].values.astype(float),
    ])
    yv = data["y"].values.astype(float)

    coef = _fit_linear_nonneg(X, yv)
    a0, a1, a2 = float(coef[0]), float(coef[1]), float(coef[2])

    base_bps = max(0.0, a0)
    alpha_vol = max(0.0, a1)
    beta_illiquidity = (a2 / base_bps) if base_bps > 1e-12 else 0.0
    beta_illiquidity = max(0.0, beta_illiquidity)

    y_hat = X @ coef
    resid = yv - y_hat
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    ss_res = float(np.sum(resid ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    params = {
        "base_bps": float(base_bps),
        "alpha_vol": float(alpha_vol),
        "beta_illiquidity": float(beta_illiquidity),
    }
    stats = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n": int(len(data)),
    }
    return params, stats


# ---------------------------------------------------------------------------
# Service layer
# ---------------------------------------------------------------------------


@dataclass
class TCostCalibrateConfig:
    """Configuration for T-cost calibration."""

    sandbox_config: str
    out: str
    target: str = "hl"
    k: float = 0.25
    dry_run: bool = False


def run(cfg: TCostCalibrateConfig) -> Dict[str, object]:
    """Execute calibration according to ``cfg``.

    Returns the report that is also written to ``cfg.out``.
    """

    sandbox = load_sandbox_config(cfg.sandbox_config)
    dpath = sandbox.data.path if sandbox.data else ""
    if not dpath or not os.path.exists(dpath):
        raise FileNotFoundError(f"Не найден файл данных: {dpath!r}")

    ts_col = sandbox.data.ts_col if sandbox.data else "ts_ms"
    sym_col = sandbox.data.symbol_col if sandbox.data else "symbol"
    price_col = sandbox.data.price_col if sandbox.data else "ref_price"

    dspread = sandbox.dynamic_spread or {}
    vol_mode = str(dspread.get("vol_mode", "hl"))
    liq_col = str(dspread.get("liq_col", "number_of_trades"))
    liq_ref = float(dspread.get("liq_ref", 1000.0))

    df = pd.read_parquet(dpath) if dpath.endswith(".parquet") else pd.read_csv(dpath)
    if sym_col not in df.columns:
        df[sym_col] = sandbox.symbol

    params, stats = calibrate(
        df=df,
        price_col=price_col,
        vol_mode=vol_mode,
        target_mode=cfg.target,
        target_k=float(cfg.k),
        liq_col=liq_col,
        liq_ref=float(liq_ref),
    )

    out_cfg_path = cfg.sandbox_config
    updated = sandbox.model_dump()
    updated.setdefault("dynamic_spread", {})
    updated["dynamic_spread"]["base_bps"] = float(params["base_bps"])
    updated["dynamic_spread"]["alpha_vol"] = float(params["alpha_vol"])
    updated["dynamic_spread"]["beta_illiquidity"] = float(params["beta_illiquidity"])

    if not cfg.dry_run:
        with open(out_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(updated, f, sort_keys=False, allow_unicode=True)

    report = {
        "config_in": os.path.abspath(cfg.sandbox_config),
        "config_out": os.path.abspath(out_cfg_path),
        "data_path": os.path.abspath(dpath),
        "price_col": price_col,
        "vol_mode": vol_mode,
        "target": cfg.target,
        "k": float(cfg.k),
        "liq_col": liq_col,
        "liq_ref": float(liq_ref),
        "fitted_params": params,
        "stats": stats,
    }

    _ensure_dir(cfg.out)
    with open(cfg.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def from_config(config_path: str, *, out: str) -> Dict[str, object]:
    """Convenience wrapper used by CLI scripts."""

    cfg = TCostCalibrateConfig(sandbox_config=config_path, out=out)
    return run(cfg)


__all__ = [
    "TCostCalibrateConfig",
    "calibrate",
    "run",
    "from_config",
]

