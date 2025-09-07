# training/tcost.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from legacy_sandbox_config import load_config as load_sandbox_config


class DynSpreadCfg(BaseModel):
    base_bps: float = 3.0
    alpha_vol: float = 0.5
    beta_illiquidity: float = 1.0
    vol_mode: str = "hl"
    liq_col: str = "number_of_trades"
    liq_ref: float = 1000.0
    min_bps: float = 1.0
    max_bps: float = 25.0


def load_dyn_spread_config(path: str) -> DynSpreadCfg:
    cfg = load_sandbox_config(path)
    return DynSpreadCfg(**(cfg.dynamic_spread or {}))


def _vol_factor(row: pd.Series, *, ref: float, vol_mode: str, last_ref: Optional[float]) -> Tuple[float, float]:
    """
    Возвращает (vol_factor, new_last_ref).
    vol_factor — безразмерный (доли), не bps.
    """
    try:
        if vol_mode.lower() == "hl" and ("high" in row.index) and ("low" in row.index):
            hi = float(row["high"])
            lo = float(row["low"])
            if ref > 0:
                return max(0.0, (hi - lo) / float(ref)), ref
    except Exception:
        pass
    # fallback: |log return| к предыдущей цене
    if last_ref is None or last_ref <= 0 or ref <= 0:
        return 0.0, ref
    return abs(math.log(float(ref) / float(last_ref))), ref


def _liquidity(row: pd.Series, *, liq_col: str) -> float:
    try:
        if liq_col in row.index:
            return float(row[liq_col])
        if "volume" in row.index:
            return float(row["volume"])
    except Exception:
        pass
    return 1.0


def dyn_spread_bps_row(
    row: pd.Series,
    *,
    cfg: DynSpreadCfg,
    last_ref: Optional[float],
    ref_price_col: str = "ref_price",
) -> Tuple[float, float]:
    """
    Возвращает (spread_bps_clamped, new_last_ref).
    """
    ref = float(row[ref_price_col])
    vf, new_last_ref = _vol_factor(row, ref=ref, vol_mode=cfg.vol_mode, last_ref=last_ref)
    liq = _liquidity(row, liq_col=cfg.liq_col)
    base = float(cfg.base_bps)
    vol_term = float(cfg.alpha_vol) * float(vf) * 10000.0
    ratio = 0.0
    if float(cfg.liq_ref) > 0 and liq == liq:  # NaN check
        ratio = max(0.0, (float(cfg.liq_ref) - float(liq)) / float(cfg.liq_ref))
    illq = float(cfg.beta_illiquidity) * ratio * base
    spread_bps = base + vol_term + illq
    spread_bps = max(float(cfg.min_bps), min(float(cfg.max_bps), float(spread_bps)))
    return float(spread_bps), new_last_ref


def effective_return_series(
    df: pd.DataFrame,
    *,
    horizon_bars: int,
    fees_bps_total: float,
    sandbox_yaml_path: str,
    ts_col: str = "ts_ms",
    symbol_col: str = "symbol",
    ref_price_col: str = "ref_price",
    label_threshold: Optional[float] = None,
    roundtrip_spread: bool = True,
) -> pd.DataFrame:
    """
    Считает r_eff и (опционально) бинарную метку по всему DataFrame.
    Добавляет колонки:
      - eff_ret_<h>
      - slippage_bps
      - fees_bps_total
      - y_eff_<h>  (если задан threshold)
    """
    cfg = load_dyn_spread_config(sandbox_yaml_path)

    # сортировка и группировка
    df = df.sort_values([symbol_col, ts_col]).reset_index(drop=True)
    # будущее значение
    future_price = df.groupby(symbol_col)[ref_price_col].shift(-int(horizon_bars))
    ref = pd.to_numeric(df[ref_price_col], errors="coerce").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ret = (future_price - ref) / ref

    # динамический спред по каждой строке
    slipp = np.full(len(df), np.nan, dtype=float)
    last_ref_by_sym: Dict[str, Optional[float]] = {}
    syms = df[symbol_col].astype(str).tolist()
    refs = ref.to_numpy()
    for i in range(len(df)):
        sym = syms[i]
        last = last_ref_by_sym.get(sym)
        row = df.iloc[i]
        spread_bps, new_last = dyn_spread_bps_row(row, cfg=cfg, last_ref=last, ref_price_col=ref_price_col)
        # как обсуждали: используем spread_bps как оценку round-trip спред-стоимости.
        slipp[i] = spread_bps if roundtrip_spread else (spread_bps * 0.5)
        last_ref_by_sym[sym] = new_last

    slippage_bps = pd.Series(slipp, index=df.index)

    # итоговый эффективный ретёрн
    r_eff = raw_ret - (float(fees_bps_total) * 1e-4) - (slippage_bps * 1e-4)

    out = df.copy()
    out[f"eff_ret_{horizon_bars}"] = r_eff.astype(float)
    out["slippage_bps"] = slippage_bps.astype(float)
    out["fees_bps_total"] = float(fees_bps_total)

    if label_threshold is not None:
        thr = float(label_threshold)
        out[f"y_eff_{horizon_bars}"] = (out[f"eff_ret_{horizon_bars}"] > thr).astype(int)

    # убираем «хвост» без будущей цены
    valid_mask = future_price.notna() & np.isfinite(out[f"eff_ret_{horizon_bars}"])
    return out.loc[valid_mask].reset_index(drop=True)
