# scripts/calibrate_tcost.py
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


def _load_cfg(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if d:
        os.makedirs(d, exist_ok=True)


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
    """
    Простая NNLS-заглушка: сначала обычный lstsq, затем обрезаем <0 до 0.
    """
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
    """
    Возвращает (params, stats) где:
      params: base_bps, alpha_vol, beta_illiquidity (см. описание ниже)
      stats: rmse, mae, r2, n
    Модель калибровки:
      y_bps ≈ a0 + a1 * vol_bps + a2 * illq_ratio
      затем base_bps = a0; alpha_vol = a1; beta_illiquidity = a2 / base_bps (если base_bps>0)
    """
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


def main():
    ap = argparse.ArgumentParser(description="Калибровка параметров T-cost (base_bps, alpha_vol, beta_illiquidity) по историческим данным.")
    ap.add_argument("--config", default="configs/sandbox.yaml", help="Путь к sandbox.yaml")
    ap.add_argument("--data", default="", help="Путь к данным (если пусто — берём из config.data.path)")
    ap.add_argument("--ts_col", default="", help="Колонка времени (по умолчанию из конфигурации)")
    ap.add_argument("--symbol_col", default="", help="Колонка символа (по умолчанию из конфигурации)")
    ap.add_argument("--price_col", default="", help="Колонка цены-референса (по умолчанию из конфигурации)")
    ap.add_argument("--vol_mode", choices=["hl", "ret"], default="", help="Режим волатильности (по умолчанию из dynamic_spread.vol_mode)")
    ap.add_argument("--target", choices=["hl", "oc"], default="hl", help="Целевая метрика спреда: hl=(H-L)/C, oc=|C-O|/C")
    ap.add_argument("--k", type=float, default=0.25, help="Коэффициент шкалирования таргета (например, 0.25 для hl)")
    ap.add_argument("--liq_col", default="", help="Колонка ликвидности (по умолчанию dynamic_spread.liq_col или 'volume')")
    ap.add_argument("--liq_ref", type=float, default=0.0, help="Опорная ликвидность (по умолчанию dynamic_spread.liq_ref)")
    ap.add_argument("--out_cfg", default="", help="Куда сохранить обновлённый sandbox.yaml (если пусто — перезапись входного)")
    ap.add_argument("--out_json", default="logs/tcost_calibration.json", help="Куда сохранить JSON с результатами")
    ap.add_argument("--dry_run", action="store_true", help="Не записывать конфиг, только расчёт")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    dpath = args.data or (((cfg.get("data") or {}).get("path")) or "")
    if not dpath or not os.path.exists(dpath):
        raise FileNotFoundError(f"Не найден файл данных: {dpath!r}")

    ts_col = args.ts_col or ((cfg.get("data") or {}).get("ts_col") or "ts_ms")
    sym_col = args.symbol_col or ((cfg.get("data") or {}).get("symbol_col") or "symbol")
    price_col = args.price_col or ((cfg.get("data") or {}).get("price_col") or "ref_price")

    dspread = cfg.get("dynamic_spread") or {}
    vol_mode = args.vol_mode or str(dspread.get("vol_mode", "hl"))
    liq_col = args.liq_col or str(dspread.get("liq_col", "number_of_trades"))
    liq_ref = float(args.liq_ref or float(dspread.get("liq_ref", 1000.0)))

    df = pd.read_parquet(dpath) if dpath.endswith(".parquet") else pd.read_csv(dpath)
    if sym_col not in df.columns:
        df[sym_col] = (cfg.get("symbol") or "BTCUSDT")

    # калибровка одна на все символы (или отфильтруй заранее по cfg.symbol)
    params, stats = calibrate(
        df=df,
        price_col=price_col,
        vol_mode=vol_mode,
        target_mode=args.target,
        target_k=float(args.k),
        liq_col=liq_col,
        liq_ref=float(liq_ref),
    )

    # обновление конфига
    out_cfg_path = args.out_cfg or args.config
    updated = dict(cfg)
    updated.setdefault("dynamic_spread", {})
    updated["dynamic_spread"]["base_bps"] = float(params["base_bps"])
    updated["dynamic_spread"]["alpha_vol"] = float(params["alpha_vol"])
    updated["dynamic_spread"]["beta_illiquidity"] = float(params["beta_illiquidity"])

    if not args.dry_run:
        with open(out_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(updated, f, sort_keys=False, allow_unicode=True)

    # отчёт
    report = {
        "config_in": os.path.abspath(args.config),
        "config_out": os.path.abspath(out_cfg_path),
        "data_path": os.path.abspath(dpath),
        "price_col": price_col,
        "vol_mode": vol_mode,
        "target": args.target,
        "k": float(args.k),
        "liq_col": liq_col,
        "liq_ref": float(liq_ref),
        "fitted_params": params,
        "stats": stats,
    }
    _ensure_dir(args.out_json)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
