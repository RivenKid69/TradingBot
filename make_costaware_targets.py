# scripts/make_costaware_targets.py
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from trainingtcost import effective_return_series


class SimFeesConfig(BaseModel):
    fees_bps_total: Optional[float] = None
    fees: Dict[str, Any] = Field(default_factory=dict)


def load_sim_fees_config(path: str) -> SimFeesConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return SimFeesConfig(**data)


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Неизвестный формат файла данных: {ext}")


def _write_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
        return
    if ext in (".csv", ".txt"):
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Неизвестный формат файла вывода: {ext}")


def _try_read_fees_bps_total(sim_yaml_path: Optional[str]) -> Optional[float]:
    """
    Пытаемся считать комиссии из configs/config_sim.yaml.
    Поддерживаем варианты:
      fees_bps_total: <float>      # уже готовая «кругорейсовая» комиссия
      fees: { maker_bps, taker_bps, roundtrip_mode: "taker"|"maker"|"mixed" }
    Если ничего не нашли — вернём None.
    """
    if not sim_yaml_path or not os.path.exists(sim_yaml_path):
        return None
    try:
        cfg = load_sim_fees_config(sim_yaml_path)
    except Exception:
        return None

    if cfg.fees_bps_total is not None:
        try:
            return float(cfg.fees_bps_total)
        except Exception:
            pass

    fees = cfg.fees
    try:
        maker = float(fees.get("maker_bps", 0.0))
        taker = float(fees.get("taker_bps", 0.0))
        mode = str(fees.get("roundtrip_mode", "taker")).lower()
        if mode == "maker":
            return 2.0 * maker
        if mode == "mixed":
            return maker + taker
        # по умолчанию — обе ноги как тейкер:
        return 2.0 * taker
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Сделать cost-aware таргет (eff_ret_h и, опционально, y_eff_h) с учётом комиссий и динамич. спреда.")
    ap.add_argument("--data", required=True, help="Путь к входным данным (CSV/Parquet) с колонками ts_ms,symbol,ref_price,(high/low или ret_1m), ликвидность (number_of_trades или volume)")
    ap.add_argument("--out", default="", help="Путь к выходному файлу (CSV/Parquet). По умолчанию — рядом с суффиксом _costaware.")
    ap.add_argument("--sandbox_config", default="configs/legacy_sandbox.yaml", help="Путь к legacy_sandbox.yaml (берём dynamic_spread)")
    ap.add_argument("--sim_config", default="configs/config_sim.yaml", help="Путь к config_sim.yaml (пытаемся взять комиссии)")
    ap.add_argument("--fees_bps_total", type=float, default=None, help="Кругорейсовая комиссия в bps (перебивает sim.yaml)")
    ap.add_argument("--horizon_bars", type=int, default=60, help="Горизонт таргета в барах")
    ap.add_argument("--threshold", type=float, default=None, help="Порог для бинарной метки (если задан — добавим y_eff_h)")
    ap.add_argument("--ts_col", default="ts_ms", help="Колонка метки времени")
    ap.add_argument("--symbol_col", default="symbol", help="Колонка символа")
    ap.add_argument("--price_col", default="ref_price", help="Колонка референс-цены")
    ap.add_argument("--roundtrip_spread", action="store_true", help="Использовать spread_bps как полную кругорейсовую издержку (по умолчанию так и делаем)")
    args = ap.parse_args()

    df = _read_table(args.data)

    fees_total = args.fees_bps_total
    if fees_total is None:
        fees_total = _try_read_fees_bps_total(args.sim_config)
    if fees_total is None:
        # разумная дефолтная комиссия (пример для USDT-фьючей как тейкер с обеих ног ~ 5 bps * 2 = 10 bps)
        fees_total = 10.0

    out = effective_return_series(
        df,
        horizon_bars=int(args.horizon_bars),
        fees_bps_total=float(fees_total),
        sandbox_yaml_path=args.sandbox_config,
        ts_col=args.ts_col,
        symbol_col=args.symbol_col,
        ref_price_col=args.price_col,
        label_threshold=(float(args.threshold) if args.threshold is not None else None),
        roundtrip_spread=bool(args.roundtrip_spread or True),
    )

    out_path = args.out.strip()
    if not out_path:
        base, ext = os.path.splitext(args.data)
        out_path = f"{base}_costaware{ext if ext.lower() in ('.csv', '.parquet', '.pq', '.txt') else '.parquet'}"

    _write_table(out, out_path)
    print(f"Готово. Записано: {out_path}")
    print(f"Добавлены колонки: eff_ret_{int(args.horizon_bars)}, slippage_bps, fees_bps_total"
          + (f", y_eff_{int(args.horizon_bars)}" if args.threshold is not None else ""))
    

if __name__ == "__main__":
    main()
