# scripts/build_training_table.py
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import pandas as pd

from asof_join import AsofMerger, AsofSpec
from leakguard import LeakConfig, LeakGuard
from labels import LabelConfig, LabelBuilder


def _read_df(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    p = argparse.ArgumentParser(description="Build training table with asof-merge and leak protection.")
    p.add_argument("--base", required=True, help="CSV/Parquet: базовая таблица фичей/сигналов. Должна содержать ts_ms, symbol.")
    p.add_argument("--sources", nargs="*", default=[], help="JSON-строки спецификаций asof-источников. Пример: '{\"name\":\"book\",\"path\":\"book.parquet\",\"time_col\":\"ts_ms\",\"keys\":[\"symbol\"],\"direction\":\"backward\",\"tolerance_ms\":60000}'")
    p.add_argument("--prices", required=True, help="CSV/Parquet: таблица цен с колонками ts_ms, symbol, price (или другой столбец цен).")
    p.add_argument("--price-col", default="price", help="Название колонки цены в prices.")
    p.add_argument("--decision-delay-ms", type=int, default=0)
    p.add_argument("--label-horizon-ms", type=int, default=60000)
    p.add_argument("--label-returns", choices=["log", "arith"], default="log")
    p.add_argument("--out", required=True, help="Выходной CSV/Parquet (.parquet по расширению).")
    args = p.parse_args()

    base = _read_df(args.base)
    if "ts_ms" not in base.columns or "symbol" not in base.columns:
        raise SystemExit("base должен содержать 'ts_ms' и 'symbol'")

    # asof-источники
    specs: List[AsofSpec] = []
    for s in args.sources:
        obj = json.loads(s)
        df = _read_df(obj["path"])
        spec = AsofSpec(
            name=obj["name"],
            df=df,
            time_col=obj.get("time_col", "ts_ms"),
            keys=obj.get("keys", ["symbol"]),
            prefix=obj.get("prefix"),
            direction=obj.get("direction", "backward"),
            tolerance_ms=obj.get("tolerance_ms"),
            allow_exact_matches=bool(obj.get("allow_exact_matches", True)),
        )
        specs.append(spec)

    merger = AsofMerger(base_df=base, time_col="ts_ms", keys=["symbol"])
    merged = merger.merge(specs)

    # leakguard: добавим decision_ts
    lg = LeakGuard(LeakConfig(decision_delay_ms=int(args.decision_delay_ms), min_lookback_ms=0))
    merged = lg.attach_decision_time(merged, ts_col="ts_ms")

    # labels
    prices = _read_df(args.prices)
    lb = LabelBuilder(LabelConfig(horizon_ms=int(args.label_horizon_ms), price_col=args.price_col, returns=args.label_returns))
    out = lb.build(merged, prices, ts_col="ts_ms", symbol_col="symbol")

    # сохранить
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.out.lower().endswith(".parquet"):
        out.to_parquet(args.out, index=False)
    else:
        out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
