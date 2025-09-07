# scripts/check_feature_parity.py
from __future__ import annotations

import argparse
from typing import Any, Dict, List

import pandas as pd

from transformers import FeatureSpec, OnlineFeatureTransformer, apply_offline_features


def _read_any(path: str) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Сравнить оффлайн и онлайн расчёт фич на одном датасете.")
    ap.add_argument("--data", required=True, help="CSV или Parquet с колонками ts_ms,symbol,price")
    ap.add_argument("--threshold", type=float, default=1e-9,
                    help="Допустимое абсолютное отклонение")
    ap.add_argument("--price-col", default="price", help="Имя колонки цены")
    ap.add_argument("--lookbacks", default="5,15,60",
                    help="Окна SMA/ret через запятую")
    ap.add_argument("--rsi-period", type=int, default=14,
                    help="Период RSI (Wilder)")
    args = ap.parse_args()

    df = _read_any(args.data)
    if df is None or df.empty:
        raise SystemExit("входной файл пуст")

    lookbacks = [int(s.strip()) for s in str(args.lookbacks).split(",") if s.strip()]
    spec = FeatureSpec(lookbacks_prices=lookbacks, rsi_period=int(args.rsi_period))

    offline = apply_offline_features(
        df, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col=args.price_col
    )

    d = df[["ts_ms", "symbol", args.price_col]].dropna().copy()
    d["ts_ms"] = d["ts_ms"].astype("int64")
    d["symbol"] = d["symbol"].astype(str)
    d = d.sort_values(["symbol", "ts_ms"]).reset_index(drop=True)

    tr = OnlineFeatureTransformer(spec)
    rows: List[Dict[str, Any]] = []
    for row in d.itertuples(index=False):
        price = getattr(row, args.price_col)
        rows.append(tr.update(symbol=row.symbol, ts_ms=row.ts_ms, close=price))
    online = pd.DataFrame(rows)

    merged = offline.merge(online, on=["ts_ms", "symbol"], suffixes=("_off", "_on"))
    features = [c for c in offline.columns if c not in ("ts_ms", "symbol")]

    diffs: List[str] = []
    for feat in features:
        diff = (merged[f"{feat}_off"] - merged[f"{feat}_on"]).abs()
        mask = diff > float(args.threshold)
        if mask.any():
            for idx in merged.index[mask]:
                diffs.append(
                    f"ts={merged.loc[idx, 'ts_ms']} symbol={merged.loc[idx, 'symbol']} "
                    f"feat={feat} offline={merged.loc[idx, f'{feat}_off']} "
                    f"online={merged.loc[idx, f'{feat}_on']} diff={diff.loc[idx]}"
                )

    if diffs:
        print("Найдены расхождения выше порога:")
        for line in diffs:
            print(line)
    else:
        print("Паритет подтверждён: различий выше порога не найдено.")


if __name__ == "__main__":
    main()
