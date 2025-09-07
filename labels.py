# data/labels.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class LabelConfig:
    """
    Построение меток без утечки:
      - horizon_ms: горизонт удержания после decision_ts
      - price_col: колонка цены из price_df для оценки результата (например, mid/close/mark)
      - returns: "log" или "arith"
    """
    horizon_ms: int = 60_000
    price_col: str = "price"
    returns: str = "log"


def _safe_price(x: float) -> float:
    try:
        v = float(x)
        if v <= 0.0 or not math.isfinite(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


class LabelBuilder:
    """
    Строит таргет, строго начиная с decision_ts.
    """
    def __init__(self, cfg: Optional[LabelConfig] = None):
        self.cfg = cfg or LabelConfig()

    def build(self, base_with_decision: pd.DataFrame, price_df: pd.DataFrame, *, ts_col: str = "ts_ms", symbol_col: str = "symbol") -> pd.DataFrame:
        """
        Возвращает датафрейм base_with_decision + колонки:
          - label_t1_ts
          - label_price0
          - label_price1
          - label_ret
        Правила:
          - t0 = decision_ts
          - t1 = decision_ts + horizon_ms
          - price0 = ближайшая вперёд (asof forward/nearest) цена >= t0
          - price1 = ближайшая вперёд цена >= t1
        """
        if "decision_ts" not in base_with_decision.columns:
            raise ValueError("ожидается колонка 'decision_ts'. Сначала вызови LeakGuard.attach_decision_time().")
        if ts_col not in price_df.columns:
            raise ValueError(f"price_df не содержит '{ts_col}'")
        if symbol_col not in base_with_decision.columns or symbol_col not in price_df.columns:
            raise ValueError(f"требуется колонка '{symbol_col}' в обоих датафреймах")

        base = base_with_decision.copy()
        px = price_df.copy()
        base = base.sort_values([symbol_col, "decision_ts"]).reset_index(drop=True)
        px = px.sort_values([symbol_col, ts_col]).reset_index(drop=True)

        # Для t0: ближайшая вперёд цена (allow_exact_matches=True, direction='forward')
        b0 = base[[symbol_col, "decision_ts"]].rename(columns={"decision_ts": ts_col})
        m0 = pd.merge_asof(
            b0,
            px[[symbol_col, ts_col, self.cfg.price_col]],
            by=[symbol_col],
            left_on=ts_col,
            right_on=ts_col,
            direction="forward",
            allow_exact_matches=True,
        ).rename(columns={self.cfg.price_col: "label_price0", ts_col: "label_t0_ts"})

        # Для t1: то же, но от decision_ts + horizon
        b1 = base[[symbol_col, "decision_ts"]].copy()
        b1[ts_col] = b1["decision_ts"].astype("int64") + int(self.cfg.horizon_ms)
        m1 = pd.merge_asof(
            b1,
            px[[symbol_col, ts_col, self.cfg.price_col]],
            by=[symbol_col],
            left_on=ts_col,
            right_on=ts_col,
            direction="forward",
            allow_exact_matches=True,
        ).rename(columns={self.cfg.price_col: "label_price1", ts_col: "label_t1_ts"})

        out = base.join(m0[["label_t0_ts", "label_price0"]])
        out = out.join(m1[["label_t1_ts", "label_price1"]])

        # посчитать доходность
        p0 = out["label_price0"].map(_safe_price)
        p1 = out["label_price1"].map(_safe_price)
        if self.cfg.returns.lower() == "log":
            ret = (p1 / p0).map(lambda x: math.log(x) if (isinstance(x, float) and math.isfinite(x)) else float("nan"))
        else:
            ret = p1 / p0 - 1.0
        out["label_ret"] = ret
        return out
