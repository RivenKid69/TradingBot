# features/transformers.py
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class FeatureSpec:
    """
    Единая спецификация фич:
      - lookbacks_prices: окна для SMA и лог-ретёрнов (в минутах для 1m входа)
      - rsi_period: период RSI по Вайльдеру (EMA-уподоблённое сглаживание)
    """

    lookbacks_prices: List[int]
    rsi_period: int = 14

    def __post_init__(self) -> None:
        if (
            not isinstance(self.lookbacks_prices, list)
            or len(self.lookbacks_prices) == 0
        ):
            self.lookbacks_prices = [5, 15, 60]
        self.lookbacks_prices = [
            int(abs(x)) for x in self.lookbacks_prices if int(abs(x)) > 0
        ]
        self.rsi_period = int(self.rsi_period)


class OnlineFeatureTransformer:
    """
    Онлайн-трансформер: состояние на символ, детерминистичное поведение.
    Полностью соответствует онлайновой логике (как раньше в FeaturePipe):
      - SMA и ретёрны из окна цен (1 точка в минуту)
      - RSI по Вайльдеру: скользящие avg_gain/avg_loss с периодом p
    """

    def __init__(self, spec: FeatureSpec) -> None:
        self.spec = spec
        self._state: Dict[str, Dict[str, Any]] = {}

    def _ensure_state(self, symbol: str) -> Dict[str, Any]:
        st = self._state.get(symbol)
        if st is None:
            maxlen = max(self.spec.lookbacks_prices + [self.spec.rsi_period + 1])
            st = {
                "prices": deque(maxlen=maxlen),  # type: deque[float]
                "avg_gain": None,  # type: Optional[float]
                "avg_loss": None,  # type: Optional[float]
                "last_close": None,  # type: Optional[float]
            }
            self._state[symbol] = st
        return st

    def update(self, *, symbol: str, ts_ms: int, close: float) -> Dict[str, Any]:
        sym = str(symbol).upper()
        price = float(close)
        st = self._ensure_state(sym)

        last = st["last_close"]
        if last is not None:
            delta = price - float(last)
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)
            if st["avg_gain"] is None or st["avg_loss"] is None:
                st["avg_gain"] = float(gain)
                st["avg_loss"] = float(loss)
            else:
                p = self.spec.rsi_period
                st["avg_gain"] = ((float(st["avg_gain"]) * (p - 1)) + gain) / p
                st["avg_loss"] = ((float(st["avg_loss"]) * (p - 1)) + loss) / p
        st["last_close"] = price

        st["prices"].append(price)

        feats: Dict[str, Any] = {
            "ts_ms": int(ts_ms),
            "symbol": sym,
            "ref_price": float(price),
        }

        seq = list(st["prices"])
        for lb in self.spec.lookbacks_prices:
            if len(seq) >= lb:
                window = seq[-lb:]
                sma = sum(window) / float(lb)
                feats[f"sma_{lb}"] = float(sma)
                first = float(window[0])
                feats[f"ret_{lb}m"] = (
                    float(math.log(price / first)) if first > 0 else 0.0
                )

        if (
            st["avg_gain"] is not None
            and st["avg_loss"] is not None
            and float(st["avg_loss"]) > 0.0
        ):
            rs = float(st["avg_gain"]) / float(st["avg_loss"])
            feats["rsi"] = float(100.0 - (100.0 / (1.0 + rs)))
        else:
            feats["rsi"] = float("nan")

        return feats


def apply_offline_features(
    df: pd.DataFrame,
    *,
    spec: FeatureSpec,
    ts_col: str = "ts_ms",
    symbol_col: str = "symbol",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Оффлайн-расчёт фич с точным соответствием онлайновому трансформеру.
    На входе ожидается таблица 1m-просэмплированных цен (price).
    На выходе: ts_ms, symbol, ref_price, sma_*, ret_*m, rsi.
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[ts_col, symbol_col, "ref_price", "rsi"]
            + [f"sma_{x}" for x in spec.lookbacks_prices]
            + [f"ret_{x}m" for x in spec.lookbacks_prices]
        )

    d = df.copy()
    if symbol_col not in d.columns or ts_col not in d.columns:
        raise ValueError(f"Вход должен содержать колонки '{symbol_col}' и '{ts_col}'")
    if price_col not in d.columns:
        raise ValueError(f"Вход должен содержать колонку цены '{price_col}'")

    d = d[[ts_col, symbol_col, price_col]].dropna().copy()
    d[ts_col] = d[ts_col].astype("int64")
    d[symbol_col] = d[symbol_col].astype(str)

    d = d.sort_values([symbol_col, ts_col]).reset_index(drop=True)

    out_rows: List[Dict[str, Any]] = []
    current_symbol: Optional[str] = None
    transformer: Optional[OnlineFeatureTransformer] = None

    for _, row in d.iterrows():
        sym = str(row[symbol_col]).upper()
        ts = int(row[ts_col])
        px = float(row[price_col])

        if transformer is None or current_symbol != sym:
            current_symbol = sym
            transformer = OnlineFeatureTransformer(spec)

        feats = transformer.update(symbol=sym, ts_ms=ts, close=px)
        out_rows.append(feats)

    out = pd.DataFrame(out_rows)
    out = out.sort_values([symbol_col, ts_col]).reset_index(drop=True)
    return out
