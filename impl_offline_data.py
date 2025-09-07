# -*- coding: utf-8 -*-
"""
impl_offline_data.py
Источники офлайн-данных: CSV/Parquet бары единым интерфейсом MarketDataSource.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence
import glob
import os

import pandas as pd  # предполагается в зависимостях

from core_contracts import MarketDataSource
from core_models import Bar, to_decimal
from market_utils import ensure_timeframe, timeframe_to_ms, to_ms


@dataclass
class OfflineCSVConfig:
    paths: List[str]                     # список путей/глобов к CSV, можно со звёздочками
    timeframe: str                       # например "1m"
    symbol_col: str = "symbol"
    ts_col: str = "ts"                   # миллисекунды или ISO8601
    o_col: str = "open"
    h_col: str = "high"
    l_col: str = "low"
    c_col: str = "close"
    v_col: str = "volume"
    bid_col: Optional[str] = None
    ask_col: Optional[str] = None
    n_trades_col: Optional[str] = None
    sort_by_ts: bool = True


class OfflineCSVBarSource(MarketDataSource):
    """Simple offline CSV reader yielding bars."""

    def __init__(self, cfg: OfflineCSVConfig) -> None:
        self.cfg = cfg
        ensure_timeframe(self.cfg.timeframe)

    def stream_bars(self, symbols: Sequence[str], interval_ms: int) -> Iterator[Bar]:
        if interval_ms != timeframe_to_ms(self.cfg.timeframe):
            raise ValueError(f"Timeframe mismatch. Source={self.cfg.timeframe}, requested_ms={interval_ms}")
        symbols_set = {s.upper() for s in symbols}

        cols = [self.cfg.ts_col, self.cfg.symbol_col, self.cfg.o_col, self.cfg.h_col,
                self.cfg.l_col, self.cfg.c_col, self.cfg.v_col]
        opt = [self.cfg.bid_col, self.cfg.ask_col, self.cfg.n_trades_col]
        cols += [c for c in opt if c]

        files: List[str] = []
        for p in self.cfg.paths:
            files.extend(glob.glob(p))
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            raise FileNotFoundError(f"No CSV files matched: {self.cfg.paths}")

        for path in sorted(files):
            df = pd.read_csv(path, usecols=lambda c: c in cols)
            if self.cfg.sort_by_ts and self.cfg.ts_col in df.columns:
                df = df.sort_values(self.cfg.ts_col, kind="mergesort")
            for _, r in df.iterrows():
                sym = str(r[self.cfg.symbol_col]).upper()
                if symbols_set and sym not in symbols_set:
                    continue
                bar = Bar(
                    ts=to_ms(r[self.cfg.ts_col]),
                    symbol=sym,
                    open=to_decimal(r[self.cfg.o_col]),
                    high=to_decimal(r[self.cfg.h_col]),
                    low=to_decimal(r[self.cfg.l_col]),
                    close=to_decimal(r[self.cfg.c_col]),
                    volume_base=to_decimal(r[self.cfg.v_col]),
                    trades=(None if not self.cfg.n_trades_col else int(r[self.cfg.n_trades_col])),
                )
                yield bar
