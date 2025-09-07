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

from market_data_port import Bar, MarketEvent, EventKind, MarketDataSource, ensure_timeframe, to_ms


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
    vendor: Optional[str] = "offline"
    sort_by_ts: bool = True


class OfflineCSVBarSource(MarketDataSource):
    def __init__(self, cfg: OfflineCSVConfig) -> None:
        self.cfg = cfg
        self._symbols: List[str] = []
        self._opened = False
        self._files: List[str] = []
        ensure_timeframe(self.cfg.timeframe)

    def open(self) -> None:
        files: List[str] = []
        for p in self.cfg.paths:
            files.extend(glob.glob(p))
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            raise FileNotFoundError(f"No CSV files matched: {self.cfg.paths}")
        self._files = sorted(files)
        self._opened = True

    def subscribe(self, symbols: Sequence[str], *, timeframe: Optional[str] = None) -> None:
        if timeframe and ensure_timeframe(timeframe) != self.cfg.timeframe:
            raise ValueError(f"Timeframe mismatch. Source={self.cfg.timeframe}, requested={timeframe}")
        self._symbols = list(dict.fromkeys([s.upper() for s in symbols]))

    def iter_events(self) -> Iterator[MarketEvent]:
        if not self._opened:
            raise RuntimeError("Source not opened")

        cols = [self.cfg.ts_col, self.cfg.symbol_col, self.cfg.o_col, self.cfg.h_col,
                self.cfg.l_col, self.cfg.c_col, self.cfg.v_col]
        opt = [self.cfg.bid_col, self.cfg.ask_col, self.cfg.n_trades_col]
        cols += [c for c in opt if c]

        for path in self._files:
            df = pd.read_csv(path, usecols=lambda c: c in cols)
            if self.cfg.sort_by_ts and self.cfg.ts_col in df.columns:
                df = df.sort_values(self.cfg.ts_col, kind="mergesort")
            for _, r in df.iterrows():
                sym = str(r[self.cfg.symbol_col]).upper()
                if self._symbols and sym not in self._symbols:
                    continue
                bar = Bar(
                    ts=to_ms(r[self.cfg.ts_col]),
                    symbol=sym,
                    timeframe=self.cfg.timeframe,
                    open=float(r[self.cfg.o_col]),
                    high=float(r[self.cfg.h_col]),
                    low=float(r[self.cfg.l_col]),
                    close=float(r[self.cfg.c_col]),
                    volume=float(r[self.cfg.v_col]),
                    bid=(None if not self.cfg.bid_col else float(r[self.cfg.bid_col])),
                    ask=(None if not self.cfg.ask_col else float(r[self.cfg.ask_col])),
                    n_trades=(None if not self.cfg.n_trades_col else int(r[self.cfg.n_trades_col])),
                    vendor=self.cfg.vendor,
                    meta={"file": os.path.basename(path)},
                )
                yield MarketEvent(kind=EventKind.BAR, bar=bar)

    def close(self) -> None:
        self._opened = False
        self._files = []
