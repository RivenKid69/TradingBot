# -*- coding: utf-8 -*-
"""
impl_offline_data.py
Источники офлайн-данных: CSV/Parquet бары единым интерфейсом MarketDataSource.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterator, List, Optional, Sequence
import glob
import os
import random
import time
import logging

import pandas as pd  # предполагается в зависимостях

import clock
from core_models import Bar, Tick
from core_contracts import MarketDataSource
from utils_time import parse_time_to_ms, bar_close_ms, is_bar_closed
from config import DataDegradationConfig

logger = logging.getLogger(__name__)


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
    enforce_closed_bars: bool = True
    close_lag_ms: int = 2000


class OfflineCSVBarSource(MarketDataSource):
    def __init__(
        self,
        cfg: OfflineCSVConfig,
        data_degradation: DataDegradationConfig | None = None,
    ) -> None:
        self.cfg = cfg
        ensure_timeframe(self.cfg.timeframe)
        if data_degradation is None:
            data_degradation = getattr(cfg, "data_degradation", None)
        self._degradation = data_degradation or DataDegradationConfig.default()
        self._rng = random.Random(self._degradation.seed)

    def stream_bars(self, symbols: Sequence[str], interval_ms: int) -> Iterator[Bar]:
        interval_ms_cfg = timeframe_to_ms(self.cfg.timeframe)
        if interval_ms != interval_ms_cfg:
            raise ValueError(
                f"Timeframe mismatch. Source={self.cfg.timeframe}, requested={interval_ms}ms"
            )

        logger.info(
            "OfflineCSVBarSource closed bar enforcement: enforce=%s close_lag_ms=%d",
            self.cfg.enforce_closed_bars,
            self.cfg.close_lag_ms,
        )

        symbols_u = list(dict.fromkeys([s.upper() for s in symbols]))
        last_ts: Dict[str, int] = {}
        prev_bar: Optional[Bar] = None
        total = drop_cnt = stale_cnt = delay_cnt = skip_cnt = 0

        files: List[str] = []
        for p in self.cfg.paths:
            files.extend(glob.glob(p))
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            raise FileNotFoundError(f"No CSV files matched: {self.cfg.paths}")

        cols = [
            self.cfg.ts_col,
            self.cfg.symbol_col,
            self.cfg.o_col,
            self.cfg.h_col,
            self.cfg.l_col,
            self.cfg.c_col,
            self.cfg.v_col,
        ]
        if self.cfg.n_trades_col:
            cols.append(self.cfg.n_trades_col)

        for path in sorted(files):
            df = pd.read_csv(path, usecols=lambda c: c in cols)
            if self.cfg.sort_by_ts and self.cfg.ts_col in df.columns:
                df = df.sort_values(self.cfg.ts_col, kind="mergesort")
            for _, r in df.iterrows():
                sym = str(r[self.cfg.symbol_col]).upper()
                if symbols_u and sym not in symbols_u:
                    continue
                ts = to_ms(r[self.cfg.ts_col])
                if ts % interval_ms_cfg != 0:
                    raise ValueError(
                        f"Timestamp {ts} not aligned with interval {interval_ms_cfg}ms"
                    )
                prev = last_ts.get(sym)
                if prev is not None:
                    if ts == prev:
                        raise ValueError(f"Duplicate bar for {sym} at {ts}")
                    if ts - prev > interval_ms_cfg:
                        missing = list(range(prev + interval_ms_cfg, ts, interval_ms_cfg))
                        raise ValueError(f"Missing bars for {sym}: {missing}")
                last_ts[sym] = ts
                total += 1
                if self._rng.random() < self._degradation.drop_prob:
                    drop_cnt += 1
                    continue
                if prev_bar is not None and self._rng.random() < self._degradation.stale_prob:
                    stale_cnt += 1
                    if self._rng.random() < self._degradation.dropout_prob:
                        delay_ms = self._rng.randint(0, self._degradation.max_delay_ms)
                        if delay_ms > 0:
                            delay_cnt += 1
                            time.sleep(delay_ms / 1000.0)
                    yield prev_bar
                    continue
                close_ts = bar_close_ms(ts, interval_ms_cfg)
                is_final = True
                if self.cfg.enforce_closed_bars and not is_bar_closed(
                    close_ts, clock.now_ms(), self.cfg.close_lag_ms
                ):
                    is_final = False
                    skip_cnt += 1
                bar = Bar(
                    ts=ts,
                    symbol=sym,
                    open=Decimal(str(r[self.cfg.o_col])),
                    high=Decimal(str(r[self.cfg.h_col])),
                    low=Decimal(str(r[self.cfg.l_col])),
                    close=Decimal(str(r[self.cfg.c_col])),
                    volume_base=Decimal(str(r[self.cfg.v_col])),
                    trades=(
                        None
                        if not self.cfg.n_trades_col
                        else int(r[self.cfg.n_trades_col])
                    ),
                    is_final=is_final,
                )
                if not is_final:
                    prev_bar = bar
                    yield bar
                    continue
                if self._rng.random() < self._degradation.dropout_prob:
                    delay_ms = self._rng.randint(0, self._degradation.max_delay_ms)
                    if delay_ms > 0:
                        delay_cnt += 1
                        time.sleep(delay_ms / 1000.0)
                prev_bar = bar
                yield bar

        if total:
            logger.info(
                "OfflineCSVBarSource degradation: drop=%0.2f%% (%d/%d), stale=%0.2f%% (%d/%d), delay=%0.2f%% (%d/%d), skip=%0.2f%% (%d/%d)",
                drop_cnt / total * 100.0,
                drop_cnt,
                total,
                stale_cnt / total * 100.0,
                stale_cnt,
                total,
                delay_cnt / total * 100.0,
                delay_cnt,
                total,
                skip_cnt / total * 100.0,
                skip_cnt,
                total,
            )
        else:
            logger.info("OfflineCSVBarSource degradation: no bars processed")

    def stream_ticks(self, symbols: Sequence[str]) -> Iterator[Tick]:
        return iter([])


# ----- utilities -----

_VALID_TF = {
    "1s",
    "5s",
    "10s",
    "15s",
    "30s",
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
}


def ensure_timeframe(tf: str) -> str:
    tf = str(tf).lower()
    if tf not in _VALID_TF:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return tf


def timeframe_to_ms(tf: str) -> int:
    tf = ensure_timeframe(tf)
    mult = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return int(tf[:-1]) * mult[tf[-1]]


def to_ms(dt: Any) -> int:
    if isinstance(dt, int):
        return dt
    if isinstance(dt, float):
        return int(dt)
    if isinstance(dt, str):
        return parse_time_to_ms(dt)
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return int(dt.timestamp() * 1000)
    raise TypeError(f"Unsupported datetime type: {type(dt)}")
