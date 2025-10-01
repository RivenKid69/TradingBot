import logging
import os
import sys
from typing import Iterable, Iterator, List

import pandas as pd
import pytest


sys.path.append(os.getcwd())

import clock
import impl_offline_data
from config import DataDegradationConfig
from impl_offline_data import OfflineCSVBarSource, OfflineCSVConfig


class _FakeRandom:
    """Deterministic random provider feeding predefined values."""

    def __init__(self, random_values: Iterable[float], randint_values: Iterable[int]):
        self._random_iter = iter(random_values)
        self._randint_iter = iter(randint_values)

    def random(self) -> float:
        try:
            return next(self._random_iter)
        except StopIteration:
            return 1.0

    def randint(self, a: int, b: int) -> int:
        try:
            val = next(self._randint_iter)
        except StopIteration:
            val = a
        if not (a <= val <= b):
            raise AssertionError(f"randint value {val} not in range [{a}, {b}]")
        return val


def _write_csv(path: str, rows: List[dict]) -> str:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def test_stream_bars_degradation_branches(tmp_path, monkeypatch, caplog):
    csv_path = tmp_path / "bars.csv"
    _write_csv(
        csv_path,
        [
            {"ts": 0, "symbol": "BTC", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            {"ts": 60_000, "symbol": "BTC", "open": 2, "high": 2, "low": 2, "close": 2, "volume": 2},
            {"ts": 120_000, "symbol": "BTC", "open": 3, "high": 3, "low": 3, "close": 3, "volume": 3},
            {"ts": 180_000, "symbol": "BTC", "open": 4, "high": 4, "low": 4, "close": 4, "volume": 4},
        ],
    )

    cfg = OfflineCSVConfig(paths=[str(csv_path)], timeframe="1m")
    degradation = DataDegradationConfig(
        drop_prob=0.4,
        stale_prob=0.5,
        dropout_prob=0.7,
        max_delay_ms=100,
        seed=123,
    )
    src = OfflineCSVBarSource(cfg, data_degradation=degradation)
    src._rng = _FakeRandom(
        random_values=[0.9, 0.1, 0.8, 0.2, 0.3, 0.85, 0.9, 0.6],
        randint_values=[50, 80],
    )

    times: Iterator[int] = iter([0, 300_000, 300_000])
    monkeypatch.setattr(clock, "now_ms", lambda: next(times, 300_000))

    sleep_calls: List[float] = []

    def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(impl_offline_data.time, "sleep", _fake_sleep)

    caplog.set_level(logging.INFO, logger="impl_offline_data")

    bars = list(src.stream_bars(["BTC"], 60_000))

    assert [(bar.ts, bar.is_final) for bar in bars] == [
        (0, False),
        (0, False),
        (180_000, True),
    ]
    assert bars[0] is bars[1]

    assert sleep_calls == [0.05, 0.08]

    summary = next(
        (record.message for record in caplog.records if "OfflineCSVBarSource degradation" in record.message),
        None,
    )
    assert summary is not None
    assert "drop=25.00% (1/4)" in summary
    assert "stale=25.00% (1/4)" in summary
    assert "delay=50.00% (2/4)" in summary
    assert "skip=25.00% (1/4)" in summary
