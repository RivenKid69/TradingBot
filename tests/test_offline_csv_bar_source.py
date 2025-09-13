import os
import sys
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

sys.path.append(os.getcwd())
import clock
from utils_time import floor_to_timeframe
from impl_offline_data import OfflineCSVConfig, OfflineCSVBarSource, to_ms


def _write_csv(tmp_path, rows):
    path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_duplicate_bar_raises(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {"ts": 0, "symbol": "BTC", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            {"ts": 0, "symbol": "BTC", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        ],
    )
    cfg = OfflineCSVConfig(paths=[path], timeframe="1m")
    src = OfflineCSVBarSource(cfg)
    with pytest.raises(ValueError, match="Duplicate bar for BTC.*0"):
        list(src.stream_bars(["BTC"], 60_000))


def test_duplicate_hour_bar_raises(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {"ts": 0, "symbol": "BTC", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            {"ts": 0, "symbol": "BTC", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        ],
    )
    cfg = OfflineCSVConfig(paths=[path], timeframe="1h")
    src = OfflineCSVBarSource(cfg)
    with pytest.raises(ValueError, match="Duplicate bar for BTC.*0"):
        list(src.stream_bars(["BTC"], 3_600_000))


def test_missing_bar_raises(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {"ts": 0, "symbol": "BTC", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            {"ts": 120_000, "symbol": "BTC", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        ],
    )
    cfg = OfflineCSVConfig(paths=[path], timeframe="1m")
    src = OfflineCSVBarSource(cfg)
    with pytest.raises(ValueError) as exc:
        list(src.stream_bars(["BTC"], 60_000))
    msg = str(exc.value)
    assert "Missing bars for BTC" in msg
    assert "60000" in msg


def test_missing_hour_bar_raises(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            {"ts": 0, "symbol": "BTC", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            {"ts": 7_200_000, "symbol": "BTC", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        ],
    )
    cfg = OfflineCSVConfig(paths=[path], timeframe="1h")
    src = OfflineCSVBarSource(cfg)
    with pytest.raises(ValueError) as exc:
        list(src.stream_bars(["BTC"], 3_600_000))
    msg = str(exc.value)
    assert "Missing bars for BTC" in msg
    assert "3600000" in msg


def test_timezone_bar_passes(tmp_path):
    tz = timezone(timedelta(hours=3))
    dt_local = datetime(1970, 1, 1, 3, 0, tzinfo=tz)
    assert to_ms(dt_local) == 0
    path = _write_csv(
        tmp_path,
        [
            {
                "ts": dt_local,
                "symbol": "BTC",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
            }
        ],
    )
    cfg = OfflineCSVConfig(paths=[path], timeframe="1m")
    src = OfflineCSVBarSource(cfg)
    bars = list(src.stream_bars(["BTC"], 60_000))
    assert bars[0].ts == 0


def test_enforce_closed_bar(tmp_path, monkeypatch):
    now = clock.now_ms()
    start = floor_to_timeframe(now, 60_000)
    path = _write_csv(
        tmp_path,
        [
            {
                "ts": start,
                "symbol": "BTC",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
            }
        ],
    )
    cfg = OfflineCSVConfig(paths=[path], timeframe="1m")
    monkeypatch.setattr(clock, "now_ms", lambda: start + 30_000)
    src = OfflineCSVBarSource(cfg)
    bars = list(src.stream_bars(["BTC"], 60_000))
    assert len(bars) == 1
    assert bars[0].is_final is False
