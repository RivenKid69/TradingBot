import pandas as pd
import pytest

from impl_offline_data import OfflineCSVConfig, OfflineCSVBarSource


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
