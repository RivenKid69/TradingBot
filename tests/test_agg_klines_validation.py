import importlib.util
import pathlib

import pandas as pd
import pytest

_MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "agg_klines.py"
spec = importlib.util.spec_from_file_location("agg_klines", _MODULE_PATH)
_mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(_mod)  # type: ignore[assignment]
_agg = _mod._agg


def _mk(ts_list):
    return pd.DataFrame({
        "ts_ms": ts_list,
        "symbol": ["BTCUSDT"] * len(ts_list),
        "open": [1.0] * len(ts_list),
        "high": [1.0] * len(ts_list),
        "low": [1.0] * len(ts_list),
        "close": [1.0] * len(ts_list),
        "volume": [1.0] * len(ts_list),
        "number_of_trades": [1] * len(ts_list),
        "taker_buy_base": [0.0] * len(ts_list),
        "taker_buy_quote": [0.0] * len(ts_list),
    })


def test_gap_detection_raises():
    df = _mk([0, 120_000])
    with pytest.raises(ValueError):
        _agg(df, "1m")


def test_valid_no_gap():
    df = _mk([0, 60_000, 120_000])
    out = _agg(df, "1m")
    assert len(out) == 3


def test_partial_start_raises():
    df = _mk([60_000, 120_000, 180_000, 240_000, 300_000])
    with pytest.raises(ValueError):
        _agg(df, "3m")


def test_partial_end_raises():
    df = _mk([0, 60_000, 120_000, 180_000, 240_000])
    with pytest.raises(ValueError):
        _agg(df, "3m")


def test_drop_partial():
    df = _mk([60_000, 120_000, 180_000, 240_000, 300_000, 360_000, 420_000])
    out = _agg(df, "3m", drop_partial=True)
    assert out["ts_ms"].tolist() == [180_000]


def test_drop_partial_keeps_aligned_head():
    df = _mk([0, 60_000, 120_000, 180_000, 240_000])
    out = _agg(df, "3m", drop_partial=True)
    assert out["ts_ms"].tolist() == [0]


def test_hourly_aggregation_alignment():
    df = _mk(list(range(0, 2 * 3_600_000, 60_000)))
    out = _agg(df, "1h")
    assert out["ts_ms"].tolist() == [0, 3_600_000]
    assert all(ts % 3_600_000 == 0 for ts in out["ts_ms"])
