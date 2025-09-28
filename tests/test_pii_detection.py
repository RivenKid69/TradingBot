import importlib.util
import pathlib
import pandas as pd
import pytest

BASE = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("data_validation", BASE / "data_validation.py")
dv_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dv_module)
DataValidator = dv_module.DataValidator


def _make_df(note):
    idx = pd.date_range('2024-01-01', periods=1, freq='1T')
    data = {
        'timestamp': idx,
        'symbol': ['BTCUSDT'],
        'open': [1.0],
        'high': [1.0],
        'low': [1.0],
        'close': [1.0],
        'volume': [1.0],
        'quote_asset_volume': [1.0],
        'number_of_trades': [1],
        'taker_buy_base_asset_volume': [1.0],
        'taker_buy_quote_asset_volume': [1.0],
        'note': [note],
    }
    return pd.DataFrame(data, index=idx)


def test_rejects_email():
    df = _make_df('user@example.com')
    validator = DataValidator()
    with pytest.raises(ValueError):
        validator.validate(df)


def test_accepts_clean_data():
    df = _make_df('seasonal pattern')
    validator = DataValidator()
    validator.validate(df)


def test_integer_index_with_custom_step_passes():
    idx = pd.Index([0, 5, 10], name="row")
    timestamps = pd.date_range('2024-01-01', periods=3, freq='1T')
    data = {
        'timestamp': timestamps,
        'symbol': ['BTCUSDT'] * 3,
        'open': [1.0, 1.1, 1.2],
        'high': [1.1, 1.2, 1.3],
        'low': [0.9, 1.0, 1.1],
        'close': [1.05, 1.15, 1.25],
        'volume': [1.0, 1.0, 1.0],
        'quote_asset_volume': [1.0, 1.0, 1.0],
        'number_of_trades': [1, 1, 1],
        'taker_buy_base_asset_volume': [1.0, 1.0, 1.0],
        'taker_buy_quote_asset_volume': [1.0, 1.0, 1.0],
        'note': ['clean'] * 3,
    }
    df = pd.DataFrame(data, index=idx)
    validator = DataValidator()
    validator.validate(df)
