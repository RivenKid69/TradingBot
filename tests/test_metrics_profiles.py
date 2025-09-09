import pandas as pd
from services.metrics import calculate_metrics
import json


def make_trades() -> pd.DataFrame:
    return pd.DataFrame({
        "ts_ms": [1, 2],
        "pnl": [1.0, -0.5],
        "side": ["BUY", "SELL"],
        "qty": [1, 1],
    })


def make_equity() -> pd.DataFrame:
    return pd.DataFrame({
        "ts_ms": [1, 2],
        "equity": [1.0, 0.5],
    })


def test_calculate_metrics_multiple_profiles():
    trades = {"a": make_trades(), "b": make_trades()}
    equity = {"a": make_equity(), "b": make_equity()}
    metrics = calculate_metrics(trades, equity)
    data = json.loads(json.dumps(metrics))
    assert "a" in data and "b" in data
