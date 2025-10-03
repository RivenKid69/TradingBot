import os

import pandas as pd

import fetch_all_data_patch


def _build_base_frame(symbol: str) -> pd.DataFrame:
    timestamps = [0, 3600, 7200]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0, 2.0, 3.0],
            "high": [1.1, 2.1, 3.1],
            "low": [0.9, 1.9, 2.9],
            "close": [1.05, 2.05, 3.05],
            "volume": [10.0, 11.0, 12.0],
            "quote_asset_volume": [20.0, 21.0, 22.0],
            "number_of_trades": [100, 101, 102],
            "taker_buy_base_asset_volume": [5.0, 5.1, 5.2],
            "taker_buy_quote_asset_volume": [6.0, 6.1, 6.2],
            "symbol": [symbol] * 3,
        }
    )


def test_load_all_data_preserves_single_fear_greed_column(tmp_path, monkeypatch):
    symbol = "BTCUSDT"
    df = _build_base_frame(symbol)
    df["fear_greed_value"] = [1.0, 2.0, 3.0]
    candle_path = tmp_path / f"{symbol}.feather"
    candle_path.write_text("dummy")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fng_path = data_dir / "fear_greed.csv"
    fng_df = pd.DataFrame(
        {
            "timestamp": [0, 3600, 7200],
            "fear_greed_value": [10.0, 20.0, 30.0],
        }
    )
    fng_df.to_csv(fng_path, index=False)

    monkeypatch.setattr(fetch_all_data_patch, "FNG_PATH", os.fspath(fng_path))
    monkeypatch.setattr(
        fetch_all_data_patch.pd,
        "read_feather",
        lambda path, **kwargs: df.copy(),
    )

    all_dfs, _ = fetch_all_data_patch.load_all_data([os.fspath(candle_path)])
    loaded = all_dfs[symbol]

    assert "fear_greed_value" in loaded.columns
    assert list(loaded["fear_greed_value"].astype(float)) == [10.0, 20.0, 30.0]
    assert "fear_greed_value_orig" not in loaded.columns


def test_load_all_data_converts_millisecond_timestamps(tmp_path, monkeypatch):
    symbol = "ETHUSDT"
    base_ts_ms = 1_650_000_000_000
    df = _build_base_frame(symbol)
    df["timestamp"] = [base_ts_ms, base_ts_ms + 3_600_000, base_ts_ms + 7_200_000]

    candle_path = tmp_path / f"{symbol}.feather"
    candle_path.write_text("dummy")

    monkeypatch.setattr(fetch_all_data_patch, "FNG_PATH", os.fspath(tmp_path / "fear_greed.csv"))
    monkeypatch.setattr(
        fetch_all_data_patch.pd,
        "read_feather",
        lambda path, **kwargs: df.copy(),
    )

    all_dfs, _ = fetch_all_data_patch.load_all_data([os.fspath(candle_path)])
    loaded = all_dfs[symbol]

    expected_start = (base_ts_ms // 1000 // 3600) * 3600
    expected_timestamps = [expected_start + offset for offset in (0, 3600, 7200)]

    assert list(loaded["timestamp"]) == expected_timestamps
