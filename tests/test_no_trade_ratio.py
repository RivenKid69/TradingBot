import numpy as np
import pandas as pd
import pytest
import os
import sys
import yaml

sys.path.append(os.getcwd())

from no_trade import compute_no_trade_mask, estimate_block_ratio
from no_trade_config import get_no_trade_config


def test_blocked_share_matches_legacy_config():
    cfg = get_no_trade_config("configs/legacy_sandbox.yaml")
    ts = np.arange(0, 24 * 60, dtype=np.int64) * 60_000
    df = pd.DataFrame({"ts_ms": ts})
    mask = compute_no_trade_mask(df, sandbox_yaml_path="configs/legacy_sandbox.yaml")
    est = estimate_block_ratio(df, cfg)
    expected = 28 / 1440
    assert est == pytest.approx(expected, abs=1e-6)
    assert mask.mean() == pytest.approx(expected, abs=1e-6)


def test_dynamic_guard_blocks_and_logs_reasons(tmp_path):
    cfg_data = {
        "no_trade": {
            "funding_buffer_min": 0,
            "daily_utc": [],
            "custom_ms": [],
            "dynamic_guard": {
                "enable": True,
                "sigma_window": 2,
                "atr_window": 2,
                "vol_abs": 0.2,
                "hysteresis": 0.5,
                "cooldown_bars": 1,
            },
        }
    }
    cfg_path = tmp_path / "sandbox.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    df = pd.DataFrame(
        {
            "ts_ms": np.arange(6, dtype=np.int64) * 60_000,
            "symbol": ["BTC"] * 6,
            "close": [100.0, 150.0, 152.0, 153.0, 154.0, 154.5],
        }
    )

    mask = compute_no_trade_mask(df, sandbox_yaml_path=str(cfg_path))
    expected_mask = [False, False, True, True, False, False]
    assert mask.tolist() == expected_mask

    reasons = mask.attrs.get("reasons")
    assert isinstance(reasons, pd.DataFrame)
    assert reasons.index.equals(df.index)
    for col in ["window", "dynamic_guard", "dyn_vol_abs", "dyn_guard_raw", "dyn_guard_hold"]:
        assert col in reasons.columns
    assert reasons["window"].sum() == 0
    assert reasons["dynamic_guard"].tolist() == expected_mask
    assert bool(reasons.loc[df.index[2], "dyn_vol_abs"])
    assert bool(reasons.loc[df.index[2], "dyn_guard_raw"])
    assert bool(reasons.loc[df.index[3], "dyn_guard_hold"])
    assert not bool(reasons.loc[df.index[2], "dyn_guard_hold"])

    labels = mask.attrs.get("reason_labels")
    assert isinstance(labels, dict)
    assert "dynamic_guard" in labels

    cfg = get_no_trade_config(str(cfg_path))
    ratio = estimate_block_ratio(df, cfg)
    assert ratio == pytest.approx(mask.mean())


def test_dynamic_guard_skipped_when_data_missing(tmp_path):
    cfg_data = {
        "no_trade": {
            "funding_buffer_min": 0,
            "daily_utc": [],
            "custom_ms": [],
            "dynamic_guard": {
                "enable": True,
                "sigma_window": 3,
                "vol_abs": 0.1,
            },
        }
    }
    cfg_path = tmp_path / "sandbox.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    df = pd.DataFrame({"ts_ms": np.arange(5, dtype=np.int64) * 60_000})

    mask = compute_no_trade_mask(df, sandbox_yaml_path=str(cfg_path))
    assert not mask.any()

    reasons = mask.attrs.get("reasons")
    assert isinstance(reasons, pd.DataFrame)
    assert "dynamic_guard" in reasons.columns
    assert not reasons["dynamic_guard"].any()

    meta = mask.attrs.get("meta")
    assert isinstance(meta, dict)
    dyn_meta = meta.get("dynamic_guard")
    assert isinstance(dyn_meta, dict)
    assert dyn_meta.get("skipped")
    assert "volatility" in dyn_meta.get("missing", [])
