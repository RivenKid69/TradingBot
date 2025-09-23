import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.append(os.getcwd())

from no_trade import compute_no_trade_mask, estimate_block_ratio, _load_maintenance_calendar
from no_trade_config import (
    NoTradeState,
    get_no_trade_config,
    load_no_trade_state,
    save_no_trade_state,
)


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
    expected_cols = {
        "window",
        "dynamic_guard",
        "dyn_vol_abs",
        "dyn_ret_anomaly",
        "dyn_guard_raw",
        "dyn_guard_hold",
        "dyn_guard_next_block",
        "dyn_guard_state",
        "dyn_vol_extreme",
        "dyn_spread_wide",
        "dyn_guard_warmup",
        "dyn_cooldown",
    }
    assert expected_cols.issubset(set(reasons.columns))
    assert "maintenance_calendar" in reasons.columns
    assert reasons["window"].sum() == 0
    assert reasons["dynamic_guard"].tolist() == expected_mask
    assert bool(reasons.loc[df.index[2], "dyn_vol_abs"])
    assert bool(reasons.loc[df.index[2], "dyn_ret_anomaly"])
    assert bool(reasons.loc[df.index[2], "dyn_guard_raw"])
    assert bool(reasons.loc[df.index[3], "dyn_guard_hold"])
    assert not bool(reasons.loc[df.index[2], "dyn_guard_hold"])
    assert list(reasons.loc[df.index[:2], "dyn_guard_warmup"]) == [True, True]
    assert bool(reasons.loc[df.index[3], "dyn_cooldown"])
    assert not reasons["dyn_guard_next_block"].any()
    assert not reasons["dyn_guard_state"].any()

    labels = mask.attrs.get("reason_labels")
    assert isinstance(labels, dict)
    for key in ["dynamic_guard", "dyn_ret_anomaly", "dyn_guard_next_block"]:
        assert key in labels

    state = mask.attrs.get("state")
    assert isinstance(state, dict)
    anomaly_state = state.get("anomaly_block_until_ts")
    assert isinstance(anomaly_state, dict)
    assert anomaly_state.get("BTC") == int(df["ts_ms"].iloc[3])
    maintenance_state = state.get("maintenance")
    assert isinstance(maintenance_state, dict)
    assert maintenance_state.get("funding_buffer_min") == 0
    assert maintenance_state.get("daily_utc") == []
    assert "calendar" not in maintenance_state

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


def test_next_bars_block_extends_mask_and_state(tmp_path):
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
                "hysteresis": 0.0,
                "cooldown_bars": 0,
                "next_bars_block": {"anomaly": 2},
            },
        }
    }
    cfg_path = tmp_path / "sandbox.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    df = pd.DataFrame(
        {
            "ts_ms": np.arange(7, dtype=np.int64) * 60_000,
            "symbol": ["BTC"] * 7,
            "close": [100.0, 150.0, 152.0, 152.5, 152.6, 152.7, 152.8],
        }
    )

    mask = compute_no_trade_mask(df, sandbox_yaml_path=str(cfg_path))
    expected_mask = [False, False, True, True, True, False, False]
    assert mask.tolist() == expected_mask

    reasons = mask.attrs.get("reasons")
    assert isinstance(reasons, pd.DataFrame)
    assert "maintenance_calendar" in reasons.columns
    assert list(reasons.loc[df.index, "dyn_guard_next_block"]) == [False, False, False, True, True, False, False]

    state = mask.attrs.get("state")
    assert isinstance(state, dict)
    anomaly_state = state.get("anomaly_block_until_ts")
    assert anomaly_state.get("BTC") == int(df["ts_ms"].iloc[4])
    dyn_state = state.get("dynamic_guard") or {}
    assert int(dyn_state["BTC"]["block_until_ts"]) == int(df["ts_ms"].iloc[4])
    assert dyn_state["BTC"]["next_block_left"] == 0
    maintenance_state = state.get("maintenance")
    assert isinstance(maintenance_state, dict)


def test_state_blocks_rows_without_guard(tmp_path):
    cfg_data = {
        "no_trade": {
            "funding_buffer_min": 0,
            "daily_utc": [],
            "custom_ms": [],
            "dynamic_guard": {"enable": False},
        }
    }
    cfg_path = tmp_path / "sandbox.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    df = pd.DataFrame(
        {
            "ts_ms": np.arange(4, dtype=np.int64) * 60_000,
            "symbol": ["BTC"] * 4,
        }
    )

    state = {"anomaly_block_until_ts": {"BTC": int(df["ts_ms"].iloc[1])}}
    mask = compute_no_trade_mask(df, sandbox_yaml_path=str(cfg_path), state=state)
    expected_mask = [True, True, False, False]
    assert mask.tolist() == expected_mask

    reasons = mask.attrs.get("reasons")
    assert isinstance(reasons, pd.DataFrame)
    assert list(reasons.loc[df.index, "dyn_guard_state"]) == expected_mask

    state_payload = mask.attrs.get("state")
    assert isinstance(state_payload, dict)
    maintenance_state = state_payload.get("maintenance")
    assert isinstance(maintenance_state, dict)

    cfg = get_no_trade_config(str(cfg_path))
    ratio = estimate_block_ratio(df, cfg, state=state)
    assert ratio == pytest.approx(sum(expected_mask) / len(expected_mask))


def test_maintenance_calendar_blocks_rows(tmp_path):
    calendar_df = pd.DataFrame(
        [
            {"start_ts_ms": 60_000, "end_ts_ms": 180_000, "symbol": "BTC"},
            {"start_ts_ms": 300_000, "end_ts_ms": 360_000},
        ]
    )
    calendar_path = tmp_path / "maintenance.csv"
    calendar_df.to_csv(calendar_path, index=False)

    cfg_data = {
        "no_trade": {
            "maintenance": {
                "path": "maintenance.csv",
                "daily_utc": [],
                "custom_ms": [],
                "funding_buffer_min": 0,
            },
            "dynamic_guard": {"enable": False},
        }
    }
    cfg_path = tmp_path / "sandbox.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    df = pd.DataFrame(
        {
            "ts_ms": [0, 60_000, 120_000, 240_000, 320_000],
            "symbol": ["BTC", "BTC", "ETH", "BTC", "ETH"],
        }
    )

    mask = compute_no_trade_mask(df, sandbox_yaml_path=str(cfg_path))
    expected_mask = [False, True, False, False, True]
    actual_mask = mask.tolist()

    reasons = mask.attrs.get("reasons")
    assert isinstance(reasons, pd.DataFrame)
    actual_calendar = reasons["maintenance_calendar"].tolist()
    assert reasons["maintenance_daily"].tolist() == [False] * len(df)
    assert reasons["maintenance_custom"].tolist() == [False] * len(df)

    state_payload = mask.attrs.get("state")
    assert isinstance(state_payload, dict)
    maintenance_state = state_payload.get("maintenance") or {}
    calendar_state = maintenance_state.get("calendar") or {}
    assert calendar_state.get("exists") is True
    assert Path(calendar_state.get("path")).name == "maintenance.csv"
    assert calendar_state.get("source") == "maintenance.csv"
    assert calendar_state.get("format") == "csv"
    assert calendar_state.get("global") == [(300_000, 360_000)]
    assert calendar_state.get("per_symbol", {}).get("BTC") == [(60_000, 180_000)]
    windows = calendar_state.get("windows") or []
    assert any(entry.get("symbol") == "BTC" for entry in windows)
    assert any(entry.get("symbol") is None for entry in windows), windows

    assert actual_mask == expected_mask == actual_calendar, (
        actual_mask,
        actual_calendar,
        windows,
    )

    meta = mask.attrs.get("meta") or {}
    maintenance_meta = meta.get("maintenance_calendar") or {}
    assert maintenance_meta.get("exists") is True
    assert maintenance_meta.get("format") == "csv"
    assert maintenance_meta.get("source") == "maintenance.csv"


def test_missing_maintenance_calendar_logs_warning(tmp_path, caplog):
    cfg_data = {
        "no_trade": {
            "maintenance": {
                "path": "missing.json",
                "daily_utc": [],
                "custom_ms": [],
                "funding_buffer_min": 0,
            },
            "dynamic_guard": {"enable": False},
        }
    }
    cfg_path = tmp_path / "sandbox.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    df = pd.DataFrame({"ts_ms": [0, 60_000]})

    with caplog.at_level(logging.WARNING):
        mask = compute_no_trade_mask(df, sandbox_yaml_path=str(cfg_path))

    assert any("Maintenance calendar file not found" in rec.getMessage() for rec in caplog.records)

    state_payload = mask.attrs.get("state")
    maintenance_state = state_payload.get("maintenance") or {}
    calendar_state = maintenance_state.get("calendar") or {}
    assert calendar_state.get("exists") is False
    assert calendar_state.get("source") == "missing.json"
    assert calendar_state.get("format") == "auto"

    meta = mask.attrs.get("meta") or {}
    maintenance_meta = meta.get("maintenance_calendar") or {}
    assert maintenance_meta.get("exists") is False
    assert maintenance_meta.get("format") == "auto"


def test_stale_maintenance_calendar_emits_warning(tmp_path, caplog):
    calendar_path = tmp_path / "maintenance.json"
    calendar_payload = [{"start_ts_ms": 0, "end_ts_ms": 60_000}]
    calendar_path.write_text(json.dumps(calendar_payload), encoding="utf-8")
    old_time = time.time() - 10.0
    os.utime(calendar_path, (old_time, old_time))

    cfg_data = {
        "no_trade": {
            "maintenance": {
                "path": str(calendar_path),
                "daily_utc": [],
                "custom_ms": [],
                "funding_buffer_min": 0,
                "max_age_sec": 1,
            },
            "dynamic_guard": {"enable": False},
        }
    }
    cfg_path = tmp_path / "sandbox.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    df = pd.DataFrame({"ts_ms": [0, 120_000]})

    with caplog.at_level(logging.WARNING):
        mask = compute_no_trade_mask(df, sandbox_yaml_path=str(cfg_path))

    assert any("looks stale" in rec.getMessage() for rec in caplog.records)
    reasons = mask.attrs.get("reasons")
    assert reasons["maintenance_calendar"].tolist() == [True, False]

    state_payload = mask.attrs.get("state")
    calendar_state = (state_payload.get("maintenance") or {}).get("calendar") or {}
    assert calendar_state.get("stale") is True
    assert calendar_state.get("format") == "json"

    meta = mask.attrs.get("meta") or {}
    maintenance_meta = meta.get("maintenance_calendar") or {}
    assert maintenance_meta.get("stale") is True
    assert maintenance_meta.get("format") == "json"


def test_calendar_format_override_and_merge(tmp_path):
    calendar_path = tmp_path / "schedule.data"
    payload = [
        {"start_ts_ms": 60_000, "end_ts_ms": 120_000, "symbol": "BTC"},
        {"start_ts_ms": 90_000, "end_ts_ms": 150_000, "symbol": "BTC"},
        {"start_ts_ms": 200_000, "end_ts_ms": 260_000},
        {"start_ts_ms": 250_000, "end_ts_ms": 300_000},
    ]
    calendar_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg_data = {
        "no_trade": {
            "maintenance": {
                "path": "schedule.data",
                "format": "json",
                "daily_utc": [],
                "custom_ms": [],
                "funding_buffer_min": 0,
            },
            "dynamic_guard": {"enable": False},
        }
    }
    cfg_path = tmp_path / "sandbox.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    cfg = get_no_trade_config(str(cfg_path))
    calendar, meta = _load_maintenance_calendar(cfg)

    assert meta.get("format") == "json"
    assert meta.get("source") == "schedule.data"
    assert Path(meta.get("path")).name == "schedule.data"
    assert calendar.get("per_symbol", {}).get("BTC") == [(60_000, 150_000)]
    assert calendar.get("global") == [(200_000, 300_000)]

    df = pd.DataFrame(
        {
            "ts_ms": [0, 60_000, 110_000, 210_000, 290_000],
            "symbol": ["ETH", "BTC", "BTC", "ETH", "BTC"],
        }
    )

    mask = compute_no_trade_mask(df, sandbox_yaml_path=str(cfg_path))
    reasons = mask.attrs.get("reasons")
    assert reasons is not None
    expected = [False, True, True, True, True]
    assert mask.tolist() == expected
    assert reasons["maintenance_calendar"].tolist() == expected
    assert reasons["maintenance_daily"].tolist() == [False] * len(df)
    assert reasons["maintenance_custom"].tolist() == [False] * len(df)

    state_payload = mask.attrs.get("state") or {}
    calendar_state = (state_payload.get("maintenance") or {}).get("calendar") or {}
    assert calendar_state.get("format") == "json"
    assert calendar_state.get("per_symbol", {}).get("BTC") == [(60_000, 150_000)]
    assert calendar_state.get("global") == [(200_000, 300_000)]

    meta_payload = mask.attrs.get("meta") or {}
    maintenance_meta = meta_payload.get("maintenance_calendar") or {}
    assert maintenance_meta.get("format") == "json"
    assert maintenance_meta.get("source") == "schedule.data"


def test_no_trade_state_migration_and_save(tmp_path):
    legacy_payloads = [
        ({"BTCUSDT": 123}, {"BTCUSDT": 123}),
        ({"anomaly_block_until_ts_ms": {"ETHUSDT": "456"}}, {"ETHUSDT": 456}),
        (
            {
                "anomaly_block_until_ts": [
                    {"symbol": "XRPUSDT", "ts": 789},
                    {"pair": "LTCUSDT", "timestamp_ms": "1000"},
                ]
            },
            {"XRPUSDT": 789, "LTCUSDT": 1000},
        ),
    ]

    for payload, expected in legacy_payloads:
        path = tmp_path / "legacy_state.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        state = load_no_trade_state(path)
        assert state.anomaly_block_until_ts == expected
        assert state.dynamic_guard == {}

    state = NoTradeState(anomaly_block_until_ts={"BTCUSDT": 123})
    out_path = tmp_path / "saved_state.json"
    save_no_trade_state(state, out_path)
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved == {
        "anomaly_block_until_ts": {"BTCUSDT": 123},
        "dynamic_guard": {},
    }
