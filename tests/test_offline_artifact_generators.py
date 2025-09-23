import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import build_adv as build_adv_cli
from scripts import build_hourly_seasonality as hourly_cli
from scripts import build_spread_seasonality as spread_cli
from scripts import refresh_fees as refresh_fees_cli
from scripts.offline_utils import apply_split_tag
from utils_time import parse_time_to_ms


@pytest.mark.parametrize("symbol", ["BTCUSDT"])
def test_build_adv_split_uses_config(tmp_path, monkeypatch, symbol):
    adv_base = tmp_path / "adv" / "adv.json"
    config_path = tmp_path / "offline.yaml"
    config_payload = {
        "datasets": {
            "sample": {
                "version": "v99-train",
                "start": "2020-01-01T00:00:00Z",
                "end": "2020-01-31T00:00:00Z",
                "adv": {
                    "input": {
                        "start": "2020-01-01T00:00:00Z",
                        "end": "2020-01-21T00:00:00Z",
                    },
                    "output_path": str(adv_base),
                },
            }
        }
    }
    config_path.write_text(yaml.safe_dump(config_payload))
    rest_cfg = tmp_path / "rest.yaml"
    rest_cfg.write_text("{}\n")

    start_ms = parse_time_to_ms("2020-01-01T00:00:00Z")
    end_ms = parse_time_to_ms("2020-01-21T00:00:00Z")

    fetch_kwargs: dict = {}

    def fake_fetch(session, symbols, **kwargs):
        fetch_kwargs.update(kwargs)
        ts_values = [start_ms + i * 86_400_000 for i in range(5)]
        df = pd.DataFrame(
            {
                "ts_ms": ts_values,
                "quote_asset_volume": np.linspace(1_000.0, 1_500.0, num=5),
            }
        )
        return {symbols[0]: df}

    class DummySession:
        def __init__(self, cfg):
            self.cfg = cfg

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write_stats(self, path):
            self.stats_path = path

    monkeypatch.setattr(build_adv_cli, "fetch_klines_for_symbols", fake_fetch)
    monkeypatch.setattr(build_adv_cli, "RestBudgetSession", DummySession)
    monkeypatch.setattr(build_adv_cli, "_load_universe", lambda _: ({}, []))

    args = [
        "--split",
        "sample",
        "--config",
        str(config_path),
        "--symbols",
        symbol,
        "--rest-budget-config",
        str(rest_cfg),
    ]
    build_adv_cli.main(args)

    assert fetch_kwargs["start_ms"] == start_ms
    assert fetch_kwargs["end_ms"] == end_ms

    out_path = apply_split_tag(adv_base, "v99-train")
    assert out_path.exists()

    payload = json.loads(out_path.read_text())
    meta = payload["meta"]
    assert meta["split"] == {"name": "sample", "version": "v99-train"}
    assert meta["data_window"]["actual"]["start_ms"] == start_ms
    assert meta["data_window"]["actual"]["end_ms"] == end_ms
    assert meta["data_window"]["config"]["end_ms"] == end_ms


def test_refresh_fees_split_output(tmp_path, monkeypatch):
    fees_base = tmp_path / "fees" / "fees.json"
    config_path = tmp_path / "offline.yaml"
    config_payload = {
        "datasets": {
            "sample": {
                "version": "v2",
                "start": "2019-01-01T00:00:00Z",
                "end": "2019-06-30T00:00:00Z",
                "fees": {
                    "input": {
                        "start": "2019-01-01T00:00:00Z",
                        "end": "2019-04-01T00:00:00Z",
                    },
                    "output_path": str(fees_base),
                },
            }
        }
    }
    config_path.write_text(yaml.safe_dump(config_payload))

    monkeypatch.setattr(refresh_fees_cli, "fetch_exchange_symbols", lambda timeout: ["BTCUSDT"])

    snapshot_payload = {
        "metadata": {"built_at": "2019-04-01T00:00:00Z"},
        "fees": {"BTCUSDT": {"maker_bps": 10, "taker_bps": 10}},
    }
    dummy_snapshot = SimpleNamespace(
        payload=snapshot_payload,
        records={"BTCUSDT": object()},
        symbols={"BTCUSDT"},
        maker_bps_default=None,
        taker_bps_default=None,
        taker_discount_mult=None,
        maker_discount_mult=None,
    )
    monkeypatch.setattr(refresh_fees_cli, "load_public_fee_snapshot", lambda **_: dummy_snapshot)

    refresh_fees_cli.main(["--split", "sample", "--config", str(config_path)])

    out_path = apply_split_tag(fees_base, "v2")
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    meta = payload["metadata"]
    assert meta["split"] == {"name": "sample", "version": "v2"}
    assert meta["data_window"]["actual"]["end"] == "2019-04-01T00:00:00Z"
    assert meta["data_window"]["config"]["end"] == "2019-04-01T00:00:00Z"
    assert meta["output_path"].endswith("_v2.json")


def test_build_hourly_seasonality_split(tmp_path, monkeypatch):
    data_path = tmp_path / "seasonality.csv"
    start_ms = parse_time_to_ms("2021-04-01T00:00:00Z")
    end_ms = parse_time_to_ms("2021-05-01T00:00:00Z")
    extra_ms = parse_time_to_ms("2021-06-01T00:00:00Z")
    ts_values = [start_ms + i * 3_600_000 for i in range(10)] + [end_ms, extra_ms]
    df = pd.DataFrame(
        {
            "ts_ms": ts_values,
            "liquidity": np.linspace(1.0, 2.0, num=len(ts_values)),
            "latency_ms": np.linspace(10, 20, num=len(ts_values)),
            "spread": np.linspace(0.1, 0.2, num=len(ts_values)),
        }
    )
    df.to_csv(data_path, index=False)

    config_path = tmp_path / "offline.yaml"
    output_base = tmp_path / "seasonality" / "profile.json"
    config_payload = {
        "datasets": {
            "sample": {
                "version": "v-season",
                "start": "2021-03-01T00:00:00Z",
                "end": "2021-06-30T00:00:00Z",
                "seasonality": {
                    "input": {
                        "start": "2021-04-01T00:00:00Z",
                        "end": "2021-05-01T00:00:00Z",
                    },
                    "output_path": str(output_base),
                },
            }
        }
    }
    config_path.write_text(yaml.safe_dump(config_payload))

    args = [
        "prog",
        "--data",
        str(data_path),
        "--split",
        "sample",
        "--config",
        str(config_path),
    ]
    monkeypatch.setattr(sys, "argv", args)
    hourly_cli.main()

    out_path = apply_split_tag(output_base, "v-season")
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    meta = payload["metadata"]
    assert meta["split"] == {"name": "sample", "version": "v-season"}
    assert meta["data_window"]["config"]["end"] == "2021-05-01T00:00:00Z"
    assert meta["data_window"]["actual"]["end"] == "2021-05-01T00:00:00Z"
    assert meta["data_window"]["actual"]["start"] == "2021-04-01T00:00:00Z"


def test_build_spread_seasonality_split(tmp_path, monkeypatch):
    data_path = tmp_path / "spread.csv"
    start_ms = parse_time_to_ms("2022-07-01T00:00:00Z")
    end_ms = parse_time_to_ms("2022-07-10T00:00:00Z")
    rows = []
    for i in range(15):
        ts = start_ms + i * 3_600_000
        rows.append({"timestamp": ts, "high": 102.0 + i, "low": 100.0 + i})
    rows.append({"timestamp": end_ms, "high": 200.0, "low": 199.0})
    rows.append({"timestamp": parse_time_to_ms("2022-08-01T00:00:00Z"), "high": 150.0, "low": 149.0})
    pd.DataFrame(rows).to_csv(data_path, index=False)

    config_path = tmp_path / "offline.yaml"
    output_base = tmp_path / "spread" / "profile.json"
    config_payload = {
        "datasets": {
            "sample": {
                "version": "v-spread",
                "start": "2022-06-01T00:00:00Z",
                "end": "2022-08-31T00:00:00Z",
                "seasonality": {
                    "input": {
                        "start": "2022-07-01T00:00:00Z",
                        "end": "2022-07-10T00:00:00Z",
                    },
                    "output_path": str(output_base),
                },
            }
        }
    }
    config_path.write_text(yaml.safe_dump(config_payload))

    args = [
        "prog",
        "--data",
        str(data_path),
        "--split",
        "sample",
        "--config",
        str(config_path),
        "--ts-col",
        "timestamp",
    ]
    monkeypatch.setattr(sys, "argv", args)
    spread_cli.main()

    out_path = apply_split_tag(output_base, "v-spread")
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    meta = payload["metadata"]
    assert meta["split"] == {"name": "sample", "version": "v-spread"}
    assert meta["data_window"]["config"]["start"] == "2022-07-01T00:00:00Z"
    assert meta["data_window"]["actual"]["end"].startswith("2022-07-10")
