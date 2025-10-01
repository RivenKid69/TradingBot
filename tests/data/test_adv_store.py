import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import pytest

from adv_store import ADVStore


def _write_dataset(path, data: Mapping[str, Any], meta: Mapping[str, Any] | None = None) -> None:
    payload = {"data": data}
    if meta is not None:
        payload["meta"] = meta
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_bar_dataset(
    path, data: Mapping[str, Any], meta: Mapping[str, Any] | None = None
) -> None:
    payload: dict[str, Any] = {"bars": data}
    if meta is not None:
        payload["meta"] = meta
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_path_and_refresh_staleness(tmp_path, caplog):
    dataset_name = "adv_sample.json"
    # Create an additional candidate directory to ensure resolution walks options.
    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()
    dataset_path = extra_dir / dataset_name

    now_ms = int(time.time() * 1000)
    _write_dataset(
        dataset_path,
        {"BTCUSDT": {"adv_quote": 100.0}},
        meta={"generated_at_ms": now_ms},
    )

    cfg = {
        "path": os.fspath(tmp_path / "missing.json"),
        "dataset": dataset_name,
        "extra": {"adv_path": os.fspath(extra_dir)},
        "refresh_days": 1,
    }

    store = ADVStore(cfg)
    # Path resolution should pick the dataset inside the extra directory.
    assert store.path == os.fspath(dataset_path)

    # Initial load is fresh because timestamp is current.
    assert store.get_adv_quote("BTCUSDT") == 100.0
    assert store.is_dataset_stale is False

    # Update dataset with an older timestamp to trigger stale detection and refresh logic.
    stale_ts = now_ms - int(3 * 86_400_000)
    _write_dataset(
        dataset_path,
        {"BTCUSDT": {"adv_quote": 125.0}},
        meta={"generated_at_ms": stale_ts},
    )
    future_time = time.time() + 2
    os.utime(dataset_path, (future_time, future_time))

    caplog.set_level(logging.WARNING)
    # Second call reloads due to updated mtime and marks dataset stale.
    assert store.get_adv_quote("BTCUSDT") is None
    assert store.is_dataset_stale is True
    assert any("older than" in record.getMessage() for record in caplog.records)
    # Metadata reflects the stale dataset that was just loaded.
    assert store.metadata["generated_at_ms"] == stale_ts


def test_handles_malformed_entries(tmp_path):
    dataset_path = tmp_path / "adv_bad.json"
    _write_dataset(
        dataset_path,
        {
            "BTCUSDT": {"adv_quote": 150.0},
            "ETHUSDT": {"adv_quote": -1},  # Negative values ignored.
            "LTCUSDT": "NaN",  # Non numeric ignored.
            "": 200,
        },
    )

    store = ADVStore({"path": os.fspath(dataset_path)})

    assert store.get_adv_quote("BTCUSDT") == 150.0
    assert store.get_adv_quote("ETHUSDT") is None
    assert store.get_adv_quote("LTCUSDT") is None
    # Metadata should reflect only valid symbols.
    assert store.metadata["symbol_count"] == 1


@pytest.mark.parametrize("policy,level", [("warn", logging.WARNING), ("error", logging.ERROR)])
def test_missing_symbol_policy_logs_once(tmp_path, caplog, policy, level):
    dataset_path = tmp_path / "adv_missing.json"
    _write_dataset(dataset_path, {"BTCUSDT": {"adv_quote": 90.0}})

    store = ADVStore({"path": os.fspath(dataset_path), "missing_symbol_policy": policy})

    caplog.set_level(logging.DEBUG)
    assert store.get_adv_quote("MISSING") is None
    assert sum("ADV quote missing" in record.getMessage() for record in caplog.records) == 1
    assert any(record.levelno == level for record in caplog.records)

    caplog.clear()
    assert store.get_adv_quote("MISSING") is None
    # No additional log should be emitted for the same symbol.
    assert all("ADV quote missing" not in record.getMessage() for record in caplog.records)


def test_get_bar_capacity_quote_applies_defaults_and_floor(tmp_path):
    dataset_path = tmp_path / "adv_defaults.json"
    _write_dataset(dataset_path, {"BTCUSDT": {"adv_quote": 80.0}, "ETHUSDT": {"adv_quote": 50.0}})

    store = ADVStore(
        {
            "path": os.fspath(dataset_path),
            "default_quote": 70,
            "floor_quote": 60,
        }
    )

    # Existing symbol should respect floor enforcement.
    assert store.get_bar_capacity_quote("ETHUSDT") == 60.0
    # Higher values are unaffected by the floor.
    assert store.get_bar_capacity_quote("BTCUSDT") == 80.0
    # Missing symbols fall back to default quote which is then floored.
    assert store.get_bar_capacity_quote("ADAUSDT") == 70.0

    # If floor is above default, the floor acts as the lower bound.
    store_high_floor = ADVStore(
        {
            "path": os.fspath(dataset_path),
            "default_quote": 55,
            "floor_quote": 90,
        }
    )
    assert store_high_floor.get_bar_capacity_quote("ADAUSDT") == 90.0


def test_concurrent_access_uses_single_payload_load(tmp_path, monkeypatch):
    dataset_path = tmp_path / "adv_concurrent.json"
    _write_dataset(dataset_path, {"BTCUSDT": {"adv_quote": 110.0}})

    store = ADVStore({"path": os.fspath(dataset_path)})

    call_count = 0
    orig_reader = store._read_payload

    def _wrapped_reader(path):
        nonlocal call_count
        call_count += 1
        time.sleep(0.01)
        return orig_reader(path)

    monkeypatch.setattr(store, "_read_payload", _wrapped_reader)

    results: list[float | None] = []
    errors: list[BaseException] = []

    def worker_get_adv():
        try:
            results.append(store.get_adv_quote("BTCUSDT"))
        except BaseException as exc:  # pragma: no cover - defensive
            errors.append(exc)

    threads = [threading.Thread(target=worker_get_adv) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert results == [110.0] * len(threads)
    assert call_count == 1


def test_extract_bar_meta_populates_path_and_symbol_count(tmp_path):
    dataset_path = tmp_path / "bars.json"
    payload = {
        "meta": {"generated_at": "2023-12-10T00:00:00Z", "other": "value"},
        "bars": {
            "btcusdt": {"adv": 1},
            " ETHUSDT ": {"adv": 2},
            "": {"adv": 3},
        },
    }

    store = ADVStore({})
    bars, meta = store._extract_bar_meta(payload, os.fspath(dataset_path))

    assert bars == {"BTCUSDT": {"adv": 1}, "ETHUSDT": {"adv": 2}}
    assert meta["path"] == os.fspath(dataset_path)
    assert meta["symbol_count"] == 2
    # Existing metadata should be preserved.
    assert meta["other"] == "value"


def test_maybe_reset_bar_cache_detects_file_change(tmp_path):
    dataset_path = tmp_path / "adv_bars.json"
    _write_bar_dataset(dataset_path, {"BTCUSDT": {"adv": 10}})

    store = ADVStore({"bar_cache_limit": 4})

    data_one, _ = store.load_bar_dataset(os.fspath(dataset_path))
    assert data_one["BTCUSDT"] == {"adv": 10}

    _write_bar_dataset(dataset_path, {"BTCUSDT": {"adv": 25}})
    future_ts = time.time() + 1
    os.utime(dataset_path, (future_ts, future_ts))

    data_two, _ = store.load_bar_dataset(os.fspath(dataset_path))
    assert data_two["BTCUSDT"] == {"adv": 25}


def test_extract_timestamp_and_stale_transitions(tmp_path):
    dataset_path = tmp_path / "adv_stale.json"
    past_dt = datetime.now(tz=timezone.utc) - timedelta(days=5)
    _write_dataset(
        dataset_path,
        {"BTCUSDT": {"adv_quote": 10.0}},
        meta={"generated_at": past_dt.isoformat()},
    )

    store = ADVStore({"path": os.fspath(dataset_path), "refresh_days": 2})

    # Initial load should mark dataset stale because timestamp is too old.
    assert store.get_adv_quote("BTCUSDT") is None
    assert store.is_dataset_stale is True

    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    _write_dataset(
        dataset_path,
        {"BTCUSDT": {"adv_quote": 32.0}},
        meta={"generated_at_ms": now_ms},
    )
    future_ts = time.time() + 1
    os.utime(dataset_path, (future_ts, future_ts))

    assert store.get_adv_quote("BTCUSDT") == 32.0
    assert store.is_dataset_stale is False


def test_trim_bar_cache_and_symbol_normalisation(tmp_path, monkeypatch):
    store = ADVStore({"bar_cache_limit": 2})

    paths: list[str] = []
    for idx in range(3):
        path = tmp_path / f"bars_{idx}.json"
        symbol = f"sym{idx}usd"
        _write_bar_dataset(path, {symbol: {"adv": idx + 1}})
        paths.append(os.fspath(path))

    call_count: dict[str, int] = {}
    original_reader = store._read_bar_payload

    def _wrapped_reader(path: str):
        call_count[path] = call_count.get(path, 0) + 1
        return original_reader(path)

    monkeypatch.setattr(store, "_read_bar_payload", _wrapped_reader)

    # Load first two datasets into the cache.
    for path in paths[:2]:
        data, meta = store.load_bar_dataset(path)
        assert meta["symbol_count"] == 1
        assert call_count[path] == 1
        # Symbol lookup should be case-insensitive and trimmed.
        symbol = next(iter(data))
        assert store.get_bar_entry(path, symbol.lower()) == data[symbol]
        assert store.get_bar_entry(path, f"  {symbol.lower()}  ") == data[symbol]

    # Loading a third dataset should evict the least recently used (first) entry.
    store.load_bar_dataset(paths[2])
    assert call_count[paths[2]] == 1

    # Re-loading the first dataset should trigger a fresh read.
    store.load_bar_dataset(paths[0])
    assert call_count[paths[0]] == 2
