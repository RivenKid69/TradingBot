import json
import logging
import os
import threading
import time
from typing import Any, Mapping

import pytest

from adv_store import ADVStore


def _write_dataset(path, data: Mapping[str, Any], meta: Mapping[str, Any] | None = None) -> None:
    payload = {"data": data}
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

    # Concurrent bar-capacity calls should reuse the cached payload without extra loads.
    results.clear()
    threads = [threading.Thread(target=lambda: results.append(store.get_bar_capacity_quote("BTCUSDT"))) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [110.0] * len(threads)
    assert call_count == 1
