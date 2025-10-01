import csv
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pytest

if "exchange" not in sys.modules:
    exchange_pkg = types.ModuleType("exchange")
    specs_mod = types.ModuleType("exchange.specs")
    specs_mod.load_specs = lambda *args, **kwargs: None
    specs_mod.round_price_to_tick = lambda price, tick=None: price
    exchange_pkg.specs = specs_mod
    sys.modules["exchange"] = exchange_pkg
    sys.modules["exchange.specs"] = specs_mod

from service_backtest import (
    ServiceBacktest,
    _apply_bar_capacity_base_config,
    _finalise_bar_capacity_payload,
    _yield_bar_capacity_meta,
)


class _StubAdvStore:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path


class _RecordingSim:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def set_bar_capacity_base_config(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def test_finalise_bar_capacity_payload_with_fallbacks() -> None:
    adv_store = _StubAdvStore(path="/tmp/adv.csv")
    payload, fallbacks, missing = _finalise_bar_capacity_payload(
        {"enabled": True},
        adv_store=adv_store,
        default_timeframe_ms=15_000,
    )

    assert payload == {
        "enabled": True,
        "adv_base_path": "/tmp/adv.csv",
        "timeframe_ms": 15_000,
    }
    assert fallbacks == [
        ("adv_base_path", "/tmp/adv.csv"),
        ("timeframe_ms", 15_000),
    ]
    assert missing == []


def test_finalise_bar_capacity_payload_records_missing_fields() -> None:
    payload, fallbacks, missing = _finalise_bar_capacity_payload(
        {"enabled": True},
        adv_store=None,
        default_timeframe_ms=None,
    )

    assert payload == {"enabled": True}
    assert fallbacks == []
    assert missing == ["adv_base_path", "timeframe_ms"]


def test_apply_bar_capacity_base_config_calls_setter_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    sim = _RecordingSim()
    adv_store = _StubAdvStore(path="/tmp/adv.csv")
    caplog.set_level("WARNING")

    _apply_bar_capacity_base_config(
        sim,
        {"enabled": True},
        adv_store=adv_store,
        default_timeframe_ms=60_000,
        context="test",
    )

    assert sim.calls == [
        {
            "enabled": True,
            "adv_base_path": "/tmp/adv.csv",
            "timeframe_ms": 60_000,
        }
    ]
    warning_messages = [rec.getMessage() for rec in caplog.records]
    assert any("falling back to ADV dataset" in msg for msg in warning_messages)
    assert any("falling back to timeframe" in msg for msg in warning_messages)


def test_apply_bar_capacity_base_config_logs_missing_fields(caplog: pytest.LogCaptureFixture) -> None:
    sim = _RecordingSim()
    caplog.set_level("WARNING")

    _apply_bar_capacity_base_config(
        sim,
        {"enabled": True},
        adv_store=None,
        default_timeframe_ms=None,
        context="ctx",
    )

    assert sim.calls == [{"enabled": True}]
    warning_messages = [rec.getMessage() for rec in caplog.records]
    assert any(
        "ctx: bar_capacity_base.adv_base_path not configured and no fallback available"
        in msg
        for msg in warning_messages
    )
    assert any(
        "ctx: bar_capacity_base.timeframe_ms not configured and no fallback available"
        in msg
        for msg in warning_messages
    )


def test_apply_bar_capacity_base_config_bypasses_when_unavailable(caplog: pytest.LogCaptureFixture) -> None:
    class NoConfigSim:
        pass

    sim = NoConfigSim()
    caplog.set_level("WARNING")

    _apply_bar_capacity_base_config(
        sim,
        {"enabled": True},
        adv_store=_StubAdvStore(path="/tmp/adv.csv"),
        default_timeframe_ms=1_000,
        context="simctx",
    )

    warning_messages = [rec.getMessage() for rec in caplog.records]
    assert warning_messages == [
        "simctx: ExecutionSimulator lacks set_bar_capacity_base_config(); bar capacity base disabled"
    ]


def test_yield_bar_capacity_meta_prefers_core_reports() -> None:
    report = {
        "core_exec_reports": [
            {"meta": {"bar_capacity_base": {"source": "core"}}},
            {"meta": {"bar_capacity_base": {"source": "core2"}}},
        ],
        "trades": [
            {"meta": {"bar_capacity_base": {"source": "trade"}}},
            {"capacity_reason": "BAR_CAPACITY_BASE"},
        ],
    }

    result = _yield_bar_capacity_meta(report)
    assert result == [{"source": "core"}, {"source": "core2"}]


def test_yield_bar_capacity_meta_falls_back_to_trades() -> None:
    report = {
        "core_exec_reports": [
            {"meta": {"bar_capacity_base": "not_a_mapping"}},
        ],
        "trades": [
            {"meta": {"bar_capacity_base": {"order": 1}}},
            {"capacity_reason": "BAR_CAPACITY_BASE", "order": 2},
        ],
    }

    result = _yield_bar_capacity_meta(report)
    assert result == [{"order": 1}, {"capacity_reason": "BAR_CAPACITY_BASE", "order": 2}]


def test_write_bar_reports_creates_files_and_handles_summary(tmp_path: Path) -> None:
    sb = ServiceBacktest.__new__(ServiceBacktest)
    target_dir = tmp_path / "bars" / "nested"
    path = target_dir / "report.csv"
    records: List[Mapping[str, Any]] = [
        {"symbol": "BTC", "value": 1},
        {"symbol": "ETH", "value": 2},
    ]
    summary: Mapping[str, Any] = {"total": 3}

    sb._write_bar_reports(
        str(path),
        records=records,
        summary=summary,
    )

    assert path.exists()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows == [
        {"symbol": "BTC", "value": "1"},
        {"symbol": "ETH", "value": "2"},
    ]

    summary_path = sb._bar_summary_path(str(path))
    assert os.path.exists(summary_path)
    with open(summary_path, newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows == [{"total": "3"}]

    other_path = tmp_path / "bars_no_summary" / "bars.csv"
    sb._write_bar_reports(
        str(other_path),
        records=records,
        summary=None,
    )
    assert other_path.exists()
    assert not os.path.exists(sb._bar_summary_path(str(other_path)))
