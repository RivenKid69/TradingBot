from __future__ import annotations

import csv
import math
import sys
import types
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

# ``service_backtest`` depends on ``exchange.specs`` which is not required for these
# tests. Provide a lightweight shim before importing the module.
if "exchange" not in sys.modules:
    exchange_module = types.ModuleType("exchange")
    specs_module = types.ModuleType("exchange.specs")

    def _identity_round(price: float, *args: Any, **kwargs: Any) -> float:
        return price

    specs_module.load_specs = lambda *args, **kwargs: {}
    specs_module.round_price_to_tick = _identity_round
    exchange_module.specs = specs_module
    sys.modules["exchange"] = exchange_module
    sys.modules["exchange.specs"] = specs_module

from service_backtest import BarBacktestSimBridge, ServiceBacktest


class _FakeExecutor:
    """Minimal stand-in for :class:`impl_bar_executor.BarExecutor`."""

    def __init__(self, symbol: str = "FAKE") -> None:
        self.symbol = symbol


@pytest.fixture
def bridge() -> BarBacktestSimBridge:
    executor = _FakeExecutor("bridge_symbol")
    return BarBacktestSimBridge(
        executor,
        symbol="override",
        timeframe_ms=60_000,
        initial_equity=100.0,
    )


def test_bridge_build_bar_and_vol_estimator(bridge: BarBacktestSimBridge) -> None:
    estimator = bridge.vol_estimator
    assert estimator.observe(symbol="BTCUSDT") == 0.0
    assert estimator.value("BTCUSDT") is None
    assert estimator.last("BTCUSDT") is None

    bar = bridge._build_bar(
        ts_ms=123_456,
        symbol="btcusdt",
        open_price=None,
        high_price=Decimal("101.5"),
        low_price="100.0",
        close_price=101.25,
    )

    assert bar.ts == 123_456
    assert bar.symbol == "BTCUSDT"
    # ``open`` falls back to ``close`` when coercion fails.
    assert bar.open == Decimal("101.25")
    assert bar.high == Decimal("101.5")
    assert bar.low == Decimal("100.0")
    assert bar.close == Decimal("101.25")


@pytest.mark.parametrize(
    "value, default, expected, is_nan",
    [
        (1, 0.0, 1.0, False),
        (Decimal("3.14"), 0.0, 3.14, False),
        ("2.5", 1.0, 2.5, False),
        (None, 7.0, 7.0, False),
        ("NaN", 5.0, math.nan, True),
        ("oops", 9.0, 9.0, False),
    ],
)
def test_bridge_safe_float_coercion(
    bridge: BarBacktestSimBridge,
    value: Any,
    default: float,
    expected: float,
    is_nan: bool,
) -> None:
    result = bridge._safe_float(value, default)
    if is_nan:
        assert math.isnan(result)
    else:
        assert result == pytest.approx(expected)


def _read_csv(path: Path) -> Sequence[Mapping[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_write_bar_reports_creates_expected_outputs(tmp_path: Path) -> None:
    service = ServiceBacktest.__new__(ServiceBacktest)

    target = tmp_path / "bars" / "report.csv"
    records = [
        {"symbol": "BTC", "value": 1},
        {"symbol": "ETH", "value": 2},
    ]

    service._write_bar_reports(
        str(target),
        records=records,
        summary={"total": 3},
    )

    assert target.exists()
    assert _read_csv(target) == [
        {"symbol": "BTC", "value": "1"},
        {"symbol": "ETH", "value": "2"},
    ]

    summary_path = Path(service._bar_summary_path(str(target)))
    assert summary_path.name.endswith("_summary.csv")
    assert summary_path.exists()
    assert _read_csv(summary_path) == [{"total": "3"}]

    no_summary_target = tmp_path / "bars_no_summary" / "bars.csv"
    service._write_bar_reports(
        str(no_summary_target),
        records=records,
        summary=None,
    )
    assert no_summary_target.exists()
    assert not Path(service._bar_summary_path(str(no_summary_target))).exists()

    malformed_target = tmp_path / "bars_malformed" / "bars.csv"
    service._write_bar_reports(
        str(malformed_target),
        records=records,
        summary="not-a-mapping",
    )
    assert malformed_target.exists()
    assert not Path(service._bar_summary_path(str(malformed_target))).exists()
