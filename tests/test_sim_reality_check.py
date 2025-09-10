from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.metrics import calculate_metrics, equity_from_trades
from scripts.sim_reality_check import (
    _latency_stats,
    _order_fill_stats,
    _cancel_stats,
)


DATA_DIR = Path(__file__).parent / "data"


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def test_kpi_computation() -> None:
    sim = _read_csv("sim_trades.csv")
    hist = _read_csv("hist_trades.csv")

    # equity & trade metrics
    metrics = calculate_metrics(sim, equity_from_trades(sim))
    assert metrics["trades"]["n_trades"] == 3
    assert metrics["equity"]["pnl_total"] == 5.0

    # latency KPIs
    sim_lat = _latency_stats(sim)
    hist_lat = _latency_stats(hist)
    assert sim_lat["p50_ms"] == pytest.approx(70.0)
    assert sim_lat["p95_ms"] == pytest.approx(97.0)
    assert hist_lat["p50_ms"] == pytest.approx(60.0)

    # order fill fractions
    fill = _order_fill_stats(sim)
    assert fill["fraction_partially_filled"] == pytest.approx(1 / 3)
    assert fill["fraction_unfilled"] == pytest.approx(1 / 3)

    # cancellation counts
    cancel = _cancel_stats(sim)
    assert cancel["counts"]["TOO_LATE"] == 1
    assert cancel["shares"]["TOO_LATE"] == pytest.approx(1.0)


def _check(values: dict, specs: dict, prefix: str = "", flags: dict | None = None) -> dict:
    if flags is None:
        flags = {}
    for key, spec in specs.items():
        if isinstance(spec, dict) and {"min", "max"} <= set(spec.keys()):
            actual = values.get(key)
            if actual is None or not (spec["min"] <= actual <= spec["max"]):
                flags[prefix + key] = "нереалистично"
        elif isinstance(spec, dict):
            _check(values.get(key, {}), spec, prefix + key + ".", flags)
    return flags


def test_flagging_out_of_tolerance() -> None:
    sim = _read_csv("sim_trades.csv")
    kpi_values = {
        "latency_ms": _latency_stats(sim),
        "order_fill": _order_fill_stats(sim),
    }
    thresholds = {
        "latency_ms": {"p50_ms": {"min": 0, "max": 60}},
        "order_fill": {"fraction_unfilled": {"min": 0, "max": 0.2}},
    }
    flags = _check(kpi_values, thresholds)
    assert "latency_ms.p50_ms" in flags
    assert "order_fill.fraction_unfilled" in flags
