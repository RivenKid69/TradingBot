import math

import pytest

from execution_sim import ExecutionSimulator, SymbolFilterSnapshot


def test_min_qty_threshold_aligns_with_step_rounding_up():
    filters = SymbolFilterSnapshot(qty_min=0.0011, qty_step=0.0005)

    assert filters.min_qty_threshold == pytest.approx(0.0015)


def test_min_qty_threshold_without_step_uses_min_qty():
    filters = SymbolFilterSnapshot(qty_min=0.25, qty_step=0.0)

    assert filters.min_qty_threshold == pytest.approx(0.25)


def test_min_qty_threshold_rounds_up_when_above_multiple():
    filters = SymbolFilterSnapshot(qty_min=0.010000000000001, qty_step=0.01)

    assert filters.min_qty_threshold == pytest.approx(0.02)


@pytest.mark.parametrize(
    "qty_min, qty_step, expected",
    [
        (0.0, 0.0, 0.0),
        (0.0, 0.001, 0.001),
        (-0.5, 0.1, 0.1),
        (1e-12, 0.1, 0.1),
    ],
)
def test_min_qty_threshold_handles_edge_cases(qty_min: float, qty_step: float, expected: float):
    filters = SymbolFilterSnapshot(qty_min=qty_min, qty_step=qty_step)

    assert math.isclose(filters.min_qty_threshold, expected, rel_tol=0, abs_tol=1e-15)


def test_percent_price_bounds_buy_prefers_bid_sell_prefers_ask():
    filters = SymbolFilterSnapshot(
        multiplier_up=1.2,
        multiplier_down=0.8,
        ask_multiplier_up=1.15,
        ask_multiplier_down=0.85,
        bid_multiplier_up=1.25,
        bid_multiplier_down=0.75,
    )

    buy_up, buy_down = filters.percent_price_bounds("BUY")
    sell_up, sell_down = filters.percent_price_bounds("SELL")

    assert buy_up == pytest.approx(1.25)
    assert buy_down == pytest.approx(0.75)
    assert sell_up == pytest.approx(1.15)
    assert sell_down == pytest.approx(0.85)


def test_percent_price_bounds_fallback_to_generic_when_missing():
    filters = SymbolFilterSnapshot(
        multiplier_up=1.2,
        multiplier_down=0.8,
        ask_multiplier_up=None,
        ask_multiplier_down=None,
        bid_multiplier_up=None,
        bid_multiplier_down=0.7,
    )

    buy_up, buy_down = filters.percent_price_bounds("BUY")
    sell_up, sell_down = filters.percent_price_bounds("SELL")

    assert buy_up == pytest.approx(1.2)
    assert buy_down == pytest.approx(0.7)
    assert sell_up == pytest.approx(1.2)
    assert sell_down == pytest.approx(0.8)


def _make_market_legacy_sim(filters: SymbolFilterSnapshot) -> ExecutionSimulator:
    sim = ExecutionSimulator.__new__(ExecutionSimulator)
    sim.symbol = "TESTUSDT"
    sim.strict_filters = True
    sim.quantizer = None
    sim._current_symbol_filters = lambda: filters
    return sim


def test_market_legacy_allows_large_step_aligned_quantity():
    filters = SymbolFilterSnapshot(qty_step=1e-3, qty_min=0.0, qty_max=1e9)
    sim = _make_market_legacy_sim(filters)

    qty, rejection = sim._apply_filters_market_legacy("BUY", 123_456.789, None)

    assert rejection is None
    assert qty == pytest.approx(123_456.789)


def test_market_legacy_rejects_misaligned_quantity():
    filters = SymbolFilterSnapshot(qty_step=1e-3, qty_min=0.0, qty_max=1e9)
    sim = _make_market_legacy_sim(filters)

    qty, rejection = sim._apply_filters_market_legacy("BUY", 123_456.7891, None)

    assert qty == 0.0
    assert rejection is not None
    assert rejection.code == "LOT_SIZE"
