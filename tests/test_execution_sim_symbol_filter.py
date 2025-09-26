import math

import pytest

from execution_sim import SymbolFilterSnapshot


def test_min_qty_threshold_aligns_with_step_rounding_up():
    filters = SymbolFilterSnapshot(qty_min=0.0011, qty_step=0.0005)

    assert filters.min_qty_threshold == pytest.approx(0.0015)


def test_min_qty_threshold_without_step_uses_min_qty():
    filters = SymbolFilterSnapshot(qty_min=0.25, qty_step=0.0)

    assert filters.min_qty_threshold == pytest.approx(0.25)


@pytest.mark.parametrize(
    "qty_min, qty_step, expected",
    [
        (0.0, 0.0, 0.0),
        (0.0, 0.001, 0.0),
        (-0.5, 0.1, 0.0),
        (1e-12, 0.1, 0.1),
    ],
)
def test_min_qty_threshold_handles_edge_cases(qty_min: float, qty_step: float, expected: float):
    filters = SymbolFilterSnapshot(qty_min=qty_min, qty_step=qty_step)

    assert math.isclose(filters.min_qty_threshold, expected, rel_tol=0, abs_tol=1e-15)
