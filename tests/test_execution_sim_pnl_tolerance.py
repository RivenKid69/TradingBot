import logging

import pytest

from execution_sim import (
    _PNL_RECONCILE_REL_TOL,
    _check_pnl_reconciliation,
)


def test_pnl_reconciliation_large_notional(caplog: pytest.LogCaptureFixture) -> None:
    expected = 1e12
    allowed_diff = expected * _PNL_RECONCILE_REL_TOL * 0.5

    with caplog.at_level(logging.WARNING):
        _check_pnl_reconciliation(expected, expected + allowed_diff)

    assert not caplog.records

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _check_pnl_reconciliation(expected, expected + allowed_diff * 4)

    assert any(
        "PnL reconciliation drift exceeds tolerance" in record.getMessage()
        for record in caplog.records
    )
