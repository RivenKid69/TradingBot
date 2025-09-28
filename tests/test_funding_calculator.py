import math

from fees import FundingCalculator


def test_funding_schedule_reset_on_position_close():
    calc = FundingCalculator(
        enabled=True,
        const_rate_per_interval=0.01,
        interval_seconds=60,
        align_to_epoch=False,
    )

    total_open, events_open = calc.accrue(
        position_qty=1.0,
        mark_price=100.0,
        now_ts_ms=0,
    )

    assert math.isclose(total_open, 0.0)
    assert events_open == []
    assert calc._next_ts_ms == 60_000

    total_close, events_close = calc.accrue(
        position_qty=0.0,
        mark_price=100.0,
        now_ts_ms=30_000,
    )

    assert math.isclose(total_close, 0.0)
    assert events_close == []
    assert calc._next_ts_ms is None

    total_reopen, events_reopen = calc.accrue(
        position_qty=1.0,
        mark_price=100.0,
        now_ts_ms=90_000,
    )

    assert math.isclose(total_reopen, 0.0)
    assert events_reopen == []
    assert calc._next_ts_ms == 150_000
