import pytest

from risk import RiskConfig, RiskManager


def _event_codes(manager: RiskManager) -> set[str]:
    return {ev.code for ev in manager.pop_events()}


def test_total_notional_limit_clamps_quantity():
    cfg = RiskConfig(
        enabled=True,
        max_total_notional=600.0,
        exposure_buffer_frac=0.1,
    )
    rm = RiskManager(cfg)

    qty = rm.pre_trade_adjust(
        ts_ms=0,
        side="BUY",
        intended_qty=2.0,
        price=100.0,
        position_qty=5.0,
    )

    assert qty == pytest.approx(0.9090909, rel=1e-6)
    assert "TOTAL_NOTIONAL_CLAMP" in _event_codes(rm)


def test_total_notional_limit_blocks_when_no_capacity():
    cfg = RiskConfig(
        enabled=True,
        max_total_notional=600.0,
        exposure_buffer_frac=0.05,
    )
    rm = RiskManager(cfg)

    qty = rm.pre_trade_adjust(
        ts_ms=0,
        side="BUY",
        intended_qty=1.0,
        price=100.0,
        position_qty=6.0,
    )

    assert qty == 0.0
    assert "TOTAL_NOTIONAL_BLOCK" in _event_codes(rm)


def test_total_exposure_pct_uses_equity():
    cfg = RiskConfig(enabled=True, max_total_exposure_pct=0.5)
    rm = RiskManager(cfg)
    rm.on_mark(ts_ms=0, equity=1_000.0)

    qty = rm.pre_trade_adjust(
        ts_ms=1,
        side="BUY",
        intended_qty=2.0,
        price=100.0,
        position_qty=4.0,
    )

    assert qty == pytest.approx(1.0, rel=1e-9)
    assert "TOTAL_EXPOSURE_CLAMP" in _event_codes(rm)


def test_total_notional_respects_external_total():
    cfg = RiskConfig(enabled=True, max_total_notional=550.0)
    rm = RiskManager(cfg)

    qty = rm.pre_trade_adjust(
        ts_ms=0,
        side="BUY",
        intended_qty=5.0,
        price=100.0,
        position_qty=2.0,
        total_notional=500.0,
    )

    assert qty == pytest.approx(0.5, rel=1e-9)
    assert "TOTAL_NOTIONAL_CLAMP" in _event_codes(rm)


def test_total_notional_allows_reduction():
    cfg = RiskConfig(enabled=True, max_total_notional=500.0)
    rm = RiskManager(cfg)

    qty = rm.pre_trade_adjust(
        ts_ms=0,
        side="SELL",
        intended_qty=2.0,
        price=100.0,
        position_qty=5.0,
    )

    assert qty == pytest.approx(2.0, rel=1e-9)
    assert "TOTAL_NOTIONAL_BLOCK" not in _event_codes(rm)
