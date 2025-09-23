import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from risk import RiskManager, RiskConfig


def test_liquidity_multiplier_scales_limits():
    cfg = RiskConfig(enabled=True, max_abs_position_qty=10.0)
    rm = RiskManager(cfg)
    allowed = rm.pre_trade_adjust(
        ts_ms=0,
        side="BUY",
        intended_qty=8.0,
        price=None,
        position_qty=0.0,
        liquidity_mult=0.5,
    )
    assert allowed == 5.0


def test_latency_multiplier_scales_order_rate():
    cfg = RiskConfig(enabled=True, max_orders_per_min=10, max_orders_window_s=60)
    rm = RiskManager(cfg)
    for i in range(5):
        assert rm.can_send_order(ts_ms=i * 1000, latency_mult=2.0)
        rm.on_new_order(i * 1000)
    assert not rm.can_send_order(ts_ms=5000, latency_mult=2.0)


def test_daily_entry_limit_blocks_after_threshold():
    cfg = RiskConfig(enabled=True, max_entries_per_day=2)
    rm = RiskManager(cfg)

    qty = rm.pre_trade_adjust(
        ts_ms=0,
        side="BUY",
        intended_qty=1.0,
        price=100.0,
        position_qty=0.0,
    )
    assert qty == 1.0
    rm.pop_events()

    qty = rm.pre_trade_adjust(
        ts_ms=1,
        side="SELL",
        intended_qty=1.0,
        price=100.0,
        position_qty=1.0,
    )
    assert qty == 1.0
    rm.pop_events()

    qty = rm.pre_trade_adjust(
        ts_ms=2,
        side="SELL",
        intended_qty=1.0,
        price=100.0,
        position_qty=0.0,
    )
    assert qty == 1.0
    rm.pop_events()

    blocked = rm.pre_trade_adjust(
        ts_ms=3,
        side="BUY",
        intended_qty=2.0,
        price=100.0,
        position_qty=-1.0,
    )
    assert blocked == 0.0
    events = rm.pop_events()
    assert any(ev.code == "ENTRY_LIMIT_BLOCK" for ev in events)

    allowed_next_day = rm.pre_trade_adjust(
        ts_ms=86_400_000,
        side="BUY",
        intended_qty=1.0,
        price=100.0,
        position_qty=0.0,
    )
    assert allowed_next_day == 1.0


def test_daily_loss_limit_handles_none_initial_equity():
    cfg = RiskConfig(enabled=True, daily_loss_limit=50.0, pause_seconds_on_violation=60)
    rm = RiskManager(cfg)

    rm.on_mark(ts_ms=0, equity=None)
    assert rm._equity_day_start is None
    assert rm.pop_events() == []

    rm.on_mark(ts_ms=1, equity=1_000.0)
    assert rm._equity_day_start == 1_000.0
    assert rm.pop_events() == []

    rm.on_mark(ts_ms=2, equity=940.0)
    events = rm.pop_events()
    assert any(ev.code == "DAILY_LOSS_PAUSE" for ev in events)
    assert rm.paused_until_ms > 0
