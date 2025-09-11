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
