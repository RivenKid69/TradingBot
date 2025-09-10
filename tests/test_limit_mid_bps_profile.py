import pytest
from test_execution_profiles import base_sim, ActionProto, ActionType


def test_limit_mid_bps_profile(base_sim):
    sim = base_sim
    params = {"limit_offset_bps": 200}
    sim.set_execution_profile("LIMIT_MID_BPS", params)
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)
    rep = sim.run_step(
        ts=0,
        ref_price=100.0,
        bid=99.0,
        ask=101.0,
        vol_factor=1.0,
        liquidity=5.0,
        actions=[(ActionType.MARKET, proto)],
    )
    assert len(rep.trades) == 1
    trade = rep.trades[0]
    assert trade.price == pytest.approx(101.0)
    assert trade.proto_type == ActionType.LIMIT
    assert rep.fee_total == pytest.approx(trade.price * trade.qty * 0.001)
    assert rep.position_qty == pytest.approx(trade.qty)
