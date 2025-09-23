import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from risk_guard import RiskGuard, RiskConfig, RiskEvent
from action_proto import ActionProto, ActionType

class DummyState:
    def __init__(self):
        self.cash = 100.0
        self.units = 1.0
        self.max_position = 1.0


def test_risk_guard_reset():
    rg = RiskGuard(RiskConfig())
    st = DummyState()
    # Trigger post-trade update to populate internal stats
    rg.on_post_trade(st, 10.0)
    assert rg._nw_hist and rg._peak_nw_window
    rg.reset()
    assert not rg._nw_hist
    assert not rg._peak_nw_window
    assert rg._last_event == RiskEvent.NONE


class DummyStateNoMax:
    def __init__(self):
        self.cash = 0.0
        self.units = 0.0


class DummyStateExposure:
    def __init__(self, cash: float, units: float):
        self.cash = cash
        self.units = units


def test_risk_guard_uses_cfg_limit_when_state_max_missing():
    cfg = RiskConfig(max_abs_position=5.0)
    rg = RiskGuard(cfg)
    st = DummyStateNoMax()
    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)

    event = rg.on_action_proposed(st, proto)

    assert event == RiskEvent.NONE
    max_pos = rg._get_max_position_from_state_or_cfg(st, cfg)
    assert max_pos == cfg.max_abs_position
    proposed_units = st.units + proto.volume_frac * max_pos
    assert proposed_units == cfg.max_abs_position


def test_notional_limit_based_on_position_exposure():
    cfg = RiskConfig(max_notional=100.0)
    rg = RiskGuard(cfg)
    st = DummyStateExposure(cash=1_000_000.0, units=0.0)

    event = rg.on_post_trade(st, 50.0)
    assert event == RiskEvent.NONE

    st.units = 5.0  # exposure = 250.0 > max_notional
    event = rg.on_post_trade(st, 50.0)
    assert event == RiskEvent.NOTIONAL_LIMIT
