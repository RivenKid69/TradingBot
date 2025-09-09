import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from risk_guard import RiskGuard, RiskConfig, RiskEvent

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
