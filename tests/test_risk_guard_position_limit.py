import pathlib
import sys


sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from action_proto import ActionProto, ActionType
from risk_guard import RiskGuard, RiskConfig, RiskEvent


class StateWithoutMax:
    def __init__(self) -> None:
        self.cash = 0.0
        self.units = 0.0


def test_missing_state_max_position_uses_config_limit():
    cfg = RiskConfig(max_abs_position=10.0)
    guard = RiskGuard(cfg)
    state = StateWithoutMax()

    max_pos = guard._get_max_position_from_state_or_cfg(state, cfg)
    assert max_pos == 10.0

    proto = ActionProto(action_type=ActionType.MARKET, volume_frac=1.0)

    event = guard.on_action_proposed(state, proto)
    assert event == RiskEvent.NONE

    # движение на полный объём (10 контрактов) допустимо, но превышение должно блокироваться
    state.units = 9.0
    event_limit = guard.on_action_proposed(state, proto)
    assert event_limit == RiskEvent.POSITION_LIMIT
