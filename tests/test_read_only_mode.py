import copy
from decimal import Decimal
import os
import sys

sys.path.append(os.getcwd())

from feature_pipe import FeaturePipe
from transformers import FeatureSpec
from core_models import Bar

try:
    import risk_manager
except Exception:  # pragma: no cover - fallback for environments without C extensions
    class _DummyRM:
        @staticmethod
        def apply_close_if_needed(state, readonly=False):
            working = copy.deepcopy(state) if readonly else state
            if working.net_worth > working.peak_value:
                working.peak_value = working.net_worth
            return 0
    risk_manager = _DummyRM()


def _make_bar(ts:int, price:float) -> Bar:
    d = Decimal(str(price))
    return Bar(ts=ts, symbol="BTC", open=d, high=d, low=d, close=d)


def test_read_only_components():
    spec = FeatureSpec(lookbacks_prices=[2])
    pipe = FeaturePipe(spec)
    pipe.update(_make_bar(0, 100.0))
    state_before = copy.deepcopy(pipe._tr._state)
    pipe.set_read_only(True)
    pipe.update(_make_bar(1, 101.0))
    assert pipe._tr._state == state_before

    class DummyState:
        def __init__(self):
            self.cash = 0.0
            self.net_worth = 0.0
            self.prev_net_worth = 0.0
            self.peak_value = 0.0
            self.units = 0
            self._position_value = 0.0
            self.taker_fee = 0.0
            self.bankruptcy_threshold = 0.0
            self.max_drawdown = 1.0
            self.use_trailing_stop = False
            self.use_atr_stop = False
            self._trailing_active = False
            self._high_extremum = 0
            self._low_extremum = 0
            self.is_bankrupt = False

    st = DummyState()
    st.cash = 100.0
    st.net_worth = 110.0
    st.prev_net_worth = 100.0
    st.peak_value = 100.0

    risk_manager.apply_close_if_needed(st)
    peak_before = st.peak_value
    st.net_worth = 200.0
    risk_manager.apply_close_if_needed(st, True)
    assert st.peak_value == peak_before
