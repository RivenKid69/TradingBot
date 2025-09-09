import os
import sys
import types
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
# stub minimal infra modules required for import
infra_pkg = types.ModuleType("infra")
event_bus_stub = types.ModuleType("event_bus")
event_bus_stub.publish = lambda *a, **k: None
event_bus_stub.Topics = object
time_provider_stub = types.ModuleType("time_provider")
time_provider_stub.TimeProvider = object
time_provider_stub.RealTimeProvider = object
sys.modules["infra"] = infra_pkg
sys.modules["infra.event_bus"] = event_bus_stub
sys.modules["infra.time_provider"] = time_provider_stub
lob_state_stub = types.ModuleType("lob_state_cython")
lob_state_stub.N_FEATURES = 1
sys.modules["lob_state_cython"] = lob_state_stub
mediator_stub = types.ModuleType("mediator")
class _Mediator:
    def __init__(self, env):
        self.env = env
    def step(self, proto):
        return np.zeros(1), 0.0, False, False, {}
    def reset(self):
        return np.zeros(1, dtype=np.float32), {}
mediator_stub.Mediator = _Mediator
sys.modules["mediator"] = mediator_stub

from trading_patchnew import TradingEnv


def test_close_not_in_observation():
    df = pd.DataFrame(
        {
            "open": [1, 1, 1, 1],
            "high": [1, 1, 1, 1],
            "low": [1, 1, 1, 1],
            "close": [100.0, 101.0, 102.0, 103.0],
            "price": [1, 1, 1, 1],
            "quote_asset_volume": [1, 1, 1, 1],
        }
    )
    env = TradingEnv(df)
    obs, _ = env.reset()
    assert not np.isclose(obs, 100.0).any()
    expected = pd.Series([np.nan, 100.0, 101.0, 102.0], name="close")
    pd.testing.assert_series_equal(env.df["close"], expected)
