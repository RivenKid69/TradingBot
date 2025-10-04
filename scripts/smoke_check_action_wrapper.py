"""Quick smoke-check for the Dict->MultiDiscrete action wrapper."""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd
from gymnasium import spaces

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


class _DummyResponse:
    def __init__(self, payload: dict | None = None) -> None:
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return dict(self._payload)


def _dummy_get(*args, **kwargs):
    return _DummyResponse()


if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace(get=_dummy_get)


if "lob_state_cython" not in sys.modules:
    lob_state_stub = types.ModuleType("lob_state_cython")
    lob_state_stub._compute_n_features = lambda: 1
    sys.modules["lob_state_cython"] = lob_state_stub


if "mediator" not in sys.modules:
    mediator_stub = types.ModuleType("mediator")

    class _Mediator:
        def __init__(self, env) -> None:
            self.env = env

        def step(self, proto):
            return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

        def reset(self):
            return np.zeros(1, dtype=np.float32), {}

    mediator_stub.Mediator = _Mediator
    sys.modules["mediator"] = mediator_stub


if "domain.adapters" not in sys.modules:
    adapters_stub = types.ModuleType("domain.adapters")

    def gym_to_action_v1(action):
        return action

    class _DummyProto:
        def __init__(self, payload):
            self.payload = payload

    def action_v1_to_proto(action):
        return _DummyProto(action)

    adapters_stub.gym_to_action_v1 = gym_to_action_v1
    adapters_stub.action_v1_to_proto = action_v1_to_proto
    if "domain" not in sys.modules:
        sys.modules["domain"] = types.ModuleType("domain")
    sys.modules["domain"].adapters = adapters_stub
    sys.modules["domain.adapters"] = adapters_stub


from trading_patchnew import TradingEnv
from wrappers.action_space import DictToMultiDiscreteActionWrapper


def make_single_env() -> TradingEnv:
    """Construct a minimal trading environment instance for smoke tests."""

    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.0, 1.2, 1.3, 1.4],
            "low": [1.0, 1.0, 1.1, 1.2],
            "close": [1.0, 1.1, 1.2, 1.3],
            "price": [1.0, 1.1, 1.2, 1.3],
            "quote_asset_volume": [10.0, 11.0, 12.0, 13.0],
        }
    )
    return TradingEnv(df)


def main() -> None:
    env = make_single_env()
    if isinstance(env.action_space, spaces.Dict):
        env = DictToMultiDiscreteActionWrapper(env, bins_vol=101)

    assert isinstance(env.action_space, spaces.MultiDiscrete)
    assert env.action_space.nvec.tolist()[-1] == 101

    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(
        "OK",
        type(obs),
        float(reward),
        bool(np.any(terminated)),
        bool(np.any(truncated)),
    )


if __name__ == "__main__":
    main()
