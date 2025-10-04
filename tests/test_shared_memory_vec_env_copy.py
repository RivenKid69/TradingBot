import numpy as np
import pathlib, sys
import pytest
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

try:
    from gymnasium import Env, spaces
    from shared_memory_vec_env import SharedMemoryVecEnv
except ModuleNotFoundError:
    pytest.skip("stable-baselines3 or gymnasium is not available", allow_module_level=True)

class DummyEnv(Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.ones(2, dtype=np.float32), 0.0, True, False, {}


def test_step_returns_copies():
    vec_env = SharedMemoryVecEnv([lambda: DummyEnv()])
    obs = vec_env.reset()
    assert obs.base is None
    obs, rewards, dones, _ = vec_env.step(np.zeros((1,1), dtype=np.float32))
    assert obs.base is None
    assert rewards.base is None
    assert dones.base is None
    vec_env.close()
