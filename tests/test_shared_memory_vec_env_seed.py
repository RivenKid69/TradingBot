import numpy as np
import pathlib, sys
import pytest
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

try:
    from gymnasium import Env, spaces
    from shared_memory_vec_env import SharedMemoryVecEnv
except ModuleNotFoundError:
    pytest.skip("stable-baselines3 or gymnasium is not available", allow_module_level=True)

class RNGEnv(Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        return np.array([np.random.random()], dtype=np.float32), {}

    def step(self, action):
        obs = np.array([np.random.random()], dtype=np.float32)
        return obs, 0.0, False, False, {}

def test_independent_rng_per_worker():
    vec_env = SharedMemoryVecEnv([lambda: RNGEnv(), lambda: RNGEnv()], base_seed=123)
    obs = vec_env.reset()
    assert obs[0] != obs[1]
    vec_env.step_async(np.zeros((2,1), dtype=np.float32))
    obs2, _, _, _ = vec_env.step_wait()
    assert obs2[0] != obs2[1]
    vec_env.close()
    for p in vec_env.processes:
        assert not p.is_alive()
