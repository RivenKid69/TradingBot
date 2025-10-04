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
        reward = float(np.random.random())
        return obs, reward, False, False, {}


def rollout(seed, steps):
    vec_env = SharedMemoryVecEnv([lambda: RNGEnv()], base_seed=seed)
    obs = vec_env.reset()
    obs_seq = [obs.copy()]
    reward_seq = []
    for _ in range(steps):
        obs, rewards, _, _ = vec_env.step(np.zeros((1,1), dtype=np.float32))
        obs_seq.append(obs.copy())
        reward_seq.append(rewards.copy())
    vec_env.close()
    return obs_seq, reward_seq


def test_reproducible_rollout():
    steps = 5
    obs_seq1, reward_seq1 = rollout(123, steps)
    obs_seq2, reward_seq2 = rollout(123, steps)
    for a, b in zip(obs_seq1, obs_seq2):
        assert np.allclose(a, b)
    for a, b in zip(reward_seq1, reward_seq2):
        assert np.allclose(a, b)
