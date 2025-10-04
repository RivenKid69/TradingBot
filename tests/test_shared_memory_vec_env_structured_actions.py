import numpy as np
import pathlib
import sys
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

try:
    from gymnasium import Env, spaces
    from shared_memory_vec_env import SharedMemoryVecEnv
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    pytest.skip("stable-baselines3 or gymnasium is not available", allow_module_level=True)


class DictActionEnv(Env):
    def __init__(self):
        self.action_space = spaces.Dict(
            {
                "box": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                "disc": spaces.Discrete(5),
            }
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        assert isinstance(action, dict)
        box = np.asarray(action["box"], dtype=np.float32)
        disc = int(action["disc"])
        info = {
            "echo": {
                "box": box.copy(),
                "disc": disc,
            }
        }
        return np.ones(1, dtype=np.float32), float(disc), False, False, info


def test_dict_action_round_trip():
    vec_env = SharedMemoryVecEnv([lambda: DictActionEnv()])
    try:
        obs = vec_env.reset()
        assert obs.shape == (1, 1)

        action = {
            "box": np.array([0.25, -0.75], dtype=np.float32),
            "disc": 3,
        }

        obs, rewards, dones, infos = vec_env.step([action])
        np.testing.assert_allclose(obs, np.ones((1, 1), dtype=np.float32))
        np.testing.assert_allclose(rewards, np.array([3.0], dtype=np.float32))
        np.testing.assert_array_equal(dones, np.array([False]))

        echoed = infos[0]["echo"]
        np.testing.assert_array_equal(echoed["box"], action["box"])
        assert echoed["disc"] == action["disc"]
    finally:
        vec_env.close()
