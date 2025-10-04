from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from gymnasium import spaces


class DictToMultiDiscreteActionWrapper:
    """
    Convert Dict action space:
      { price_offset_ticks: Discrete(201),
        ttl_steps:          Discrete(33),
        type:               Discrete(4),
        volume_frac:        Box(-1,1,(1,),float32) }
    -> MultiDiscrete([201, 33, 4, bins_vol])

    Agent outputs [i_price, i_ttl, i_type, i_vol]; wrapper decodes to Dict.
    Observation space is proxied unchanged.
    """

    def __init__(self, env: Any, bins_vol: int = 101):
        assert int(bins_vol) >= 2, "bins_vol must be >= 2"
        self.env = env
        self.bins_vol = int(bins_vol)
        self.action_space = spaces.MultiDiscrete([201, 33, 4, self.bins_vol])
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def _vol_center(self, idx: int) -> float:
        idx = int(np.clip(idx, 0, self.bins_vol - 1))
        step = 2.0 / (self.bins_vol - 1)
        return float(-1.0 + step * idx)

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        a = np.asarray(action, dtype=np.int64).reshape(-1)
        assert a.size == 4, f"Expected 4-dim MultiDiscrete, got shape {a.shape}"
        price_i, ttl_i, type_i, vol_i = a.tolist()

        dict_action = {
            "price_offset_ticks": int(np.clip(price_i, 0, 200)),
            "ttl_steps":          int(np.clip(ttl_i,   0, 32)),
            "type":               int(np.clip(type_i,  0, 3)),
            "volume_frac":        np.array([self._vol_center(vol_i)], dtype=np.float32),
        }
        obs, rew, terminated, truncated, info = self.env.step(dict_action)
        return obs, rew, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


__all__ = ["DictToMultiDiscreteActionWrapper"]
