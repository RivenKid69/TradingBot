from __future__ import annotations
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium import ActionWrapper  # <-- ключевое: наследуемся от ActionWrapper


class DictToMultiDiscreteActionWrapper(ActionWrapper):
    """
    Convert Dict action space:
      { price_offset_ticks: Discrete(201),
        ttl_steps:          Discrete(33),
        type:               Discrete(4),
        volume_frac:        Box(-1,1,(1,),float32) }
    -> MultiDiscrete([201, 33, 4, bins_vol])

    Agent outputs [i_price, i_ttl, i_type, i_vol]; wrapper maps to Dict and
    delegates to underlying env.step(...). Observation space is proxied unchanged.
    """

    def __init__(self, env: Any, bins_vol: int = 101):
        # делаем класс полноценным Gymnasium-энвом
        super().__init__(env)
        assert int(bins_vol) >= 2, "bins_vol must be >= 2"
        self.bins_vol = int(bins_vol)

        # обновляем action_space на MultiDiscrete; observation_space оставляем как у базовой среды
        self.action_space = spaces.MultiDiscrete([201, 33, 4, self.bins_vol])
        self.observation_space = env.observation_space

    def _vol_center(self, idx: int) -> float:
        idx = int(np.clip(idx, 0, self.bins_vol - 1))
        step = 2.0 / (self.bins_vol - 1)
        return float(-1.0 + step * idx)

    # Метод ActionWrapper.action(a) преобразует действие ПЕРЕД вызовом env.step(...)
    def action(self, action):
        a = np.asarray(action, dtype=np.int64).reshape(-1)
        if a.size != 4:
            raise ValueError(f"Expected 4-dim MultiDiscrete action, got shape {a.shape}")
        price_i, ttl_i, type_i, vol_i = a.tolist()

        # Собираем dict-действие для исходной среды
        dict_action = {
            "price_offset_ticks": int(np.clip(price_i, 0, 200)),
            "ttl_steps":          int(np.clip(ttl_i,   0, 32)),
            "type":               int(np.clip(type_i,  0, 3)),
            "volume_frac":        np.array([self._vol_center(vol_i)], dtype=np.float32),
        }
        return dict_action


__all__ = ["DictToMultiDiscreteActionWrapper"]
