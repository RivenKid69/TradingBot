from __future__ import annotations

"""Action-space wrappers for trading environments."""

from typing import Any, Iterable, Tuple

import numpy as np
from gymnasium import spaces


class DictToMultiDiscreteActionWrapper:
    """Adapt a ``Dict`` action space to ``MultiDiscrete`` for SB3 PPO.

    Parameters
    ----------
    env:
        Environment exposing the dictionary action space with keys
        ``price_offset_ticks``, ``ttl_steps``, ``type`` and ``volume_frac``.
    bins_vol:
        Number of equally spaced bins to quantise ``volume_frac`` on
        ``[-1.0, 1.0]``. The default 101 bins includes the end points.
    """

    def __init__(self, env: Any, *, bins_vol: int = 101) -> None:
        if bins_vol < 2:
            raise ValueError("bins_vol must be >= 2")
        self.env = env
        self.bins_vol = int(bins_vol)
        self.action_space = spaces.MultiDiscrete([201, 33, 4, self.bins_vol])
        self.observation_space = env.observation_space

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _vol_center(self, idx: int) -> float:
        """Return the centre value for the selected volume bin."""

        clipped = int(np.clip(idx, 0, self.bins_vol - 1))
        step = 2.0 / (self.bins_vol - 1)
        return float(-1.0 + step * clipped)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action: Iterable[int]) -> Tuple[Any, float, bool, bool, dict]:
        array = np.asarray(action, dtype=np.int64).reshape(-1)
        if array.size != 4:
            raise ValueError(
                "DictToMultiDiscreteActionWrapper expected action of length 4"
            )
        price_i, ttl_i, type_i, vol_i = array.tolist()
        dict_action = {
            "price_offset_ticks": int(np.clip(price_i, 0, 200)),
            "ttl_steps": int(np.clip(ttl_i, 0, 32)),
            "type": int(np.clip(type_i, 0, 3)),
            "volume_frac": np.array([self._vol_center(vol_i)], dtype=np.float32),
        }
        return self.env.step(dict_action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self) -> None:
        return self.env.close()

    def __getattr__(self, item: str) -> Any:
        return getattr(self.env, item)


def _wrap_action_space_if_needed(env: Any, *, bins_vol: int = 101) -> Any:
    """Wrap environments with the expected dict action space into ``MultiDiscrete``."""

    if isinstance(getattr(env, "action_space", None), spaces.Dict):
        try:
            keys = set(env.action_space.spaces.keys())
        except Exception:
            keys = set()
        expected = {"price_offset_ticks", "ttl_steps", "type", "volume_frac"}
        if expected.issubset(keys):
            return DictToMultiDiscreteActionWrapper(env, bins_vol=bins_vol)
    return env


__all__ = ["DictToMultiDiscreteActionWrapper", "_wrap_action_space_if_needed"]
