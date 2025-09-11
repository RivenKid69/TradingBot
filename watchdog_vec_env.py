from __future__ import annotations

"""
WatchdogVecEnv — обёртка вокруг SharedMemoryVecEnv c автоматическим
перезапуском упавшей векторной среды.

Поведение:
- Любая ошибка в step_wait() перехватывается.
- Текущая базовая среда закрывается, создаётся новая через сохранённые env_fns.
- Возвращаются obs после reset() новой среды,
  rewards=zeros, dones=ones (эпизод завершён),
  infos — список словарей с ключом {"watchdog_restart": True}.

Совместимость:
- Ставит тонкий фолбэк на sb3 VecEnv (если не установлен, не наследуемся жёстко).
- Проксирует reset/step_async/step_wait/close/get_attr/set_attr/
  env_method/env_is_wrapped.
"""

from typing import Any, Callable, Sequence

import numpy as np

try:
    from stable_baselines3.common.vec_env import VecEnv  # type: ignore
except Exception:

    class VecEnv:  # минимальный интерфейс для типовой совместимости
        pass


from shared_memory_vec_env import SharedMemoryVecEnv


class WatchdogVecEnv(VecEnv):
    def __init__(
        self,
        env_fns: Sequence[Callable[[], Any]],
        *,
        verbose: bool = True,
        max_restarts: int = 100,
    ):
        """
        env_fns: последовательность фабрик, создающих отдельные окружения.
        verbose: печатать события рестартов.
        max_restarts: предохранитель от бесконечных рестартов.
        """
        self._env_fns = list(env_fns)
        if not self._env_fns:
            raise ValueError("WatchdogVecEnv requires a non-empty list of env_fns")
        self._verbose = bool(verbose)
        self._max_restarts = int(max_restarts)
        self._restarts = 0

        self.env: SharedMemoryVecEnv = SharedMemoryVecEnv(self._env_fns)

    # ------------- внутреннее -------------

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"[WatchdogVecEnv] {msg}")

    def _reinit(self) -> None:
        """Закрыть текущую и создать новую базовую среду."""
        self._restarts += 1
        if self._restarts > self._max_restarts:
            raise RuntimeError(
                f"WatchdogVecEnv exceeded max_restarts={self._max_restarts}"
            )
        try:
            self.env.close()
        except Exception:
            pass
        self._log(f"Restarting underlying env (restart #{self._restarts})")
        self.env = SharedMemoryVecEnv(self._env_fns)

    # ------------- прокси-API VecEnv -------------

    @property
    def num_envs(self) -> int:
        return getattr(self.env, "num_envs", len(self._env_fns))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step_async(self, actions):
        return self.env.step_async(actions)

    def step_wait(self):
        try:
            return self.env.step_wait()
        except Exception as e:
            self._log(f"Error in step_wait: {e!r}")
            self._reinit()
            obs = self.env.reset()
            n = self.num_envs
            rewards = np.zeros((n,), dtype=np.float32)
            dones = np.ones((n,), dtype=bool)
            infos = [{"watchdog_restart": True} for _ in range(n)]
            return obs, rewards, dones, infos

    def close(self):
        try:
            self.env.close()
        finally:
            pass

    # вспомогательные прокси (как в sb3 VecEnv)
    def get_attr(self, attr_name: str, indices=None):
        return self.env.get_attr(attr_name, indices=indices)

    def set_attr(self, attr_name: str, value, indices=None):
        return self.env.set_attr(attr_name, value, indices=indices)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        return self.env.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self.env.env_is_wrapped(wrapper_class, indices=indices)

    # совместимость с рендерами (не обязательно реализовано для SharedMemoryVecEnv)
    def render(self, *args, **kwargs):
        if hasattr(self.env, "render"):
            return self.env.render(*args, **kwargs)
        return None
