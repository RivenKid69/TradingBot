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
        def __init__(self, num_envs, observation_space, action_space):
            self.num_envs = num_envs
            self.observation_space = observation_space
            self.action_space = action_space


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
        self._sync_spaces()

        try:
            super().__init__(self.num_envs, self.observation_space, self.action_space)
        except TypeError:
            # если базовый VecEnv не требует инициализации (фолбэк выше)
            pass

    # ------------- внутреннее -------------

    def _log(self, msg: str) -> None:
        if self._verbose:
            print(f"[WatchdogVecEnv] {msg}")

    def _sync_spaces(self) -> None:
        """Скопировать пространства из обёрнутой среды."""
        self.num_envs = getattr(self.env, "num_envs", len(self._env_fns))
        self.observation_space = getattr(self.env, "observation_space", None)
        self.action_space = getattr(self.env, "action_space", None)

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
        self._sync_spaces()

    # ------------- прокси-API VecEnv -------------

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Синхронизируем reset_infos с базовой средой (SharedMemoryVecEnv)
        self.reset_infos = getattr(self.env, "reset_infos", [{} for _ in range(self.num_envs)])
        return obs

    def step_async(self, actions):
        return self.env.step_async(actions)

    def step_wait(self):
        try:
            return self.env.step_wait()
        except Exception as e:
            self._log(f"Error in step_wait: {e!r}")
            self._reinit()
            obs = self.env.reset()
            self.reset_infos = getattr(self.env, "reset_infos", [{} for _ in range(self.num_envs)])
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
