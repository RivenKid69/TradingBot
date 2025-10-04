from __future__ import annotations

import importlib
import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env.util import is_vecenv_wrapped
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

_cy_evaluate_episode: Callable[..., tuple[list[float], list[list[float]]]] | None = None
_cy_spec = importlib.util.find_spec("cy_eval_core")
if _cy_spec is not None:
    module = importlib.import_module("cy_eval_core")
    _cy_evaluate_episode = getattr(module, "evaluate_episode", None)


CallbackType = Callable[[dict[str, Any], dict[str, Any]], bool | None]


def _append_equity(buffer: list[float], value: Any) -> None:
    if value is None:
        return
    try:
        equity = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(equity):
        buffer.append(equity)


def _maybe_vec_env(env: Any) -> tuple[VecEnv, bool]:
    if isinstance(env, VecEnv):
        return env, False
    dummy_env = DummyVecEnv([lambda: env])
    return dummy_env, True


def evaluate_policy_custom_cython(
    model: BaseAlgorithm,
    env: VecEnv | Any | None = None,
    *,
    num_episodes: int = 5,
    deterministic: bool = True,
    callback: CallbackType | None = None,
    warn: bool = True,
    render: bool = False,
) -> tuple[list[float], list[list[float]]]:
    """Evaluate a policy and collect equity curves.

    Parameters
    ----------
    model:
        Trained Stable-Baselines3 algorithm.
    env:
        Environment to evaluate on. If ``None`` the model's environment is used.
    num_episodes:
        Number of evaluation episodes.
    deterministic:
        Whether to use deterministic actions.
    callback:
        Optional callback executed after every environment step. Returning ``False`` stops evaluation.
    warn:
        Emit a warning when evaluation finishes with empty results.
    render:
        If ``True`` request environment rendering on every step.

    Returns
    -------
    tuple[list[float], list[list[float]]]
        Episode rewards and matching equity curves.
    """

    if num_episodes <= 0:
        return [], []

    eval_env = env if env is not None else model.get_env()
    if eval_env is None:
        raise ValueError("No evaluation environment provided and the model has no associated env.")

    vec_env, should_close = _maybe_vec_env(eval_env)

    is_monitor_wrapped = is_vecenv_wrapped(vec_env, VecMonitor) or vec_env.env_is_wrapped(Monitor)[0]
    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards if wrappers change them. "
            "Consider wrapping the environment with ``Monitor`` before evaluation.",
            UserWarning,
            stacklevel=2,
        )

    if _cy_evaluate_episode is not None:
        try:
            return _cy_evaluate_episode(
                model,
                vec_env,
                num_episodes=num_episodes,
                deterministic=deterministic,
            )
        except TypeError:
            warnings.warn(
                "cy_eval_core.evaluate_episode signature mismatch; falling back to Python implementation.",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception as exc:  # pragma: no cover - safety net for unexpected C-level failures
            warnings.warn(
                f"cy_eval_core evaluation failed ({exc!r}); falling back to Python implementation.",
                RuntimeWarning,
                stacklevel=2,
            )

    episode_rewards: list[float] = []
    equity_curves: list[list[float]] = []

    policy = model.policy
    training_mode = getattr(policy, "training", False)

    try:
        policy.set_training_mode(False)

        obs_reset = vec_env.reset()
        if isinstance(obs_reset, tuple):
            observations = obs_reset[0]
        else:
            observations = obs_reset

        n_envs = vec_env.num_envs
        episode_counts = np.zeros(n_envs, dtype=int)
        episode_count_targets = np.array([(num_episodes + i) // n_envs for i in range(n_envs)], dtype=int)
        current_rewards = np.zeros(n_envs, dtype=float)
        current_equity = [[] for _ in range(n_envs)]
        episode_starts = np.ones(n_envs, dtype=bool)
        states: Any = None

        with th.no_grad():
            while (episode_counts < episode_count_targets).any():
                actions, states = model.predict(
                    observations,
                    state=states,
                    episode_start=episode_starts,
                    deterministic=deterministic,
                )

                new_observations, rewards, dones, infos = vec_env.step(actions)

                current_rewards += np.array(rewards, dtype=float)

                for env_idx in range(n_envs):
                    if episode_counts[env_idx] >= episode_count_targets[env_idx]:
                        episode_starts[env_idx] = True
                        continue

                    reward = float(rewards[env_idx])
                    done = bool(dones[env_idx])
                    info = infos[env_idx]

                    equity_val = info.get("equity")
                    if equity_val is None:
                        equity_val = info.get("portfolio_value", info.get("net_worth"))
                    _append_equity(current_equity[env_idx], equity_val)

                    episode_starts[env_idx] = done

                    if callback is not None:
                        callback_result = callback(locals(), globals())
                        if callback_result is False:
                            return episode_rewards[:num_episodes], equity_curves[:num_episodes]

                    if not done:
                        continue

                    if is_monitor_wrapped and "episode" in info:
                        episode_rewards.append(float(info["episode"]["r"]))
                    else:
                        episode_rewards.append(float(current_rewards[env_idx]))

                    equity_curves.append(current_equity[env_idx].copy())
                    episode_counts[env_idx] += 1
                    current_rewards[env_idx] = 0.0
                    current_equity[env_idx].clear()

                    reset_info = info.get("reset_info")
                    if isinstance(reset_info, dict):
                        equity_reset = reset_info.get("equity")
                        if equity_reset is None:
                            equity_reset = reset_info.get("portfolio_value", reset_info.get("net_worth"))
                        _append_equity(current_equity[env_idx], equity_reset)

                observations = new_observations

                if render:
                    vec_env.render()

        if not equity_curves and warn:
            warnings.warn("Evaluation finished without producing equity curves.", RuntimeWarning, stacklevel=2)

    finally:
        policy.set_training_mode(training_mode)
        if should_close:
            vec_env.close()

    return episode_rewards[:num_episodes], equity_curves[:num_episodes]

