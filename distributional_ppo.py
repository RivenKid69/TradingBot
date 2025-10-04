# Имя файла: distributional_ppo.py
# ИЗМЕНЕНИЯ (ФАЗА 5 - Векторизация CVaR):
# 1. Функция `calculate_cvar` полностью переписана с использованием
#    векторизованных операций PyTorch.
# 2. Удален неэффективный цикл `for` по элементам батча, что значительно
#    ускоряет вычисления и снижает нагрузку на CPU.
# 3. Логика вычислений и результат функции остались прежними, изменился
#    только способ их получения (без циклов).

import torch
import numpy as np
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
try:
    from stable_baselines3.common.vec_env.vec_normalize import unwrap_vec_normalize as _sb3_unwrap
except Exception:
    _sb3_unwrap = None


def unwrap_vec_normalize(env):
    """Backcompat для SB3>=2.x: пройти обёртки и найти VecNormalize."""

    if _sb3_unwrap is not None:
        return _sb3_unwrap(env)
    try:
        from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
    except Exception:
        return None
    e = env
    while isinstance(e, VecEnvWrapper):
        if isinstance(e, VecNormalize):
            return e
        e = getattr(e, "venv", None)
    return None

import torch.nn.functional as F
from typing import Any, Dict, Optional, Type, Union

torch.set_float32_matmul_precision("high")


import time
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn

def safe_explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

# --- Векторизованная версия функции CVaR (без изменений) ---
def calculate_cvar(probs: torch.Tensor, atoms: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Вычисляет Conditional Value at Risk (CVaR) для заданного распределения.
    Эта версия векторизована для эффективной работы с батчами.
    """
    batch_size, num_atoms = probs.shape
    device = probs.device

    sorted_atoms, sort_indices = torch.sort(atoms)
    sorted_probs = probs.gather(1, sort_indices.unsqueeze(0).expand_as(probs))
    cumulative_probs = torch.cumsum(sorted_probs, dim=1)

    alpha_tensor = torch.full((batch_size, 1), fill_value=alpha, dtype=torch.float32, device=device)
    var_indices = torch.searchsorted(cumulative_probs.detach(), alpha_tensor).clamp(max=num_atoms - 1)

    atom_indices_j = torch.arange(num_atoms, device=device).view(1, -1)
    tail_mask = atom_indices_j < var_indices

    per_atom_expectation = sorted_probs * sorted_atoms.view(1, -1).detach()
    tail_expectation = torch.sum(per_atom_expectation * tail_mask, dim=1)

    tail_mass_indices = (var_indices - 1).clamp(min=0)
    tail_mass = cumulative_probs.gather(1, tail_mass_indices).squeeze(1)
    tail_mass[var_indices.squeeze(1) == 0] = 0.0

    weight_on_var = alpha - tail_mass
    var_atom_values = sorted_atoms[var_indices.squeeze(1)].detach()
    expectation_on_var = weight_on_var * var_atom_values

    cvar = (tail_expectation + expectation_on_var) / (alpha + 1e-8)

    return cvar


class DistributionalPPO(RecurrentPPO):
    """
    Дистрибутивный вариант RecurrentPPO без использования смешанной точности.
    """
    def __init__(
        self,
        policy: Union[str, Type[RecurrentActorCriticPolicy]],
        env: Union[VecEnv, str],
        cql_alpha: float = 1.0,
        cql_beta: float = 5.0,
        cvar_alpha: float = 0.05,
        cvar_weight: float = 0.5,
        v_range_ema_alpha: float = 0.01,
        use_torch_compile: bool = False,
        **kwargs: Any,
    ):
        self._last_lstm_states = None
        super().__init__(policy=policy, env=env, **kwargs)

        self.cql_alpha = cql_alpha
        self.cql_beta = cql_beta
        self.cvar_alpha = cvar_alpha
        self.cvar_weight = cvar_weight
        self.v_range_ema_alpha = v_range_ema_alpha
        self.running_v_min = 0.0
        self.running_v_max = 0.0
        self.v_range_initialized = False

        self.lr_scheduler = None
        
        if use_torch_compile and self.device.type == "cuda":
            print("--> Compiling the policy with torch.compile...")
            # Оптимизатор уже создан и ссылается на параметры,
            # torch.compile будет работать с ними.
            self.policy = torch.compile(self.policy, mode="reduce-overhead")
            print("--> Policy compilation complete.")
        print(f"--> CVaR-in-loss settings: (alpha={self.cvar_alpha:.2f}, weight={self.cvar_weight:.2f}).")

        if hasattr(self.policy, "optimizer_scheduler") and self.policy.optimizer_scheduler is not None:
            self.lr_scheduler = self.policy.optimizer_scheduler
    
    def parameters(self, recurse: bool = True):
        """Позволяет обращаться к параметрам агента как к nn.Module."""
        return self.policy.parameters(recurse)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Собирает траектории. Эта версия адаптирована для дистрибутивного RL.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        if self._last_lstm_states is None:
            # self.policy.recurrent_initial_state создается в RecurrentActorCriticPolicy
            # и является кортежем из двух тензоров (для pi и vf)
            initial_state = self.policy.recurrent_initial_state
            # Перемещаем начальное состояние на правильное устройство
            self._last_lstm_states = (initial_state[0].to(self.device), initial_state[1].to(self.device))


        self.policy.set_training_mode(False)

        vec_normalize_env: Optional[VecNormalize] = None
        for candidate_env in (env, getattr(self, "env", None)):
            if candidate_env is None:
                continue
            if isinstance(candidate_env, VecNormalize):
                vec_normalize_env = candidate_env
            else:
                try:
                    vec_normalize_env = unwrap_vec_normalize(candidate_env)
                except ValueError:
                    vec_normalize_env = None
            if vec_normalize_env is not None:
                break

        if vec_normalize_env is not None and getattr(vec_normalize_env, "norm_reward", False):
            raise AssertionError("VecNormalize reward normalization must be disabled to recover raw ΔPnL.")

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                obs_tensor = self.policy.obs_to_tensor(self._last_obs)[0]
                episode_starts = torch.as_tensor(self._last_episode_starts, dtype=torch.float32).to(self.device)

                actions, _, log_probs, self._last_lstm_states = self.policy.forward(
                    obs_tensor, self._last_lstm_states, episode_starts
                )

                value_logits = self.policy.last_value_logits

            if value_logits is None:
                raise RuntimeError("Policy did not cache value logits during forward pass")

            probs = torch.softmax(value_logits, dim=1)
            scalar_values = (probs * self.policy.atoms).sum(dim=1, keepdim=True).detach()
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            normalized_rewards = rewards
            raw_rewards = rewards
            if vec_normalize_env is not None and hasattr(vec_normalize_env, "get_original_reward"):
                raw_rewards = vec_normalize_env.get_original_reward()
            raw_rewards = np.asarray(raw_rewards)
            rewards = raw_rewards
            if raw_rewards.size > 0:
                self.logger.record("rollout/raw_reward_mean", float(np.mean(raw_rewards)))

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.reshape(-1, 1)

            

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                scalar_values.squeeze(-1),
                log_probs,
                lstm_states=self._last_lstm_states,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            obs_tensor = self.policy.obs_to_tensor(new_obs)[0]
            episode_starts = torch.as_tensor(dones, dtype=torch.float32).to(self.device)
            _, _, _, _ = self.policy.forward(obs_tensor, self._last_lstm_states, episode_starts)

            last_value_logits = self.policy.last_value_logits
            if last_value_logits is None:
                raise RuntimeError("Policy did not cache value logits during terminal forward pass")

            last_probs = torch.softmax(last_value_logits, dim=1)
            last_scalar_values = (last_probs * self.policy.atoms).sum(dim=1)

        rollout_buffer.compute_returns_and_advantage(last_values=last_scalar_values, dones=dones)
        callback.on_rollout_end()
        return True


    def train(self) -> None:
        """
        Обновляет параметры политики стандартными градиентными шагами.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
 
        with torch.no_grad():
            # Вычисляем мин/макс по ВСЕМУ буферу роллаута
            v_min = self.rollout_buffer.returns.min().item()
            v_max = self.rollout_buffer.returns.max().item()
            # Сразу обновляем атомы в политике
            self.policy.update_atoms(v_min, v_max)

        # Логируем эти границы
        self.logger.record("train/v_min", v_min)
        self.logger.record("train/v_max", v_max)

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                
                

                
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    rollout_data.actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                # --- РАСЧЕТ ПОЛИТИЧЕСКОЙ ПОТЕРИ (POLICY LOSS) ---
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss_ppo = -torch.min(policy_loss_1, policy_loss_2).mean()

                with torch.no_grad():
                    weights = torch.exp(advantages / self.cql_beta)
                    weights = torch.clamp(weights, max=100.0).detach()
                policy_loss_bc = (-log_prob * weights).mean()

                policy_loss = policy_loss_ppo + self.cql_alpha * policy_loss_bc

                # --- РАСЧЕТ ПОТЕРИ ЭНТРОПИИ ---
                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                # --- РАСЧЕТ КРИТИЧЕСКОЙ ПОТЕРИ (CRITIC LOSS) В ПОЛНОЙ ТОЧНОСТИ ---
                value_logits = self.policy.last_value_logits
                if value_logits is None:
                    raise RuntimeError("Policy did not cache value logits during training forward pass")

                value_logits_fp32 = value_logits.float()
                with torch.no_grad():
                    target_returns = rollout_data.returns
                    delta_z = (self.policy.v_max - self.policy.v_min) / (self.policy.num_atoms - 1)
                    clamped_targets = target_returns.clamp(self.policy.v_min, self.policy.v_max)
                    b = (clamped_targets - self.policy.v_min) / (delta_z + 1e-8)
                    lower_bound = b.floor().long()
                    upper_bound = b.ceil().long()
                    lower_bound[(upper_bound > 0) & (lower_bound == upper_bound)] -= 1
                    upper_bound[(lower_bound < (self.policy.num_atoms - 1)) & (lower_bound == upper_bound)] += 1
                    target_distribution = torch.zeros_like(value_logits_fp32)
                    upper_prob = (b - lower_bound.float())
                    lower_prob = (upper_bound.float() - b)
                    upper_prob = upper_prob.to(target_distribution.dtype)
                    lower_prob = lower_prob.to(target_distribution.dtype)
                    target_distribution.scatter_add_(1, lower_bound.unsqueeze(1), lower_prob.unsqueeze(1))
                    target_distribution.scatter_add_(1, upper_bound.unsqueeze(1), upper_prob.unsqueeze(1))

                pred_probs_fp32 = torch.clamp(F.softmax(value_logits_fp32, dim=1), min=1e-8, max=1.0)
                log_predictions = torch.log(pred_probs_fp32)
                critic_loss = -(target_distribution * log_predictions).sum(dim=1).mean()

                # --- РАСЧЕТ ПОТЕРИ CVaR (ПОЛНАЯ ТОЧНОСТЬ ДЛЯ РАСПРЕДЕЛЕНИЙ) ---
                predicted_cvar = calculate_cvar(pred_probs_fp32, self.policy.atoms, self.cvar_alpha)
                cvar_loss = -predicted_cvar.mean()

                # --- ИТОГОВАЯ ФУНКЦИЯ ПОТЕРЬ ---
                loss = (
                    policy_loss.float()
                    + self.ent_coef * entropy_loss.float()
                    + self.vf_coef * critic_loss
                    + self.cvar_weight * cvar_loss
                )

                self.policy.optimizer.zero_grad(set_to_none=True)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()


                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_divs.append(torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy())

        self._n_updates += self.n_epochs

        with torch.no_grad():
            pred_probs = F.softmax(value_logits, dim=1)
            mean_pred_values = (pred_probs * self.policy.atoms).sum(dim=1)

        y_true_np = rollout_data.returns.flatten().cpu().numpy()
        y_pred_np = mean_pred_values.flatten().cpu().numpy()
        explained_var = np.nan_to_num(safe_explained_variance(y_true_np, y_pred_np))

        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/critic_loss", critic_loss.item())
        self.logger.record("train/cvar_loss", cvar_loss.item())
        self.logger.record("train/policy_loss_ppo", policy_loss_ppo.item())
        self.logger.record("train/policy_loss_bc", policy_loss_bc.item())
        if len(approx_kl_divs) > 0:
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/clip_range", clip_range)
