# Имя файла: train_model_multi_patch.py
# ИЗМЕНЕНИЯ (ФАЗА 3 - CVaR):
# 1. Добавлен гиперпараметр `cvar_alpha` в пространство поиска Optuna.
# 2. `cvar_alpha` передается в конструктор агента DistributionalPPO.
# ИЗМЕНЕНИЯ (ФАЗА 6 - Устранение утечки данных в HPO):
# 1. Статистики нормализации из тренировочной среды (env_tr) теперь
#    сохраняются в файл ВНАЧАЛЕ испытания.
# 2. Валидационная среда (env_va) для прунинга теперь создается
#    путем ЗАГРУЗКИ этих сохраненных статистик.
# 3. Это устраняет утечку данных, когда env_va вычисляла статистики
#    на валидационных данных, и обеспечивает корректную оценку модели.
# ИЗМЕНЕНИЯ (ФАЗА 7 - Исправление HPO для CVaR):
# 1. Гиперпараметр `cvar_weight` добавлен в пространство поиска Optuna.
# 2. `cvar_weight` теперь корректно передается в конструктор модели
#    DistributionalPPO, что делает HPO полноценным.
# ИЗМЕНЕНИЯ (ОПТИМИЗАЦИЯ ПРОИЗВОДИТЕЛЬНОСТИ):
# 1. Добавлены настройки PyTorch для ускорения вычислений на GPU.
# 2. Добавлена компиляция модели (`torch.compile`) для значительного ускорения.
# 3. Цикл оценки вынесен в быстрый Cython-модуль.

import os
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime
import shutil
import random
import math
from features_pipeline import FeaturePipeline
from pathlib import Path

from distributional_ppo import DistributionalPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.exceptions import TrialPruned
from torch.optim.lr_scheduler import OneCycleLR
import multiprocessing as mp
class AdversarialCallback(BaseCallback):
    """
    Проводит стресс-тесты в специальных рыночных режимах и СОХРАНЯЕТ
    результаты (Sortino Ratio) для каждого режима.
    """
    def __init__(self, eval_env: VecEnv, eval_freq: int, regimes: list, regime_duration: int):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.regimes = regimes
        self.regime_duration = regime_duration
        # Словарь для хранения метрик: {'regime_name': sortino_score}
        self.regime_metrics = {}

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print("\n--- Starting Adversarial Regime Stress Tests ---")
            
            for regime in self.regimes:
                print(f"Testing regime: {regime}...")
                # Устанавливаем режим в среде
                self.eval_env.env_method("set_market_regime", regime=regime, duration=self.regime_duration)
                
                # Запускаем оценку в этом режиме
                _rewards, equity_curves = evaluate_policy_custom_cython(
                    self.model,
                    self.eval_env,
                    num_episodes=1 # Один длинный эпизод для каждого режима
                )
                
                # Считаем Sortino и сохраняем
                all_returns = [pd.Series(c).pct_change().dropna().to_numpy() for c in equity_curves if len(c) > 1]
                flat_returns = np.concatenate(all_returns) if all_returns else np.array([0.0])
                score = sortino_ratio(flat_returns)
                self.regime_metrics[regime] = score
                
                print(f"Regime '{regime}' | Sortino: {score:.4f}")

            # Сбрасываем среду в нормальный режим
            self.eval_env.env_method("set_market_regime", regime='normal', duration=0)
            print("--- Adversarial Tests Finished ---\n")
        return True

    def get_regime_metrics(self) -> dict:
        """Возвращает словарь с результатами тестов."""
        return self.regime_metrics
from shared_memory_vec_env import SharedMemoryVecEnv

# --- ИЗМЕНЕНИЕ: Включаем оптимизации PyTorch ---
# Если GPU и версия CUDA >= 11, включаем высокую точность
if torch.cuda.is_available() and int(torch.version.cuda.split(".")[0]) >= 11:
    torch.set_float32_matmul_precision("high")
# Позволяет cuDNN находить лучшие алгоритмы для текущей конфигурации
torch.backends.cudnn.benchmark = True


from trading_patchnew import TradingEnv
from custom_policy_patch import CustomActorCriticPolicy
from fetch_all_data_patch import load_all_data
# --- ИЗМЕНЕНИЕ: Импортируем быструю Cython-функцию оценки ---
from evaluation_utils import evaluate_policy_custom_cython
from data_validation import DataValidator
from utils.model_io import save_sidecar_metadata, check_model_compat
from watchdog_vec_env import WatchdogVecEnv

# === КОНФИГУРАЦИЯ ИНДИКАТОРОВ (ЕДИНЫЙ ИСТОЧНИК ПРАВДЫ) ===
MA5_WINDOW = 5
MA20_WINDOW = 20
ATR_WINDOW = 14
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MOMENTUM_WINDOW = 10
CCI_WINDOW = 14
BB_WINDOW = 20
OBV_MA_WINDOW = 50


class NanGuardCallback(BaseCallback):
    """
    Проверяет loss и градиенты на NaN/Inf.
    При обнаружении прерывает текущий Optuna-trial.
    """
    def __init__(self, threshold: float = float("inf"), verbose: int = 0):
        super().__init__(verbose)
        self.threshold = threshold       # Можно задать лимит «слишком большой» loss

    def _on_rollout_end(self) -> None:
        # 1) Проверяем loss, если SB3 положил его в локалы
        last_loss = self.locals.get("loss", None)
        if last_loss is not None:
            if (not torch.isfinite(last_loss)) or (torch.abs(last_loss) > self.threshold):
                print("🚨  NaN/Inf обнаружен в loss  —  прерываем trial")
                raise TrialPruned("NaN detected in loss")

        # 2) Проверяем градиенты всех параметров
        for p in self.model.parameters():
            if p.grad is not None and (not torch.isfinite(p.grad).all()):
                print("🚨  NaN/Inf обнаружен в градиентах  —  прерываем trial")
                raise TrialPruned("NaN detected in gradients")

    def _on_step(self) -> bool:
        return True

class SortinoPruningCallback(BaseCallback):
    """
    Кастомный callback для Optuna, который оценивает модель и принимает
    решение о прунинге на основе КОЭФФИЦИЕНТА СОРТИНО, а не среднего вознаграждения.
    """
    def __init__(self, trial: optuna.Trial, eval_env: VecEnv, n_eval_episodes: int = 5, eval_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        # Запускаем оценку в заданные интервалы
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            
            # Используем быструю Cython-функцию для оценки
            rewards, equity_curves = evaluate_policy_custom_cython(
                self.model, 
                self.eval_env, 
                num_episodes=self.n_eval_episodes
            )
            
            # Рассчитываем Sortino на основе полученных кривых капитала
            if not equity_curves:
                # Если по какой-то причине оценка не вернула результатов, считаем Sortino равным 0
                current_sortino = 0.0
            else:
                all_returns = [
                    pd.Series(curve).pct_change().dropna().to_numpy() 
                    for curve in equity_curves if len(curve) > 1
                ]
                flat_returns = np.concatenate(all_returns) if all_returns else np.array([0.0])
                current_sortino = sortino_ratio(flat_returns)

            if self.verbose > 0:
                print(f"Step {self.n_calls}: Pruning check with Sortino Ratio = {current_sortino:.4f}")

            # 1. Сообщаем Optuna о промежуточном результате (теперь это Sortino)
            self.trial.report(current_sortino, self.n_calls)

            # 2. Проверяем, не нужно ли остановить этот trial
            if self.trial.should_prune():
                raise TrialPruned(f"Trial pruned at step {self.n_calls} with Sortino Ratio: {current_sortino:.4f}")

        return True

class ObjectiveScorePruningCallback(BaseCallback):
    """
    Callback для прунинга, использующий полную взвешенную метрику objective_score.
    Работает реже, чем SortinoPruningCallback, так как оценка занимает больше времени.
    """
    def __init__(self, trial: optuna.Trial, eval_env: VecEnv, eval_freq: int = 40000, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        
        # Веса, идентичные финальной оценке
        self.main_weight = 0.5
        self.choppy_weight = 0.3
        self.trend_weight = 0.2
        self.regime_duration = 2_500

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
            
            print(f"\n--- Step {self.n_calls}: Starting comprehensive pruning check with Objective Score ---")
            
            regimes_to_evaluate = ['normal', 'choppy_flat', 'strong_trend']
            evaluated_metrics = {}

            try:
                for regime in regimes_to_evaluate:
                    if self.verbose > 0:
                        print(f"Pruning evaluation: testing regime '{regime}'...")

                    # Устанавливаем адверсариальный режим
                    if regime != 'normal':
                        self.eval_env.env_method("set_market_regime", regime=regime, duration=self.regime_duration)

                    # Для прунинга достаточно меньшего числа эпизодов, чем в финале
                    num_episodes = 5  # Всегда использовать 5 эпизодов для более стабильной оценки
                    
                    _rewards, equity_curves = evaluate_policy_custom_cython(
                        self.model, self.eval_env, num_episodes=num_episodes
                    )
                    
                    all_returns = [pd.Series(c).pct_change().dropna().to_numpy() for c in equity_curves if len(c) > 1]
                    flat_returns = np.concatenate(all_returns) if all_returns else np.array([0.0])
                    score = sortino_ratio(flat_returns)
                    evaluated_metrics[regime] = score

            finally:
                # КРИТИЧЕСКИ ВАЖНО: всегда сбрасываем среду в нормальный режим,
                # чтобы не влиять на следующий шаг обучения или другие колбэки.
                self.eval_env.env_method("set_market_regime", regime='normal', duration=0)

            # Рассчитываем взвешенную метрику
            main_sortino = evaluated_metrics.get('normal', -1.0)
            choppy_score = evaluated_metrics.get('choppy_flat', -1.0)
            trend_score = evaluated_metrics.get('strong_trend', -1.0)
            
            objective_score = (self.main_weight * main_sortino + 
                               self.choppy_weight * choppy_score + 
                               self.trend_weight * trend_score)

            if self.verbose > 0:
                print(f"Comprehensive pruning check complete. Objective Score: {objective_score:.4f}")
                print(f"Components -> Main: {main_sortino:.4f}, Choppy: {choppy_score:.4f}, Trend: {trend_score:.4f}\n")

            # Сообщаем Optuna о результате и проверяем необходимость прунинга
            self.trial.report(objective_score, self.n_calls)
            if self.trial.should_prune():
                raise TrialPruned(f"Trial pruned at step {self.n_calls} with Objective Score: {objective_score:.4f}")

        return True

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    std = np.std(returns)
    return np.mean(returns - risk_free_rate) / (std + 1e-9) * np.sqrt(365 * 24)

def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    downside = returns[returns < risk_free_rate] - risk_free_rate
    if downside.size == 0:
        # Если нет убытков, используем стандартное отклонение (как в Шарпе).
        # Это более адекватно оценивает риск, чем возврат константы.
        std = np.std(returns)
        # Предотвращаем деление на ноль, если все доходности одинаковы.
        if std < 1e-9:
            return 0.0
        return np.mean(returns - risk_free_rate) / std * np.sqrt(365 * 24)

    downside_std = np.sqrt(np.mean(downside**2)) + 1e-9
    # Эта проверка становится избыточной, если downside_std используется только один раз, 
    # но оставим для безопасности.
    return np.mean(returns - risk_free_rate) / downside_std * np.sqrt(365 * 24)

# --- ИЗМЕНЕНИЕ: Старая Python-функция удалена, так как заменена на Cython-версию ---

def objective(trial: optuna.Trial,
              total_timesteps: int,
              train_data_by_token: dict,
              train_obs_by_token: dict,
              val_data_by_token: dict,
              val_obs_by_token: dict,
              norm_stats: dict):

    print(f">>> Trial {trial.number+1} with budget={total_timesteps}")
    

    # ИСПРАВЛЕНО: window_size возвращен в пространство поиска HPO
    params = {
        "window_size": trial.suggest_categorical("window_size", [10, 20, 30]),
        "trade_frequency_penalty": trial.suggest_float("trade_frequency_penalty", 1e-5, 5e-4, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        "n_epochs": trial.suggest_int("n_epochs", 1, 5),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "ent_coef": trial.suggest_float("ent_coef", 5e-5, 5e-3, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True),
        "risk_aversion_drawdown": trial.suggest_float("risk_aversion_drawdown", 0.05, 0.3),
        "risk_aversion_variance": trial.suggest_float("risk_aversion_variance", 0.005, 0.01),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "gamma": trial.suggest_float("gamma", 0.97, 0.995),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
        "clip_range": trial.suggest_float("clip_range", 0.12, 0.18),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "atr_multiplier": trial.suggest_float("atr_multiplier", 1.5, 3.0),
        "trailing_atr_mult": trial.suggest_float("trailing_atr_mult", 1.0, 2.0),
        "tp_atr_mult": trial.suggest_float("tp_atr_mult", 2.0, 4.0),
        "cql_alpha": trial.suggest_float("cql_alpha", 0.1, 10.0, log=True),
        "cql_beta": trial.suggest_float("cql_beta", 1.0, 10.0),
        "cvar_alpha": trial.suggest_float("cvar_alpha", 0.01, 0.20),
        "momentum_factor": trial.suggest_float("momentum_factor", 0.1, 0.7),
        "mean_reversion_factor": trial.suggest_float("mean_reversion_factor", 0.2, 0.8),
        "adversarial_factor": trial.suggest_float("adversarial_factor", 0.3, 0.9),
        "cvar_weight": trial.suggest_float("cvar_weight", 0.1, 2.0, log=True),
        "vf_coef": trial.suggest_float("vf_coef", 0.05, 0.5, log=True), # <-- ДОБАВЛЕНО
        "v_range_ema_alpha": trial.suggest_float("v_range_ema_alpha", 0.005, 0.05, log=True),
    }
    # 1. Определяем окно самого "медленного" индикатора на основе статических параметров.
    #    Эти параметры передаются в C++ симулятор.
    #    (В данном проекте они жестко заданы в TradingEnv, но для надежности берем их из HPO)
    # 1. Определяем окно самого "медленного" индикатора на основе констант.
    #    Это гарантирует синхронизацию с параметрами среды по умолчанию.
    slowest_window = max(
        params["window_size"],
        MA20_WINDOW,
        MACD_SLOW,
        ATR_WINDOW,
        RSI_WINDOW,
        CCI_WINDOW,
        BB_WINDOW,
        OBV_MA_WINDOW
    )
    
    # 2. Устанавливаем период прогрева с запасом (рекомендуется x2).
    #    Это гарантирует полную стабилизацию даже для вложенных сглаживаний,
    #    как в сигнальной линии MACD.
    warmup_period = slowest_window * 2

    policy_arch_params = {
        "hidden_dim": params["hidden_dim"],
        "use_memory": True,
        "num_atoms": 51,  
    }

    if not train_data_by_token: raise ValueError("Нет данных для обучения в этом trial.")

    n_envs = 8
    print(f"Запускаем {n_envs} параллельных сред...")
    def make_env_train(rank: int):
        def _init():
            # Создаем уникальный seed для каждого trial-а и каждого воркера
            # Это гарантирует воспроизводимость и декорреляцию
            unique_seed = trial.number * 100 + rank 
            env_params = {
                # 1. Параметры, которые подбирает Optuna
                "risk_aversion_drawdown": params["risk_aversion_drawdown"],
                "risk_aversion_variance": params["risk_aversion_variance"],
                "atr_multiplier": params["atr_multiplier"],
                "trailing_atr_mult": params["trailing_atr_mult"],
                "tp_atr_mult": params["tp_atr_mult"],
                "window_size": params.get("window_size", 20),
                "gamma": params["gamma"],
                "trade_frequency_penalty": params["trade_frequency_penalty"],
                "momentum_factor": params["momentum_factor"],
                "mean_reversion_factor": params["mean_reversion_factor"],
                "adversarial_factor": params["adversarial_factor"],

                # 2. Данные, которые передаются в функцию objective
                "df_dict": train_data_by_token,
                "obs_dict": train_obs_by_token,
                "norm_stats": norm_stats,

                # 3. Статические параметры и параметры индикаторов
                "mode": "train",
                "reward_shaping": True,
                "warmup_period": warmup_period,
                "ma5_window": MA5_WINDOW,
                "ma20_window": MA20_WINDOW,
                "atr_window": ATR_WINDOW,
                "rsi_window": RSI_WINDOW,
                "macd_fast": MACD_FAST,
                "macd_slow": MACD_SLOW,
                "macd_signal": MACD_SIGNAL,
                "momentum_window": MOMENTUM_WINDOW,
                "cci_window": CCI_WINDOW,
                "bb_window": BB_WINDOW,
                "obv_ma_window": OBV_MA_WINDOW,
                 
            }
            
            # Создаем и возвращаем экземпляр среды
            return TradingEnv(**env_params)
        return _init

    env_constructors = [make_env_train(rank=i) for i in range(n_envs)]

    stats_path = f"models/trials/vec_normalize_{trial.number}.pkl"
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    base_env_tr = WatchdogVecEnv(env_constructors)
    monitored_env_tr = VecMonitor(base_env_tr)
    env_tr = VecNormalize(monitored_env_tr, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=params["gamma"])

    

    env_tr.save(stats_path)
    save_sidecar_metadata(stats_path, extra={"kind": "vecnorm_stats"})

    def make_env_val():
        env_val_params = {
            "df_dict": val_data_by_token,
            "obs_dict": val_obs_by_token,
            "norm_stats": norm_stats,
            "window_size": params["window_size"],
            "gamma": params["gamma"],
            "atr_multiplier": params["atr_multiplier"],
            "trailing_atr_mult": params["trailing_atr_mult"],
            "tp_atr_mult": params["tp_atr_mult"],
            "trade_frequency_penalty": params["trade_frequency_penalty"],
            "mode": "val",
            "reward_shaping": False,
            "warmup_period": warmup_period,
            "ma5_window": MA5_WINDOW,
            "ma20_window": MA20_WINDOW,
            "atr_window": ATR_WINDOW,
            "rsi_window": RSI_WINDOW,
            "macd_fast": MACD_FAST,
            "macd_slow": MACD_SLOW,
            "macd_signal": MACD_SIGNAL,
            "momentum_window": MOMENTUM_WINDOW,
            "cci_window": CCI_WINDOW,
            "bb_window": BB_WINDOW,
            "obv_ma_window": OBV_MA_WINDOW
        }
        return TradingEnv(**env_val_params)

    monitored_env_va = VecMonitor(DummyVecEnv([make_env_val]))
    check_model_compat(stats_path)
    env_va = VecNormalize.load(stats_path, monitored_env_va)
    env_va.training = False
    env_va.norm_reward = False
    env_va.training = False
    env_va.norm_reward = False

    policy_kwargs = {
        "arch_params": policy_arch_params,
        "optimizer_class": torch.optim.AdamW,
        "optimizer_kwargs": {"weight_decay": params["weight_decay"]},
    }
    # Рассчитываем, сколько раз будет собран полный буфер данных (rollout)
    num_rollouts = math.ceil(total_timesteps / (params["n_steps"] * n_envs))
    
    # Рассчитываем, на сколько мини-батчей делится каждый роллаут
    num_minibatches_per_rollout = (params["n_steps"] * n_envs) // params["batch_size"]
    
    # Итоговое количество шагов оптимизатора за все обучение
    total_optimizer_steps = num_rollouts * params["n_epochs"] * num_minibatches_per_rollout
    
    # Создаем lambda-функцию для планировщика
    # SB3 вызовет ее со своим внутренним, правильным оптимизатором
    def scheduler_fn(optimizer):
        return OneCycleLR(optimizer=optimizer, max_lr=params["learning_rate"] * 3, total_steps=total_optimizer_steps)
    
    # Оборачиваем ее в словарь для передачи в policy_kwargs
    policy_kwargs["lr_scheduler"] = scheduler_fn
    model = DistributionalPPO(
        use_torch_compile=True,
        v_range_ema_alpha=params["v_range_ema_alpha"],
        policy=CustomActorCriticPolicy,
        env=env_tr,
        cql_alpha=params["cql_alpha"],
        cql_beta=params["cql_beta"],
        cvar_alpha=params["cvar_alpha"],
        vf_coef=params["vf_coef"],
        cvar_weight=params["cvar_weight"],
        
        learning_rate=params["learning_rate"],
        n_steps=params["n_steps"],
        n_epochs=params["n_epochs"],
        batch_size=params["batch_size"],
        ent_coef=params["ent_coef"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        clip_range=params["clip_range"],
        max_grad_norm=params["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        tensorboard_log="tensorboard_logs",
        verbose=1
    )

    



    nan_guard = NanGuardCallback()

    # Быстрый колбэк для раннего отсечения по простой метрике
    sortino_pruner = SortinoPruningCallback(trial, eval_env=env_va, eval_freq=8_000, n_eval_episodes=10)

    # Медленный, но точный колбэк для позднего отсечения по финальной метрике
    objective_pruner = ObjectiveScorePruningCallback(trial, eval_env=env_va, eval_freq=40_000, verbose=1)

    all_callbacks = [nan_guard, sortino_pruner, objective_pruner]

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks,
            progress_bar=True
        )
    finally:
        env_tr.close()
        env_va.close()

    trial_model_path = f"models/trials/trial_{trial.number}_model.zip"
    model.save(trial_model_path)
    save_sidecar_metadata(trial_model_path, extra={"kind": "sb3_model", "trial": int(trial.number)})

    

    print(f"<<< Trial {trial.number+1} finished training, starting unified final evaluation…")

    # 1. Определяем все режимы для оценки
    regimes_to_evaluate = ['normal', 'choppy_flat', 'strong_trend']
    final_metrics = {}
    regime_duration = 2_500 

    # 2. Последовательно оцениваем модель в каждом режиме
    for regime in regimes_to_evaluate:
        def make_final_eval_env():
            final_env_params = {
                "df_dict": val_data_by_token, "obs_dict": val_obs_by_token,
                "norm_stats": norm_stats, "window_size": params["window_size"],
                "gamma": params["gamma"], "atr_multiplier": params["atr_multiplier"],
                "trailing_atr_mult": params["trailing_atr_mult"], "tp_atr_mult": params["tp_atr_mult"],
                "trade_frequency_penalty": params["trade_frequency_penalty"], "mode": "val",
                "reward_shaping": False, "warmup_period": warmup_period,
                "ma5_window": MA5_WINDOW, "ma20_window": MA20_WINDOW, "atr_window": ATR_WINDOW,
                "rsi_window": RSI_WINDOW, "macd_fast": MACD_FAST, "macd_slow": MACD_SLOW,
                "macd_signal": MACD_SIGNAL, "momentum_window": MOMENTUM_WINDOW,
                "cci_window": CCI_WINDOW, "bb_window": BB_WINDOW, "obv_ma_window": OBV_MA_WINDOW
            }
            return TradingEnv(**final_env_params)

        check_model_compat(stats_path)
        final_eval_env = VecMonitor(
            VecNormalize.load(stats_path, DummyVecEnv([make_final_eval_env]))
        )
        final_eval_env.training = False
        final_eval_env.norm_reward = False

        if regime != 'normal':
            final_eval_env.env_method("set_market_regime", regime=regime, duration=regime_duration)

        num_episodes = len(val_data_by_token.keys())
        _rewards, equity_curves = evaluate_policy_custom_cython(model, final_eval_env, num_episodes=num_episodes)

        all_returns = [pd.Series(c).pct_change().dropna().to_numpy() for c in equity_curves if len(c) > 1]
        flat_returns = np.concatenate(all_returns) if all_returns else np.array([0.0])
        final_metrics[regime] = sortino_ratio(flat_returns)

        nt_stats = final_eval_env.env_method("get_no_trade_stats")
        total_steps = sum(s["total_steps"] for s in nt_stats if s)
        blocked_steps = sum(s["blocked_steps"] for s in nt_stats if s)
        ratio = blocked_steps / total_steps if total_steps else 0.0
        print(
            f"No-trade blocks: {blocked_steps}/{total_steps} steps ({ratio:.2%})"
        )

        final_eval_env.close()

    # --- РАСЧЕТ ИТОГОВОЙ ВЗВЕШЕННОЙ МЕТРИКИ ---
    main_sortino = final_metrics.get('normal', -1.0)
    choppy_score = final_metrics.get('choppy_flat', -1.0)
    trend_score = final_metrics.get('strong_trend', -1.0)

    main_weight = 0.5
    choppy_weight = 0.3
    trend_weight = 0.2
 
    objective_score = (main_weight * main_sortino + choppy_weight * choppy_score + trend_weight * trend_score)

    # Сохраняем все компоненты для анализа
    trial.set_user_attr("main_sortino", main_sortino)
    trial.set_user_attr("choppy_sortino", choppy_score)
    trial.set_user_attr("trend_sortino", trend_score)
    trial.set_user_attr("final_objective", objective_score)

    
    trial.set_user_attr("final_return", 0.0) # Устанавливаем в 0, т.к. корректный расчет усложнен

    print(f"\n[✅ Trial {trial.number}] COMPLETE. Final Weighted Score: {objective_score:.4f}")
    print(f"   Components -> Main Sortino: {main_sortino:.4f}, Choppy: {choppy_score:.4f}, Trend: {trend_score:.4f}\n")

    return objective_score

def main():
    PROCESSED_DATA_DIR = "data/processed"
    N_ENSEMBLE = 5
    # --- Добавленный блок очистки ---
    trials_dir = "models/trials"
    if os.path.exists(trials_dir):
        print(f"Очистка старых артефактов из директории '{trials_dir}'...")
        shutil.rmtree(trials_dir)
    # Пересоздаем пустую директорию
    os.makedirs(trials_dir, exist_ok=True)

    print("Loading all pre-processed data...")
    all_feather_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "*.feather"))
    if not all_feather_files:
        raise FileNotFoundError(
            f"No .feather files found in {PROCESSED_DATA_DIR}. "
            f"Run prepare_advanced_data.py (Fear&Greed), prepare_events.py (macro events), "
            f"incremental_klines.py (1h candles), then prepare_and_run.py (merge/export)."
        )

    all_dfs_dict, all_obs_dict = load_all_data(all_feather_files, synthetic_fraction=0, seed=42)

    # Fit/Save pipeline on full training data and add standardized features (_z)
    PREPROC_PATH = Path("models") / "preproc_pipeline.json"
    pipe = FeaturePipeline()
    pipe.fit(all_dfs_dict)
    PREPROC_PATH.parent.mkdir(parents=True, exist_ok=True)
    pipe.save(str(PREPROC_PATH))
    all_dfs_dict = pipe.transform_dict(all_dfs_dict, add_suffix="_z")
    print(f"Feature pipeline fitted and saved to {PREPROC_PATH}. Standardized columns *_z added.")
    print("To run inference over processed data, execute: python infer_signals.py")
    # --- Гейт качества данных: строгая валидация OHLCV перед сплитом ---
    _validator = DataValidator()
    for _key, _df in all_dfs_dict.items():
        try:
            # frequency=None -> автоопределение внутри валидатора
            _validator.validate(_df, frequency=None)
        except Exception as e:
            raise RuntimeError(f"Data validation failed for asset '{_key}': {e}")
    print("✓ Data validation passed for all assets.")

    
    
    print("Splitting data into train/validation sets...")

    full_train_data, full_val_data = {}, {}
    full_train_obs, full_val_obs = {}, {}

    rng = random.Random(42)
    all_keys = list(all_dfs_dict.keys())
    rng.shuffle(all_keys)

    split_idx = int(0.8 * len(all_keys))
    train_keys = all_keys[:split_idx]
    val_keys   = all_keys[split_idx:]

    for key in train_keys:
        full_train_data[key] = all_dfs_dict[key]
        if key in all_obs_dict:
            full_train_obs[key] = all_obs_dict[key]

    for key in val_keys:
        full_val_data[key] = all_dfs_dict[key]
        if key in all_obs_dict:
            full_val_obs[key] = all_obs_dict[key]


    print(f"Data split: {len(full_train_data)} assets for training, {len(full_val_data)} for validation.")

    print("Calculating per-asset normalization stats from the training set...")
    norm_stats = {}
    
    # Итерируем по каждому активу в ТРЕНИРОВОЧНОМ наборе данных
    for asset_key, train_df in full_train_data.items():
        
        # 1. Находим признаки для нормализации в ДАННОМ конкретном активе
        features_to_normalize = [
            col for col in train_df.columns 
            if '_norm' in col and col not in ['log_volume_norm', 'fear_greed_value_norm']
        ]
        
        if not features_to_normalize:
            continue # Пропускаем, если у этого ассета нет таких признаков
            
        # 2. Рассчитываем статистики ТОЛЬКО по данным этого ассета
        mean_stats = train_df[features_to_normalize].mean().to_dict()
        std_stats = train_df[features_to_normalize].std().to_dict()
        
        # 3. Находим ID токена, связанный с этим ассетом
        # (Предполагаем, что один файл = один основной токен)
        if 'token_id' in train_df.columns:
            # Убедимся, что в датафрейме есть данные
            if not train_df.empty:
                token_id = train_df['token_id'].iloc[0]
                
                # 4. Сохраняем индивидуальные статистики для этого токена
                norm_stats[str(token_id)] = {'mean': mean_stats, 'std': std_stats}

    with open("models/norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=4)
    print(f"Per-asset normalization stats for {len(norm_stats)} tokens calculated and saved.")

    HPO_TRIALS = 20 # Общее количество испытаний
    HPO_BUDGET_PER_TRIAL = 1_000_000 # Таймстепы для каждого испытания

    print(f"\n===== Starting Unified HPO Process ({HPO_TRIALS} trials) =====")

    sampler = TPESampler(n_startup_trials=5, seed=42)
    pruner = HyperbandPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    # Запускаем оптимизацию на ПОЛНОМ, диверсифицированном наборе данных
    study.optimize(
        lambda t: objective(t, HPO_BUDGET_PER_TRIAL,
                            full_train_data, full_train_obs,
                            full_val_data, full_val_obs,
                            norm_stats),
        n_trials=HPO_TRIALS,
        n_jobs=1
    )

    # Сохраняем итоговое исследование
    final_study = study
    if not final_study: print("No final study completed. Exiting."); return
    # <-- КОНЕЦ БЛОКА ДЛЯ ЗАМЕНЫ -->

    print(f"\nSaving best {N_ENSEMBLE} models from the final stage...")
    ensemble_dir = "models/ensemble"
    if os.path.exists(ensemble_dir): shutil.rmtree(ensemble_dir)
    os.makedirs(ensemble_dir)

    top_trials = sorted(final_study.trials, key=lambda tr: tr.value or -1e9, reverse=True)[:N_ENSEMBLE]

    ensemble_meta = []
    for i, trial in enumerate(top_trials):
        model_idx = i + 1
        src_model = f"models/trials/trial_{trial.number}_model.zip"
        src_stats = f"models/trials/vec_normalize_{trial.number}.pkl"

        if os.path.exists(src_model):
            shutil.copyfile(src_model, os.path.join(ensemble_dir, f"model_{model_idx}.zip"))
            if os.path.exists(src_stats):
                shutil.copyfile(src_stats, os.path.join(ensemble_dir, f"vec_normalize_{model_idx}.pkl"))

            ensemble_meta.append({"ensemble_index": model_idx, "trial_number": trial.number, "value": trial.value, "params": trial.params})
        else:
            print(f"⚠️ WARNING: Could not find model for trial {trial.number}. Skipping.")
    # Копируем единый файл со статистиками нормализации наблюдений,
    # так как он является неотъемлемой частью всех моделей в ансамбле.
    src_norm_stats = "models/norm_stats.json"
    if os.path.exists(src_norm_stats):
        shutil.copyfile(src_norm_stats, os.path.join(ensemble_dir, "norm_stats.json"))
    else:
        print(f"⚠️ CRITICAL WARNING: Could not find the global 'norm_stats.json' file. The saved ensemble will not be usable for inference.")
    with open(os.path.join(ensemble_dir, "ensemble_meta.json"), "w") as f: json.dump(ensemble_meta, f, indent=4)
    print(f"\n✅ Ensemble of {len(ensemble_meta)} models saved to '{ensemble_dir}'. HPO complete.")

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # --- gradient sanity check (включается флагом окружения) ---что 
    from runtime_flags import get_bool
    if get_bool("GRAD_SANITY", False):
        from tools.grad_sanity import run_check
        run_check()

    main()