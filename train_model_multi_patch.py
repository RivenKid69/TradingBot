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
import math
import argparse
import yaml
import sys
import hashlib
from functools import lru_cache
from collections.abc import Mapping, Sequence
from typing import Any, Callable
from features_pipeline import FeaturePipeline
from pathlib import Path

from core_config import (
    load_config,
    ExecutionProfile,
    load_timing_profiles,
    resolve_execution_timing,
    TrainConfig,
)

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.exceptions import TrialPruned
from torch.optim.lr_scheduler import OneCycleLR
import multiprocessing as mp
from leakguard import LeakGuard, LeakConfig
class AdversarialCallback(BaseCallback):
    """
    Проводит стресс-тесты в специальных рыночных режимах и СОХРАНЯЕТ
    результаты (Sortino Ratio) для каждого режима.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int,
        regimes: list,
        regime_duration: int,
        regime_config_path: str | None = None,
        liquidity_seasonality_path: str | None = None,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.regimes = regimes
        self.regime_duration = regime_duration
        # Словарь для хранения метрик: {'regime_name': sortino_score}
        self.regime_metrics = {}

        if regime_config_path:
            os.environ["MARKET_REGIMES_JSON"] = regime_config_path
        self._liq_seasonality_path = liquidity_seasonality_path

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

            # Validate distributions against reference after stress tests
            try:
                dist_metrics, _ = compare_regime_distributions(
                    env=self.eval_env,
                    reference_path="configs/reference_regime_distributions.json",
                    n_steps=self.regime_duration,
                    tolerance=0.2,
                    raise_on_fail=False,
                )
                for r_name, vals in dist_metrics.items():
                    for m_name, val in vals.items():
                        if self.logger is not None:
                            self.logger.record(f"regime_dist/{r_name}/{m_name}", val)
            except Exception as exc:
                print(f"Distribution validation failed: {exc}")
        return True

    def get_regime_metrics(self) -> dict:
        """Возвращает словарь с результатами тестов."""
        return self.regime_metrics
from shared_memory_vec_env import SharedMemoryVecEnv


@lru_cache(maxsize=1)
def _get_distributional_ppo():
    from distributional_ppo import DistributionalPPO

    return DistributionalPPO

# --- ИЗМЕНЕНИЕ: Включаем оптимизации PyTorch ---
# Если GPU и версия CUDA >= 11, включаем высокую точность
if torch.cuda.is_available() and int(torch.version.cuda.split(".")[0]) >= 11:
    torch.set_float32_matmul_precision("high")
# Позволяет cuDNN находить лучшие алгоритмы для текущей конфигурации
torch.backends.cudnn.benchmark = True


from trading_patchnew import TradingEnv, DecisionTiming
from gymnasium import spaces
from wrappers.action_space import DictToMultiDiscreteActionWrapper
from custom_policy_patch1 import CustomActorCriticPolicy
from fetch_all_data_patch import load_all_data
# --- ИЗМЕНЕНИЕ: Импортируем быструю Cython-функцию оценки ---
from evaluate_policy_custom_cython import evaluate_policy_custom_cython
from scripts.validate_regime_distributions import compare_regime_distributions
from data_validation import DataValidator
from utils.model_io import save_sidecar_metadata, check_model_compat
from watchdog_vec_env import WatchdogVecEnv
from scripts.offline_utils import load_offline_payload, resolve_split_bundle

# --- helper to compute SHA256 of liquidity seasonality file ---
def _file_sha256(path: str | None) -> str | None:
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        return None


def _wrap_action_space_if_needed(env, bins_vol: int = 101):
    """
    If env.action_space is Dict with expected keys, wrap it into MultiDiscrete.
    Otherwise return as is.
    """

    try:
        if isinstance(env.action_space, spaces.Dict):
            keys = set(getattr(env.action_space, "spaces", {}).keys())
            expected = {"price_offset_ticks", "ttl_steps", "type", "volume_frac"}
            if expected.issubset(keys):
                return DictToMultiDiscreteActionWrapper(env, bins_vol=bins_vol)
    except Exception:
        # If anything goes wrong, fail open (no wrapping)
        return env
    return env


def _coerce_timestamp(value) -> int | None:
    """Normalize timestamp representations to integer seconds."""

    if value is None:
        return None
    if isinstance(value, str):
        txt = value.strip()
        if not txt or txt.lower() == "none":
            return None
        try:
            value = int(txt)
        except ValueError:
            try:
                ts = pd.Timestamp(txt)
            except Exception as exc:  # pragma: no cover - defensive branch
                raise ValueError(f"Unable to parse timestamp string '{value}'.") from exc
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return int(ts.value // 10**9)
    if isinstance(value, pd.Timestamp):
        ts = value
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.value // 10**9)
    if isinstance(value, (np.integer, int)):
        iv = int(value)
        if abs(iv) > 10_000_000_000:  # heuristic: values in ms
            return iv // 1000
        return iv
    if isinstance(value, (np.floating, float)):
        if math.isnan(value):
            return None
        return _coerce_timestamp(int(value))
    raise ValueError(f"Unsupported timestamp value: {value!r}")


def _normalize_interval(item) -> tuple[int | None, int | None]:
    if item is None:
        return (None, None)
    if isinstance(item, dict):
        start = item.get("start_ts")
        if start is None:
            start = item.get("start") or item.get("from")
        end = item.get("end_ts")
        if end is None:
            end = item.get("end") or item.get("to")
    elif isinstance(item, (list, tuple)) and len(item) == 2:
        start, end = item
    else:
        raise TypeError(f"Unsupported interval specification: {item!r}")
    start_ts = _coerce_timestamp(start)
    end_ts = _coerce_timestamp(end)
    if start_ts is not None and end_ts is not None and end_ts < start_ts:
        raise ValueError(f"Invalid interval with start {start_ts} after end {end_ts}")
    return (start_ts, end_ts)


def _extract_offline_split_overrides(
    payload: Mapping[str, Any] | None,
    dataset_key: str,
    *,
    fallback_split: str = "time",
) -> dict[str, list[dict[str, int | None]]]:
    """Extract inline time split overrides from an offline payload."""

    if payload is None:
        return {}

    datasets = payload.get("datasets") if isinstance(payload, Mapping) else None
    dataset_entry = datasets.get(dataset_key) if isinstance(datasets, Mapping) else None

    def _select_split_block(entry: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if not isinstance(entry, Mapping):
            return None
        splits_block = entry.get("splits")
        if isinstance(splits_block, Mapping):
            return splits_block
        # Some payloads embed train/val/test directly at the dataset level.
        direct = {k: entry.get(k) for k in ("train", "val", "test") if entry.get(k) is not None}
        return direct or None

    def _pick_split_entry(block: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if not isinstance(block, Mapping) or not block:
            return None
        if dataset_key in block and isinstance(block[dataset_key], Mapping):
            return block[dataset_key]
        if fallback_split in block and isinstance(block[fallback_split], Mapping):
            return block[fallback_split]
        # As a last resort take the first mapping value.
        for value in block.values():
            if isinstance(value, Mapping):
                return value
        return None

    split_entry = _pick_split_entry(_select_split_block(dataset_entry))
    if split_entry is None:
        # fall back to top-level split registry if provided
        split_entry = _pick_split_entry(
            payload.get("splits") if isinstance(payload, Mapping) else None
        )

    if not isinstance(split_entry, Mapping):
        return {}

    overrides: dict[str, list[dict[str, int | None]]] = {}
    for phase, entries in split_entry.items():
        if phase not in {"train", "val", "test"}:
            continue
        if isinstance(entries, (list, tuple)):
            iterable: Sequence[Any] = entries
        else:
            iterable = (entries,)
        normalized_items: list[dict[str, int | None]] = []
        for item in iterable:
            try:
                start_ts, end_ts = _normalize_interval(item)
            except Exception:
                continue
            normalized_items.append({"start_ts": start_ts, "end_ts": end_ts})
        if normalized_items:
            overrides[phase] = normalized_items
    return overrides


def _load_time_splits(data_cfg) -> tuple[str | None, dict[str, list[tuple[int | None, int | None]]]]:
    """Derive train/val/test time windows from config or an external manifest."""

    splits: dict[str, list[tuple[int | None, int | None]]] = {"train": [], "val": [], "test": []}
    version: str | None = getattr(data_cfg, "split_version", None)

    overrides = getattr(data_cfg, "split_overrides", None)
    if isinstance(overrides, Mapping):
        for phase, entries in overrides.items():
            if phase not in splits:
                splits[phase] = []
            if isinstance(entries, (list, tuple)):
                iterable = entries
            else:
                iterable = [entries]
            splits[phase].extend(_normalize_interval(item) for item in iterable)

    split_path = getattr(data_cfg, "split_path", None)
    if split_path:
        manifest_path = Path(split_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {split_path}")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            if manifest_path.suffix.lower() in (".yaml", ".yml"):
                raw = yaml.safe_load(fh) or {}
            else:
                raw = json.load(fh)
        version = raw.get("version") or raw.get("name") or version or manifest_path.stem
        raw_splits = raw.get("splits")
        if raw_splits is None:
            raw_splits = {k: raw.get(k) for k in ("train", "val", "test") if raw.get(k) is not None}
        if raw_splits is None:
            raise ValueError("Split manifest must contain 'splits' or per-phase keys (train/val/test).")
        for phase in ("train", "val", "test"):
            entries = raw_splits.get(phase)
            if entries is None:
                continue
            if isinstance(entries, (list, tuple)):
                iterable = entries
            else:
                iterable = [entries]
            splits[phase].extend(_normalize_interval(item) for item in iterable)

    for phase in ("train", "val", "test"):
        start_attr = getattr(data_cfg, f"{phase}_start_ts", None)
        end_attr = getattr(data_cfg, f"{phase}_end_ts", None)
        if start_attr is not None or end_attr is not None:
            splits[phase].append(_normalize_interval({"start_ts": start_attr, "end_ts": end_attr}))

    if not splits["train"]:
        fallback = _normalize_interval({"start_ts": getattr(data_cfg, "start_ts", None), "end_ts": getattr(data_cfg, "end_ts", None)})
        if fallback != (None, None):
            splits["train"].append(fallback)

    for phase, items in splits.items():
        cleaned = [item for item in items if not (item[0] is None and item[1] is None)]
        cleaned.sort(key=lambda it: it[0] if it[0] is not None else -float("inf"))
        splits[phase] = cleaned

    if not splits["train"]:
        raise ValueError("Training split is empty: provide train_start/train_end or a manifest")

    return version, splits


def _ensure_validation_split_present(
    dfs_with_roles: dict[str, pd.DataFrame],
    intervals: dict[str, list[tuple[int | None, int | None]]],
    timestamp_column: str,
    role_column: str,
) -> None:
    """Abort execution when the validation split yields zero rows."""

    val_rows = 0
    for df in dfs_with_roles.values():
        val_rows += int((df[role_column].astype(str) == "val").sum())
    if val_rows > 0:
        return

    configured = intervals.get("val", [])
    configured_desc = (
        ", ".join(_format_interval(it) for it in configured)
        if configured
        else "(not configured)"
    )
    observed_start, observed_end = _phase_bounds(dfs_with_roles, timestamp_column)
    coverage_desc = f"[{_fmt_ts(observed_start)} .. {_fmt_ts(observed_end)}]"

    msg_lines = [
        "Validation split is empty after applying configured intervals.",
        f"Configured validation intervals: {configured_desc}",
        f"Observed data coverage: {coverage_desc}",
    ]

    if configured:
        overlap_detected = None
        if observed_start is not None and observed_end is not None:
            overlap_detected = False
            for start, end in configured:
                start_cmp = observed_start if start is None else start
                end_cmp = observed_end if end is None else end
                if start_cmp <= observed_end and end_cmp >= observed_start:
                    overlap_detected = True
                    break
        if overlap_detected is False:
            msg_lines.append(
                "Configured validation window does not overlap with available data; "
                "regenerate or refresh the offline dataset."
            )
        else:
            msg_lines.append(
                "Adjust the validation split configuration or refresh the offline dataset "
                "to include the desired range."
            )
    else:
        msg_lines.append(
            "Provide validation split overrides or refresh the offline dataset to include validation data."
        )

    raise SystemExit("\n".join(msg_lines))


def _apply_role_column(
    df: pd.DataFrame,
    intervals: dict[str, list[tuple[int | None, int | None]]],
    timestamp_column: str,
    role_column: str,
) -> tuple[pd.DataFrame, bool]:
    """Annotate ``df`` with walk-forward roles using the provided intervals."""

    if timestamp_column not in df.columns:
        raise KeyError(f"DataFrame is missing timestamp column '{timestamp_column}'")
    ts = pd.to_numeric(df[timestamp_column], errors="coerce")
    roles = pd.Series(np.full(len(df), "none", dtype=object), index=df.index)

    for phase in ("train", "val", "test"):
        masks = []
        for start, end in intervals.get(phase, []):
            cur = pd.Series(True, index=df.index)
            if start is not None:
                cur &= ts >= start
            if end is not None:
                cur &= ts <= end
            masks.append(cur)
        if not masks:
            continue
        phase_mask = masks[0]
        for extra in masks[1:]:
            phase_mask |= extra
        assignable = (roles == "none") & phase_mask
        if assignable.any():
            roles.loc[assignable] = phase

    inferred_test = False
    if not intervals.get("test"):
        leftover = roles == "none"
        if leftover.any():
            roles.loc[leftover] = "test"
            inferred_test = True

    df_out = df.copy()
    df_out[role_column] = roles.values
    return df_out, inferred_test


def _phase_bounds(mapping: dict[str, pd.DataFrame], ts_col: str) -> tuple[int | None, int | None]:
    start: int | None = None
    end: int | None = None
    for df in mapping.values():
        ts = pd.to_numeric(df[ts_col], errors="coerce").dropna()
        if ts.empty:
            continue
        cur_start = int(ts.min())
        cur_end = int(ts.max())
        start = cur_start if start is None or cur_start < start else start
        end = cur_end if end is None or cur_end > end else end
    return start, end


def _fmt_ts(ts: int | None) -> str:
    if ts is None:
        return "None"
    return pd.to_datetime(int(ts), unit="s", utc=True).isoformat()


def _format_interval(interval: tuple[int | None, int | None]) -> str:
    start, end = interval
    return f"[{_fmt_ts(start)} .. {_fmt_ts(end)}]"

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

            if self.logger is not None:
                self.logger.record("pruning/sortino_ratio", current_sortino)

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

            if self.logger is not None:
                self.logger.record("pruning/objective_score", objective_score)
                self.logger.record("pruning/objective_main", main_sortino)
                self.logger.record("pruning/objective_choppy", choppy_score)
                self.logger.record("pruning/objective_trend", trend_score)

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
              cfg: TrainConfig,
              total_timesteps: int,
              train_data_by_token: dict,
              train_obs_by_token: dict,
              val_data_by_token: dict,
              val_obs_by_token: dict,
              test_data_by_token: dict,
              test_obs_by_token: dict,
              norm_stats: dict,
              sim_config: dict,
              timing_env_kwargs: dict,
              leak_guard_kwargs: dict,
              trials_dir: Path,
              tensorboard_log_dir: Path | None):

    print(f">>> Trial {trial.number+1} with budget={total_timesteps}")

    def _extract_bins_vol_from_cfg(cfg, default=101):
        try:
            aw = getattr(getattr(cfg, "algo", None), "action_wrapper", None)
            val = getattr(aw, "bins_vol", None) if aw is not None else None
            if val is None and hasattr(aw, "__dict__"):
                val = aw.__dict__.get("bins_vol")
            return max(2, int(val)) if val is not None else int(default)
        except Exception:
            return int(default)

    bins_vol = _extract_bins_vol_from_cfg(cfg, default=101)


    # ИСПРАВЛЕНО: window_size возвращен в пространство поиска HPO
    params = {
        "window_size": trial.suggest_categorical("window_size", [10, 20, 30]),
        "trade_frequency_penalty": trial.suggest_float("trade_frequency_penalty", 1e-5, 5e-4, log=True),
        "turnover_penalty_coef": trial.suggest_float("turnover_penalty_coef", 0.0, 5e-4),
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
    leak_guard_train = LeakGuard(LeakConfig(**leak_guard_kwargs))
    leak_guard_val = LeakGuard(LeakConfig(**leak_guard_kwargs))

    train_symbol_items = sorted(train_data_by_token.items())
    if not train_symbol_items:
        raise ValueError("Нет тренировочных символов для создания сред.")

    def make_env_train(rank: int):
        symbol_idx = rank % len(train_symbol_items)
        symbol, df = train_symbol_items[symbol_idx]

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
                "turnover_penalty_coef": params["turnover_penalty_coef"],
                "momentum_factor": params["momentum_factor"],
                "mean_reversion_factor": params["mean_reversion_factor"],
                "adversarial_factor": params["adversarial_factor"],

                # 2. Данные, которые передаются в функцию objective
                "norm_stats": norm_stats,

                # 3. Статические параметры и параметры индикаторов
                "mode": "train",
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
                "obv_ma_window": OBV_MA_WINDOW,

            }
            env_params.update(sim_config)
            env_params.update(timing_env_kwargs)

            # Создаем и возвращаем экземпляр среды
            env = TradingEnv(
                df,
                **env_params,
                leak_guard=leak_guard_train,
                seed=unique_seed,
            )
            setattr(env, "selected_symbol", symbol)
            env = _wrap_action_space_if_needed(env, bins_vol=bins_vol)
            return env
        return _init

    env_constructors = [make_env_train(rank=i) for i in range(n_envs)]
    train_stats_path = trials_dir / f"vec_normalize_train_{trial.number}.pkl"
    os.makedirs(trials_dir, exist_ok=True)

    base_env_tr = WatchdogVecEnv(env_constructors)
    monitored_env_tr = VecMonitor(base_env_tr)
    env_tr = VecNormalize(
        monitored_env_tr,
        norm_obs=False,
        norm_reward=False,
        clip_reward=None,
        gamma=params["gamma"],
    )

    env_tr.save(str(train_stats_path))
    save_sidecar_metadata(str(train_stats_path), extra={"kind": "vecnorm_stats", "phase": "train"})

    val_symbol_items = sorted(val_data_by_token.items())
    if not val_symbol_items:
        raise ValueError("Нет валидационных символов для создания сред.")

    def _make_val_env_factory(symbol: str, df: pd.DataFrame):
        env_val_params = {
            "norm_stats": norm_stats,
            "window_size": params["window_size"],
            "gamma": params["gamma"],
            "atr_multiplier": params["atr_multiplier"],
            "trailing_atr_mult": params["trailing_atr_mult"],
            "tp_atr_mult": params["tp_atr_mult"],
            "trade_frequency_penalty": params["trade_frequency_penalty"],
            "turnover_penalty_coef": params["turnover_penalty_coef"],
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
        env_val_params.update(sim_config)
        env_val_params.update(timing_env_kwargs)
        env = TradingEnv(
            df,
            **env_val_params,
            leak_guard=leak_guard_val,
        )
        setattr(env, "selected_symbol", symbol)
        env = _wrap_action_space_if_needed(env, bins_vol=bins_vol)
        return env

    val_env_fns = [
        lambda symbol=symbol, df=df: _make_val_env_factory(symbol, df)
        for symbol, df in val_symbol_items
    ]
    monitored_env_va = VecMonitor(DummyVecEnv(val_env_fns))
    check_model_compat(str(train_stats_path))
    env_va = VecNormalize.load(str(train_stats_path), monitored_env_va)
    env_va.training = False
    env_va.norm_reward = False
    env_va.clip_reward = None

    val_stats_path = trials_dir / f"vec_normalize_val_{trial.number}.pkl"
    env_va.save(str(val_stats_path))
    save_sidecar_metadata(str(val_stats_path), extra={"kind": "vecnorm_stats", "phase": "val"})

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
    policy_kwargs["optimizer_scheduler_fn"] = scheduler_fn
    DistributionalPPO = _get_distributional_ppo()

    tb_log_path: Path | None = None
    if tensorboard_log_dir is not None:
        tb_log_path = tensorboard_log_dir / f"trial_{trial.number:03d}"
        tb_log_path.mkdir(parents=True, exist_ok=True)

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
        tensorboard_log=str(tb_log_path) if tb_log_path is not None else None,
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
            progress_bar=True,
            tb_log_name=f"trial_{trial.number:03d}" if tb_log_path is not None else "run"
        )
    finally:
        env_tr.close()
        env_va.close()

    trial_model_path = trials_dir / f"trial_{trial.number}_model.zip"
    model.save(str(trial_model_path))
    save_sidecar_metadata(str(trial_model_path), extra={"kind": "sb3_model", "trial": int(trial.number)})

    

    print(f"<<< Trial {trial.number+1} finished training, starting unified final evaluation…")

    eval_phase_data = test_data_by_token if test_data_by_token else val_data_by_token
    eval_phase_obs = test_obs_by_token if test_data_by_token else val_obs_by_token
    eval_phase_name = "test" if test_data_by_token else "val"
    if not eval_phase_data:
        raise ValueError("No data available for validation/test evaluation. Check time split configuration.")

    # 1. Определяем все режимы для оценки
    regimes_to_evaluate = ['normal', 'choppy_flat', 'strong_trend']
    final_metrics = {}
    regime_duration = 2_500 

    # 2. Последовательно оцениваем модель в каждом режиме
    for regime in regimes_to_evaluate:
        symbol_equity_curves: list[list[float]] = []
        test_stats_path = trials_dir / f"vec_normalize_test_{trial.number}.pkl"

        for symbol, df in sorted(eval_phase_data.items()):
            def make_final_eval_env(symbol: str = symbol, df: pd.DataFrame = df):
                final_env_params = {
                    "norm_stats": norm_stats, "window_size": params["window_size"],
                    "gamma": params["gamma"], "atr_multiplier": params["atr_multiplier"],
                    "trailing_atr_mult": params["trailing_atr_mult"], "tp_atr_mult": params["tp_atr_mult"],
                    "trade_frequency_penalty": params["trade_frequency_penalty"],
                    "turnover_penalty_coef": params["turnover_penalty_coef"], "mode": eval_phase_name,
                    "reward_shaping": False, "warmup_period": warmup_period,
                    "ma5_window": MA5_WINDOW, "ma20_window": MA20_WINDOW, "atr_window": ATR_WINDOW,
                    "rsi_window": RSI_WINDOW, "macd_fast": MACD_FAST, "macd_slow": MACD_SLOW,
                    "macd_signal": MACD_SIGNAL, "momentum_window": MOMENTUM_WINDOW,
                    "cci_window": CCI_WINDOW, "bb_window": BB_WINDOW, "obv_ma_window": OBV_MA_WINDOW,
                }
                final_env_params.update(sim_config)
                final_env_params.update(timing_env_kwargs)
                env = TradingEnv(
                    df,
                    **final_env_params,
                    leak_guard=LeakGuard(LeakConfig(**leak_guard_kwargs)),
                )
                setattr(env, "selected_symbol", symbol)
                env = _wrap_action_space_if_needed(env, bins_vol=bins_vol)
                return env

            check_model_compat(str(train_stats_path))
            final_eval_norm = VecNormalize.load(
                str(train_stats_path),
                DummyVecEnv([make_final_eval_env]),
            )
            final_eval_norm.training = False
            final_eval_norm.norm_reward = False
            final_eval_norm.clip_reward = None
            final_eval_env = VecMonitor(final_eval_norm)

            if regime != 'normal':
                final_eval_env.env_method("set_market_regime", regime=regime, duration=regime_duration)

            _rewards, equity_curves = evaluate_policy_custom_cython(model, final_eval_env, num_episodes=1)
            symbol_equity_curves.extend(equity_curves)

            nt_stats = final_eval_env.env_method("get_no_trade_stats")
            if nt_stats:
                total_steps = sum(s["total_steps"] for s in nt_stats if s)
                blocked_steps = sum(s["blocked_steps"] for s in nt_stats if s)
                ratio = blocked_steps / total_steps if total_steps else 0.0
                print(
                    f"[{symbol}] No-trade blocks: {blocked_steps}/{total_steps} steps ({ratio:.2%})"
                )

            final_eval_env.close()

        if not test_stats_path.exists():
            final_eval_norm.save(str(test_stats_path))
            save_sidecar_metadata(str(test_stats_path), extra={"kind": "vecnorm_stats", "phase": eval_phase_name})

        all_returns = [pd.Series(c).pct_change().dropna().to_numpy() for c in symbol_equity_curves if len(c) > 1]
        flat_returns = np.concatenate(all_returns) if all_returns else np.array([0.0])
        final_metrics[regime] = sortino_ratio(flat_returns)

    # --- РАСЧЕТ ИТОГОВОЙ ВЗВЕШЕННОЙ МЕТРИКИ ---
    main_sortino = final_metrics.get('normal', -1.0)
    choppy_score = final_metrics.get('choppy_flat', -1.0)
    trend_score = final_metrics.get('strong_trend', -1.0)

    main_weight = 0.5
    choppy_weight = 0.3
    trend_weight = 0.2
 
    objective_score = (main_weight * main_sortino + choppy_weight * choppy_score + trend_weight * trend_score)

    if model.logger is not None:
        for regime_name, score in final_metrics.items():
            model.logger.record(f"evaluation/{regime_name}_sortino", score)
        model.logger.record("evaluation/final_objective_score", objective_score)
        model.logger.dump(step=model.num_timesteps)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_train.yaml", help="Path to YAML config")
    parser.add_argument(
        "--regime-config",
        default="configs/market_regimes.json",
        help="Path to market regime parameters",
    )
    parser.add_argument(
        "--offline-config",
        default="configs/offline.yaml",
        help="Path to offline dataset configuration with artefact declarations",
    )
    parser.add_argument(
        "--dataset-split",
        default="val",
        help="Dataset split identifier declared in the offline config",
    )
    parser.add_argument(
        "--liquidity-seasonality",
        default=None,
        help="Override path to liquidity seasonality coefficients (defaults to offline bundle)",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        default="tensorboard_logs",
        help="Directory where TensorBoard events will be written (use 'none' to disable)",
    )
    args, unknown = parser.parse_known_args()

    raw_tensorboard_dir = (args.tensorboard_log_dir or "").strip()
    if raw_tensorboard_dir.lower() in {"", "none", "null"}:
        tensorboard_log_dir = None
    else:
        tensorboard_log_dir = Path(raw_tensorboard_dir).expanduser()

    os.environ["MARKET_REGIMES_JSON"] = args.regime_config

    split_key = (args.dataset_split or "").strip()
    offline_bundle = None
    offline_payload: Mapping[str, Any] | None = None
    offline_split_overrides: dict[str, list[dict[str, int | None]]] = {}
    if split_key and split_key.lower() not in {"none", "null"}:
        try:
            offline_bundle = resolve_split_bundle(args.offline_config, split_key)
        except FileNotFoundError as exc:
            raise SystemExit(f"Offline config not found: {args.offline_config}") from exc
        except KeyError as exc:
            raise SystemExit(
                f"Dataset split '{split_key}' not found in offline config {args.offline_config}"
            ) from exc
        except ValueError as exc:
            raise SystemExit(f"Failed to resolve offline split '{split_key}': {exc}") from exc
        offline_payload = load_offline_payload(args.offline_config)
        offline_split_overrides = _extract_offline_split_overrides(
            offline_payload, split_key, fallback_split="time"
        )

    seasonality_path = args.liquidity_seasonality
    fees_path: str | None = None
    adv_path: str | None = None
    seasonality_hash: str | None = None
    if offline_bundle is not None:
        if offline_bundle.version:
            print(
                f"Resolved offline dataset split '{offline_bundle.name}' version {offline_bundle.version}"
            )
        else:
            print(f"Resolved offline dataset split '{offline_bundle.name}'")
        seasonality_art = offline_bundle.artifacts.get("seasonality")
        if seasonality_art:
            if seasonality_path is None:
                seasonality_path = seasonality_art.path.as_posix()
            raw_hash = seasonality_art.info.artifact.get("verification_hash")
            if raw_hash:
                seasonality_hash = str(raw_hash)
        fees_art = offline_bundle.artifacts.get("fees")
        if fees_art:
            fees_path = fees_art.path.as_posix()
        adv_art = offline_bundle.artifacts.get("adv")
        if adv_art:
            adv_path = adv_art.path.as_posix()

    if seasonality_path is None:
        seasonality_path = "configs/liquidity_seasonality.json"
    args.liquidity_seasonality = seasonality_path

    if not Path(seasonality_path).exists():
        raise FileNotFoundError(
            f"Liquidity seasonality file not found: {seasonality_path}. Run offline builders first."
        )

    cfg = load_config(args.config)

    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--") and "." in unknown[i]:
            key = unknown[i][2:]
            if i + 1 >= len(unknown):
                raise ValueError(f"Missing value for argument {unknown[i]}")
            block, field = key.split('.', 1)
            value = yaml.safe_load(unknown[i + 1])
            overrides.setdefault(block, {})[field] = value
            i += 2
        else:
            i += 1

    cfg_dict = cfg.dict()
    if offline_bundle is not None:
        data_block = cfg_dict.setdefault("data", {})
        if offline_bundle.version:
            data_block["split_version"] = offline_bundle.version
        if offline_split_overrides and not data_block.get("split_overrides"):
            data_block["split_overrides"] = offline_split_overrides
        if adv_path:
            adv_block = cfg_dict.setdefault("adv", {})
            if not adv_block.get("path"):
                adv_block["path"] = adv_path
        if fees_path:
            fees_block = cfg_dict.setdefault("fees", {})
            if not fees_block.get("path"):
                fees_block["path"] = fees_path
    for block, params in overrides.items():
        cfg_dict.setdefault(block, {}).update(params)
    cfg_dict["liquidity_seasonality_path"] = args.liquidity_seasonality
    cfg_dict["latency_seasonality_path"] = args.liquidity_seasonality
    latency_block = cfg_dict.setdefault("latency", {})
    if not latency_block.get("latency_seasonality_path"):
        latency_block["latency_seasonality_path"] = args.liquidity_seasonality
    if seasonality_hash:
        cfg_dict["liquidity_seasonality_hash"] = seasonality_hash
    cfg = cfg.__class__.parse_obj(cfg_dict)

    bins_vol = 101
    try:
        maybe = None
        algo_cfg = getattr(cfg, "algo", None)
        if algo_cfg is not None and hasattr(algo_cfg, "action_wrapper"):
            aw = algo_cfg.action_wrapper
            maybe = getattr(aw, "bins_vol", None)
            if maybe is None and hasattr(aw, "__dict__"):
                maybe = aw.__dict__.get("bins_vol")
        if maybe is None and hasattr(cfg, "__dict__"):
            maybe = (cfg.__dict__.get("algo", {}) or {}).get("action_wrapper", {}).get("bins_vol")
        if maybe is not None:
            bins_vol = int(maybe)
    except Exception:
        bins_vol = 101
    try:
        bins_vol = max(2, int(bins_vol))
    except Exception:
        bins_vol = 101

    timing_defaults, timing_profiles = load_timing_profiles()
    exec_profile = getattr(cfg, "execution_profile", ExecutionProfile.MKT_OPEN_NEXT_H1)
    resolved_timing = resolve_execution_timing(exec_profile, timing_defaults, timing_profiles)
    timing_env_kwargs = {
        "decision_mode": DecisionTiming[resolved_timing.decision_mode],
        "decision_delay_ms": resolved_timing.decision_delay_ms,
        "latency_steps": resolved_timing.latency_steps,
    }
    leak_guard_kwargs = {
        "decision_delay_ms": resolved_timing.decision_delay_ms,
        "min_lookback_ms": resolved_timing.min_lookback_ms,
    }

    sim_config = {k: getattr(cfg, k) for k in ("quantizer", "slippage", "fees", "latency", "risk", "no_trade")}
    sim_config["liquidity_seasonality_path"] = args.liquidity_seasonality
    liq_hash = _file_sha256(args.liquidity_seasonality)
    if liq_hash:
        sim_config["liquidity_seasonality_hash"] = liq_hash
        print(f"Liquidity seasonality hash: {liq_hash}")
    else:
        print(f"Warning: could not compute hash for {args.liquidity_seasonality}; seasonality consistency not enforced")

    processed_data_dir = getattr(cfg.data, "processed_dir", "data/processed")
    run_id = getattr(cfg, "run_id", "default-run")
    artifacts_root = Path(getattr(cfg, "artifacts_dir", "models")) / run_id
    trials_dir = artifacts_root / "trials"
    N_ENSEMBLE = int(cfg.model.params.get("n_ensemble", 5))

    if os.path.exists(trials_dir):
        print(f"Очистка старых артефактов из директории '{trials_dir}'...")
        shutil.rmtree(trials_dir)
    os.makedirs(trials_dir, exist_ok=True)

    print("Loading all pre-processed data...")
    all_feather_files = glob.glob(os.path.join(processed_data_dir, "*.feather"))
    if not all_feather_files:
        raise FileNotFoundError(
            f"No .feather files found in {processed_data_dir}. "
            f"Run prepare_advanced_data.py (Fear&Greed), prepare_events.py (macro events), "
            f"incremental_klines.py (1h candles), then prepare_and_run.py (merge/export).",
        )
    all_dfs_dict, all_obs_dict = load_all_data(all_feather_files, synthetic_fraction=0, seed=42)

    split_version, time_splits = _load_time_splits(cfg.data)
    if split_version:
        print(f"Using split version: {split_version}")
    timestamp_column = getattr(cfg.data, "timestamp_column", "timestamp")
    role_column = getattr(cfg.data, "role_column", "wf_role")

    dfs_with_roles: dict[str, pd.DataFrame] = {}
    inferred_test_any = False
    for symbol, df in all_dfs_dict.items():
        annotated, inferred_flag = _apply_role_column(df, time_splits, timestamp_column, role_column)
        dfs_with_roles[symbol] = annotated
        inferred_test_any = inferred_test_any or inferred_flag

    _ensure_validation_split_present(dfs_with_roles, time_splits, timestamp_column, role_column)

    train_intervals = time_splits.get("train", [])
    train_start_candidates = [start for start, _ in train_intervals if start is not None]
    train_end_candidates = [end for _, end in train_intervals if end is not None]
    train_start_ts = min(train_start_candidates) if train_start_candidates else None
    train_end_ts = max(train_end_candidates) if train_end_candidates else None

    PREPROC_PATH = artifacts_root / "preproc_pipeline.json"
    pipe = FeaturePipeline()
    pipe.fit(
        dfs_with_roles,
        train_mask_column=role_column,
        train_mask_values={"train"},
        train_start_ts=train_start_ts,
        train_end_ts=train_end_ts,
        timestamp_column=timestamp_column,
        split_version=split_version,
        train_intervals=train_intervals,
    )
    PREPROC_PATH.parent.mkdir(parents=True, exist_ok=True)
    pipe.save(str(PREPROC_PATH))
    try:
        save_sidecar_metadata(
            str(PREPROC_PATH),
            extra={
                "kind": "feature_pipeline",
                "split_version": split_version,
                "train_start_ts": train_start_ts,
                "train_end_ts": train_end_ts,
            },
        )
    except Exception:
        # Sidecar metadata helper may not be available in all environments.
        pass
    all_dfs_with_roles = pipe.transform_dict(dfs_with_roles, add_suffix="_z")
    print(f"Feature pipeline fitted and saved to {PREPROC_PATH}. Standardized columns *_z added.")
    print("To run inference over processed data, execute: python infer_signals.py")

    # --- Гейт качества данных: строгая валидация OHLCV перед сплитом ---
    _validator = DataValidator()
    for _key, _df in all_dfs_with_roles.items():
        try:
            # frequency=None -> автоопределение внутри валидатора
            _validator.validate(_df, frequency=None)
        except Exception as e:
            raise RuntimeError(f"Data validation failed for asset '{_key}': {e}")
    print("✓ Data validation passed for all assets.")

    def _extract_phase(phase: str) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for sym, df in all_dfs_with_roles.items():
            mask = df[role_column].astype(str) == phase
            phase_df = df.loc[mask].copy()
            if not phase_df.empty:
                out[sym] = phase_df.reset_index(drop=True)
        return out

    train_data_by_token = _extract_phase("train")
    val_data_by_token = _extract_phase("val")
    test_data_by_token = _extract_phase("test")

    def _select_obs(mapping: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
        return {sym: all_obs_dict[sym] for sym in mapping if sym in all_obs_dict}

    train_obs_by_token = _select_obs(train_data_by_token)
    val_obs_by_token = _select_obs(val_data_by_token)
    test_obs_by_token = _select_obs(test_data_by_token)

    unused_rows = {
        sym: int((df[role_column].astype(str) == "none").sum())
        for sym, df in all_dfs_with_roles.items()
        if (df[role_column].astype(str) == "none").any()
    }
    if unused_rows:
        total_unused = sum(unused_rows.values())
        print(
            f"Warning: {total_unused} rows across {len(unused_rows)} symbols were not assigned to train/val/test and will be ignored."
        )

    print("Time-based split summary:")
    for phase, mapping in (
        ("train", train_data_by_token),
        ("val", val_data_by_token),
        ("test", test_data_by_token),
    ):
        intervals = time_splits.get(phase, [])
        interval_desc = ", ".join(_format_interval(it) for it in intervals) if intervals else "(inferred remainder)"
        total_rows = sum(len(df) for df in mapping.values())
        observed_start, observed_end = _phase_bounds(mapping, timestamp_column)
        print(
            f"  {phase}: {len(mapping)} symbols, {total_rows} rows, intervals={interval_desc}, "
            f"observed=[{_fmt_ts(observed_start)} .. {_fmt_ts(observed_end)}]"
        )
    if inferred_test_any and not time_splits.get("test"):
        print("  Note: test split inferred from remaining rows (no explicit interval provided).")

    print("Calculating per-asset normalization stats from the training set...")
    norm_stats = {}

    # Итерируем по каждому активу в ТРЕНИРОВОЧНОМ наборе данных
    for asset_key, train_df in train_data_by_token.items():
        
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

    norm_stats_path = artifacts_root / "norm_stats.json"
    with open(norm_stats_path, "w") as f:
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
        lambda t: objective(
            t,
            cfg,
            HPO_BUDGET_PER_TRIAL,
            train_data_by_token,
            train_obs_by_token,
            val_data_by_token,
            val_obs_by_token,
            test_data_by_token,
            test_obs_by_token,
            norm_stats,
            sim_config,
            timing_env_kwargs,
            leak_guard_kwargs,
            trials_dir,
            tensorboard_log_dir,
        ),
        n_trials=HPO_TRIALS,
        n_jobs=1,
    )

    # Сохраняем итоговое исследование
    final_study = study
    if not final_study:
        print("No final study completed. Exiting.")
        return
    # <-- КОНЕЦ БЛОКА ДЛЯ ЗАМЕНЫ -->

    print(f"\nSaving best {N_ENSEMBLE} models from the final stage...")
    ensemble_dir = artifacts_root / "ensemble"
    if os.path.exists(ensemble_dir):
        shutil.rmtree(ensemble_dir)
    os.makedirs(ensemble_dir)

    top_trials = sorted(final_study.trials, key=lambda tr: tr.value or -1e9, reverse=True)[:N_ENSEMBLE]

    ensemble_meta = []
    for i, trial in enumerate(top_trials):
        model_idx = i + 1
        src_model = trials_dir / f"trial_{trial.number}_model.zip"
        src_stats = trials_dir / f"vec_normalize_{trial.number}.pkl"

        if os.path.exists(src_model):
            shutil.copyfile(src_model, ensemble_dir / f"model_{model_idx}.zip")
            if os.path.exists(src_stats):
                shutil.copyfile(src_stats, ensemble_dir / f"vec_normalize_{model_idx}.pkl")

            ensemble_meta.append({"ensemble_index": model_idx, "trial_number": trial.number, "value": trial.value, "params": trial.params})
        else:
            print(f"⚠️ WARNING: Could not find model for trial {trial.number}. Skipping.")
    # Копируем единый файл со статистиками нормализации наблюдений,
    # так как он является неотъемлемой частью всех моделей в ансамбле.
    src_norm_stats = artifacts_root / "norm_stats.json"
    if os.path.exists(src_norm_stats):
        shutil.copyfile(src_norm_stats, ensemble_dir / "norm_stats.json")
    else:
        print(f"⚠️ CRITICAL WARNING: Could not find the global 'norm_stats.json' file. The saved ensemble will not be usable for inference.")
    with open(ensemble_dir / "ensemble_meta.json", "w") as f:
        json.dump(ensemble_meta, f, indent=4)
    print(f"\n✅ Ensemble of {len(ensemble_meta)} models saved to '{ensemble_dir}'. HPO complete.")

    # --- Validation of the best model for reproducibility ---
    best_model_path = ensemble_dir / "model_1.zip"
    best_stats_path = ensemble_dir / "vec_normalize_1.pkl"
    if best_model_path.exists() and best_stats_path.exists():
        print("\nRunning validation on the best ensemble model...")
        best_trial = top_trials[0]

        final_eval_data = test_data_by_token if test_data_by_token else val_data_by_token
        final_eval_obs = test_obs_by_token if test_data_by_token else val_obs_by_token
        final_eval_mode = "test" if test_data_by_token else "val"
        if not final_eval_data:
            print("⚠️ Skipping final validation: evaluation split is empty.")
        else:
            def _make_env_val(symbol: str, df: pd.DataFrame):
                params = best_trial.params
                env_val_params = {
                    "norm_stats": norm_stats,
                    "window_size": params["window_size"],
                    "gamma": params["gamma"],
                    "atr_multiplier": params["atr_multiplier"],
                    "trailing_atr_mult": params["trailing_atr_mult"],
                    "tp_atr_mult": params["tp_atr_mult"],
                    "trade_frequency_penalty": params["trade_frequency_penalty"],
                    "turnover_penalty_coef": params["turnover_penalty_coef"],
                    "mode": final_eval_mode,
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
                    "obv_ma_window": OBV_MA_WINDOW,
                }
                env_val_params.update(sim_config)
                env_val_params.update(timing_env_kwargs)
                env = TradingEnv(
                    df,
                    **env_val_params,
                    leak_guard=LeakGuard(LeakConfig(**leak_guard_kwargs))
                )
                setattr(env, "selected_symbol", symbol)
                env = _wrap_action_space_if_needed(env, bins_vol=bins_vol)
                return env

            eval_env_fns: list[Callable[[], TradingEnv]] = []
            for symbol, df in sorted(final_eval_data.items()):
                def _factory(symbol=symbol, df=df):
                    return _make_env_val(symbol, df)

                eval_env_fns.append(_factory)

            monitored_eval_env = VecMonitor(DummyVecEnv(eval_env_fns))
            check_model_compat(str(best_stats_path))
            eval_env = VecNormalize.load(str(best_stats_path), monitored_eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
            eval_env.clip_reward = None

            DistributionalPPO = _get_distributional_ppo()
            best_model = DistributionalPPO.load(str(best_model_path), env=eval_env)

            rewards, equity_curves = evaluate_policy_custom_cython(
                best_model, eval_env, num_episodes=max(1, len(final_eval_data))
            )
            all_returns = [
                pd.Series(curve).pct_change().dropna().to_numpy()
                for curve in equity_curves if len(curve) > 1
            ]
            flat_returns = np.concatenate(all_returns) if all_returns else np.array([0.0])
            sortino = sortino_ratio(flat_returns)
            sharpe = sharpe_ratio(flat_returns)

            report = {
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "sortino_ratio": float(sortino),
                "sharpe_ratio": float(sharpe),
            }
            with open(ensemble_dir / "validation_report.json", "w") as f:
                json.dump(report, f, indent=4)
            print(
                f"Validation metrics -> Sortino: {sortino:.4f}, Sharpe: {sharpe:.4f}. "
                f"Report saved to '{ensemble_dir / 'validation_report.json'}'"
            )
    else:
        print(
            "⚠️ Could not find best model or normalization stats for validation evaluation."
        )

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    def _extract_grad_sanity(argv: list[str]) -> str | None:
        for idx, arg in enumerate(argv):
            if arg == "--grad-sanity":
                if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
                    return argv[idx + 1]
                return "1"
            if arg.startswith("--grad-sanity="):
                value = arg.split("=", 1)[1]
                return value if value else "1"
        return None

    flag_value = _extract_grad_sanity(sys.argv[1:])
    if flag_value is not None:
        os.environ["GRAD_SANITY"] = flag_value

    # --- gradient sanity check (включается флагом окружения) ---
    from runtime_flags import get_bool
    if get_bool("GRAD_SANITY", False):
        from tools.grad_sanity import run_check
        run_check()

    main()
