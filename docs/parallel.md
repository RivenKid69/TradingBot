# Параллельные окружения

## SharedMemoryVecEnv

`SharedMemoryVecEnv` запускает несколько экземпляров среды в отдельных процессах и
обменивается данными через сегменты `shared_memory`. Конструктору передаётся
список фабрик `env_fns`; каждая фабрика должна возвращать *новый* экземпляр
среды. Пример создания двух параллельных сред:

```python
from shared_memory_vec_env import SharedMemoryVecEnv
from gymnasium.envs.classic_control import CartPoleEnv

def make_env():
    return CartPoleEnv()

vec_env = SharedMemoryVecEnv([make_env, make_env], base_seed=123)
obs = vec_env.reset()
reset_infos = vec_env.reset_infos  # информация о каждом воркере после reset
# ... работа со средой ...
vec_env.close()
```

Поле `reset_infos` хранит словари `info` для каждого воркера и обновляется при
каждом вызове `reset()`, повторяя поведение стандартных векторных обёрток SB3.

## Передача seed и управление ГПСЧ

Параметр `base_seed` задаёт зерно для первого воркера. Для воркера с индексом
`i` вычисляется `seed = base_seed + i`. Внутри процесса устанавливаются:

* `np.random.seed(seed)` — глобальный генератор `NumPy`;
* `env._rng = np.random.default_rng(seed)`;
* `env.reset(seed=seed)` при первом запуске.

Таким образом каждый воркер получает независимую последовательность случайных
чисел, а повторный запуск с тем же `base_seed` воспроизводим. Для полной
детерминированности рекомендуется дополнительно фиксировать `random.seed` и, при
использовании PyTorch, `torch.manual_seed`.

## Порядок train/val/test

1. **train** — обучение модели на тренировочных данных;
2. **val** — подбор гиперпараметров и ранняя остановка на валидации при том же
   `base_seed`;
3. **test** — финальная проверка на отложенной выборке (при желании можно
   изменить `base_seed`, чтобы исключить утечку информации).

## Минимальный воспроизводимый пример

```python
import numpy as np
from gymnasium.envs.classic_control import CartPoleEnv
from shared_memory_vec_env import SharedMemoryVecEnv

def make_env():
    return CartPoleEnv()

def rollout(seed, steps=5):
    env = SharedMemoryVecEnv([make_env, make_env], base_seed=seed)
    obs_seq = []
    obs = env.reset()
    obs_seq.append(obs.copy())
    for _ in range(steps):
        actions = np.zeros((env.num_envs, 1), dtype=np.float32)
        obs, _, _, _ = env.step(actions)
        obs_seq.append(obs.copy())
    env.close()
    return obs_seq
seq1 = rollout(42)
seq2 = rollout(42)
for a, b in zip(seq1, seq2):
    assert np.allclose(a, b)
```

Сохраните код в файл `example.py` и запустите:

```bash
python example.py
```
