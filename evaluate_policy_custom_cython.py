import numpy as np
try:
    from cy_eval_core import evaluate_episode as _cy_evaluate_episode  # fast path
except Exception:
    _cy_evaluate_episode = None  # fallback to a safe no-op

def evaluate_policy_custom_cython(policy, n_eval_episodes=5):
    """
    Быстрая оценка политики через Cython-ядро, если доступно.
    Если расширение отсутствует — возвращает (0.0, 0.0) и предупреждает.
    """
    if _cy_evaluate_episode is None:
        import warnings
        warnings.warn(
            "cy_eval_core не найден: возвращаю (0.0, 0.0) без оценки. "
            "Установите/соберите расширение cy_eval_core для ускоренной проверки.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0, 0.0

    rewards = [_cy_evaluate_episode(policy) for _ in range(n_eval_episodes)]
    return float(np.mean(rewards)), float(np.std(rewards))
