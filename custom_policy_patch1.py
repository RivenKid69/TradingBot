# Имя файла: custom_policy_patch.py
# ИЗМЕНЕНИЯ (АРХИТЕКТУРНЫЙ РЕФАКТОРИНГ):
# 1. Устранен конфликт двух механизмов памяти (Трансформер vs GRU).
# 2. Выбран путь "чистой рекуррентности": GRU становится основным и единственным
#    механизмом памяти.
# 3. CustomMlpExtractor радикально упрощен. Теперь это не сложный анализатор
#    окон, а простой MLP-кодировщик признаков ОДНОГО временного шага.
# 4. Все неиспользуемые более классы (Attention, Conv блоки и т.д.) удалены.
# 
# --- ИСПРАВЛЕНИЕ IndexError ---
# Применены точечные исправления к CustomActorCriticPolicy, чтобы она
# корректно работала с новой GRU-архитектурой, не затрагивая остальной код.

import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from functools import partial
from gymnasium import spaces
from typing import Tuple, Type, Optional, Dict, Any, Callable

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule  # тип коллбэка lr_schedule



class CustomMlpExtractor(nn.Module):
    """
    Простой MLP-экстрактор для кодирования признаков одного временного шага.
    Он заменяет сложную трансформер-подобную архитектуру, передавая
    задачу обработки последовательности рекуррентной сети (GRU).
    """
    def __init__(self, feature_dim: int, hidden_dim: int, activation: Type[nn.Module]):
        super().__init__()
        # Определяем размерность для actor и critic
        self.latent_dim_pi = hidden_dim
        self.latent_dim_vf = hidden_dim

        # Простая полносвязная сеть для проекции признаков
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Просто прогоняем признаки через MLP
        return self.input_projection(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return features


class CustomActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        # правильное имя в SB3:
        lr_schedule: Optional[Schedule] = None,
        *args,
        # бэк-компат: если кто-то ещё передаёт старое имя:
        lr_scheduler: Optional[Schedule] = None,
        optimizer_class=None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        arch_params=None,
        optimizer_scheduler_fn: Optional[Callable[[Optimizer], Any]] = None,
        **kwargs,
    ):
        # если нам пришло lr_scheduler (старое имя) — мапим на lr_schedule
        if lr_schedule is None and lr_scheduler is not None:
            lr_schedule = lr_scheduler

        arch_params = arch_params or {}
        hidden_dim = arch_params.get('hidden_dim', 64)
        # SB3 ожидает, что lstm_hidden_size задаёт фактическую размерность скрытого
        # состояния. Если мы не пробросим это значение, политика внутри базового
        # класса создаст LSTM размером по умолчанию (256), а дальнейшие головы,
        # построенные на «hidden_dim», начнут конфликтовать по размерностям.
        kwargs = dict(kwargs)
        kwargs.setdefault("lstm_hidden_size", hidden_dim)
        self.hidden_dim = hidden_dim

        # super().__init__ вызывает _build, поэтому заранее сохраняем размерность действия
        if isinstance(action_space, spaces.Box):
            self.action_dim = int(np.prod(action_space.shape))
            self._multi_discrete_nvec: Optional[np.ndarray] = None
        elif isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
            self._multi_discrete_nvec = None
        elif isinstance(action_space, spaces.MultiDiscrete):
            # MultiDiscrete actions are modeled via a MultiCategorical distribution
            # whose logits are concatenated for every sub-action.
            self._multi_discrete_nvec = action_space.nvec.astype(np.int64)
            self.action_dim = int(self._multi_discrete_nvec.sum())
        else:
            raise NotImplementedError(
                f"Action space {type(action_space)} is not supported by CustomActorCriticPolicy"
            )

        act_str = arch_params.get('activation', 'relu').lower()
        if act_str == 'relu':
            self.activation = nn.ReLU
        elif act_str == 'tanh':
            self.activation = nn.Tanh
        elif act_str == 'leakyrelu':
            self.activation = nn.LeakyReLU
        else:
            self.activation = nn.ReLU

        # Параметры n_res_blocks, n_attn_blocks, attn_heads больше не нужны
        
        self.use_memory = True  # Память обеспечивается рекуррентными слоями SB3

        self.num_atoms = arch_params.get("num_atoms", 51)
        self.v_min = -1.0  # Начальное значение-заглушка
        self.v_max = 1.0

        self.optimizer_scheduler_fn = optimizer_scheduler_fn

        # dist_head создаётся позже в _build, но атрибут инициализируем заранее,
        # чтобы на него можно было безопасно ссылаться до сборки модели.
        self.dist_head: Optional[nn.Linear] = None

        # Сохраняем lr_schedule для последующей реконфигурации оптимизатора,
        # когда рекуррентные блоки будут полностью инициализированы.
        self._stored_lr_schedule: Optional[Schedule] = lr_schedule

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )

        # После инициализации базовый класс знает фактическую размерность скрытого
        # состояния (self.lstm_output_dim). Синхронизируем её с кастомным полем,
        # чтобы избежать расхождений при создании голов актёра и критика.
        self.hidden_dim = self.lstm_output_dim

        # буфер с опорой атомов остаётся
        atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms)
        self.register_buffer("atoms", atoms)

        # совместимость: где-то в старом коде могли обращаться к self.value_net
        self.value_net = self.dist_head

        

        if isinstance(self.action_space, spaces.Box):
            self.unconstrained_log_std = nn.Parameter(torch.zeros(self.action_dim))

        state_shape = (self.lstm_hidden_state_shape[0], 1, self.lstm_output_dim)
        self.recurrent_initial_state = (
            torch.zeros(state_shape, device=self.device),
            torch.zeros(state_shape, device=self.device),
        )

        # Сохраняем ссылки на рекуррентные слои, созданные базовым классом.
        self.pi_lstm = self.lstm_actor
        self.vf_lstm = self.lstm_critic if self.lstm_critic is not None else self.lstm_actor

        # После того как рекуррентные блоки стали доступны, переинициализируем
        # оптимизатор, чтобы он видел новые параметры (в частности, GRU/LSTM).
        self._configure_optimizer()

    @torch.no_grad()
    def update_atoms(self, v_min: float, v_max: float) -> None:
        """
        Динамически обновляет диапазон и сами атомы для value-распределения.
        """
        # Проверяем, изменился ли диапазон, чтобы не делать лишнюю работу
        if self.v_min == v_min and self.v_max == v_max:
            return

        self.v_min = v_min
        self.v_max = v_max
        
        # Создаем временный тензор с новыми значениями
        new_atoms = torch.linspace(v_min, v_max, self.num_atoms, device=self.atoms.device)
        
        # Копируем данные "in-place" в существующий буфер.
        # Это сохраняет регистрацию и гарантирует правильное сохранение/загрузку.
        self.atoms.copy_(new_atoms)

    def _build_mlp_extractor(self) -> None:
        # Теперь создается простой MLP экстрактор
        self.mlp_extractor = CustomMlpExtractor(
            feature_dim=self.features_dim, 
            hidden_dim=self.hidden_dim,
            activation=self.activation
        )
    def _build(self, lr_schedule) -> None:
        """
        Создаёт архитектуру сети, используя базовую реализацию SB3, а затем
        заменяет value-голову на дистрибутивную.
        """
        super()._build(lr_schedule)

        # Перестраиваем value-голову на распределение атомов.
        self.dist_head = nn.Linear(self.lstm_output_dim, self.num_atoms)
        if self.ortho_init:
            self.dist_head.apply(partial(self.init_weights, gain=1.0))
        self.value_net = self.dist_head

        # Сохраняем последний lr_schedule, чтобы впоследствии корректно
        # восстановить оптимизатор уже с рекуррентными блоками.
        self._stored_lr_schedule = lr_schedule

        # Временный оптимизатор, созданный базовым классом, больше не нужен —
        # его параметры будут заменены позже в _configure_optimizer().
        if hasattr(self, "optimizer_scheduler") and self.optimizer_scheduler is not None:
            # scheduler может хранить ссылки на старый оптимизатор; сбрасываем.
            self.optimizer_scheduler = None
        # Старый оптимизатор больше не используется.
        self.optimizer = None

    def _configure_optimizer(self) -> None:
        """Настраивает оптимизатор под актуальный набор параметров модели."""
        lr_schedule = self._stored_lr_schedule
        if lr_schedule is None:
            raise ValueError("lr_schedule is not defined; не удалось сконфигурировать оптимизатор")

        modules: list[nn.Module] = [self.mlp_extractor, self.action_net, self.dist_head]

        lstm_actor = getattr(self, "pi_lstm", getattr(self, "lstm_actor", None))
        if lstm_actor is not None:
            modules.append(lstm_actor)

        lstm_critic = getattr(self, "vf_lstm", getattr(self, "lstm_critic", None))
        if lstm_critic is not None and lstm_critic is not lstm_actor:
            modules.append(lstm_critic)

        params: list[nn.Parameter] = []
        for module in modules:
            params.extend(module.parameters())

        if hasattr(self, "log_std") and self.log_std is not None:
            params.append(self.log_std)
        if hasattr(self, "unconstrained_log_std"):
            params.append(self.unconstrained_log_std)

        self.optimizer = self.optimizer_class(params, lr=lr_schedule(1), **self.optimizer_kwargs)
        if self.optimizer_scheduler_fn is not None:
            self.optimizer_scheduler = self.optimizer_scheduler_fn(self.optimizer)
        else:
            self.optimizer_scheduler = None

    # --- ИСПРАВЛЕНИЕ: Метод переименован с forward_rnn на _forward_recurrent ---
    def _forward_recurrent(
        self,
        features: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, ...],
        episode_starts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Обрабатывает последовательность признаков, используя раздельные рекуррентные
        блоки для актёра и критика.
        Возвращает скрытые состояния, которые будут использованы как latent_pi и latent_vf.
        """
        pi_hidden_state, vf_hidden_state = lstm_states

        mask = (1.0 - episode_starts).reshape(1, -1, 1)
        pi_hidden_state = pi_hidden_state * mask
        vf_hidden_state = vf_hidden_state * mask

        features_rnn = features.unsqueeze(0)

        # ИЗМЕНЕНО: latent_pi и latent_vf теперь являются прямыми выходами рекуррентного блока
        latent_pi, new_pi_hidden_state = self.pi_lstm(features_rnn, pi_hidden_state)
        latent_vf, new_vf_hidden_state = self.vf_lstm(features_rnn, vf_hidden_state)

        latent_pi = latent_pi.squeeze(0)
        latent_vf = latent_vf.squeeze(0)

        new_lstm_states = (new_pi_hidden_state, new_vf_hidden_state)

        return latent_pi, latent_vf, new_lstm_states

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        if isinstance(self.action_space, spaces.Box):
            mean_actions = self.action_net(latent_pi)
            # Smoothly map the unconstrained parameter into the range [-5, 0]
            # torch.tanh returns [-1, 1]; rescale and shift it accordingly.
            log_std = -2.5 + 2.5 * torch.tanh(self.unconstrained_log_std)
            return self.action_dist.proba_distribution(mean_actions, log_std)
        elif isinstance(self.action_space, spaces.Discrete):
            action_logits = self.action_net(latent_pi)
            return self.action_dist.proba_distribution(action_logits)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            action_logits = self.action_net(latent_pi)
            # The underlying MultiCategorical distribution expects a concatenated
            # logits tensor of shape [batch_size, sum(nvec)]. The policy head
            # already produces the required dimensionality.
            return self.action_dist.proba_distribution(action_logits)
        else:
            raise NotImplementedError(f"Action space {type(self.action_space)} not supported")

    def _get_value_from_latent(self, latent_vf: torch.Tensor) -> torch.Tensor:
        """
        Переопределяем базовый метод SB3.
        Вместо отдельной линейной головы берём logits → probs → ожидание.
        """
        dist_logits = self.dist_head(latent_vf)        # [B, n_atoms]
        probs = torch.softmax(dist_logits, dim=-1)     # [B, n_atoms]
        value = (probs * self.atoms).sum(dim=-1, keepdim=True)  # [B, 1]
        return value
