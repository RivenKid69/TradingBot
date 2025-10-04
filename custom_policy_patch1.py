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
from gymnasium import spaces
from typing import Tuple, Type, Optional, Dict, Any, Callable

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.policies import ActorCriticPolicy
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
        enable_critic_lstm = arch_params.get("enable_critic_lstm")
        if enable_critic_lstm is not None:
            kwargs.setdefault("enable_critic_lstm", enable_critic_lstm)
        self.hidden_dim = hidden_dim

        if optimizer_class is None:
            optimizer_class = torch.optim.Adam

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
        self._last_value_logits: Optional[torch.Tensor] = None

        # lr_schedule уже используется базовым классом во время вызова super().__init__.
        # Сохраняем ссылку заранее, чтобы можно было переинициализировать оптимизатор
        # после того, как базовый класс создаст рекуррентные слои (LSTM).
        self._pending_lr_schedule = lr_schedule

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

        def _zeros_for(module: Optional[nn.Module]) -> Tuple[torch.Tensor, ...]:
            zeros = torch.zeros(self.lstm_hidden_state_shape, device=self.device)
            if isinstance(module, nn.GRU):
                return (zeros.clone(),)
            return (zeros.clone(), torch.zeros_like(zeros))

        pi_initial = _zeros_for(self.lstm_actor)
        if self.lstm_critic is not None:
            vf_initial = _zeros_for(self.lstm_critic)
        else:
            vf_initial = tuple(t.clone() for t in pi_initial)

        self.recurrent_initial_state = RNNStates(pi=pi_initial, vf=vf_initial)

        # После того как LSTM-слои фактически созданы, переинициализируем оптимизатор
        # с нужным набором параметров и (опционально) кастомным планировщиком.
        self._setup_custom_optimizer()

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
        self.value_net = self.dist_head

        # Во время вызова _build из базового класса рекуррентные слои ещё не созданы,
        # поэтому откладываем настройку оптимизатора до завершения __init__.
        self._pending_lr_schedule = lr_schedule

    def _setup_custom_optimizer(self) -> None:
        """
        Настраивает оптимизатор и (опционально) планировщик, используя только те
        параметры, которые должны участвовать в обучении кастомной политики.
        """

        lr_schedule = getattr(self, "_pending_lr_schedule", None)
        if lr_schedule is None:
            return

        modules: list[nn.Module] = [self.mlp_extractor]

        lstm_actor = getattr(self, "lstm_actor", None)
        if lstm_actor is not None:
            modules.append(lstm_actor)

        lstm_critic = getattr(self, "lstm_critic", None)
        if lstm_critic is not None and lstm_critic is not lstm_actor:
            modules.append(lstm_critic)

        modules.extend([self.action_net, self.dist_head])

        params: list[nn.Parameter] = []
        for module in modules:
            params.extend(module.parameters())

        if getattr(self, "log_std", None) is not None:
            params.append(self.log_std)
        if hasattr(self, "unconstrained_log_std"):
            params.append(self.unconstrained_log_std)

        self.optimizer = self.optimizer_class(params, lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.optimizer_scheduler_fn is not None:
            self.optimizer_scheduler = self.optimizer_scheduler_fn(self.optimizer)
        else:
            self.optimizer_scheduler = None

        # После настройки оптимизатора ссылка на lr_schedule больше не нужна.
        self._pending_lr_schedule = None
    # --- ИСПРАВЛЕНИЕ: Метод переименован с forward_rnn на _forward_recurrent ---
    def _forward_recurrent(
        self,
        features: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, RNNStates]:
        """
        Обрабатывает последовательность признаков, используя рекуррентные блоки
        актёра и критика из базовой политики.
        Возвращает скрытые состояния и обновлённые RNNStates.
        """
        if self.share_features_extractor:
            pi_features = vf_features = features  # type: ignore[assignment]
        else:
            assert isinstance(features, tuple) and len(features) == 2
            pi_features, vf_features = features

        latent_pi, new_pi_states = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )

        if self.lstm_critic is not None:
            latent_vf, new_vf_states = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            new_vf_states = tuple(state.detach() for state in new_pi_states)
        else:
            latent_vf = self.critic(vf_features)
            new_vf_states = lstm_states.vf

        return latent_pi, latent_vf, RNNStates(new_pi_states, new_vf_states)

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

    def _get_value_logits(self, latent_vf: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты распределения ценностей без их агрегации."""

        value_logits = self.dist_head(latent_vf)  # [B, n_atoms]
        self._last_value_logits = value_logits
        return value_logits

    def _value_from_logits(self, value_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(value_logits, dim=-1)
        return (probs * self.atoms).sum(dim=-1, keepdim=True)

    def _get_value_from_latent(self, latent_vf: torch.Tensor) -> torch.Tensor:
        """
        Переопределяем базовый метод SB3.
        Вместо отдельной линейной головы берём logits → probs → ожидание.
        """
        value_logits = self._get_value_logits(latent_vf)
        return self._value_from_logits(value_logits)

    @property
    def last_value_logits(self) -> Optional[torch.Tensor]:
        return self._last_value_logits

    def forward(
        self,
        obs: torch.Tensor,
        lstm_states: Optional[RNNStates],
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        if lstm_states is None:
            lstm_states = self.recurrent_initial_state

        features = self.extract_features(obs)
        latent_pi, latent_vf, new_states = self._forward_recurrent(features, lstm_states, episode_starts)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        values = self._get_value_from_latent(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob, new_states

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        lstm_states: Optional[RNNStates],
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if lstm_states is None:
            lstm_states = self.recurrent_initial_state

        features = self.extract_features(obs)
        latent_pi, latent_vf, _ = self._forward_recurrent(features, lstm_states, episode_starts)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self._get_value_from_latent(latent_vf)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, ...],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self._get_value_from_latent(latent_vf)
