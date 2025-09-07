"""
Configuration dataclasses for Trading Environment.
"""

 

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

 

@dataclass
class ExecutionParams:
"""Настройки исполнения торговых действий (например, комиссии)."""
taker_fee: float = 0.001 # Комиссия тейкера (доля от объёма сделки)
maker_fee: float = 0.0005 # Комиссия мейкера (доля от объёма сделки)
# (Дополнительные параметры исполнения могут быть добавлены здесь)

@classmethod
def default(cls) -> "ExecutionParams":
    """Создает ExecutionParams с параметрами по умолчанию."""
    return cls()

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "ExecutionParams":
    """
    Создает ExecutionParams из словаря. Неуказанные поля принимают значения по умолчанию.
    Неизвестные ключи игнорируются.
    """
    cfg = cls.default()
    for key, val in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg

def to_dict(self) -> Dict[str, Any]:
    """Возвращает словарь с параметрами ExecutionParams."""
    return {"taker_fee": self.taker_fee, "maker_fee": self.maker_fee}


@dataclass
class RiskParams:
"""Настройки риск-менеджмента и авто-закрытия позиций."""
use_atr_stop: bool = False # Включить статический ATR-stop (стоп-лосс по ATR)
atr_multiplier: float = 1.5 # Множитель ATR для статического стоп-лосса
use_trailing_stop: bool = False # Включить трейлинг-стоп
trailing_atr_mult: float = 2.0 # Множитель ATR для трейлинг-стопа (от пикового значения)
tp_atr_mult: float = 0.0 # Множитель ATR для тейк-профита (0 = не использовать)
terminate_on_sl_tp: bool = False # Завершать эпизод при срабатывании SL/TP
use_dynamic_risk: bool = False # Включить динамический риск-профиль (через индекс "страх/жадность")
risk_off_level: float = 0.3 # Порог индекса для режима risk-off (низкий риск)
risk_on_level: float = 0.7 # Порог индекса для режима risk-on (высокий риск)
max_position_risk_off: float = 0.3 # Максимальная доля позиции в режиме risk-off
max_position_risk_on: float = 1.0 # Максимальная доля позиции в режиме risk-on
fear_greed_value: float = 0.5 # Текущее значение индекса "страх/жадность" (0-1, используется при динамическом риске)
bankruptcy_threshold: float = 0.0 # Порог банкротства (доля от начального капитала, 0 = банкротство при <= 0 балансе)
bankruptcy_penalty: float = 0.0 # Дополнительный штраф за банкротство (добавляется к вознаграждению при событии)
max_drawdown: float = 1.0 # Максимальная допустимая просадка капитала (доля от пикового значения, 1.0 = не ограничено)

@classmethod
def default(cls) -> "RiskParams":
    """Создает RiskParams с параметрами по умолчанию."""
    return cls()

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "RiskParams":
    """
    Создает RiskParams из словаря. Неуказанные поля получают значения по умолчанию.
    Неизвестные ключи игнорируются.
    """
    cfg = cls.default()
    for key, val in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    # Validation: ensure levels are in order
    if cfg.risk_off_level > cfg.risk_on_level:
        raise ValueError("risk_off_level must be <= risk_on_level")
    return cfg

def to_dict(self) -> Dict[str, Any]:
    """Возвращает словарь с параметрами RiskParams."""
    return {
        "use_atr_stop": self.use_atr_stop,
        "atr_multiplier": self.atr_multiplier,
        "use_trailing_stop": self.use_trailing_stop,
        "trailing_atr_mult": self.trailing_atr_mult,
        "tp_atr_mult": self.tp_atr_mult,
        "terminate_on_sl_tp": self.terminate_on_sl_tp,
        "use_dynamic_risk": self.use_dynamic_risk,
        "risk_off_level": self.risk_off_level,
        "risk_on_level": self.risk_on_level,
        "max_position_risk_off": self.max_position_risk_off,
        "max_position_risk_on": self.max_position_risk_on,
        "fear_greed_value": self.fear_greed_value,
        "bankruptcy_threshold": self.bankruptcy_threshold,
        "bankruptcy_penalty": self.bankruptcy_penalty,
        "max_drawdown": self.max_drawdown
    }


@dataclass
class RewardParams:
"""Настройки функции вознаграждения."""
use_potential_shaping: bool = False # Использовать potential-based shaping
gamma: float = 0.99 # Коэффициент дисконтирования для shaping (γ)
potential_shaping_coef: float = 1.0 # Масштаб коэффициента потенциала
risk_aversion_variance: float = 1.0 # Коэффициент штрафа за открытый риск (размер позиции/волатильность)
risk_aversion_drawdown: float = 1.0 # Коэффициент штрафа за просадку от пика
trade_frequency_penalty: float = 0.0 # Штраф за каждую сделку (частоту торговли)
event_reward: bool = False # Включить бонусы/штрафы за события закрытия позиций
profit_close_bonus: float = 0.0 # Бонус за закрытие позиции с прибылью
loss_close_penalty: float = 0.0 # Штраф за закрытие позиции с убытком

@classmethod
def default(cls) -> "RewardParams":
    """Создает RewardParams с параметрами по умолчанию."""
    return cls()

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "RewardParams":
    """
    Создает RewardParams из словаря. Неуказанные поля получают значения по умолчанию.
    Неизвестные ключи игнорируются.
    """
    cfg = cls.default()
    for key, val in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg

def to_dict(self) -> Dict[str, Any]:
    """Возвращает словарь с параметрами RewardParams."""
    return {
        "use_potential_shaping": self.use_potential_shaping,
        "gamma": self.gamma,
        "potential_shaping_coef": self.potential_shaping_coef,
        "risk_aversion_variance": self.risk_aversion_variance,
        "risk_aversion_drawdown": self.risk_aversion_drawdown,
        "trade_frequency_penalty": self.trade_frequency_penalty,
        "event_reward": self.event_reward,
        "profit_close_bonus": self.profit_close_bonus,
        "loss_close_penalty": self.loss_close_penalty
    }


@dataclass
class ObsParams:
"""Настройки формирования наблюдений (признаков) для агента."""
external_columns: int = 0 # Число нормализованных внешних признаков, подаваемых в среду на каждом шаге
include_fear_greed: bool = False # Включать ли индекс "страх/жадность" в вектор признаков
max_num_tokens: int = 0 # Максимальное число токенов событий, кодируемых в наблюдении (0 = не использовать)
token_vocab_size: int = 0 # Размер словаря возможных токенов (для one-hot кодирования, если используется)

@property
def n_features(self) -> int:
    """Полная длина вектора признаков (N_FEATURES) с учетом всех компонентов."""
    # Базовые признаки, индикаторы, производные, состояние агента, микроструктура, метаданные
    base_count = 3
    indicator_count = 13
    derived_count = 2
    agent_state_count = 6
    micro_count = 3
    metadata_base = 2  # важность события, время с события
    total = base_count + indicator_count + derived_count + agent_state_count + micro_count + metadata_base
    # Добавить внешние колонки и индекс страха/жадности (если указан)
    total += self.external_columns
    if self.include_fear_greed:
        total += 1
    # Токены: если используются, добавить размер вектора токенов
    if self.max_num_tokens > 0 and self.token_vocab_size > 0:
        # Здесь предполагается уплощенное one-hot представление для max_num_tokens токенов
        total += self.token_vocab_size * self.max_num_tokens
    return total

@classmethod
def default(cls) -> "ObsParams":
    """Создает ObsParams с параметрами по умолчанию."""
    return cls()

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "ObsParams":
    """
    Создает ObsParams из словаря. Неуказанные поля принимают значения по умолчанию.
    Неизвестные ключи игнорируются.
    """
    cfg = cls.default()
    for key, val in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg

def to_dict(self) -> Dict[str, Any]:
    """Возвращает словарь с параметрами ObsParams."""
    return {
        "external_columns": self.external_columns,
        "include_fear_greed": self.include_fear_greed,
        "max_num_tokens": self.max_num_tokens,
        "token_vocab_size": self.token_vocab_size
    }


@dataclass
class MicroParams:
"""Настройки генерации микроструктурных событий рынка."""
events_per_step: float = 5.0 # Среднее количество внешних событий (заявок/сделок/отмен) за шаг
p_limit_order: float = 0.5 # Вероятность события PUBLIC_LIMIT_ADD
p_market_order: float = 0.3 # Вероятность события PUBLIC_MARKET_MATCH
p_cancel_order: float = 0.2 # Вероятность события PUBLIC_CANCEL_RANDOM

@classmethod
def default(cls) -> "MicroParams":
    """Создает MicroParams с параметрами по умолчанию."""
    return cls()

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "MicroParams":
    """
    Создает MicroParams из словаря. Неуказанные поля принимают значения по умолчанию.
    Неизвестные ключи игнорируются.
    """
    cfg = cls.default()
    for key, val in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    # Normalize probabilities if provided
    total_p = cfg.p_limit_order + cfg.p_market_order + cfg.p_cancel_order
    if total_p > 0:
        cfg.p_limit_order /= total_p
        cfg.p_market_order /= total_p
        cfg.p_cancel_order /= total_p
    return cfg

def to_dict(self) -> Dict[str, Any]:
    """Возвращает словарь с параметрами MicroParams."""
    return {
        "events_per_step": self.events_per_step,
        "p_limit_order": self.p_limit_order,
        "p_market_order": self.p_market_order,
        "p_cancel_order": self.p_cancel_order
    }


@dataclass
class MarketParams:
"""Настройки рыночной среды и начальных условий."""
initial_balance: float = 10000.0 # Начальный баланс (начальный капитал)
initial_price: float = 100.0 # Начальная цена инструмента
price_scale: int = 100 # Масштаб цены (количество тиков в единице цены, для перевода цен в целые числа)
initial_atr: float = 1.0 # Начальное значение ATR (для оценок волатильности, если требуется)
max_steps: Optional[int] = None # Максимальная длительность эпизода в шагах (None или 0 = не ограничено)

@classmethod
def default(cls) -> "MarketParams":
    """Создает MarketParams с параметрами по умолчанию."""
    return cls()

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "MarketParams":
    """
    Создает MarketParams из словаря. Неуказанные поля получают значения по умолчанию.
    Неизвестные ключи игнорируются.
    """
    cfg = cls.default()
    for key, val in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg

def to_dict(self) -> Dict[str, Any]:
    """Возвращает словарь с параметрами MarketParams."""
    return {
        "initial_balance": self.initial_balance,
        "initial_price": self.initial_price,
        "price_scale": self.price_scale,
        "initial_atr": self.initial_atr,
        "max_steps": self.max_steps
    }


@dataclass
class EnvConfig:
"""Общая конфигурация торговой среды."""
execution_mode: str = "FULL_LOB" # Режим исполнения: "FULL_LOB" (полная симуляция книги заявок) или "FAST" (упрощенный режим)
execution: ExecutionParams = field(default_factory=ExecutionParams.default)
risk: RiskParams = field(default_factory=RiskParams.default)
reward: RewardParams = field(default_factory=RewardParams.default)
obs: ObsParams = field(default_factory=ObsParams.default)
micro: MicroParams = field(default_factory=MicroParams.default)
market: MarketParams = field(default_factory=MarketParams.default)

@classmethod
def default(cls) -> "EnvConfig":
    """Создает EnvConfig со всеми вложенными параметрами по умолчанию."""
    return cls()

@classmethod
def from_dict(cls, data: Dict[str, Any]) -> "EnvConfig":
    """
    Создает EnvConfig из словаря, включая вложенные группы параметров.
    Неуказанные разделы будут заполнены значениями по умолчанию.
    """
    cfg = cls.default()
    for key, val in data.items():
        if key == "execution" and isinstance(val, dict):
            cfg.execution = ExecutionParams.from_dict(val)
        elif key == "risk" and isinstance(val, dict):
            cfg.risk = RiskParams.from_dict(val)
        elif key == "reward" and isinstance(val, dict):
            cfg.reward = RewardParams.from_dict(val)
        elif key == "obs" and isinstance(val, dict):
            cfg.obs = ObsParams.from_dict(val)
        elif key == "micro" and isinstance(val, dict):
            cfg.micro = MicroParams.from_dict(val)
        elif key == "market" and isinstance(val, dict):
            cfg.market = MarketParams.from_dict(val)
        elif key == "execution_mode":
            mode = str(val).upper()
            if mode not in ("FULL_LOB", "FAST"):
                raise ValueError(f"Invalid execution_mode: {mode}")
            cfg.execution_mode = mode
        else:
            # Ignore unknown top-level keys
            pass
    return cfg

def to_dict(self) -> Dict[str, Any]:
    """Возвращает словарь со всеми параметрами конфигурации окружения."""
    return {
        "execution_mode": self.execution_mode,
        "execution": self.execution.to_dict(),
        "risk": self.risk.to_dict(),
        "reward": self.reward.to_dict(),
        "obs": self.obs.to_dict(),
        "micro": self.micro.to_dict(),
        "market": self.market.to_dict()
    }
