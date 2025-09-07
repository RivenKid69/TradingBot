from __future__ import annotations
try:
    import infra_shim  # noqa: F401
except Exception:
    pass
"""
TradingEnv – Phase 11
Fully modern pipeline (Dict action‑space). Legacy box/array paths removed.
"""
import os
import random
from dataclasses import dataclass
from typing import Any, Sequence, Tuple
from core_constants import PRICE_SCALE
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces as spaces
import event_bus
from infra.event_bus import publish, Topics
from infra.time_provider import TimeProvider, RealTimeProvider

# --- auto‑import compiled C++ simulator ---
try:
    from fast_market import MarketSimulatorWrapper, CyMicrostructureGenerator
    _HAVE_FAST_MARKET = True
except ImportError:
    _HAVE_FAST_MARKET = False
# --- runtime switch for mini/full core ---
try:
    from runtime_flags import USE_MINI_CORE  # project-local
except Exception:
    import os as _os
    def _to_bool(v: object) -> bool:
        return str(v).strip().lower() in {"1", "true", "yes", "on"}
    USE_MINI_CORE = _to_bool(_os.environ.get("USE_MINI_CORE", "0"))

if USE_MINI_CORE:
    _HAVE_FAST_MARKET = False  # принудительно mini-режим


# --- unified MarketRegime enum (try C++ version first) ---
try:
    from core_constants import MarketRegime  # C++ enum
except ImportError:
    # -----------------------------------------------------------------
    # MarketRegime enum – single source of truth
    # -----------------------------------------------------------------
    try:
        from cy_constants import MarketRegime  # из C++ хедера через Cython
except Exception:
        try:
            from core_constants import MarketRegime  # старый путь, если cy_constants не собран
        except Exception:
            class MarketRegime(int):
                NORMAL        = 0
                CHOPPY_FLAT   = 1
                STRONG_TREND  = 2
                ILLIQUID      = 3
                # aliases for backward compatibility
                FLAT          = CHOPPY_FLAT
                TREND         = STRONG_TREND
                _COUNT        = 4
                def __new__(cls, value: int):
                    if not 0 <= value < cls._COUNT:
                        raise ValueError("invalid MarketRegime")
                    return int.__new__(cls, int(value))
    # PATCH‑ID:P15_TENV_enum
from action_proto import ActionProto, ActionType
from mediator import Mediator


class _AgentOrders(set):
def count(self) -> int: # noqa: D401
return len(self)

@dataclass(slots=True)
class _EnvState:
cash: float
units: float
net_worth: float
step_idx: int
peak_value: float
agent_orders: _AgentOrders
max_position: float
max_position_risk_on: float
is_bankrupt: bool = False

───────────────────────────────── Environment ─────────────────────────────
class TradingEnv(gym.Env):
# -----------------------------------------------------------------
# External control: force regime for adversarial evaluation
# -----------------------------------------------------------------
def set_market_regime(self, regime: str = "normal", duration: int = 0):
    """Proxy to MarketSimulator.force_market_regime()."""
    if not _HAVE_FAST_MARKET:
        print("⚠️  fast_market not available – regime control ignored")
        return
    mapping = {
        "normal":       MarketRegime.NORMAL,
        "choppy_flat":  MarketRegime.FLAT,
        "flat":         MarketRegime.FLAT,
        "strong_trend": MarketRegime.TREND,
        "trend":        MarketRegime.TREND,
        "liquidity_shock": MarketRegime.ILLIQUID,
        "illiquid":     MarketRegime.ILLIQUID,
    }
    # duration=0 → действовать до конца эпизода
    self.market_sim.force_market_regime(
        mapping.get(regime, MarketRegime.NORMAL),
        self.state.step_idx if self.state else 0,
        duration,
    8)
metadata = {"render.modes": []}


# ------------------------------------------------ ctor
def __init__(
    self,
    df: pd.DataFrame,
    *,
    seed: int | None = None,
    initial_cash: float = 1_000.0,
    latency_steps: int | None = None,
    slip_k: float | None = None,
    # risk parameters
    max_abs_position: float | None = None,
    max_drawdown_pct: float | None = None,
    bankruptcy_cash_th: float | None = None,
    # regime / shock
    regime_dist: Sequence[float] | None = None,
    enable_shocks: bool = False,
    flash_prob: float = 0.01,
    rng: np.random.Generator | None = None,
    validate_data: bool = False,
    time_provider: TimeProvider | None = None,
    **_: Any,
) -> None:
    super().__init__()

    # deterministic seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    self.seed_value = seed

    # per‑instance RNG
    # для каждой среды используем seed + rank (rank определяется в shared_memory_vec_env.worker)
    rank_offset = getattr(self, "rank", 0)
    base_seed = seed or 0
    self._rng: np.random.Generator = rng or np.random.default_rng(base_seed + rank_offset)
    self.time = time_provider or RealTimeProvider()
    # price data
    self.df = df.reset_index(drop=True).copy()

    # optional strict data validation
    if validate_data or os.getenv("DATA_VALIDATE") == "1":
        try:
            from data_validation import DataValidator
            DataValidator().validate(self.df)
            import time as _time
            publish(Topics.RISK, {
                "step": 0,
                "ts": int(_time.time()),
                "reason": "data_validation_ok",
                "details": {"rows": int(len(self.df))},
            })
        except Exception as e:
            # лог + немедленный fail: некондиционные данные нам не нужны
            import time as _time
            publish(Topics.RISK, {
                "step": 0,
                "ts": int(_time.time()),
                "reason": "data_validation_fail",
                "details": {"error": str(e)},
            })
            raise

    self.initial_cash = float(initial_cash)
    self._max_steps = len(self.df)

    # store config for Mediator
    self.latency_steps = latency_steps or 0
    self.slip_k = slip_k or 0.0
    self.max_abs_position = max_abs_position or 1e12
    self.max_drawdown_pct = max_drawdown_pct or 1.0
    self.bankruptcy_cash_th = bankruptcy_cash_th or -1e12

    # spaces
    self.action_space = spaces.Dict(
        {
            "type": spaces.Discrete(4),
            "price_offset_ticks": spaces.Discrete(201),
            "volume_frac": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            "ttl_steps": spaces.Discrete(33),
        }
    )
    from lob_state_cython import N_FEATURES  # импорт числа признаков
    self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(N_FEATURES+2,), dtype=np.float32
    )

    # Phase 09 regime machinery
    self._init_regime_dist = regime_dist
    self._init_enable_shocks = bool(enable_shocks)
    self._init_flash_prob = float(flash_prob)

    # attach minimal market simulator stub
    if _HAVE_FAST_MARKET:
        # --- allocate contiguous arrays (C‑contiguous float32) ---
        price_arr   = self.df["price"].to_numpy(dtype="float32", copy=True)
        open_arr    = self.df["open" ].to_numpy(dtype="float32", copy=True)
        high_arr    = self.df["high" ].to_numpy(dtype="float32", copy=True)
        low_arr     = self.df["low"  ].to_numpy(dtype="float32", copy=True)
        vol_usd_arr = self.df["quote_asset_volume"].to_numpy(dtype="float32", copy=True)
        # создаём MarketSimulatorWrapper; параметры RSI_WINDOW и др. должны быть определены в вашей конфигурации
        self.market_sim = MarketSimulatorWrapper(
            price_arr, open_arr, high_arr, low_arr, vol_usd_arr,
            RSI_WINDOW, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
            MOMENTUM_WINDOW, CCI_WINDOW, BB_WINDOW, 2.0,
            MA5_WINDOW, MA20_WINDOW, ATR_WINDOW,
            window_size=1, obv_ma_window=OBV_MA_WINDOW
        )
        # создаём генератор микроструктурных событий
        self.flow_gen = CyMicrostructureGenerator(
            momentum_factor=0.3, mean_reversion_factor=0.5,
            adversarial_factor=0.6
        )
        import os
        try:
            # уникальный seed: исходный seed XOR PID
            self.flow_gen.set_seed(int(self.seed_value or 0) ^ os.getpid())
        except Exception:
            pass  # fallback, если метода нет
    else:
        # fallback: используем Python‑stub для совместимости
        from trading_patchnew import _SimpleMarketSim as SMS  # self‑import safe
        self.market_sim = SMS(self._rng)
        self.flow_gen = None
    from trading_patchnew import MarketRegime  # self‑import safe


    # runtime state / orchestrator
    self.state: _EnvState | None = None
    self._mediator = Mediator(self)

# ------------------------------------------------ helpers
def _init_state(self) -> Tuple[np.ndarray, dict]:
    self.state = _EnvState(
        cash=self.initial_cash,
        units=0.0,
        net_worth=self.initial_cash,
        step_idx=0,
        peak_value=self.initial_cash,
        agent_orders=_AgentOrders(),
        max_position=1.0,
        max_position_risk_on=1.0,
    )
    obs = np.zeros(self.observation_space.shape, dtype=np.float32)
    return obs, {}

def _to_proto(self, action) -> ActionProto:
    if isinstance(action, ActionProto):
        return action
    if isinstance(action, dict):
        from domain.adapters import gym_to_action_v1, action_v1_to_proto
        v1 = gym_to_action_v1(action)
        return action_v1_to_proto(v1)
    if isinstance(action, np.ndarray):
        # старые массивы не поддерживаем (используй legacy_bridge.from_legacy_box вручную)
        raise TypeError("NumPy array actions are no longer supported")
    raise TypeError("Unsupported action type")

# ------------------------------------------------ Gym API
def reset(self, *args, **kwargs):
    # инициализируем собственный ГПСЧ для этой среды
    rank_offset = getattr(self, "rank", 0)
    base_seed = self.seed_value or 0
    self._rng = np.random.default_rng(base_seed + rank_offset)
    obs, info = self._init_state()

    # prepare regime & shocks
    p_vec = (
        np.asarray(self._init_regime_dist, dtype=float)
        if self._init_regime_dist is not None
        else np.asarray([0.8, 0.05, 0.10, 0.05], dtype=float)
    )
    self.market_sim.set_regime_distribution(p_vec)
    self.market_sim.enable_random_shocks(self._init_enable_shocks, self._init_flash_prob)
    regime_idx = self._rng.choice(4, p=self.market_sim.regime_distribution)
    self.market_sim.force_market_regime(MarketRegime(regime_idx))

    # mediator internal clear
    self._mediator.reset()
    return obs, info

    # mediator internal clear
    self._mediator.reset()
    return obs, info

def step(self, action):
    # полностью через Mediator: никаких прямых вызовов market_sim здесь
    proto = self._to_proto(action)
    return self._mediator.step(proto)



# ------------------------------------------------ util
def seed(self, seed: int) -> None:  # noqa: D401
    self.__init__(df=self.df, seed=seed, initial_cash=self.initial_cash)
─────────────────────── Simple market‑sim stub (unchanged) ───────────────
class _SimpleMarketSim:
def init(self, rng: np.random.Generator | None = None) -> None:
import numpy as _np

python
Копировать код
    self._rng = rng or _np.random.default_rng()
    self._regime_distribution = _np.array([0.8, 0.05, 0.10, 0.05], dtype=float)
    self._current_regime = MarketRegime(MarketRegime.NORMAL)
    self._shocks_enabled = False
    self._flash_prob = 0.01
    self._fired_steps: set[int] = set()

# --- public API mirror of C++ ------------------------------------------
def set_regime_distribution(self, p_vec: Sequence[float]) -> None:
    p = np.asarray(p_vec, dtype=float)
    if p.shape != (4,):
        raise ValueError("regime_dist must have length 4")
    s = float(p.sum())
    if s <= 0.0:
        raise ValueError("regime_dist must sum to > 0")
    self._regime_distribution = p / s

def enable_random_shocks(self, enabled: bool = True, flash_prob: float = 0.01) -> None:
    self._shocks_enabled = bool(enabled)
    self._flash_prob = float(np.clip(flash_prob, 0.0, 1.0))

def force_market_regime(self, regime: MarketRegime, *_, **__) -> None:
    self._current_regime = MarketRegime(regime)

def shock_triggered(self, step_idx: int) -> float:
    if not self._shocks_enabled or step_idx in self._fired_steps:
        return 0.0
    if self._rng.random() < self._flash_prob:
        self._fired_steps.add(step_idx)
        return 1.0 if self._rng.random() < 0.5 else -1.0
    return 0.0

# helpers for tests
@property
def regime_distribution(self) -> np.ndarray:
    return self._regime_distribution.copy()
all = ["TradingEnv", "MarketRegime"]
